import os
import logging
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph import END, MessagesState, StateGraph, START
from langgraph.graph.state import CompiledStateGraph

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Settings:
    """Settings class to manage environment variables."""
    def __init__(self):
        self.db_uri = os.getenv("DB_URI")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.connection_pool_size = int(os.getenv("CONNECTION_POOL_SIZE", 20))

        if not self.db_uri or not self.gemini_api_key:
            logger.error("Missing required environment variables: DB_URI or GEMINI_API_KEY")
            raise ValueError("Missing required environment variables.")

class DatabaseManager:
    """Manages the PostgreSQL connection pool and checkpointer."""
    def __init__(self, db_uri: str, max_size: int = 20):
        self.db_uri = db_uri
        self.max_size = max_size
        self.connection_kwargs = {"autocommit": True, "prepare_threshold": 0}
        self.pool = None
        self.checkpointer = None

    def setup(self):
        """Initialize the connection pool and checkpointer."""
        try:
            if not self.pool:
                self.pool = ConnectionPool(conninfo=self.db_uri, max_size=self.max_size, kwargs=self.connection_kwargs)
                self.checkpointer = PostgresSaver(self.pool)
                self.checkpointer.setup()
            logger.info("DatabaseManager successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize DatabaseManager: {e}")
            raise

    def teardown(self):
        """Close the connection pool."""
        try:
            if self.pool:
                self.pool.close()
                self.pool = None
                logger.info("DatabaseManager resources released.")
        except Exception as e:
            logger.error(f"Error during DatabaseManager teardown: {e}")

    def get_checkpointer(self):
        """Retrieve the checkpointer instance."""
        if not self.checkpointer:
            raise RuntimeError("DatabaseManager is not set up. Call setup() first.")
        return self.checkpointer

class ModelHandler:
    """Handles interactions with the ChatGoogleGenerativeAI model."""
    def __init__(self, model_name: str, api_key: str):
        self.model = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)

    def call_model(self, messages: list) -> list:
        """Invoke the model with the given messages."""
        try:
            return self.model.invoke(messages)
        except Exception as e:
            logger.error(f"Error invoking model: {e}")
            raise

class ConversationWorkflow:
    """Defines the workflow for managing conversations and their states."""
    class State(MessagesState):
        summary: str

    def __init__(self, model_handler: ModelHandler, checkpointer: PostgresSaver):
        self.model_handler = model_handler
        self.graph = self._build_workflow(checkpointer)

    def _call_model(self, state: State) -> State:
        summary = state.get("summary", "")
        messages = [SystemMessage(content=f"Summary of conversation earlier: {summary}")] + state["messages"] if summary else state["messages"]
        response = self.model_handler.call_model(messages)
        return {"messages": response}

    def _summarize_conversation(self, state: State) -> State:
        summary = state.get("summary", "")
        summary_message = (
            f"This is a summary of the conversation to date: {summary}\n\nExtend the summary by taking into account the new messages above:"
            if summary else
            "Create a summary of the conversation above:"
        )
        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = self.model_handler.call_model(messages)
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}

    def _should_continue(self, state: State) -> str:
        return "summarize_conversation" if len(state["messages"]) > 6 else END

    def _build_workflow(self, checkpointer: PostgresSaver) -> CompiledStateGraph:
        workflow = StateGraph(self.State)
        workflow.add_node("conversation", self._call_model)
        workflow.add_node("summarize_conversation", self._summarize_conversation)
        workflow.add_edge(START, "conversation")
        workflow.add_conditional_edges("conversation", self._should_continue)
        workflow.add_edge("summarize_conversation", END)
        return workflow.compile(checkpointer=checkpointer)

    def start_conversation(self, input_message: HumanMessage, config: dict) -> dict:
        return self.graph.invoke({"messages": [input_message]}, config)