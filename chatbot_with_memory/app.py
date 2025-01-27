import streamlit as st
import atexit
from bot import DatabaseManager, ModelHandler, ConversationWorkflow, Settings
from langchain_core.messages import HumanMessage
import logging

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
# Initialize components
def initialize_components():
    """Initialize components for the Streamlit app."""
    try:
        settings = Settings()
        
        if "db_manager" not in st.session_state:
            st.session_state.db_manager = DatabaseManager(settings.db_uri, max_size=settings.connection_pool_size)
            st.session_state.db_manager.setup()

        if "model_handler" not in st.session_state:
            st.session_state.model_handler = ModelHandler(model_name="gemini-1.5-flash", api_key=settings.gemini_api_key)

        if "conversation_workflow" not in st.session_state:
            checkpointer = st.session_state.db_manager.get_checkpointer()
            st.session_state.conversation_workflow = ConversationWorkflow(st.session_state.model_handler, checkpointer)
        logger.info("Streamlit components successfully initialized.")
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        st.error("Failed to initialize components. Check logs for details.")
        raise

# Close resources on app shutdown
def close_resources():
    """Release resources on app shutdown."""
    try:
        if "db_manager" in st.session_state and st.session_state.db_manager:
            st.session_state.db_manager.teardown()
    except Exception as e:
        logger.error(f"Error during app shutdown: {e}")

atexit.register(close_resources)

def main():
    """Main function for the Streamlit app."""
    st.title("AI-Powered Conversation Workflow")
    
    initialize_components()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_area("Enter your message:", placeholder="Type your message here...")
    thread_id = st.text_input("Thread ID:", value="1")

    if st.button("Send"):
        if user_input.strip():
            try:
                workflow = st.session_state.conversation_workflow
                input_message = HumanMessage(content=user_input)
                config = {"configurable": {"thread_id": thread_id}}
                output = workflow.start_conversation(input_message, config)
                st.session_state.messages.extend(output["messages"])
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                st.error("Error processing your message. Check logs for details.")

    st.subheader("Conversation")
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            st.markdown(f"**User:** {message.content}")
        else:
            st.markdown(f"**AI:** {message.content}")

if __name__ == "__main__":
    main()