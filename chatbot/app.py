import streamlit as st

import argparse
import sys
import time
from pathlib import Path

from bot.client.llama_cpp_client import LlamaCppClient
from bot.conversation.conversation_retrieval import ConversationRetrieval
from bot.conversation.ctx_strategy import CreateAndRefineStrategy
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from bot.model.model_registry import get_model_settings, get_models

from helpers.log import get_logger
from helpers.prettier import prettify_source


logger = get_logger(__name__)


@st.cache_resource()
def load_llm_client(model_folder: Path, model_name: str):
    model_settings = get_model_settings(model_name)
    return LlamaCppClient(model_folder=model_folder, model_settings=model_settings)


@st.cache_resource()
def load_conversational_retrieval(_llm: LlamaCppClient):
    return ConversationRetrieval(llm=_llm)


@st.cache_resource()
def load_ctx_synthesis_strategy(_llm: LlamaCppClient):
    return CreateAndRefineStrategy(llm=_llm)

# Loading the Chroma database
@st.cache_resource()
def load_index(vector_store_path: Path):
    embedding = Embedder()
    return Chroma(persist_directory=str(vector_store_path), embedding=embedding)


def init_page():
    st.set_page_config(page_title="RAG-based Chatbot")
    st.sidebar.title("Options")

@st.cache_resource()
def init_welcome_message():
    with st.chat_message("assistant"):
        st.write("How can I help you today?")

def init_chat_history(conversational_retrieval: ConversationRetrieval):
    clear_bt = st.sidebar.button("Clear chat history", key="clear")
    if clear_bt or "messages" not in st.session_state:
        st.session_state.messages = []
        conversational_retrieval.get_chat_history().clear()

def display_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def main(parameters):
    """
    Main function for the RAG-based Chatbot Streamlit app.
    """
    root_folder = Path(__file__).resolve().parent.parent
    model_folder = root_folder / "models"
    vector_store_path = root_folder / "vector_store" / "docs_index"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    model_name = parameters.model
    init_page()
    llm = load_llm_client(model_folder, model_name)
    conversational_retrieval = load_conversational_retrieval(llm)
    ctx_synthesis_strategy = load_ctx_synthesis_strategy(llm)
    index = load_index(vector_store_path)
    init_chat_history(conversational_retrieval)
    init_welcome_message()
    display_messages()

    if user_input := st.chat_input("Input your question:"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner(text="Refining the question and retrieving relevant text chunks..."):
                refined_user_input = conversational_retrieval.refine_question(user_input)
                retrived_contents, sources = index.similarity_search_w_threshold(query=refined_user_input, k=parameters.k)
                if retrived_contents:
                    full_response += "Here are the retrieved text chunks with a content preview: \n\n"
                    message_placeholder.markdown(full_response)
                    
                    for source in sources:
                        full_response += prettify_source(source)
                        full_response += "\n\n"
                        message_placeholder.markdown(full_response)

                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    full_response += "I did not detect any relevant chunks of text from the documents. \n\n"
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        start_time = time.time()
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner(text="Refining the context and generating an answer..."):
                streamer, fmt_prompts = conversational_retrieval.context_aware_answer(
                    ctx_synthesis_strategy=ctx_synthesis_strategy, question=user_input, retrived_contents=retrived_contents
                )
                for token in streamer:
                    full_response += llm.parse_token(token)
                    message_placeholder.markdown(full_response + "|||")
                
                message_placeholder.markdown(full_response)
                conversational_retrieval.update_chat_history(user_input, full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        elapsed_time = time.time() - start_time
        logger.info(f"\n--- Took {elapsed_time:.2f} seconds ---")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG-based Chatbot.")
    
    model_list = get_models()
    default_model = model_list[0]

    parser.add_argument(
        "--model", type=str, choices=model_list,
        help="The model to use for the chatbot.",
        required=False, const=default_model, nargs="?",
        default=default_model
    )

    parser.add_argument(
        "--k", type=int, default=3,
        help="The number of chunks to return from the similarity search.",
        required=False
    )

    return parser.parse_args()


# Running the app
if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True, stack_info=True)
        sys.exit(1)
