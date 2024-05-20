import os
from dotenv import load_dotenv
from pinecone import Pinecone

import streamlit as st
import time

from llama_index.core import VectorStoreIndex, Settings

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.vector_stores.pinecone import PineconeVectorStore

# Load environemtal variables
load_dotenv()

# Setup Streamlit app view
st.set_page_config(
    page_title="Chat with LlamaIndex Docs, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Chat with LlamaIndex docs 🦙")

# Methods and functions
@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Setup Pinecone index and vector_store
    pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Setup callback manager
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    Settings.callback_manager = callback_manager

    return index

index = get_index()

#####

if 'chat_engine' not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(chat_mode='context', verbose=True)

if 'messages' not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about LlamaIndex!"
        }
    ]

if prompt := st.chat_input("Your question here..."):
    st.session_state.messages.append({
        'role': 'user',
        'content': prompt
    })

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if st.session_state.messages[-1]['role'] != 'assistant':
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            response = st.session_state.chat_engine.chat(message=prompt)
            st.markdown(response.response)

            message = {
                'role': 'assistant',
                'content': response.response
            }

            st.session_state.messages.append(message)