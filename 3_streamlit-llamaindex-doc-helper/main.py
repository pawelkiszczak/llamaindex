import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
import logging

import streamlit as st

from llama_index.core import VectorStoreIndex, Settings

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.postprocessor.optimizer import SentenceEmbeddingOptimizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from node_postprocessors.duplicate_postprocessor import (
    DuplicateRemoverNodePostprocessor,
)


# Load environemtal variables
load_dotenv()
logger = logging.basicConfig(level="INFO", force=True)

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

# Add chat_engine to the st.session_state
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

if "chat_engine" not in st.session_state:
    postprocessor = SentenceEmbeddingOptimizer(
        embed_model=embed_model,
        percentile_cutoff=0.2,
        threshold_cutoff=0.7,
        context_after=0,
        context_before=0,
    )
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="best",
        verbose=True,
        node_postprocessors=[postprocessor, DuplicateRemoverNodePostprocessor()],
    )

# Add messages to the st.session_state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about LlamaIndex!"}
    ]

# User input
if prompt := st.chat_input("Your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Go through messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check for the last message in the session_state
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.stream_chat(message=prompt)

            # Nodes to retreive context found
            nodes = [node for node in response.source_nodes]

            with st.expander("Check context"):
                for idx, node in enumerate(nodes):
                    st.text_area(
                        label=f"Context {idx} - {node.node_id} - {node.score:.2f}",
                        value=node.text,
                        max_chars=5000,
                    )

            # with st.expander("Check context - column version"):
            #     for col, node in zip(st.columns(len(nodes)), nodes):
            #         with col:
            #             st.header(
            #                 f"Source node: {node.node_id} | Score: {node.score:.2f}"
            #             )
            #             st.write(node.text)

            st.write_stream(response.response_gen)

            message = {"role": "assistant", "content": response.response}

            st.session_state.messages.append(message)
