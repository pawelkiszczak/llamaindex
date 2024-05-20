import os
from dotenv import load_dotenv
from pinecone import Pinecone

from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.vector_stores.pinecone import PineconeVectorStore

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if __name__ == "__main__":
    print("RAG")

    # Setup Pinecone index and vector_store
    pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Setup callback manager
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    Settings.callback_manager = callback_manager

    # Setup OpenAI
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)

    # Sample query
    query = "What is LlamaIndex Query Engine?"

    # Setup query_engine
    query_engine = index.as_query_engine()
    
    # Get the response
    response = query_engine.query(query)

    # Print the response
    print(response)

    pass
