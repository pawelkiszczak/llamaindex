from dotenv import load_dotenv
import os

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.readers.file import UnstructuredReader
from llama_index.core import Settings
from pinecone import Pinecone, ServerlessSpec

import nltk

# CONFIG
# Load environmental variables
load_dotenv()

# Setup Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
spec = ServerlessSpec(
    cloud=os.environ["PINECONE_CLOUD"], region=os.environ["PINECONE_ENVIRONMENT"]
)

# Download nltk package for UnstructuredReader
nltk.download("averaged_perceptron_tagger")


if __name__ == "__main__":

    print("Ingesting LlamaIndex documentation into Pinecone...")

    # Setup UnstructuredReader to read .html files
    parser = UnstructuredReader()
    file_extractor = {".html": parser}
    data_dir = "llamaindex-docs"
    input_dir = os.path.join(os.path.dirname(__file__), data_dir)

    dir_reader = SimpleDirectoryReader(
        input_dir=input_dir, file_extractor=file_extractor
    )

    # Load in all of the documents
    documents = dir_reader.load_data()

    # Parse the documents
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)

    # Split text and get nodes
    nodes = node_parser.get_nodes_from_documents(documents=documents)

    # Setup the LLM
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

    # Setup embeddings
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)

    # Setup the service context - DEPRECATED
    # service_context = ServiceContext.from_defaults(
    #     llm=llm, embed_model=embed_model, node_parser=node_parser
    # )

    # From 0.10 onwards
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    ### INITIALIZE VECTORE STORE OBJECTS
    index_name = os.environ["PINECONE_INDEX_NAME"]

    # Use this if index on Pinecone is not already created manually
    # pc.create_index(
    #     name=index_name, dimension=1536, metric="cosine", spec=spec
    # )
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Actual work
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=Settings,
        show_progress=True,
    )

    print("Ingestion finished")

    pass
