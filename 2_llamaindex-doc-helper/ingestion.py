from dotenv import load_dotenv
import os

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms import openai
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (download_loader, ServiceContext, 
                              VectorStoreIndex, StorageContext)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

#CONFIG
# Load environmental variables
load_dotenv()

# Setup Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
spec = ServerlessSpec(cloud=os.environ['PINECONE_CLOUD'],
                      region=os.environ['PINECONE_ENVIRONMENT'])

if __name__ == "__main__":

    print("Ingesting LlamaIndex documentation into Pinecone...")
