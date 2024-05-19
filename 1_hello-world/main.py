import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader


def main(url: str) -> None:
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is LlamaIndex?")
    print(response)


if __name__ == "__main__":
    # Load the environment variables
    load_dotenv()

    print("Hello world LlamaIndex Course!")
    print("\n*****\n")

    url = "https://www.llamaindex.ai/"
    main(url)
