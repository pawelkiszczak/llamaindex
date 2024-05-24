import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

# Load env variables
load_dotenv()

# Setup LlamaIndex
llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])



def write_haiku(topic: str) -> str:
    """
    Write a haiku about given topic.
    """
    return llm.complete(f"Write me a haiku about {topic}")
    #return response['choices'][0]['text'].strip()

def count_characters(text: str) -> int:
    """
    Counts the number of characters in a text.
    """
    return len(text)

if __name__ == "__main__":
    print("*** Hello LlamaIndex Agents ***")

    # Give tools to the agent
    tool1 = FunctionTool.from_defaults(fn=write_haiku)
    tool2 = FunctionTool.from_defaults(fn=count_characters)

    agent = ReActAgent.from_tools(tools=[tool1, tool2], llm=llm, verbose=True)

    res = agent.query("Write me a haiku about tennis and then count the characters in it")
    print(res)