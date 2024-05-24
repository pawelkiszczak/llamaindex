import os
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.agent.openai import OpenAIAgent
import subprocess

# Load env variables
load_dotenv()

# Setup LlamaIndex
llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def write_haiku(topic: str) -> str:
    """
    Write a haiku about given topic.
    """
    return llm.complete(f"Write me a haiku about {topic}")
    # return response['choices'][0]['text'].strip()


def count_characters(text: str) -> int:
    """
    Counts the number of characters in a text.
    """
    return len(text)


def open_application(application_name: str) -> str:
    """
    Opens an application in my computer
    """
    try:
        subprocess.Popen(["/usr/bin/open", "-n", "-a", application_name])
        return f"Successfully opened {application_name}"
    except Exception as e:
        print(f"Error: {e}")


def open_url(url: str) -> str:
    """
    Opens a URL in browser (Chrome / Safari / Firefox)
    """
    try:
        subprocess.Popen(["/usr/bin/open", "--url", url])
        return f"Successfully opened {url}"
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("*** Hello LlamaIndex Agents ***")

    # Give tools to the agent
    tool1 = FunctionTool.from_defaults(fn=write_haiku)
    tool2 = FunctionTool.from_defaults(fn=count_characters)
    tool3 = FunctionTool.from_defaults(fn=open_application)
    tool4 = FunctionTool.from_defaults(fn=open_url)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])

    # agent = ReActAgent.from_tools(tools=[tool1, tool2, tool3, tool4], llm=llm, verbose=True, callback_manager=callback_manager)
    agent = OpenAIAgent.from_tools(
        tools=[tool1, tool2, tool3, tool4],
        llm=llm,
        verbose=True,
        callback_manager=callback_manager,
    )

    res = agent.query(
        "Write me a haiku about tennis and then count the characters in it"
    )
    # res = agent.query("Open Discord in my computer")
    res = agent.query("Open URL: https://www.x-kom.pl in Safari")
    print(res)