import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")

import requests
from dotenv import load_dotenv

try:
    from langchain import hub
except ImportError:
    hub = None

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool
from langchain_groq import ChatGroq


load_dotenv()


def get_available_models():
    """Calls the Groq API directly to see which models the user has access to."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return []

    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json().get("data", [])
        return [m["id"] for m in data if isinstance(m, dict) and "id" in m]
    except Exception as e:
        print(f"❌ Could not fetch available models: {e}")
        return []


def get_stable_model():
    """Finds the first available chat model from Groq."""
    available_models = get_available_models()

    for model_name in available_models:
        name = model_name.lower()
        if (
            ("llama" in name or "mixtral" in name or "gemma" in name or "qwen" in name or "gpt" in name)
            and "guard" not in name
            and "classification" not in name
            and "canopy" not in name
            and "orpheus" not in name
            and "whisper" not in name
        ):
            return model_name

    return available_models[0] if available_models else "llama-3.1-8b-instant"


def run_ai_agent():
    model_name = get_stable_model()
    print(f"🚀 Using Stable Model: {model_name}")

    llm = None
    try:
        llm = ChatGroq(
            model=model_name,
            temperature=0,
            max_tokens=512,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        llm.invoke("hi")
    except Exception as e:
        print(f"\n⚠️  Model {model_name} is not available. Searching for a valid Chat model...")
        available_models = get_available_models()
        print(f"❌ LLM Init Error: {e}")

        if not available_models:
            print("❌ No available models found. Please check your API key at console.groq.com")
            return

        chat_models = [
            m
            for m in available_models
            if (
                ("llama" in m.lower() or "mixtral" in m.lower() or "gemma" in m.lower())
                and "guard" not in m.lower()
                and "classification" not in m.lower()
                and "canopy" not in m.lower()
                and "orpheus" not in m.lower()
            )
        ]

        if not chat_models:
            chat_models = [
                m for m in available_models if "guard" not in m.lower() and "canopy" not in m.lower()
            ]

        if not chat_models:
            print("❌ No suitable model found.")
            return

        model_name = chat_models[0]
        print(f"✅ Selected Chat Model: {model_name}")
        llm = ChatGroq(
            model=model_name,
            temperature=0,
            max_tokens=512,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

    def clean_search(query: str) -> str:
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                if not results:
                    return "No results found."

                formatted_results = []
                for r in results:
                    formatted_results.append(f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}\n---")

                return "\n".join(formatted_results)
        except Exception as e:
            return f"Search failed: {e}"

    tools = [
        Tool(
            name="Internet_Search",
            func=clean_search,
            description="Use this tool when you need to find current information, news, or facts from the internet.",
        )
    ]

    if hub is not None:
        try:
            prompt = hub.pull("hwchase17/openai-functions-agent")
        except Exception as e:
            print(f"❌ Error pulling prompt: {e}")
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are JP, a helpful AI agent with access to the internet."),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
    else:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are JP, a helpful AI agent with access to the internet."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        print("ℹ️ LangChain hub is unavailable in this environment; using built-in agent prompt.")

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    print(f"\n--- 🤖 JP Agent active (Model: {model_name}) ---")
    print("Type 'exit' to stop.\n")

    while True:
        user_input = input("What should the agent do? ⮕ ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Shutting down agent.")
            break

        try:
            result = agent_executor.invoke({"input": user_input})
            print(f"\n🤖 AGENT RESPONSE:\n{result.get('output', str(result))}\n")
            print("-" * 50)
        except Exception as e:
            print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    run_ai_agent()
