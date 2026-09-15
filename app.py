import os
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Disable LangSmith tracing to avoid warnings
os.environ["LANGCHAIN_TRACING_V2"] = "false"
load_dotenv()

st.set_page_config(page_title="JP AI Agent", page_icon="🤖")

# --- Core Agent Logic ---

def get_available_models():
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
    except Exception:
        return []

def get_stable_model():
    available_models = get_available_models()
    for model_name in available_models:
        name = model_name.lower()
        if (("llama" in name or "mixtral" in name or "gemma" in name or "qwen" in name or "gpt" in name)
            and not any(x in name for x in ["guard", "classification", "canopy", "orpheus", "whisper"])):
            return model_name
    return available_models[0] if available_models else "llama-3.1-8b-instant"

def clean_search(query: str) -> str:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No results found."
            return "\n".join([f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}\n---" for r in results])
    except Exception as e:
        return f"Search failed: {e}"

@st.cache_resource
def init_agent():
    model_name = get_stable_model()
    llm = ChatGroq(
        model=model_name,
        temperature=0,
        max_tokens=512,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    tools = [
        Tool(
            name="Internet_Search",
            func=clean_search,
            description="Use this tool when you need to find current information, news, or facts from the internet.",
        )
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are JP, a helpful AI agent with access to the internet."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=5), model_name

# --- Streamlit UI ---

st.title("🤖 JP AI Agent")
st.markdown("Your internet-powered assistant.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize agent
try:
    agent_executor, model_name = init_agent()
    st.sidebar.info(f"🚀 Active Model: {model_name}")
except Exception as e:
    st.error(f"Failed to initialize agent: {e}")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What should JP do?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare chat history for the agent
                history = []
                for m in st.session_state.messages[:-1]:
                    # Convert to simplified format for agent history if needed,
                    # but create_tool_calling_agent often handles a simple list.
                    pass

                result = agent_executor.invoke({"input": prompt})
                response = result.get("output", str(result))
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
