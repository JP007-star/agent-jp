# 🤖 Agent JP (ReAct Pattern)

This project is a professional implementation of a basic AI Agent. Unlike a standard chatbot, this agent can "think" and use external tools to find real-time information.

## 🧠 Concept: What is a ReAct Agent?
This agent follows the **Reason + Act** framework:
1. **Thought**: The AI analyzes the prompt and plans a strategy.
2. **Action**: The AI selects a tool (e.g., Internet Search).
3. **Observation**: The AI reads the result of that tool.
4. **Conclusion**: The AI combines everything into a final answer.

## 🛠️ Setup Guide

### 1. Prerequisites
- Install [Python 3.10+](https://www.python.org/downloads/)
- Create a free account at [Groq Cloud](https://console.groq.com/) to get your API Key.

### 2. Installation
```bash
# Clone this repository
git clone https://github.com/your-username/agent-jp.git
cd agent-jp


# Create a virtual environment named 'venv'
python -m venv venv

# Activate the environment
# (On Windows)
.\jp\Scripts\activate

# Install the required libraries
pip install -r requirements.txt
```

### 3. Configuration
1. Rename `.env.example` to `.env`.
2. Open `.env` and replace `your_api_key_here` with your actual Groq API key.

### 4. Running the Agent
```bash
python main.py
```

## 📈 Example Prompts to Try
- *"Who is the current CEO of Nvidia and what is their latest product?"*
- *"Check the current price of Bitcoin and tell me if it went up or down today."*
- *"What is the weather in Tokyo right now?"*
