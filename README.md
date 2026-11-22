# Agentic Chatbot with Human-in-the-Loop (LangGraph)

An agentic chatbot built with LangGraph and Streamlit that supports tool calling, RAG over PDF documents, and human-in-the-loop approval workflows. The chatbot uses Google Gemini as its language model and persists conversations with SQLite checkpointing.

---

## Features

- **Agentic tool use** -- the LLM autonomously selects from multiple tools based on the user query.
- **Human-in-the-loop (HITL)** -- critical actions such as stock purchases pause the graph and wait for explicit human approval before proceeding.
- **RAG over PDFs** -- upload a PDF through the sidebar; the document is chunked, embedded with Google Generative AI Embeddings, and stored in a FAISS vector store for retrieval.
- **Web search** -- real-time web search powered by Tavily.
- **Calculator** -- evaluates math expressions safely.
- **Stock price lookup** -- fetches the latest quote from Alpha Vantage.
- **Weather reports** -- retrieves current weather data from OpenWeatherMap.
- **Multi-thread conversations** -- create new chat threads and switch between them; all threads are persisted via LangGraph's SQLite checkpointer.
- **Streaming responses** -- AI responses are streamed token-by-token with live tool-status indicators.

---

## Project Structure

```
Agentic-Chatbot-Langraph/
|-- agentic_chatbot_hitl_backend.py   # LangGraph graph, tools, and RAG logic
|-- app_hitl.py                       # Streamlit frontend with HITL support
|-- requirements.txt                  # Python dependencies
|-- .env                              # API keys (not committed)
|-- .gitignore
|-- LICENSE                           # Apache 2.0
|-- README.md
```

---

## Architecture

```
User input
    |
    v
[ chat_node ] -- LLM decides whether to answer or call a tool
    |
    |-- (direct answer) --> END
    |
    |-- (tool call) ------> [ tools node ]
                                 |
                                 |-- search_tool (Tavily)
                                 |-- calculator
                                 |-- get_stock_price (Alpha Vantage)
                                 |-- get_current_weather (OpenWeatherMap)
                                 |-- rag_tool (FAISS retriever)
                                 |-- purchase_stock (HITL interrupt)
                                 |
                                 v
                           [ chat_node ]  <-- loop back for final answer
```

The graph is compiled with a SQLite checkpointer, enabling persistent multi-turn conversations and thread management.

---

## Prerequisites

- Python 3.10 or higher
- API keys for the services listed below

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/Agentic-Chatbot-Langraph.git
   cd Agentic-Chatbot-Langraph
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root with the following keys:

   ```
   GEMINI_API_KEY=your-gemini-api-key
   TAVILY_API_KEY=your-tavily-api-key
   OPENWEATHER_API_KEY=your-openweather-api-key
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   LANGSMITH_API_KEY=your-langsmith-api-key
   LANGSMITH_PROJECT=your-langsmith-project-name
   ```

   | Variable | Description |
   |---|---|
   | `GEMINI_API_KEY` | Google Gemini API key (required) |
   | `TAVILY_API_KEY` | Tavily search API key (required for web search) |
   | `OPENWEATHER_API_KEY` | OpenWeatherMap API key (required for weather tool) |
   | `LANGSMITH_API_KEY` | LangSmith API key (optional, for tracing) |

---

## Usage

Run the Streamlit app:

```bash
streamlit run app_hitl.py
```

The application will open in your default browser. From there you can:

- Type a question in the chat input to interact with the chatbot.
- Upload a PDF via the sidebar to enable RAG-based question answering.
- Start new conversations or switch between existing threads using the sidebar.
- When the chatbot requests a stock purchase, an approval prompt will appear; click Approve or Reject to continue.

---

## Tools Reference

| Tool | Description |
|---|---|
| `search_tool` | Tavily web search with advanced depth, returning up to 5 results |
| `calculator` | Evaluates math expressions using a restricted eval with math module access |
| `get_stock_price` | Fetches latest stock quote from Alpha Vantage |
| `get_current_weather` | Returns current weather for a city using OpenWeatherMap |
| `rag_tool` | Retrieves relevant chunks from an uploaded PDF via FAISS similarity search |
| `purchase_stock` | Simulates a stock purchase with HITL interrupt for human approval |

---

## Human-in-the-Loop Workflow

1. The user asks to purchase a stock (for example, "Buy 10 shares of AAPL").
2. The LLM invokes the `purchase_stock` tool.
3. The tool calls `interrupt()`, which pauses the LangGraph execution.
4. The Streamlit UI displays an approval prompt with Approve and Reject buttons.
5. The user clicks one of the buttons.
6. The graph resumes with the human decision, and the tool returns a success or cancellation result.
7. The LLM generates a final response based on the outcome.

---

## Tech Stack

- **LangGraph** -- stateful agent graph with conditional edges and checkpointing
- **LangChain** -- tool abstraction, embeddings, document loaders, text splitters
- **Google Gemini** -- language model (gemini-3.5-flash-lite)
- **Streamlit** -- web-based chat interface
- **FAISS** -- vector similarity search for RAG
- **SQLite** -- conversation persistence via LangGraph checkpointer
- **Tavily** -- web search API
- **Alpha Vantage** -- stock market data API
- **OpenWeatherMap** -- weather data API

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
