from backend import chatbot, get_all_threads, ingest_rag_document
from langgraph.types import Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
import streamlit as st
import uuid
import os
import tempfile

def extract_text(content):
    """Return plain text from an AIMessage/HumanMessage content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""

#generate a unique thread
def generate_thread_id():
    return str(uuid.uuid4())

#add thread id to the conversation list
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


#create a completely new chat conversation

def reset_chat():
    #generate and assign a new thread id
    st.session_state["thread_id"] = generate_thread_id()

    #clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # Clear any pending interrupt
    st.session_state["pending_interrupt"] = None

    # Add the new thread to the conversation list
    add_thread(st.session_state["thread_id"])

# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])


def get_config():
    """Return the LangGraph config dict for the current thread."""
    return {
        "configurable": {
            "thread_id": st.session_state["thread_id"],
            "metadata": {"thread_id": st.session_state["thread_id"]},
            "run_name": "chat_trace",
        }
    }


def check_for_interrupt():
    """Check if the graph is paused on an interrupt and return the prompt string (or None)."""
    config = get_config()
    state = chatbot.get_state(config)

    # state.tasks is a tuple of PregelTask; each has an .interrupts list
    if state.tasks:
        for task in state.tasks:
            if task.interrupts:
                # Return the value passed to interrupt() — e.g. "Approve buying 10 shares of AAPL? (yes/no)"
                return task.interrupts[0].value
    return None


def stream_and_collect(input_data, config):
    """
    Stream the chatbot, display tool status, capture RAG sources,
    and yield AI text tokens. Returns nothing — the caller uses
    st.write_stream on this generator.
    """
    status_holder = {"box": None}
    sources_holder = {"sources": []}

    # Store holders on the function so the caller can access them after streaming
    stream_and_collect.status_holder = status_holder
    stream_and_collect.sources_holder = sources_holder

    for message_chunk, metadata in chatbot.stream(
        input_data,
        config=config,
        stream_mode="messages"
    ):
        if isinstance(message_chunk, ToolMessage):
            tool_name = getattr(message_chunk, "name", "tool")
            tool_content = extract_text(message_chunk.content)

            if status_holder["box"] is None:
                status_holder["box"] = st.status(
                    f"🔧 Using {tool_name}",
                    state="running",
                    expanded=True
                )
            else:
                status_holder['box'].update(
                    label=f"🔧 Using {tool_name}",
                    state="running",
                    expanded=True
                )

            # Show tool output inside the status box
            if tool_content:
                status_holder["box"].markdown(tool_content)

            # Capture RAG sources to display after the response
            if tool_name == "rag_tool" and tool_content:
                sources_holder["sources"].append(tool_content)

        if isinstance(message_chunk, AIMessage) and message_chunk.text:
            yield message_chunk.text


def finalize_stream():
    """Close tool status box and show RAG sources expander if any."""
    status_holder = stream_and_collect.status_holder
    sources_holder = stream_and_collect.sources_holder

    if status_holder["box"] is not None:
        status_holder["box"].update(
            label="✅ Tool finished", state="complete", expanded=False
        )

    if sources_holder["sources"]:
        with st.expander("📚 Sources & References", expanded=False):
            for source_text in sources_holder["sources"]:
                st.markdown(source_text)



# ========================= Page Setup =========================

st.title("🤖 Agentic HITL Chatbot")

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()

if "thread_id" not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if "pdf_ingested" not in st.session_state:
    st.session_state["pdf_ingested"] = os.path.exists("faiss_db")

if "pending_interrupt" not in st.session_state:
    st.session_state["pending_interrupt"] = None


# ========================= Sidebar =========================

st.sidebar.title("My Conversations")

#create a button for starting a new conversation
if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()


# ========================= PDF Upload Section =========================

st.sidebar.divider()
st.sidebar.subheader("📄 Upload PDF for RAG")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    key="pdf_uploader"
)

if uploaded_file is not None:
    if st.session_state.get("last_uploaded_file") != uploaded_file.name:
        with st.sidebar:
            with st.spinner("Processing PDF..."):
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, uploaded_file.name)

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                ingest_rag_document(temp_path)

                os.remove(temp_path)
                os.rmdir(temp_dir)

                st.session_state["pdf_ingested"] = True
                st.session_state["last_uploaded_file"] = uploaded_file.name

            st.success(f"✅ '{uploaded_file.name}' ingested successfully!")

if st.session_state["pdf_ingested"]:
    st.sidebar.info("📚 PDF loaded — you can ask questions about it.")
else:
    st.sidebar.warning("⚠️ No PDF uploaded yet. Upload one to use RAG.")

st.sidebar.divider()

# ========================= Thread List =========================

for thread_id in st.session_state["chat_threads"][::-1]:

    if st.sidebar.button(
        str(thread_id),
        key=thread_id
    ):
        st.session_state["thread_id"] = thread_id
        st.session_state["pending_interrupt"] = None

        messages = load_conversation(thread_id)

        temp_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
                content = extract_text(message.content)

            elif isinstance(message, AIMessage):
                role = "assistant"
                content = extract_text(message.content)

            else:
                continue

            if not content.strip():
                continue

            temp_messages.append({
                "role": role,
                "content": content
            })

        st.session_state["message_history"] = temp_messages

        # Check if this thread has a pending interrupt
        interrupt_value = check_for_interrupt()
        if interrupt_value:
            st.session_state["pending_interrupt"] = interrupt_value

        st.rerun()



# ========================= Main chat interface =========================

# Display all messages from the currently selected conversation
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# ========================= HITL Interrupt Handling =========================

# If the graph is paused waiting for human approval, show the approval UI
if st.session_state["pending_interrupt"] is not None:
    interrupt_prompt = st.session_state["pending_interrupt"]

    st.warning(f"⏸️ **Human approval required:** {interrupt_prompt}")

    col1, col2 = st.columns(2)

    with col1:
        approve = st.button("✅ Approve", key="hitl_approve", type="primary")
    with col2:
        reject = st.button("❌ Reject", key="hitl_reject")

    if approve or reject:
        decision = "yes" if approve else "no"

        # Clear the interrupt before resuming
        st.session_state["pending_interrupt"] = None

        config = get_config()

        # Resume the graph with the human decision
        with st.chat_message("assistant"):
            ai_message = st.write_stream(
                stream_and_collect(Command(resume=decision), config)
            )
            finalize_stream()

        if ai_message:
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": ai_message
            })

        st.rerun()



# ========================= Chat Input =========================

user_input = st.chat_input("Type here")

if user_input:

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    config = get_config()

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            stream_and_collect(
                {"messages": [HumanMessage(content=user_input)]},
                config
            )
        )
        finalize_stream()

    # After streaming, check if the graph hit an interrupt
    interrupt_value = check_for_interrupt()

    if interrupt_value:
        # Graph is paused — save state and rerun to show the approval UI
        st.session_state["pending_interrupt"] = interrupt_value

        # If there's an AI response before the interrupt, save it
        if ai_message:
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": ai_message
            })

        st.rerun()

    else:
        # Normal flow — no interrupt, save the AI response
        if ai_message:
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": ai_message
            })
