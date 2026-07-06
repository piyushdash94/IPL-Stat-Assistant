import streamlit as st
import json
import datetime

from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import  messages_to_dict
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from load_and_clean_df import load_and_clean_df as _load_and_clean_df



@st.cache_data()
def load_and_clean_df():
    return _load_and_clean_df()

def add_new_chat():
    new_chat_id = len(st.session_state.chat_history_list) + 1
    new_chat = f"chat_{new_chat_id}"
    st.session_state.chat_history_list.append(new_chat)
    st.session_state.chat_history_selector = new_chat
    new_chat_history = StreamlitChatMessageHistory(key=new_chat)
    new_chat_history.add_ai_message("Hello! I am your IPL Stat Assistant. How can I help you today?")

def clear_chat_history():
    present_chat = st.session_state.chat_history_selector
    chat_history_db = StreamlitChatMessageHistory(key = present_chat)
    if len(chat_history_db.messages) > 1:
        chat_history_db.clear()
        chat_history_db.add_ai_message("Chat cleared. Ask me anything about IPL Stats!")




if "chat_history_list" not in st.session_state:
    st.session_state.chat_history_list = ["chat_1"]
    st.session_state.chat_history_selector = "chat_1"

st.session_state["active_chat_history"] = StreamlitChatMessageHistory(key=st.session_state.chat_history_selector)
if len(st.session_state["active_chat_history"].messages) == 0:
    st.session_state["active_chat_history"].add_ai_message("Hello! I am your IPL Stat Assistant. How can I help you today?")


st.set_page_config(
    page_title="IPL Stat Assistant",
    page_icon="🏏",
    layout="wide",
    menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
)

st.markdown("<div style='text-align: center; color: lightblue;'><h2>🏏 IPL Stat Assistant</h2></div>", unsafe_allow_html=True)


MODELS_OPENAI = [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini"
]
MODELS_ANTHROPIC = [
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
]
MODELS_GROQ = [
    "qwen/qwen3.6-27b", # Can capable of searching similar names
    # "qwen/qwen3-32b", # Need Exact Names to work well
    # "openai/gpt-oss-20b", # 
]

with st.sidebar:
    st.selectbox("Select a provider", ["Groq", "OpenAI", "Anthropic"], key="provider")
    if st.session_state.provider == "OpenAI":
        MODELS = MODELS_OPENAI
    elif st.session_state.provider == "Anthropic":
        MODELS = MODELS_ANTHROPIC
    else:
        MODELS = MODELS_GROQ
    st.selectbox("Select a model", MODELS, key="model")

    st.write("API Keys")
    popover_cols = st.columns(3)

    with popover_cols[0]:
        with st.popover("OpenAI"):
            st.text_input(
                "Introduce your OpenAI API Key (https://platform.openai.com/)", 
                value=st.session_state.get("openai_api_key", None),
                type="password",
                key="openai_api_key",
            )
    with popover_cols[1]:
        with st.popover("Anthropic"):
            st.text_input(
                "Introduce your Anthropic API Key (https://console.anthropic.com/)", 
                value=st.session_state.get("anthropic_api_key", None),
                type="password",
                key="anthropic_api_key",
            )
    with popover_cols[2]:
        with st.popover("Groq"):
            st.text_input(
                "Introduce your Groq API Key (https://groq.com/pricing)", 
                value=st.session_state.get("groq_api_key", None),
                type="password",
                key="groq_api_key",
            )

    st.divider()

openai_api_key = st.session_state.get("openai_api_key", None)
anthropic_api_key = st.session_state.get("anthropic_api_key", None)
groq_api_key = st.session_state.get("groq_api_key", None)

missing_openai = openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key
missing_anthropic = anthropic_api_key == "" or anthropic_api_key is None
missing_groq = groq_api_key == "" or groq_api_key is None

if (st.session_state.provider == "OpenAI" and missing_openai) or (st.session_state.provider == "Anthropic" and missing_anthropic) or (st.session_state.provider == "Groq" and missing_groq):
    st.warning("⬅️ Please introduce an API Key of the provider selected to continue...")
    st.stop()

with st.sidebar:
    button_cols = st.columns(2)
    with button_cols[0]:
        st.button("New Chat", on_click=add_new_chat, type="secondary")
    with button_cols[1]:
        is_history_empty = len(st.session_state["active_chat_history"].messages) == 0
        with st.popover("🗑️ Clear Chat", type="primary", disabled=is_history_empty, use_container_width=True):
            st.write("⚠️ **Wipe this entire chat?**")
            st.button("Yes, Clear History", on_click=clear_chat_history, type="primary", use_container_width=True)

    st.selectbox("Select a chat history", st.session_state.chat_history_list, key="chat_history_selector")
    
    st.download_button(
        label="📥 Export Current Chat (JSON)",
        data=json.dumps(messages_to_dict(st.session_state["active_chat_history"].messages), indent=2),
        file_name=f"{st.session_state['chat_history_selector'].lower().replace(' ', '_')}_history.json",
        mime="application/json",
        use_container_width=True
    )

    # st.write("📤 **Import External Chat**")
    # uploaded_file = st.file_uploader(
    #     "Upload an exported JSON history file", 
    #     type=["json"], 
    #     label_visibility="collapsed",
    #     key="chat_file_uploader",
    #     # on_change=lambda: import_chat_history(st.session_state.chat_file_uploader)
    # )

model_provider = st.session_state.provider
if model_provider == "Groq":
    llm = ChatGroq(
        api_key=groq_api_key, # type: ignore
        model=st.session_state.model,
        temperature=0
    )
elif model_provider == "OpenAI":
    llm = ChatOpenAI(
        api_key=openai_api_key, # type: ignore
        model=st.session_state.model,
        temperature=0
    )
elif model_provider == "anthropic":
    llm = ChatAnthropic(
        api_key=anthropic_api_key,
        model=st.session_state.model, # type: ignore
        temperature=0,
        timeout=None,
        stop=None,
        model_name='',
    )
else:
    st.error("Invalid provider selected.")
    st.stop()

with st.spinner("Preparing Dataset"):
    try:
        df = load_and_clean_df()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

with st.spinner("Creating Agent"):
    system_prompt = """You are an advanced data analyst expert in Python and Pandas.
        You are working with a pre-loaded pandas DataFrame named `df`.
        CRITICAL RULES:
         - You have READ-ONLY access to the data.
         - DO NOT create, redefine, mock or copy the dataframe `df`. It already exists in your environment.
         - DO NOT import pandas as pd. It is already imported.
         - NEVER use destructive operations like `df.drop()`, `inplace=True`, `del df[...]`, or re-assigning columns via `df['col'] = ...`.
         - If you need to filter or manipulate data, ALWAYS create a separate temporary copy or variable (e.g., `filtered_df = df[df['runs'] > 50]`) or use safe methods like `.loc[]`.
         - Simply write code that queries the existing `df` variable.
         - To see what data is actually inside the dataframe, you should ALWAYS look at `df.head()` or `df.columns` first if you are unsure.
        This is the structure of the dataframe you have access to:
        {df_head}
        If you need more explanation about a column check the dictionary `df.attrs['col_description']` where in the keys are the column names and the values are the descriptions of the columns.
    """

    df_head_info = df.head(3).to_string()
    df_columns = df.columns.tolist()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.format(df_head=df_head_info)),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    try:
        agent = create_pandas_dataframe_agent(
            llm,
            df=df,
            agent_type="tool-calling",
            allow_dangerous_code=True,
            verbose=True,
            # prompt=prompt,
            # system_prompt=system_prompt.format(df_head=df_head_info),
            prefix=f"You are a helpful assistant that can answer questions about IPL Stats using the provided DataFrame df by only queries. Please answer the user's questions based on the data in df. If you don't know the answer, just say 'I can not answer'. Do not try to make up an answer. These are the column names in the DataFrame: {df_columns}.",
            agent_executor_kwargs={
                "handle_parsing_errors": True,
            },
            suffix="You can use `df.head()` to see the first few rows of the DataFrame and `df.columns` to see the column names and `df.attrs[col_description][<column_name>] to get the column verbal descriptions`. Use these commands to help you answer the user's questions.",
            max_execution_time=60
        )
    except Exception as e:
        st.error(f"Error creating agent: {e}")
        st.stop()

for msg in st.session_state["active_chat_history"].messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif msg.type == "ai":
        with st.chat_message("assistant"):
            if msg.content == "Chat cleared. Ask me anything about IPL Stats!" or msg.content == "Hello! I am your IPL Stat Assistant. How can I help you today?":
                st.markdown(msg.content)
            else:
                st.markdown(msg.content + f"\n\n*Note: The AI's response is based on the data available till 16-04-2026 PBKS vs MI match (included) and may not reflect real-time information.*")
    # st.chat_message(msg.type).write(msg.content)

if user_query := st.chat_input("Ask anything related to IPL Stats..."):
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Generating response..."):
        try:
            with st.expander("👀 View AI Thinking Process & Code Queries", expanded=False):
                st_callback = StreamlitCallbackHandler(st.container())
            response = agent.invoke(
                {
                "input": user_query,
                # "chat_history": st.session_state["active_chat_history"].messages
                },
                {"callbacks": [st_callback]}
            )
            answer = response['output'] + f"\n\nModel: {st.session_state.model}"
            st.session_state["active_chat_history"].add_user_message(user_query)
            st.session_state["active_chat_history"].add_ai_message(answer)
            # st.rerun()
            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(f"Model: {st.session_state.model} | Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.stop()

