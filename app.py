import streamlit as st
import json
import datetime

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import  messages_to_dict
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

import pandas as pd

from load_and_clean_df import load_and_clean_df as _load_and_clean_df
from match_summary import (
    list_matches,
    build_match_summary,
    render_card,
    chart_frame_from_markdown,
)

default_disclaimer = "The AI's responses are based on the data available till 16-04-2026 PBKS vs MI match (included) and may not reflect real-time information. Use at your own risk."

GREETING = "Hello! I am your IPL Stat Assistant. How can I help you today?"
CLEARED = "Chat cleared. Ask me anything about IPL Stats!"
# Turns of prior conversation fed back to the agent so follow-up questions keep context.
HISTORY_TURNS = 6

@st.cache_data()
def load_and_clean_df():
    return _load_and_clean_df()

def add_new_chat():
    new_chat_id = len(st.session_state.chat_history_list) + 1
    new_chat = f"chat_{new_chat_id}"
    st.session_state.chat_history_list.append(new_chat)
    st.session_state.chat_history_selector = new_chat
    new_chat_history = StreamlitChatMessageHistory(key=new_chat)
    new_chat_history.add_ai_message(GREETING)

def clear_chat_history():
    present_chat = st.session_state.chat_history_selector
    chat_history_db = StreamlitChatMessageHistory(key = present_chat)
    if len(chat_history_db.messages) > 1:
        chat_history_db.clear()
        chat_history_db.add_ai_message(CLEARED)




if "chat_history_list" not in st.session_state:
    st.session_state.chat_history_list = ["chat_1"]
    st.session_state.chat_history_selector = "chat_1"

st.session_state["active_chat_history"] = StreamlitChatMessageHistory(key=st.session_state.chat_history_selector)
if len(st.session_state["active_chat_history"].messages) == 0:
    st.session_state["active_chat_history"].add_ai_message(GREETING)


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


# OpenRouter serves OpenAI, Anthropic and Gemini models through a single key.
# Model ids are OpenRouter slugs (https://openrouter.ai/models) and can be edited freely.
MODELS_OPENROUTER = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.0-flash-001",
    "google/gemini-flash-1.5",
]
MODELS_GROQ = [
    "qwen/qwen3.6-27b", # Can capable of searching similar names
    # "qwen/qwen3-32b", # Need Exact Names to work well
    # "openai/gpt-oss-20b", #
]

with st.sidebar:
    st.selectbox("Select a provider", ["OpenRouter", "Groq"], key="provider")
    if st.session_state.provider == "OpenRouter":
        MODELS = MODELS_OPENROUTER
    else:
        MODELS = MODELS_GROQ
    st.selectbox("Select a model", MODELS, key="model")

    st.write("API Keys")
    popover_cols = st.columns(2)

    with popover_cols[0]:
        with st.popover("OpenRouter"):
            st.text_input(
                "Introduce your OpenRouter API Key (https://openrouter.ai/keys)",
                value=st.session_state.get("openrouter_api_key", None),
                type="password",
                key="openrouter_api_key",
            )
    with popover_cols[1]:
        with st.popover("Groq"):
            st.text_input(
                "Introduce your Groq API Key (https://groq.com/pricing)",
                value=st.session_state.get("groq_api_key", None),
                type="password",
                key="groq_api_key",
            )

    st.divider()

openrouter_api_key = st.session_state.get("openrouter_api_key", None)
groq_api_key = st.session_state.get("groq_api_key", None)

missing_openrouter = not openrouter_api_key
missing_groq = not groq_api_key

if (st.session_state.provider == "OpenRouter" and missing_openrouter) or (st.session_state.provider == "Groq" and missing_groq):
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

    st.markdown(
        f"""
        <div style="text-align: center; color: gray; font-size: 12px; margin-top: 20px;">
            © 2026 All rights reserved.<br>
            Disclaimer: {default_disclaimer}
        </div>
        """,
        unsafe_allow_html=True
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
elif model_provider == "OpenRouter":
    llm = ChatOpenAI(
        api_key=openrouter_api_key, # type: ignore
        base_url="https://openrouter.ai/api/v1",
        model=st.session_state.model,
        temperature=0,
        # Cap the completion so requests fit limited/free OpenRouter balances
        # (the default reserves 16k tokens up front and 402s on low credits).
        max_tokens=1024,
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
    # Read-only safety rules + answer-formatting, used as the agent prefix.
    # (The factory appends df.head() to the prompt itself, so no df_head here.)
    system_prompt = """You are an advanced data analyst expert in Python and Pandas answering questions about IPL Stats using a pre-loaded DataFrame named `df`.
CRITICAL RULES:
 - You have READ-ONLY access to the data.
 - DO NOT create, redefine, mock or copy the dataframe `df`. It already exists in your environment.
 - DO NOT import pandas as pd. It is already imported.
 - NEVER use destructive operations like `df.drop()`, `inplace=True`, `del df[...]`, or re-assigning columns via `df['col'] = ...`.
 - If you need to filter or manipulate data, ALWAYS create a separate temporary variable (e.g. `filtered_df = df[df['over'] < 6]`) or use safe methods like `.loc[]`.
 - To see what is in the data, look at `df.head()` or `df.columns` first if unsure. Column descriptions are in `df.attrs['col_description']`.
 - Answer only from `df`. If you don't know, say 'I can not answer' — do not make up an answer.
 - When the answer is a ranking or list of items with numeric values (e.g. top run scorers, wickets per season), present it as a GitHub-style markdown table with the label in the first column so it can be charted.
"""

    df_columns = df.columns.tolist()

    try:
        agent = create_pandas_dataframe_agent(
            llm,
            df=df,
            agent_type="tool-calling",
            allow_dangerous_code=True,
            verbose=True,
            prefix=f"{system_prompt}\nThese are the column names in the DataFrame: {df_columns}.",
            agent_executor_kwargs={
                "handle_parsing_errors": True,
            },
            suffix="You can use `df.head()` to see the first few rows of the DataFrame and `df.columns` to see the column names and `df.attrs[col_description][<column_name>] to get the column verbal descriptions`. Use these commands to help you answer the user's questions.",
            max_execution_time=60
        )
    except Exception as e:
        st.error(f"Error creating agent: {e}")
        st.stop()

with st.expander("📋 Match Summary & Charts", expanded=False):
    matches = list_matches(df)
    picked = st.selectbox(
        "Pick a match",
        matches["label"].tolist(),
        index=None,
        placeholder="Search a match (team, season, date)…",
        key="match_summary_pick",
    )
    if picked:
        match_id = matches.loc[matches["label"] == picked, "match_id"].iloc[0]
        summary = build_match_summary(df, match_id)
        st.code(render_card(summary), language=None)
        for inn in summary["innings"]:
            if inn["over_runs"]:
                st.caption(f"Runs per over — {inn['team']}")
                # Numeric over index so the x-axis stays in 1..N order
                # (string labels sort lexicographically: Ov 1, Ov 10, Ov 2…).
                st.bar_chart(
                    pd.DataFrame(
                        {"runs": inn["over_runs"]},
                        index=pd.Index(range(1, len(inn["over_runs"]) + 1), name="over"),
                    )
                )

for msg in st.session_state["active_chat_history"].messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif msg.type == "ai":
        with st.chat_message("assistant"):
            if msg.content in (CLEARED, GREETING):
                st.markdown(msg.content)
            else:
                st.markdown(msg.content + f"\n\n*Note: {default_disclaimer}*")
    # st.chat_message(msg.type).write(msg.content)

if user_query := st.chat_input("Ask anything related to IPL Stats..."):
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Generating response..."):
        try:
            with st.expander("👀 View AI Thinking Process & Code Queries", expanded=False):
                st_callback = StreamlitCallbackHandler(st.container())

            # Give the agent short-term memory: replay the last few turns as context.
            # The pandas-agent prompt has no chat_history slot, so we fold it into the input.
            prior = [
                m for m in st.session_state["active_chat_history"].messages
                if m.content not in (GREETING, CLEARED)
            ][-HISTORY_TURNS:]
            if prior:
                transcript = "\n".join(
                    f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
                    for m in prior
                )
                agent_input = (
                    "Earlier conversation (for context on follow-up questions):\n"
                    f"{transcript}\n\nCurrent question: {user_query}"
                )
            else:
                agent_input = user_query

            response = agent.invoke(
                {"input": agent_input},
                {"callbacks": [st_callback]}
            )
            answer = response['output'] + f"\n\nModel: {st.session_state.model}"
            st.session_state["active_chat_history"].add_user_message(user_query)
            st.session_state["active_chat_history"].add_ai_message(answer)
            # st.rerun()
            with st.chat_message("assistant"):
                st.markdown(answer)
                # Auto-chart: if the answer contains a numeric markdown table, plot it.
                try:
                    chart_df = chart_frame_from_markdown(answer)
                    if chart_df is not None and not chart_df.empty:
                        st.bar_chart(chart_df)
                except Exception:
                    pass
                st.caption(f"Model: {st.session_state.model} | Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.stop()

