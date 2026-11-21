"""Web interface for the chatbot. Run with:  streamlit run app.py"""

from __future__ import annotations

import streamlit as st

from qa.llm import make_llm
from qa.pipeline import CONFIGS, QAPipeline

st.set_page_config(page_title="Kikapu Help Assistant", page_icon="🛒")
st.title("Kikapu Help Assistant")
st.caption("Answers questions using only the help documents in the docs/ folder.")

with st.sidebar:
    config_name = st.selectbox("Config", list(CONFIGS), index=list(CONFIGS).index("improved"))
    provider = st.selectbox("Provider", ["llm", "mock"], help="'mock' runs offline without an API key")
    config = CONFIGS[config_name]
    st.markdown(
        f"**Chunking:** {config.chunking} ({config.chunk_size} words, overlap {config.overlap})  \n"
        f"**Chunks retrieved:** {config.top_k}"
    )


@st.cache_resource
def get_pipeline(config_name: str, provider: str) -> QAPipeline:
    return QAPipeline(CONFIGS[config_name], make_llm(provider))


try:
    pipeline = get_pipeline(config_name, provider)
except RuntimeError as error:
    st.error(str(error))
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

question = st.chat_input("Ask about delivery, payments, refunds or your account")
if question:
    st.chat_message("user").write(question)
    with st.spinner("Searching the help documents..."):
        result = pipeline.ask(question)
    with st.chat_message("assistant"):
        st.write(result.answer)
        with st.expander("Sources used"):
            for chunk in result.sources:
                st.markdown(f"**{chunk.source}** - {chunk.text}")
    st.session_state.history += [("user", question), ("assistant", result.answer)]
