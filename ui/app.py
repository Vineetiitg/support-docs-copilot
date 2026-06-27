import os

import requests
import streamlit as st


st.set_page_config(page_title="Support Docs Copilot", page_icon="SD")
st.title("Support Docs Copilot")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/chat/stream")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a support question..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        try:
            with requests.post(BACKEND_URL, json={"query": user_query}, stream=True, timeout=120) as response:
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            response_placeholder.markdown(full_response + "...")
                    response_placeholder.markdown(full_response)
                elif response.status_code == 400:
                    full_response = f"Warning: {response.json().get('detail', 'Violation.')}"
                    response_placeholder.error(full_response)
                else:
                    full_response = "Warning: server communication error."
                    response_placeholder.error(full_response)
        except requests.RequestException:
            full_response = "Backend connection error."
            response_placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
