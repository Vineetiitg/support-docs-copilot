import os
from pathlib import Path

import requests
import streamlit as st


st.set_page_config(page_title="Support Docs Copilot", page_icon="SD", layout="wide")
st.title("Support Docs Copilot")

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")
BACKEND_STREAM_URL = os.getenv("BACKEND_URL", f"{BACKEND_BASE_URL}/chat/stream")
DATA_DIR = os.getenv("DATA_DIR", "data/docs")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "token" not in st.session_state:
    st.session_state.token = ""


def headers() -> dict:
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


def get_json(path: str) -> dict:
    response = requests.get(f"{BACKEND_BASE_URL}{path}", headers=headers(), timeout=10)
    response.raise_for_status()
    return response.json()


def post_json(path: str, payload: dict | None = None) -> dict:
    response = requests.post(f"{BACKEND_BASE_URL}{path}", json=payload or {}, headers=headers(), timeout=300)
    response.raise_for_status()
    return response.json()


chat_tab, documents_tab, admin_tab, evaluation_tab, settings_tab = st.tabs(
    ["Chat", "Documents", "Admin", "Evaluation", "Settings"]
)

with chat_tab:
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
                with requests.post(
                    BACKEND_STREAM_URL,
                    json={"query": user_query, "chat_history": st.session_state.messages[:-1]},
                    headers=headers(),
                    stream=True,
                    timeout=300,
                ) as response:
                    if response.status_code == 200:
                        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                full_response += chunk
                                response_placeholder.markdown(full_response + "...")
                        response_placeholder.markdown(full_response)
                    else:
                        full_response = f"Request failed: {response.text}"
                        response_placeholder.error(full_response)
            except requests.RequestException as exc:
                full_response = f"Backend connection error: {exc}"
                response_placeholder.error(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

with documents_tab:
    try:
        payload = get_json("/documents")
        docs = payload.get("documents", [])
        st.metric("Indexed documents", len(docs))
        if docs:
            st.dataframe(docs, use_container_width=True)
        else:
            st.info("No indexed documents found. Upload documents in Admin and run ingestion.")
    except requests.RequestException as exc:
        st.error(f"Unable to load documents: {exc}")

with admin_tab:
    uploaded_files = st.file_uploader(
        "Upload support docs",
        type=["txt", "md", "pdf", "docx", "html", "htm"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Save uploaded files"):
        try:
            files = [("files", (file.name, file.getvalue(), file.type or "application/octet-stream")) for file in uploaded_files]
            response = requests.post(
                f"{BACKEND_BASE_URL}/admin/upload",
                headers=headers(),
                files=files,
                timeout=120,
            )
            response.raise_for_status()
            st.success(f"Uploaded {len(uploaded_files)} file(s) to backend.")
            st.json(response.json())
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")

    col1, col2 = st.columns(2)
    with col1:
        force = st.checkbox("Force recreate index")
        if st.button("Run ingestion"):
            try:
                st.json(post_json("/admin/ingest", {"data_dir": DATA_DIR, "force": force}))
            except requests.RequestException as exc:
                st.error(f"Ingestion failed: {exc}")
    with col2:
        if st.button("Reset index"):
            try:
                st.json(post_json("/admin/reset"))
            except requests.RequestException as exc:
                st.error(f"Reset failed: {exc}")

with evaluation_tab:
    st.subheader("Automated Quality Assessment (RAGAS)")
    st.write("Evaluate how accurately and faithfully the copilot answers support questions using the RAGAS framework.")
    
    if st.button("🚀 Run RAG Evaluation Now", type="primary"):
        with st.spinner("Running automated RAG evaluation against test questions... This may take 1-2 minutes."):
            try:
                res = post_json("/admin/eval")
                st.success("Evaluation completed successfully!")
            except requests.RequestException as exc:
                st.error(f"Evaluation failed: {exc}. Ensure you are logged in as admin under Settings and have remaining OpenRouter credits/limits.")
    
    st.divider()
    st.subheader("Latest Evaluation Report")
    try:
        res = get_json("/admin/eval")
        st.markdown(res.get("report", "No evaluation report available."))
    except requests.RequestException:
        st.info("No evaluation report available yet. Click the button above to run your first evaluation!")

with settings_tab:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        try:
            response = requests.post(f"{BACKEND_BASE_URL}/auth/login", data={"username": username, "password": password})
            if response.ok:
                st.session_state.token = response.json().get("access_token", "")
                st.success("Logged in successfully.")
            else:
                st.error("Login failed.")
        except requests.RequestException as exc:
            st.error(f"Login request failed: {exc}")

    st.divider()
    st.subheader("Observability & Tracing (LangSmith)")
    st.write("Monitor RAG agent steps, prompt tokens, and latency in real-time by adding these variables to your `.env`:")
    st.code("LANGCHAIN_TRACING_V2=true\nLANGCHAIN_API_KEY=your_langsmith_api_key\nLANGCHAIN_PROJECT=\"Support Docs Copilot\"", language="env")

    st.divider()
    st.caption(f"Backend base URL: {BACKEND_BASE_URL}")
    try:
        st.json(get_json("/ready"))
    except requests.RequestException as exc:
        st.error(f"Readiness check failed: {exc}")
