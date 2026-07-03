import os
import time
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
if "role" not in st.session_state:
    st.session_state.role = ""
if "username" not in st.session_state:
    st.session_state.username = ""


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


def poll_job_status(job_id: str, status_text: str = "Processing in background..."):
    with st.status(status_text, expanded=True) as status:
        st.write("Job enqueued in Redis...")
        for _ in range(120):
            try:
                res = get_json(f"/tasks/status/{job_id}")
                job_status = res.get("status", "unknown")
                st.write(f"Status: **{job_status}**")
                if err_msg := res.get("error"):
                    st.error(f"Error details: {err_msg}")
                if job_status in ("complete", "success"):
                    status.update(label="Job Completed Successfully!", state="complete", expanded=False)
                    return res.get("result")
                elif job_status in ("not_found", "error", "failed", "error_try_again"):
                    status.update(label=f"Job Finished ({job_status})", state="complete" if job_status == "complete" else "error", expanded=True)
                    return res.get("result")
            except Exception:
                pass
            time.sleep(1.5)
        status.update(label="Job Timed Out / Still Running", state="error")
    return None


# Authentication Sidebar
with st.sidebar:
    st.subheader("🔐 Authentication")
    if st.session_state.token and st.session_state.role:
        if st.session_state.role == "admin":
            st.success(f"Logged in as: **{st.session_state.username or 'Admin'}**\n\nRole: **👑 Administrator**")
        else:
            st.info(f"Logged in as: **{st.session_state.username or 'User'}**\n\nRole: **👤 User**")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.token = ""
            st.session_state.role = ""
            st.session_state.username = ""
            st.rerun()
    else:
        st.write("Login to access role-specific UI features.")
        username_input = st.text_input("Username", key="sb_user")
        password_input = st.text_input("Password", type="password", key="sb_pass")
        if st.button("🔑 Login", use_container_width=True, type="primary"):
            try:
                response = requests.post(
                    f"{BACKEND_BASE_URL}/auth/login",
                    data={"username": username_input, "password": password_input},
                )
                if response.ok:
                    data = response.json()
                    st.session_state.token = data.get("access_token", "")
                    st.session_state.role = data.get("role", "user")
                    st.session_state.username = username_input
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            except requests.RequestException as exc:
                st.error(f"Login request failed: {exc}")
    
    st.divider()
    st.caption(f"Backend base URL: {BACKEND_BASE_URL}")
    try:
        ready_data = get_json("/ready")
        if ready_data.get("ready"):
            st.caption("🟢 Backend System Online")
        else:
            st.caption("🟡 Backend Degraded")
    except Exception:
        st.caption("🔴 Backend Offline")


# Tab Renderers
def render_chat_tab():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a support question..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            try:
                response = requests.post(
                    f"{BACKEND_BASE_URL}/chat/stream",
                    json={"query": user_query, "chat_history": st.session_state.messages[:-1]},
                    headers=headers(),
                    stream=True,
                    timeout=120,
                )
                response.raise_for_status()
                
                placeholder = st.empty()
                full_answer = ""
                for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                    if chunk:
                        full_answer += chunk
                        if "[CANCELLED:" in full_answer or "[CANCEL:" in full_answer:
                            full_answer = "🚨 **[CANCELLED: This response violated safety guidelines and has been retracted.]**"
                            placeholder.markdown(full_answer)
                            break
                        placeholder.markdown(full_answer + "▌")
                placeholder.markdown(full_answer)
                st.session_state.messages.append({"role": "assistant", "content": full_answer})
            except requests.RequestException as exc:
                st.error(f"Chat request failed: {exc}")


def render_documents_tab():
    st.subheader("Indexed Documents")
    if st.button("Refresh list", key="ref_docs"):
        st.rerun()
    try:
        data = get_json("/documents")
        documents = data.get("documents", [])
        if not documents:
            st.info("No indexed documents found.")
        else:
            for doc in documents:
                st.write(f"**{doc.get('source')}** | id={doc.get('doc_id')} | chunks={doc.get('chunk_count')}")
    except requests.RequestException as exc:
        st.error(f"Failed to load documents: {exc}")


def render_admin_portal_tab():
    st.subheader("Document Ingestion & Management")
    uploaded_files = st.file_uploader(
        "Upload support docs",
        type=["txt", "md", "pdf", "docx", "html", "htm"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Save uploaded files", type="primary"):
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

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        force = st.checkbox("Force recreate index")
        if st.button("Run ingestion", use_container_width=True):
            try:
                res = post_json("/admin/ingest", {"data_dir": DATA_DIR, "force": force})
                st.json(res)
                if job_id := res.get("job_id"):
                    poll_job_status(job_id, "Ingesting documents via Arq Worker...")
            except requests.RequestException as exc:
                st.error(f"Ingestion failed: {exc}")
    with col2:
        if st.button("Reset index", use_container_width=True):
            try:
                st.json(post_json("/admin/reset"))
            except requests.RequestException as exc:
                st.error(f"Reset failed: {exc}")


def render_evaluation_tab():
    st.subheader("Automated Quality Assessment (RAGAS)")
    st.write("Evaluate how accurately and faithfully the copilot answers support questions using the RAGAS framework.")
    
    if st.button("🚀 Run RAG Evaluation Now", type="primary"):
        try:
            res = post_json("/admin/eval")
            st.json(res)
            if job_id := res.get("job_id"):
                poll_job_status(job_id, "Running RAGAS evaluation via Arq Worker... This may take 1-2 minutes.")
            st.success("Evaluation task dispatched!")
        except requests.RequestException as exc:
            st.error(f"Evaluation failed: {exc}. Ensure you have remaining OpenRouter credits/limits.")
    
    st.divider()
    st.subheader("Latest Evaluation Report")
    try:
        res = get_json("/admin/eval")
        st.markdown(res.get("report", "No evaluation report available."))
    except requests.RequestException:
        st.info("No evaluation report available yet. Click the button above to run your first evaluation!")


def render_langsmith_tab():
    st.subheader("Observability & Tracing (LangSmith)")
    st.write("Monitor RAG agent steps, prompt tokens, and latency in real-time by adding these variables to your `.env`:")
    st.code("LANGCHAIN_TRACING_V2=true\nLANGCHAIN_API_KEY=your_langsmith_api_key\nLANGCHAIN_PROJECT=\"Support Docs Copilot\"", language="env")

    st.divider()
    st.subheader("System Readiness Diagnostics")
    st.caption(f"Backend base URL: {BACKEND_BASE_URL}")
    try:
        st.json(get_json("/ready"))
    except requests.RequestException as exc:
        st.error(f"Readiness check failed: {exc}")


# Render Role-Specific UI Layouts
if st.session_state.role == "admin":
    chat_tab, docs_tab, admin_tab, eval_tab, langsmith_tab = st.tabs([
        "💬 Chat Copilot", 
        "📚 Documents", 
        "🛠️ Admin Portal", 
        "📊 RAGAS Evaluation", 
        "📈 LangSmith & Observability"
    ])
    with chat_tab:
        render_chat_tab()
    with docs_tab:
        render_documents_tab()
    with admin_tab:
        render_admin_portal_tab()
    with eval_tab:
        render_evaluation_tab()
    with langsmith_tab:
        render_langsmith_tab()
else:
    chat_tab, docs_tab, status_tab = st.tabs([
        "💬 Chat Copilot", 
        "📚 Documents", 
        "⚙️ System Status"
    ])
    with chat_tab:
        render_chat_tab()
    with docs_tab:
        render_documents_tab()
    with status_tab:
        st.subheader("System Status")
        try:
            st.json(get_json("/ready"))
        except requests.RequestException as exc:
            st.error(f"Readiness check failed: {exc}")
        st.divider()
        st.info("💡 **Admin Notice**: To access Document Ingestion, RAGAS Benchmarks, and LangSmith Observability tools, please login as an Administrator using the authentication sidebar on the left.")
