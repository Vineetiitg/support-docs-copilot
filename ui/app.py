import os
import time
from pathlib import Path
import requests
import streamlit as st
import uuid

st.set_page_config(page_title="Support Docs Copilot", page_icon="🚀", layout="wide")

def load_css():
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

load_css()

# Header & Banner
st.markdown("""
<div style="padding: 0.5rem 0; margin-bottom: 1rem;">
    <h1 style="font-size: 2.6rem; margin: 0;">🚀 Support Docs Copilot</h1>
    <p style="color: #94a3b8; font-size: 1.05rem; margin-top: 0.2rem;">
        Next-Gen Agentic RAG Assistant powered by LangGraph, Speculative Retrieval & Cohere Reranking
    </p>
</div>
""", unsafe_allow_html=True)

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
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


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
        st.write("Job enqueued in Redis worker queue...")
        status_placeholder = st.empty()
        for _ in range(120):
            try:
                res = get_json(f"/tasks/status/{job_id}")
                job_status = res.get("status", "unknown")
                status_placeholder.markdown(f"Status: **{job_status}** ⏳")
                if err_msg := res.get("error"):
                    st.error(f"Error details: {err_msg}")
                if job_status in ("complete", "success"):
                    status_placeholder.markdown("Status: **Complete** ✅")
                    status.update(label="Job Completed Successfully!", state="complete", expanded=False)
                    return res.get("result")
                elif job_status in ("not_found", "error", "failed", "error_try_again"):
                    status_placeholder.markdown(f"Status: **{job_status}** ❌")
                    status.update(label=f"Job Finished ({job_status})", state="complete" if job_status == "complete" else "error", expanded=True)
                    return res.get("result")
            except Exception:
                pass
            time.sleep(1.5)
        status.update(label="Job Timed Out / Still Running", state="error")
    return None


# Authentication Sidebar
with st.sidebar:
    st.markdown("### 🔐 Authentication")
    if st.session_state.token and st.session_state.role:
        if st.session_state.role == "admin":
            st.markdown(f"""
            <div class="glass-card" style="padding: 1rem; border-left: 4px solid #a855f7;">
                <p style="margin: 0; font-size: 0.9rem; color: #cbd5e1;">Logged in as:</p>
                <p style="margin: 0; font-size: 1.1rem; font-weight: 600; color: #f8fafc;">👑 {st.session_state.username or 'Admin'}</p>
                <span class="status-badge badge-purple" style="margin-top: 0.5rem;">Administrator</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="glass-card" style="padding: 1rem; border-left: 4px solid #60a5fa;">
                <p style="margin: 0; font-size: 0.9rem; color: #cbd5e1;">Logged in as:</p>
                <p style="margin: 0; font-size: 1.1rem; font-weight: 600; color: #f8fafc;">👤 {st.session_state.username or 'User'}</p>
                <span class="status-badge badge-blue" style="margin-top: 0.5rem;">User</span>
            </div>
            """, unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.token = ""
            st.session_state.role = ""
            st.session_state.username = ""
            st.rerun()
    else:
        st.write("Login to access role-specific UI features.")
        username_input = st.text_input("Username", key="sb_user", placeholder="admin or user")
        password_input = st.text_input("Password", type="password", key="sb_pass", placeholder="••••••••")
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
    st.caption(f"🔗 Backend API: `{BACKEND_BASE_URL}`")
    try:
        ready_data = get_json("/ready")
        if ready_data.get("ready"):
            st.markdown('<span class="status-badge badge-green">🟢 System Online</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-yellow">🟡 System Degraded</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span class="status-badge" style="background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid #ef4444;">🔴 Backend Offline</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 💬 Recent Chats")
    if st.button("➕ New Support Topic", use_container_width=True, type="primary", key="new_chat_btn"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    if not st.session_state.token:
        st.info("🔒 **Not logged in.** Please login above to access and resume your saved recent chat history.")
    else:
        if st.session_state.role == "admin":
            st.caption(f"Showing saved sessions for **👑 Admin ({st.session_state.username})**")
        else:
            st.caption(f"Showing saved sessions for **👤 {st.session_state.username}**")
            
        try:
            sessions_res = get_json("/api/v1/sessions")
            sessions_list = sessions_res.get("sessions", [])
            if not sessions_list:
                st.caption("No recent sessions found for your account.")
            else:
                for s in sessions_list[:10]:
                    sid = s.get("session_id", "default")
                    preview = s.get("last_preview", sid[:8] + "...")
                    btn_label = f"💬 {preview}" if sid != st.session_state.get("session_id") else f"🟢 {preview}"
                    if st.button(btn_label, key=f"sess_{sid}", use_container_width=True):
                        st.session_state.session_id = sid
                        msg_res = get_json(f"/api/v1/sessions/{sid}/messages")
                        st.session_state.messages = msg_res.get("messages", [])
                        st.rerun()
        except Exception:
            st.caption("Could not load sessions.")


# Tab Renderers
def render_chat_tab():
    col_stop, col_info = st.columns([1, 4])
    with col_stop:
        if st.button("🛑 Stop Generation", key="term_btn_top", use_container_width=True):
            if sid := st.session_state.get("session_id"):
                try:
                    post_json(f"/api/v1/sessions/{sid}/terminate")
                    st.toast("🛑 Sent termination signal to active generation!")
                except Exception:
                    pass
    with col_info:
        st.caption("💡 Tip: Click **🛑 Stop Generation** anytime during text output to immediately terminate an ongoing chat response.")

    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                conf = message.get("confidence")
                sources = message.get("sources")
                if conf is not None or sources:
                    cols = st.columns([1, 4])
                    with cols[0]:
                        if conf is not None:
                            badge_class = "badge-green" if conf >= 0.8 else ("badge-yellow" if conf >= 0.5 else "badge-purple")
                            st.markdown(f'<span class="status-badge {badge_class}">🎯 Conf: {conf*100:.0f}%</span>', unsafe_allow_html=True)
                    with cols[1]:
                        if sources:
                            with st.expander(f"📚 Cited Sources ({len(sources)} documents referenced)"):
                                for idx, src in enumerate(sources, 1):
                                    src_name = src.get("source", src.get("doc_id", "Unknown Document"))
                                    score = src.get("relevance_score", src.get("similarity_score", 0.0))
                                    st.markdown(f"**{idx}. {src_name}** `(Relevance Score: {score:.2f})`")
                                    if snippet := src.get("content_snippet"):
                                        st.caption(f'"{snippet[:200]}..."')

    if user_query := st.chat_input("Ask a support question..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            try:
                response = requests.post(
                    f"{BACKEND_BASE_URL}/chat/stream",
                    json={"query": user_query, "chat_history": st.session_state.messages[:-1], "session_id": st.session_state.get("session_id")},
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
                
                # Append locally immediately so output ALWAYS displays on screen
                new_msg = {"role": "assistant", "content": full_answer}
                st.session_state.messages.append(new_msg)
                
                # Fetch updated session messages from backend ONLY if backend has more or equal messages (preventing erasure)
                time.sleep(0.35)
                sid = st.session_state.get("session_id")
                if sid:
                    try:
                        res_msgs = get_json(f"/api/v1/sessions/{sid}/messages")
                        if res_msgs and (msgs := res_msgs.get("messages")) and len(msgs) >= len(st.session_state.messages):
                            st.session_state.messages = msgs
                    except Exception:
                        pass
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Chat request failed: {exc}")


def render_documents_tab():
    st.markdown("### 📚 Indexed Knowledge Base")
    st.write("Manage your RAG vector store documents. You can inspect chunk counts, content hashes, or remove individual files.")
    
    col_ref, col_spacer = st.columns([1, 5])
    with col_ref:
        if st.button("🔄 Refresh List", key="ref_docs", use_container_width=True):
            st.rerun()
            
    try:
        data = get_json("/documents")
        documents = data.get("documents", [])
        if not documents:
            st.info("💡 Knowledge base is currently empty. Login as an Admin and go to **🛠️ Admin Portal** to ingest documents.")
        else:
            for doc in documents:
                with st.container():
                    st.markdown(f"""
                    <div class="glass-card" style="padding: 1.2rem; margin-bottom: 0.6rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4 style="margin: 0; color: #60a5fa;">📄 {doc.get('source')}</h4>
                                <p style="margin: 0.3rem 0 0 0; color: #94a3b8; font-size: 0.85rem;">
                                    <b>ID:</b> <code>{doc.get('doc_id')}</code> | <b>Chunks:</b> <span class="status-badge badge-blue">{doc.get('chunk_count')} chunks</span> | <b>Hash:</b> <code>{str(doc.get('content_hash'))[:10]}...</code>
                                </p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.session_state.role == "admin":
                        col_del, col_blank = st.columns([1, 5])
                        with col_del:
                            if st.button("🗑️ Delete File & Embeddings", key=f"del_{doc.get('doc_id')}", use_container_width=True):
                                try:
                                    requests.delete(f"{BACKEND_BASE_URL}/admin/documents/{doc.get('doc_id')}", headers=headers(), timeout=10)
                                    st.success(f"🗑️ Deleted file '{doc.get('source')}' from disk and removed its embeddings from Qdrant!")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Delete failed: {exc}")
                st.divider()
    except requests.RequestException as exc:
        st.error(f"Failed to load documents: {exc}")


def render_admin_portal_tab():
    st.markdown("### 🛠️ Document Ingestion & Index Control")
    st.write("Upload new knowledge documents, trigger hybrid vector index rebuilds via Arq worker, or reset the collection.")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📤 Upload Documents to Storage")
    uploaded_files = st.file_uploader(
        "Select files (.pdf, .docx, .txt, .md, .html)",
        type=["txt", "md", "pdf", "docx", "html", "htm"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("💾 Save Uploaded Files to Backend", type="primary"):
        try:
            files = [("files", (file.name, file.getvalue(), file.type or "application/octet-stream")) for file in uploaded_files]
            response = requests.post(
                f"{BACKEND_BASE_URL}/admin/upload",
                headers=headers(),
                files=files,
                timeout=120,
            )
            response.raise_for_status()
            st.success(f"✅ Successfully saved {len(uploaded_files)} file(s) to backend storage!")
            st.json(response.json())
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### ⚙️ Hybrid Vector Indexing")
    col1, col2 = st.columns(2)
    with col1:
        force = st.checkbox("⚠️ Force recreate index (wipes existing embeddings)")
        if st.button("🚀 Run Document Ingestion", use_container_width=True, type="primary"):
            try:
                res = post_json("/admin/ingest", {"data_dir": DATA_DIR, "force": force})
                st.json(res)
                if job_id := res.get("job_id"):
                    result = poll_job_status(job_id, "Ingesting & embedding documents via Arq Worker...")
                    if result and result.get("status") == "SUCCESS":
                        st.success("✅ Ingestion complete! Switch to the 📚 Documents tab or click Refresh list to see your new documents.")
            except requests.RequestException as exc:
                st.error(f"Ingestion failed: {exc}")
    with col2:
        st.write("")
        if st.button("🗑️ Reset Entire Index", use_container_width=True):
            try:
                st.json(post_json("/admin/reset"))
                st.success("Index reset successfully.")
            except requests.RequestException as exc:
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🗑️ One-Click Post-Ingestion File & Embedding Management")
    st.write("Easily remove uploaded files from storage and wipe their vector embeddings in one click after running ingestion.")
    try:
        docs_res = get_json("/documents")
        ingested_docs = docs_res.get("documents", [])
        if not ingested_docs:
            st.caption("No ingested documents currently found.")
        else:
            selected_to_delete = []
            for doc in ingested_docs:
                col_name, col_btn = st.columns([3, 1])
                with col_name:
                    if st.checkbox(f"📄 **{doc.get('source')}** (`{doc.get('chunk_count')} chunks`)", key=f"adm_chk_{doc.get('doc_id')}"):
                        selected_to_delete.append(doc)
                with col_btn:
                    if st.button("🗑️ Delete (1-Click)", key=f"adm_del_{doc.get('doc_id')}", use_container_width=True):
                        try:
                            requests.delete(f"{BACKEND_BASE_URL}/admin/documents/{doc.get('doc_id')}", headers=headers(), timeout=10)
                            st.success(f"🗑️ Deleted file '{doc.get('source')}' and removed its embeddings!")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Delete failed: {exc}")
            if selected_to_delete:
                st.write("")
                if st.button(f"🗑️ Delete {len(selected_to_delete)} Selected File(s) & Embeddings in One Click", type="primary", use_container_width=True):
                    for d in selected_to_delete:
                        try:
                            requests.delete(f"{BACKEND_BASE_URL}/admin/documents/{d.get('doc_id')}", headers=headers(), timeout=10)
                        except Exception:
                            pass
                    st.success(f"🗑️ Successfully deleted {len(selected_to_delete)} file(s) and removed their embeddings!")
                    st.rerun()
    except Exception as exc:
        st.caption(f"Could not load ingested documents: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)


def render_evaluation_tab():
    st.markdown("### 📊 Automated Quality Assessment (RAGAS)")
    st.write("Evaluate how accurately and faithfully the copilot answers support questions using the RAGAS framework.")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if st.button("🚀 Run RAG Evaluation Now", type="primary"):
        try:
            res = post_json("/admin/eval")
            st.json(res)
            if job_id := res.get("job_id"):
                poll_job_status(job_id, "Running RAGAS evaluation via Arq Worker... This may take 1-2 minutes.")
            st.success("Evaluation task dispatched!")
        except requests.RequestException as exc:
            st.error(f"Evaluation failed: {exc}. Ensure you have remaining OpenRouter credits/limits.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    st.markdown("#### 📑 Latest Evaluation Report")
    try:
        res = get_json("/admin/eval")
        report_text = res.get("report", "No evaluation report available.")
        st.markdown(f'<div class="glass-card">{report_text}</div>', unsafe_allow_html=True)
    except requests.RequestException:
        st.info("💡 No evaluation report available yet. Click the button above to run your first evaluation!")


def render_langsmith_tab():
    st.markdown("### 📈 Observability & Tracing (LangSmith)")
    st.write("Monitor RAG agent steps, prompt tokens, and latency in real-time by adding these variables to your `.env`:")
    st.code("LANGCHAIN_TRACING_V2=true\nLANGCHAIN_API_KEY=your_langsmith_api_key\nLANGCHAIN_PROJECT=\"Support Docs Copilot\"", language="env")

    st.divider()
    st.markdown("#### 🔍 System Readiness Diagnostics")
    st.caption(f"Backend API Base URL: `{BACKEND_BASE_URL}`")
    try:
        ready_data = get_json("/ready")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Overall Status", "🟢 Ready" if ready_data.get("ready") else "🟡 Degraded")
        with cols[1]:
            st.metric("Vector Store", "🟢 Online" if ready_data.get("vector_store") else "🔴 Offline")
        with cols[2]:
            st.metric("LLM Provider", "🟢 Connected" if ready_data.get("llm") else "🔴 Offline")
        st.json(ready_data)
    except requests.RequestException as exc:
        st.error(f"Readiness check failed: {exc}")


def render_observability_dashboard_tab():
    st.markdown("### 👀 Live Session Observability Dashboard")
    st.write("Monitor ongoing user chat threads across the enterprise, inspect RAG source citations, and inject supervisor guidance.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### 🧵 Active Enterprise Threads")
        if st.button("🔄 Refresh Live Sessions", key="ref_obs", use_container_width=True):
            st.rerun()
        try:
            res = get_json("/api/v1/admin/sessions")
            all_sess = res.get("sessions", [])
            if not all_sess:
                st.info("No active user sessions found.")
                selected_sess = None
            else:
                # Sort all user sessions by latest timestamp in descending order
                all_sess.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
                options = {}
                for s in all_sess:
                    uid = s.get("user_id", "unknown")
                    sid = s["session_id"]
                    preview = s.get("last_preview", sid[:8] + "...")
                    t_val = s.get("updated_at", time.time())
                    t_str = time.strftime('%H:%M:%S', time.localtime(t_val)) if t_val else "recently"
                    label = f"[{t_str}] 👤 {uid} | {preview} ({sid[:6]})"
                    options[label] = (uid, sid)
                selected_label = st.radio("Select Thread (Sorted by Recent Activity):", list(options.keys()), key="obs_radio")
                selected_sess = options[selected_label] if selected_label else None
        except Exception as exc:
            st.error(f"Failed to fetch live sessions: {exc}")
            selected_sess = None
            
    with col2:
        st.markdown("#### 🔬 Live Thread Inspection & Intervention")
        if selected_sess:
            target_uid, target_sid = selected_sess
            try:
                thread_res = get_json(f"/api/v1/admin/sessions/{target_uid}/{target_sid}/messages")
                msgs = thread_res.get("messages", [])
                summary = thread_res.get("summary")
                
                if summary:
                    st.markdown(f'<div class="glass-card" style="border-left: 4px solid #38bdf8;"><b>🧠 Dense Background Memory Summary:</b><br>{summary}</div>', unsafe_allow_html=True)
                    
                st.markdown(f"**Viewing Session:** `{target_sid}` | **User Account:** `{target_uid}`")
                
                with st.container(height=400, border=True):
                    for m in msgs:
                        role_icon = "👤 User" if m["role"] == "user" else ("👑 Supervisor" if m["role"] == "supervisor" else "🤖 Copilot")
                        st.markdown(f"**{role_icon}** ({time.strftime('%H:%M:%S', time.localtime(m.get('timestamp', time.time())))}):")
                        st.markdown(m.get("content", ""))
                        if sources := m.get("sources"):
                            with st.expander(f"📚 Inspect {len(sources)} Cited RAG Sources (Confidence: {m.get('confidence', 0.0)*100:.0f}%)"):
                                st.json(sources)
                        st.divider()
                        
                st.markdown("#### 🚨 Inject Supervisor Guidance")
                intervene_msg = st.text_input("Type clarification or correction message for this thread...", key="inv_input")
                if st.button("Inject Message into Thread", type="primary", key="inv_btn"):
                    if intervene_msg:
                        post_json(f"/api/v1/admin/sessions/{target_uid}/{target_sid}/message", {"message": intervene_msg, "role": "supervisor"})
                        st.success("Supervisor intervention injected successfully!")
                        st.rerun()
            except Exception as exc:
                st.error(f"Could not load thread details: {exc}")
        else:
            st.caption("Select an active thread from the left column to inspect chat history, verify citations, and intervene.")


# Render Role-Specific UI Layouts
if st.session_state.role == "admin":
    chat_tab, obs_tab, docs_tab, admin_tab, eval_tab, langsmith_tab = st.tabs([
        "💬 Chat Copilot", 
        "👀 Live Observability",
        "📚 Documents", 
        "🛠️ Admin Portal", 
        "📊 RAGAS Evaluation", 
        "📈 LangSmith & Observability"
    ])
    with chat_tab:
        render_chat_tab()
    with obs_tab:
        render_observability_dashboard_tab()
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
        st.markdown("### ⚙️ System Status")
        try:
            ready_data = get_json("/ready")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Overall Status", "🟢 Ready" if ready_data.get("ready") else "🟡 Degraded")
            with cols[1]:
                st.metric("Vector Store", "🟢 Online" if ready_data.get("vector_store") else "🔴 Offline")
            with cols[2]:
                st.metric("LLM Provider", "🟢 Connected" if ready_data.get("llm") else "🔴 Offline")
            st.json(ready_data)
        except requests.RequestException as exc:
            st.error(f"Readiness check failed: {exc}")
        st.divider()
        st.info("💡 **Admin Notice**: To access Document Ingestion, RAGAS Benchmarks, and LangSmith Observability tools, please login as an Administrator using the authentication sidebar on the left.")
