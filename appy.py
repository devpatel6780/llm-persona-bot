import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Demo mode toggle (set DEMO_MODE="1" in Streamlit Secrets for public demo)
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
# ─────────────────────────────────────────────────────────────────────────────

import json
import yaml
import uuid
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# RAG deps
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# ─────────────────────────────────────────────────────────────────────────────
# MUST be the first Streamlit call
st.set_page_config(page_title="🤖 Groq Persona Chatbot", page_icon="🤖", layout="wide")
# ─────────────────────────────────────────────────────────────────────────────

# ========= Constants & paths =========
BASE_DIR = Path(".")
CHATS_DIR = BASE_DIR / "chats"
STORES_DIR = BASE_DIR / "stores"          # per-chat vector stores
PROFILE_PATH = BASE_DIR / "profile.json"

# Only ensure dirs when NOT in demo mode
if not DEMO_MODE:
    CHATS_DIR.mkdir(exist_ok=True)
    STORES_DIR.mkdir(exist_ok=True)

MODEL_CHOICES = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim

# ========= Env & client =========
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    st.error("❌ Missing GROQ_API_KEY (.env locally or Secrets in Streamlit Cloud).")
    st.stop()
client = Groq(api_key=API_KEY.strip())

# ========= Persona loading =========
@st.cache_data
def load_personas(path="personas.yaml"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
            return y.get("personas", {})
    except Exception:
        return {
            "career_coach": {
                "name": "Career Coach",
                "style": "Encouraging, practical, concise. Ask clarifying questions.",
                "guardrails": "Don't invent facts; be inclusive and supportive.",
                "examples": ["Help me prepare for a SWE interview."]
            },
            "math_tutor": {
                "name": "Math Tutor",
                "style": "Patient, step-by-step explanations with checks for understanding.",
                "guardrails": "Show steps; offer hints before final solutions.",
                "examples": ["Explain the chain rule with 2 examples."]
            },
        }

PERSONAS = load_personas()

# ========= User Profile =========
DEFAULT_PROFILE = {
    "name": "",
    "role_or_studies": "",
    "skills": "",
    "goals": "",
    "tone_preferences": "",
}

def load_profile() -> dict:
    if DEMO_MODE:
        # keep profile in session only (no disk)
        if "PROFILE" not in st.session_state:
            st.session_state.PROFILE = DEFAULT_PROFILE.copy()
        return st.session_state.PROFILE

    if PROFILE_PATH.exists():
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k in DEFAULT_PROFILE:
                data.setdefault(k, DEFAULT_PROFILE[k])
            return data
        except Exception:
            return DEFAULT_PROFILE.copy()
    return DEFAULT_PROFILE.copy()

def save_profile(p: dict):
    clean = DEFAULT_PROFILE.copy()
    clean.update({k: (p.get(k) or "").strip() for k in DEFAULT_PROFILE})
    if DEMO_MODE:
        st.session_state.PROFILE = clean
        return
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

PROFILE = load_profile()

def profile_summary_text(p: dict) -> str:
    bits = []
    if p.get("name"): bits.append(f'User name: {p["name"]}.')
    if p.get("role_or_studies"): bits.append(f'Role/Studies: {p["role_or_studies"]}.')
    if p.get("skills"): bits.append(f'Skills: {p["skills"]}.')
    if p.get("goals"): bits.append(f'Goals: {p["goals"]}.')
    if p.get("tone_preferences"): bits.append(f'Tone preferences: {p["tone_preferences"]}.')
    return " ".join(bits) if bits else "No additional user profile context provided."

# ========= System prompt =========
def build_system_prompt(persona_obj: dict, profile_obj: dict, rag_context: str | None) -> str:
    profile_text = profile_summary_text(profile_obj)
    context_block = f"\n\nRETRIEVAL CONTEXT (verbatim quotes; cite with [1], [2], ...):\n{rag_context}\n" if rag_context else ""
    return f"""
You are the "{persona_obj['name']}" persona.

STYLE:
{persona_obj['style']}

GUARDRAILS:
{persona_obj['guardrails']}

USER PROFILE (for personalization; do not reveal verbatim):
{profile_text}
{context_block}
GENERAL BEHAVIOR:
- Be concise by default and use Markdown.
- If you use any info from the Retrieval Context, cite it inline as [1], [2], etc.
- If the answer is not in the context, say you don't know or explain how to find it.
""".strip()

# ========= Persistence (disk) =========
def chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"

def store_dir(chat_id: str) -> Path:
    d = STORES_DIR / chat_id
    d.mkdir(exist_ok=True)
    return d

def list_chats() -> List[dict]:
    if DEMO_MODE:
        return []  # no persisted chats in demo
    chats = []
    for p in CHATS_DIR.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            chats.append({
                "id": data["id"],
                "title": data.get("title", "Untitled"),
                "persona_key": data.get("persona_key"),
                "model": data.get("model"),
                "updated_at": data.get("updated_at"),
                "created_at": data.get("created_at"),
            })
        except Exception:
            continue
    chats.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
    return chats

def load_chat(chat_id: str) -> dict:
    if DEMO_MODE:
        # keep active chat solely in session
        return st.session_state.get("ACTIVE_CHAT", None)
    with open(chat_path(chat_id), "r", encoding="utf-8") as f:
        return json.load(f)

def save_chat(chat: dict):
    if DEMO_MODE:
        st.session_state.ACTIVE_CHAT = chat
        return
    chat["updated_at"] = datetime.utcnow().isoformat()
    with open(chat_path(chat["id"]), "w", encoding="utf-8") as f:
        json.dump(chat, f, ensure_ascii=False, indent=2)

def new_chat(persona_key: str, model: str, title: str | None = None) -> dict:
    chat_id = uuid.uuid4().hex[:12]
    persona_name = PERSONAS.get(persona_key, {}).get("name", persona_key)
    if not title:
        title = f"{persona_name} — {datetime.now().strftime('%b %d, %H:%M')}"
    now = datetime.utcnow().isoformat()
    chat = {
        "id": chat_id,
        "title": title,
        "persona_key": persona_key,
        "model": model,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    save_chat(chat)
    return chat

def delete_chat(chat_id: str):
    if DEMO_MODE:
        st.session_state.ACTIVE_CHAT = None
        return
    try:
        chat_path(chat_id).unlink(missing_ok=True)
    except Exception:
        pass
    sd = STORES_DIR / chat_id
    if sd.exists():
        shutil.rmtree(sd, ignore_errors=True)

def to_markdown(chat: dict, persona_prompt: str) -> str:
    lines = [
        f"# {chat['title']}", "",
        f"**Persona:** {PERSONAS[chat['persona_key']]['name']}",
        f"**Model:** {chat['model']}", "",
        "## System Prompt", "", "```", persona_prompt, "```", "",
        "## Conversation", ""
    ]
    for m in chat["messages"]:
        role = m["role"].capitalize()
        lines += [f"**{role}:**", "", m["content"], ""]
    return "\n".join(lines)

# ========= Auto-title =========
def generate_chat_title(model: str, persona_name: str, user_text: str, assistant_text: str) -> str:
    fallback = (user_text or "New chat").strip()
    fallback = " ".join(fallback.split()[:6])
    try:
        prompt = f"""You are naming a chat thread.

Persona: {persona_name}

Rules:
- Return a concise title (3–6 words), Title Case.
- No trailing punctuation. No quotes.
- Reflect the user's goal/topic, not generic words like "Chat".

User message:
{user_text}

Assistant reply:
{assistant_text}

Return only the title text."""
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You create concise, clear titles for chat threads."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        title = resp.choices[0].message.content.strip()
        title = title.strip('"\''" ”“").rstrip(".!?")
        return title or fallback
    except Exception:
        return fallback

# ========= Streaming helper =========
def get_reply_streaming_safe(model: str, messages: list, temperature: float, placeholder):
    full = ""
    try:
        for chunk in client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        ):
            token = None
            try:
                token = chunk.choices[0].delta.content
            except Exception:
                token = None
            if token:
                full += token
                placeholder.markdown(full)
        return full
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content

# ========= Embedding / chunking =========
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks

# ========= File extractors =========
def extract_text_from_pdf_filelike(file) -> List[Tuple[str, int]]:
    # for DEMO_MODE in-memory uploads
    out = []
    reader = PdfReader(file)
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            out.append((txt, i + 1))
    return out

def extract_text_from_pdf_path(path: Path) -> List[Tuple[str, int]]:
    out = []
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt.strip():
                out.append((txt, i + 1))
    return out

def extract_text_from_txt_path(path: Path) -> List[Tuple[str, int]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return [(txt, 1)]

# ========= RAG (DEMO_MODE: in-memory) =========
def _ensure_demo_rag():
    if "rag_index" not in st.session_state:
        st.session_state.rag_index = None
    if "rag_meta" not in st.session_state:
        st.session_state.rag_meta = {"docs": []}

def demo_store_exists() -> bool:
    _ensure_demo_rag()
    return st.session_state.rag_index is not None and len(st.session_state.rag_meta["docs"]) > 0

def demo_list_store_sources() -> list:
    _ensure_demo_rag()
    return [d["source"] for d in st.session_state.rag_meta["docs"]]

def demo_clear_store():
    _ensure_demo_rag()
    st.session_state.rag_index = None
    st.session_state.rag_meta = {"docs": []}

def demo_build_store(uploaded_files):
    _ensure_demo_rag()
    embedder = load_embedder()
    added = []
    dim = 384

    if st.session_state.rag_index is None:
        st.session_state.rag_index = faiss.IndexFlatIP(dim)

    for file in uploaded_files:
        name = file.name
        ext = name.lower().rsplit(".", 1)[-1]
        if ext not in ("pdf", "txt"):
            continue

        # Extract
        if ext == "pdf":
            pages = extract_text_from_pdf_filelike(file)
        else:
            txt = file.read().decode("utf-8", errors="ignore")
            pages = [(txt, 1)]

        # Chunk, embed, add
        chunks_all = []
        for page_text, page_num in pages:
            for ch in chunk_text(page_text):
                chunks_all.append({"text": ch, "page": page_num})
        if not chunks_all:
            continue

        vecs = embedder.encode([c["text"] for c in chunks_all], batch_size=64, normalize_embeddings=True)
        st.session_state.rag_index.add(np.array(vecs, dtype="float32"))

        st.session_state.rag_meta["docs"].append({"source": name, "chunks": chunks_all})
        added.append({"file": name, "chunks": len(chunks_all)})

    return {"added": added, "total_docs": len(st.session_state.rag_meta["docs"])}

def demo_retrieve_context(query: str, k: int = 5):
    _ensure_demo_rag()
    if st.session_state.rag_index is None:
        return "", []
    flat = []
    for d in st.session_state.rag_meta["docs"]:
        src = d["source"]
        for ch in d["chunks"]:
            flat.append({"text": ch["text"], "page": ch.get("page", 1), "source": src})
    if not flat:
        return "", []

    embedder = load_embedder()
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = st.session_state.rag_index.search(q, min(k, len(flat)))
    idxs = idxs[0].tolist()

    citations, lines = [], []
    for i, idx in enumerate(idxs, start=1):
        item = flat[idx]
        snippet = " ".join((item["text"] or "").split())[:800]
        citations.append({"idx": i, "source": item["source"], "page": item["page"], "text": item["text"]})
        lines.append(f"[{i}] (source: {item['source']} p.{item['page']}): {snippet}")
    return "\n".join(lines), citations

# ========= RAG (disk) =========
def build_store_for_chat(chat_id: str, uploaded_files) -> Dict:
    if DEMO_MODE:
        st.warning("RAG storage disabled in demo mode — using in-memory only.")
        return {"added": [], "total_docs": 0}

    sd = store_dir(chat_id)
    index_path = sd / "index.faiss"
    meta_path = sd / "meta.json"

    dim = 384
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        index = faiss.IndexFlatIP(dim)
        meta = {"docs": []}

    embedder = load_embedder()
    added = []

    for file in uploaded_files:
        fname = file.name
        ext = fname.lower().rsplit(".", 1)[-1]
        tmp_path = sd / f"_tmp_{uuid.uuid4().hex}.{ext}"
        with open(tmp_path, "wb") as f:
            f.write(file.getbuffer())

        if ext == "pdf":
            pages = extract_text_from_pdf_path(tmp_path)
        elif ext == "txt":
            pages = extract_text_from_txt_path(tmp_path)
        else:
            tmp_path.unlink(missing_ok=True)
            continue

        chunks_all = []
        for page_text, page_num in pages:
            for ch in chunk_text(page_text):
                chunks_all.append({"text": ch, "page": page_num})
        if not chunks_all:
            tmp_path.unlink(missing_ok=True)
            continue

        vecs = embedder.encode([c["text"] for c in chunks_all], batch_size=64, normalize_embeddings=True)
        index.add(np.array(vecs, dtype="float32"))

        meta["docs"].append({"source": fname, "chunks": chunks_all})
        added.append({"file": fname, "chunks": len(chunks_all)})

        tmp_path.unlink(missing_ok=True)

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {"added": added, "total_docs": len(meta["docs"])}

def store_exists(chat_id: str) -> bool:
    if DEMO_MODE:
        return False
    sd = STORES_DIR / chat_id
    return (sd / "index.faiss").exists() and (sd / "meta.json").exists()

def list_store_sources(chat_id: str) -> List[str]:
    if DEMO_MODE:
        return []
    sd = STORES_DIR / chat_id
    meta_path = sd / "meta.json"
    if not meta_path.exists():
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return [d["source"] for d in meta.get("docs", [])]

def retrieve_context(chat_id: str, query: str, k: int = 5) -> Tuple[str, List[Dict]]:
    if DEMO_MODE:
        return "", []
    sd = STORES_DIR / chat_id
    index_path = sd / "index.faiss"
    meta_path = sd / "meta.json"
    if not index_path.exists() or not meta_path.exists():
        return "", []

    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    flat = []
    for d in meta.get("docs", []):
        src = d["source"]
        for ch in d["chunks"]:
            flat.append({"text": ch["text"], "page": ch.get("page", 1), "source": src})
    if not flat:
        return "", []

    embedder = load_embedder()
    q_vec = embedder.encode([query], normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype="float32")
    scores, idxs = index.search(q_vec, min(k, len(flat)))
    idxs = idxs[0].tolist()

    citations = []
    lines = []
    for i, idx in enumerate(idxs, start=1):
        item = flat[idx]
        snippet = " ".join(item["text"].split())[:800]
        citations.append({"idx": i, "source": item["source"], "page": item["page"], "text": item["text"]})
        lines.append(f"[{i}] (source: {item['source']} p.{item['page']}): {snippet}")

    return "\n".join(lines), citations

# ========= Session bootstrap =========
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "cached_chat" not in st.session_state:
    st.session_state.cached_chat = None

# ========= Sidebar =========
with st.sidebar:
    st.subheader("🗂️ Chats")
    default_persona_key = st.selectbox(
        "Default Persona for New Chat",
        options=list(PERSONAS.keys()),
        format_func=lambda k: PERSONAS[k]["name"],
        index=0,
        key="default_persona_selector",
    )
    default_model = st.selectbox("Default Model", MODEL_CHOICES, index=0, key="default_model_selector")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)

    if st.button("➕ New Chat"):
        chat = new_chat(default_persona_key, default_model)
        st.session_state.current_chat_id = chat["id"]
        st.session_state.cached_chat = chat

    # Existing chats list (hidden in demo)
    available = list_chats()
    if available:
        labels = [
            f"{c['title']}  ·  {PERSONAS.get(c['persona_key'], {}).get('name', c['persona_key'])}"
            for c in available
        ]
        sel_index_default = (
            0 if st.session_state.current_chat_id is None
            else next((i for i, c in enumerate(available) if c["id"] == st.session_state.current_chat_id), 0)
        )
        sel = st.selectbox("Open a chat", options=range(len(available)),
                           format_func=lambda i: labels[i], index=sel_index_default)
        selected_meta = available[sel]
        if selected_meta["id"] != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected_meta["id"]
            st.session_state.cached_chat = load_chat(st.session_state.current_chat_id)

        if st.session_state.cached_chat:
            with st.expander("✏️ Rename / Export / Delete"):
                current = st.session_state.cached_chat
                new_title = st.text_input("Title", value=current["title"])
                if st.button("Save Title"):
                    current["title"] = new_title.strip() or current["title"]
                    save_chat(current)
                    st.success("Saved title")

                # Auto Title
                if st.button("✨ Auto Title"):
                    if current["messages"]:
                        first_user = next((m["content"] for m in current["messages"] if m["role"] == "user"), "")
                        last_assistant = next((m["content"] for m in reversed(current["messages"]) if m["role"] == "assistant"), "")
                        new_title = generate_chat_title(
                            model=current["model"],
                            persona_name=PERSONAS[current["persona_key"]]["name"],
                            user_text=first_user,
                            assistant_text=last_assistant,
                        )
                        current["title"] = new_title
                        save_chat(current)
                        st.success(f"Renamed to: {new_title}")
                        st.rerun()
                    else:
                        st.info("Start the conversation first, then try auto-titling.")

                # Export Markdown
                persona_prompt = build_system_prompt(PERSONAS[current["persona_key"]], PROFILE, rag_context=None)
                md = to_markdown(current, persona_prompt)
                st.download_button("⬇️ Download Markdown", data=md, file_name=f"{current['title']}.md")

                # Delete
                if st.button("🗑️ Delete this chat"):
                    delete_chat(current["id"])
                    st.session_state.current_chat_id = None
                    st.session_state.cached_chat = None
                    st.rerun()
    else:
        st.caption("No persisted chats here." if DEMO_MODE else "No chats yet. Click **New Chat** to start.")

    # Profile editor
    st.markdown("---")
    st.subheader("👤 Profile (personalizes replies)")
    with st.form("profile_form"):
        name = st.text_input("Your name", value=PROFILE.get("name", ""))
        role = st.text_input("Role / Studies", value=PROFILE.get("role_or_studies", ""))
        skills = st.text_area("Skills (comma-separated)", value=PROFILE.get("skills", ""))
        goals = st.text_area("Goals (1–3 lines)", value=PROFILE.get("goals", ""))
        tone = st.text_input("Tone preferences", value=PROFILE.get("tone_preferences", ""))
        saved = st.form_submit_button("💾 Save Profile")
    if saved:
        PROFILE.update({
            "name": name,
            "role_or_studies": role,
            "skills": skills,
            "goals": goals,
            "tone_preferences": tone,
        })
        save_profile(PROFILE)
        st.success("Profile saved. New replies will use this context.")

    # RAG controls
    st.markdown("---")
    st.subheader("📚 Knowledge (RAG) for this chat")
    if DEMO_MODE:
        sources = demo_list_store_sources()
    else:
        if st.session_state.current_chat_id:
            sources = list_store_sources(st.session_state.current_chat_id)
        else:
            sources = []

    if sources:
        st.caption("Indexed files:")
        for s in sources:
            st.write(f"• {s}")
    else:
        st.caption("No files added yet.")

    files = st.file_uploader("Add PDFs or .txt files", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("➕ Add to Knowledge") and files:
        with st.spinner("Indexing…"):
            if DEMO_MODE:
                summary = demo_build_store(files)
            else:
                if st.session_state.current_chat_id:
                    summary = build_store_for_chat(st.session_state.current_chat_id, files)
                else:
                    st.warning("Create or open a chat first.")
                    summary = {"added": [], "total_docs": 0}
        st.success(f"Added {len(summary['added'])} file(s). Total docs: {summary['total_docs']}.")
        st.rerun()

    if st.button("🧹 Clear Knowledge"):
        if DEMO_MODE:
            demo_clear_store()
        else:
            if st.session_state.current_chat_id:
                clear_store(st.session_state.current_chat_id)
        st.success("Cleared knowledge store.")
        st.rerun()

# ========= Load active chat (init) =========
if st.session_state.current_chat_id and not st.session_state.cached_chat:
    st.session_state.cached_chat = load_chat(st.session_state.current_chat_id)

if not st.session_state.current_chat_id:
    # In demo mode we still keep one active in session so UI isn't empty
    default = new_chat(list(PERSONAS.keys())[0], MODEL_CHOICES[0], title="Welcome Chat")
    st.session_state.current_chat_id = default["id"]
    st.session_state.cached_chat = default

active_chat = st.session_state.cached_chat
persona_key = active_chat["persona_key"]
model = active_chat["model"]
persona = PERSONAS[persona_key]

# ========= Main area =========
st.title("🤖 Groq Persona Chatbot")
if DEMO_MODE:
    st.caption("**Demo Mode** is ON — no data is written to disk; uploads & chats live only in memory per session.")
st.caption(f"**Chat:** {active_chat['title']}  •  **Persona:** {persona['name']}  •  **Model:** {model}")

# Show history
for m in active_chat["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Examples
with st.expander("Try example prompts"):
    for ex in persona.get("examples", []):
        st.code(ex)

# Input
user_text = st.chat_input("Type your message...")

if user_text:
    active_chat["messages"].append({"role": "user", "content": user_text})
    save_chat(active_chat)

    # RAG retrieval
    rag_context, citations = "", []
    if DEMO_MODE:
        if demo_store_exists():
            rag_context, citations = demo_retrieve_context(user_text, k=5)
    else:
        if store_exists(active_chat["id"]):
            rag_context, citations = retrieve_context(active_chat["id"], user_text, k=5)

    system_prompt = build_system_prompt(persona, PROFILE, rag_context)
    api_messages = [{"role": "system", "content": system_prompt}] + active_chat["messages"]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        reply = get_reply_streaming_safe(
            model=model,
            messages=api_messages,
            temperature=temperature,
            placeholder=placeholder,
        )
        if citations:
            src_lines = [f"[{c['idx']}] {c['source']} (p.{c['page']})" for c in citations]
            sources_md = "\n\n**Sources:**\n" + "\n".join(f"- {line}" for line in src_lines)
            reply = reply + sources_md
            placeholder.markdown(reply)

    active_chat["messages"].append({"role": "assistant", "content": reply})
    save_chat(active_chat)

    # Auto-title on first exchange if looks default
    try:
        first_turn = sum(1 for m in active_chat["messages"] if m["role"] == "user") == 1
        looks_default = ("—" in active_chat["title"]) or active_chat["title"].lower().startswith(("welcome chat", "untitled"))
        if first_turn and looks_default:
            first_user = next((m["content"] for m in active_chat["messages"] if m["role"] == "user"), "")
            new_title = generate_chat_title(
                model=active_chat["model"],
                persona_name=PERSONAS[active_chat["persona_key"]]["name"],
                user_text=first_user,
                assistant_text=reply,
            )
            active_chat["title"] = new_title
            save_chat(active_chat)
            st.toast(f"📝 Titled chat: {new_title}")
    except Exception:
        pass

    st.rerun()
