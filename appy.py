# appy.py — LLM Persona Chatbot (Streamlit + Groq), NO RAG
# Run: streamlit run appy.py

# Disable TF + quiet tokenizers before any transformers-adjacent libs load
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import yaml
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# ── Streamlit page setup (must be first Streamlit call)
st.set_page_config(page_title="🤖 Groq Persona Chatbot", page_icon="🤖", layout="wide")

# ── Secrets/env helpers
load_dotenv()
def get_secret_or_env(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets[key]  # type: ignore[index]
    except Exception:
        return os.getenv(key, default)

DEMO_MODE = str(get_secret_or_env("DEMO_MODE", "0")).strip().strip('"') == "1"
API_KEY = get_secret_or_env("GROQ_API_KEY")
if not API_KEY:
    st.error("❌ Missing GROQ_API_KEY (.env locally or Secrets in Streamlit Cloud).")
    st.stop()
client = Groq(api_key=str(API_KEY).strip())

# ── Paths (used when not in demo)
BASE_DIR = Path(".")
CHATS_DIR = BASE_DIR / "chats"
PROFILE_PATH = BASE_DIR / "profile.json"
if not DEMO_MODE:
    CHATS_DIR.mkdir(exist_ok=True)

MODEL_CHOICES = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

# ── Personas
@st.cache_data
def load_personas(path: str = "personas.yaml") -> Dict[str, dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
            return y.get("personas", {}) or {}
    except Exception:
        # Fallback defaults
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

# ── Profile
DEFAULT_PROFILE = {
    "name": "", "role_or_studies": "", "skills": "", "goals": "", "tone_preferences": "",
}
def load_profile() -> dict:
    if DEMO_MODE:
        if "PROFILE" not in st.session_state:
            st.session_state.PROFILE = DEFAULT_PROFILE.copy()
        return st.session_state.PROFILE
    if PROFILE_PATH.exists():
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k in DEFAULT_PROFILE: data.setdefault(k, DEFAULT_PROFILE[k])
            return data
        except Exception:
            return DEFAULT_PROFILE.copy()
    return DEFAULT_PROFILE.copy()
def save_profile(p: dict):
    clean = DEFAULT_PROFILE.copy()
    clean.update({k: (p.get(k) or "").strip() for k in DEFAULT_PROFILE})
    if DEMO_MODE:
        st.session_state.PROFILE = clean
    else:
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
PROFILE = load_profile()

def profile_summary_text(p: dict) -> str:
    bits = []
    if p.get("name"): bits.append(f'User name: {p["name"]}.')
    if p.get("role_or_studies"): bits.append(f'Role/Studies: {p["role_or_studies"]}.')
    if p.get("skills"): bits.append(f'Skills: {p["skills"]}.')
    if p.get("goals"): bits.append(f'Goals: {p["goals"]}.')
    if p.get("tone_preferences"): bits.append(f'Tone: {p["tone_preferences"]}.')
    return " ".join(bits) if bits else "No additional user profile context provided."

# ── System prompt
def build_system_prompt(persona_obj: dict, profile_obj: dict) -> str:
    profile_text = profile_summary_text(profile_obj)
    return f"""
You are the "{persona_obj['name']}" persona.

STYLE:
{persona_obj['style']}

GUARDRAILS:
{persona_obj['guardrails']}

USER PROFILE (for personalization; do not reveal verbatim):
{profile_text}

GENERAL BEHAVIOR:
- Be concise by default and use Markdown.
- Ask for clarification if the user's request is ambiguous.
- If you are unsure, say so briefly and suggest next steps.
""".strip()

# ── Persistence (no RAG)
def chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"

def list_chats() -> List[dict]:
    if DEMO_MODE:
        return []
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
        return st.session_state.get("ACTIVE_CHAT", None)
    with open(chat_path(chat_id), "r", encoding="utf-8") as f:
        return json.load(f)

def save_chat(chat: dict):
    if DEMO_MODE:
        st.session_state.ACTIVE_CHAT = chat
        return
    chat["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(chat_path(chat["id"]), "w", encoding="utf-8") as f:
        json.dump(chat, f, ensure_ascii=False, indent=2)

def new_chat(persona_key: str, model: str, title: Optional[str] = None) -> dict:
    chat_id = uuid.uuid4().hex[:12]
    persona_name = PERSONAS.get(persona_key, {}).get("name", persona_key)
    if not title:
        title = f"{persona_name} — {datetime.now().strftime('%b %d, %H:%M')}"
    now = datetime.now(timezone.utc).isoformat()
    chat = {
        "id": chat_id, "title": title, "persona_key": persona_key,
        "model": model, "created_at": now, "updated_at": now, "messages": [],
    }
    save_chat(chat); return chat

def delete_chat(chat_id: str):
    if DEMO_MODE:
        st.session_state.ACTIVE_CHAT = None; return
    try: chat_path(chat_id).unlink(missing_ok=True)
    except Exception: pass

def to_markdown(chat: dict, persona_prompt: str) -> str:
    lines = [f"# {chat['title']}", "",
             f"**Persona:** {PERSONAS[chat['persona_key']]['name']}",
             f"**Model:** {chat['model']}", "",
             "## System Prompt", "", "```", persona_prompt, "```", "",
             "## Conversation", ""]
    for m in chat["messages"]:
        role = m["role"].capitalize(); lines += [f"**{role}:**", "", m["content"], ""]
    return "\n".join(lines)

# ── Auto-title
def generate_chat_title(model: str, persona_name: str, user_text: str, assistant_text: str) -> str:
    fallback = " ".join((user_text or "New chat").strip().split()[:6])
    try:
        prompt = f"""You are naming a chat thread.

Persona: {persona_name}
Rules:
- 3–6 words, Title Case, no quotes or punctuation.
User: {user_text}
Assistant: {assistant_text}
Return only the title text."""
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You create concise, clear titles for chat threads."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        return (content or fallback).strip('"\''" ”“").rstrip(".!?") or fallback
    except Exception:
        return fallback

# ── Streaming helper
def get_reply_streaming_safe(model: str, messages: list, temperature: float, placeholder):
    full = ""
    try:
        for chunk in client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, stream=True
        ):
            token = None
            try: token = chunk.choices[0].delta.content
            except Exception: token = None
            if token:
                full += token
                placeholder.markdown(full)
        return full
    except Exception:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        return resp.choices[0].message.content

# ── Session bootstrap
if "current_chat_id" not in st.session_state: st.session_state.current_chat_id = None
if "cached_chat" not in st.session_state: st.session_state.cached_chat = None

# ── Sidebar
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
                persona_prompt = build_system_prompt(PERSONAS[current["persona_key"]], PROFILE)
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

    # Quick utility
    if st.session_state.get("cached_chat"):
        if st.button("🧼 Clear Chat Messages"):
            st.session_state.cached_chat["messages"] = []
            save_chat(st.session_state.cached_chat)
            st.success("Cleared chat.")
            st.rerun()

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

# ── Load active chat (init)
if st.session_state.current_chat_id and not st.session_state.cached_chat:
    st.session_state.cached_chat = load_chat(st.session_state.current_chat_id)
if not st.session_state.current_chat_id:
    default = new_chat(list(PERSONAS.keys())[0], MODEL_CHOICES[0], title="Welcome Chat")
    st.session_state.current_chat_id = default["id"]
    st.session_state.cached_chat = default

active_chat = st.session_state.cached_chat
persona_key = active_chat["persona_key"]
model = active_chat["model"]
persona = PERSONAS[persona_key]

# ── Main area
st.title("🤖 Groq Persona Chatbot")
if DEMO_MODE:
    st.caption("**Demo Mode** is ON — no data is written to disk; chats live only in memory per session.")
st.caption(f"**Chat:** {active_chat['title']}  •  **Persona:** {persona['name']}  •  **Model:** {model}")

# Chat history
for m in active_chat["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Examples
with st.expander("Try example prompts"):
    for ex in persona.get("examples", []):
        st.code(ex)

# Input box
user_text = st.chat_input("Type your message…")

if user_text:
    active_chat["messages"].append({"role": "user", "content": user_text})
    save_chat(active_chat)

    system_prompt = build_system_prompt(persona, PROFILE)
    api_messages = [{"role": "system", "content": system_prompt}] + active_chat["messages"]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        reply = get_reply_streaming_safe(
            model=model, messages=api_messages, temperature=st.session_state.get("temperature", 0.4), placeholder=placeholder
        )

    active_chat["messages"].append({"role": "assistant", "content": reply})
    save_chat(active_chat)

    # Auto-title on first exchange if title is default-like
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
