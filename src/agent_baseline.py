from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            # Fallback to offline if live not fully wired
            pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
            
        session = self.sessions[thread_id]
        
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
        session.prompt_tokens_processed += prompt_tokens
        
        session.messages.append({"role": "user", "content": message})
        
        from memory_store import extract_profile_updates
        session_facts = {}
        for m in session.messages:
            if m["role"] == "user":
                facts = extract_profile_updates(m["content"])
                session_facts.update(facts)
                
        lower_msg = message.lower()
        response_text = "Tôi là AI."
        if "tên gì" in lower_msg or "tên mình" in lower_msg:
            response_text = f"Bạn tên là {session_facts.get('Name', 'không biết')}."
        elif "làm nghề" in lower_msg or "nghề gì" in lower_msg:
            response_text = f"Bạn làm nghề {session_facts.get('Profession', 'không biết')}."
        elif "sống ở" in lower_msg or "đến từ" in lower_msg or "ở đâu" in lower_msg:
            response_text = f"Bạn sống ở {session_facts.get('Location', 'không biết')}."
        elif "thích" in lower_msg or "style" in lower_msg:
            response_text = f"Sở thích/style: {session_facts.get('Preference', 'không biết')}."
            
        agent_tokens = estimate_tokens(response_text)
        session.token_usage += agent_tokens
        
        session.messages.append({"role": "assistant", "content": response_text})
        
        return {"output": response_text}

    def _maybe_build_langchain_agent(self):
        try:
            model = build_chat_model(self.config.model)
            if model:
                pass
        except Exception:
            pass
