from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            pass
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        facts = extract_profile_updates(message)
        if facts:
            current_profile = self.profile_store.read_text(user_id)
            lines = current_profile.splitlines() if current_profile else []
            new_lines = []
            for k, v in facts.items():
                found = False
                for i, line in enumerate(lines):
                    if line.startswith(f"- {k}:"):
                        lines[i] = f"- {k}: {v}"
                        found = True
                        break
                if not found:
                    new_lines.append(f"- {k}: {v}")
            
            updated = "\n".join(lines + new_lines)
            if updated.strip() != current_profile.strip():
                self.profile_store.write_text(user_id, updated.strip())

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id) + estimate_tokens(message)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        self.compact_memory.append(thread_id, "user", message)

        response_text = self._offline_response(user_id, thread_id, message)

        agent_tokens = estimate_tokens(response_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens

        self.compact_memory.append(thread_id, "assistant", response_text)

        return {"output": response_text}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_text = self.profile_store.read_text(user_id)
        tokens = estimate_tokens(profile_text)
        
        ctx = self.compact_memory.context(thread_id)
        tokens += estimate_tokens(str(ctx.get("summary", "")))
        for m in ctx.get("messages", []):
            if isinstance(m, dict) and "content" in m:
                tokens += estimate_tokens(m["content"])
            
        return tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        profile_text = self.profile_store.read_text(user_id)
        lower_msg = message.lower()
        
        response_text = "Tôi là AI tiên tiến."
        if "tên gì" in lower_msg or "tên mình" in lower_msg or "ai" in lower_msg:
            if "- Name:" in profile_text or "- name:" in profile_text.lower():
                for line in profile_text.splitlines():
                    if line.lower().startswith("- name:"):
                        # Extract the value properly without losing case
                        val = line[len("- name:"):].strip() if line.lower().startswith("- name:") else ""
                        response_text = f"Bạn tên là {val}."
            else:
                response_text = "Tôi không biết tên bạn."
        elif "làm nghề" in lower_msg or "nghề gì" in lower_msg:
            if "- profession:" in profile_text.lower():
                for line in profile_text.splitlines():
                    if line.lower().startswith("- profession:"):
                        val = line[len("- profession:"):].strip()
                        response_text = f"Bạn làm nghề {val}."
            else:
                response_text = "Tôi không biết nghề của bạn."
        elif "thích" in lower_msg or "style" in lower_msg:
            if "- preference:" in profile_text.lower():
                for line in profile_text.splitlines():
                    if line.lower().startswith("- preference:"):
                        val = line[len("- preference:"):].strip()
                        response_text = f"Sở thích/style: {val}."
            else:
                response_text = "Tôi chưa biết sở thích của bạn."
        elif "sống ở" in lower_msg or "ở đâu" in lower_msg:
            if "- location:" in profile_text.lower():
                for line in profile_text.splitlines():
                    if line.lower().startswith("- location:"):
                        val = line[len("- location:"):].strip()
                        response_text = f"Bạn sống ở {val}."
            else:
                response_text = "Tôi không biết bạn sống ở đâu."
        else:
            ctx = self.compact_memory.context(thread_id)
            summary = str(ctx.get("summary", "")).lower()
            if "thơ" in summary and "bài thơ" in lower_msg:
                response_text = "Tôi nhớ bạn có nhắc đến bài thơ."
            
        return response_text

    def _maybe_build_langchain_agent(self):
        try:
            model = build_chat_model(self.config.model)
            if model:
                pass
        except Exception:
            pass
