from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    return len(text.strip()) // 4 if text.strip() else 0


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() else "_" for c in user_id)
        return self.root_dir / f"{safe_id}.md"

    def read_text(self, user_id: str) -> str:
        p = self.path_for(user_id)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return ""

    def write_text(self, user_id: str, content: str) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        p = self.path_for(user_id)
        p.write_text(content, encoding="utf-8")
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        p = self.path_for(user_id)
        if not p.exists():
            return False
        content = p.read_text(encoding="utf-8")
        if search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            p.write_text(new_content, encoding="utf-8")
            return True
        return False

    def file_size(self, user_id: str) -> int:
        p = self.path_for(user_id)
        return p.stat().st_size if p.exists() else 0


def extract_profile_updates(message: str) -> dict[str, str]:
    facts = {}
    lower_msg = message.lower()
    
    # Confidence threshold / Reject questions: do not extract facts from questions or recall prompts
    if "?" in message or "có phải" in lower_msg or "đúng không" in lower_msg or "bạn biết" in lower_msg or "nhắc lại" in lower_msg or "tên gì" in lower_msg:
        return facts
    
    if "tên tôi là" in lower_msg or "tôi tên là" in lower_msg or "tôi tên" in lower_msg or "mình tên" in lower_msg:
        match = re.search(r'(?:tên tôi là|tôi tên là|tôi tên|mình tên)\s+([^.,;?!]+)', message, re.IGNORECASE)
        if match:
            facts["Name"] = match.group(1).strip()
            
    if "làm nghề" in lower_msg or "tôi là một" in lower_msg or "mình làm" in lower_msg:
        match = re.search(r'(?:làm nghề|tôi là một|mình làm)\s+([^.,;?!]+)', message, re.IGNORECASE)
        if match:
            facts["Profession"] = match.group(1).strip()
            
    if "sống ở" in lower_msg or "đến từ" in lower_msg:
        match = re.search(r'(?:sống ở|đến từ)\s+([^.,;?!]+)', message, re.IGNORECASE)
        if match:
            facts["Location"] = match.group(1).strip()
            
    if "hãy" in lower_msg or "thích" in lower_msg or "gọi mình là" in lower_msg:
        match = re.search(r'(?:hãy|thích|gọi mình là)\s+([^.,;?!]+)', message, re.IGNORECASE)
        if match:
            facts["Preference"] = match.group(1).strip()

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    summary = "SUMMARY:\n"
    for m in messages[-max_items:]:
        summary += f"[{m['role']}]: {m['content']}\n"
    return summary.strip()


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0}
            
        thread = self.state[thread_id]
        thread["messages"].append({"role": role, "content": content})
        
        total_tokens = sum(estimate_tokens(m["content"]) for m in thread["messages"])
        if total_tokens > self.threshold_tokens and len(thread["messages"]) > self.keep_messages:
            to_summarize = thread["messages"][:-self.keep_messages]
            new_summary_text = summarize_messages(to_summarize)
            if thread["summary"]:
                thread["summary"] = thread["summary"] + "\n" + new_summary_text
            else:
                thread["summary"] = new_summary_text
            
            # For offline benchmarking, prevent summary from growing infinitely like an append-only log
            if len(thread["summary"]) > 500:
                thread["summary"] = "...[compacted]...\n" + thread["summary"][-450:]
                
            thread["messages"] = thread["messages"][-self.keep_messages:]
            thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            return {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        if thread_id not in self.state:
            return 0
        return self.state[thread_id].get("compactions", 0)
