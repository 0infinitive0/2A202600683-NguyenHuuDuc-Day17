from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    config = load_config(Path(__file__).resolve().parent.parent)
    config.state_dir = tmp_path / "state"
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.compact_threshold_tokens = 20
    config.compact_keep_messages = 2
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    store.write_text("user1", "- Name: Alice")
    assert "Alice" in store.read_text("user1")
    store.edit_text("user1", "Alice", "Bob")
    assert "Bob" in store.read_text("user1")
    assert "Alice" not in store.read_text("user1")


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    
    for i in range(5):
        agent.reply("user1", "thread1", "Xin chào, đây là một câu dài để test threshold. " * 5)
        
    assert agent.compaction_count("thread1") > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    
    baseline.reply("user1", "thread1", "Tôi tên là Alice.")
    res_b = baseline.reply("user1", "thread2", "Tôi tên gì?")
    assert "Alice" not in res_b["output"]
    
    advanced.reply("user1", "thread1", "Tôi tên là Alice.")
    res_a = advanced.reply("user1", "thread2", "Tôi tên gì?")
    assert "Alice" in res_a["output"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    
    for i in range(10):
        msg = f"Nói cho tôi nghe câu chuyện thứ {i}. " * 5
        baseline.reply("user1", "thread1", msg)
        advanced.reply("user1", "thread1", msg)
        
    b_prompt = baseline.prompt_token_usage("thread1")
    a_prompt = advanced.prompt_token_usage("thread1")
    
    assert a_prompt < b_prompt
