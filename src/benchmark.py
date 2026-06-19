from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = sum(1 for e in expected if e.lower() in ans_lower)
    if matches == len(expected):
        return 1.0
    elif matches > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    score = recall_points(answer, expected)
    if len(answer) > 200:
        score *= 0.8
    return score


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    total_recall = 0.0
    total_quality = 0.0
    total_questions = 0
    threads = set()
    user_id = None
    
    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = str(uuid.uuid4())
        threads.add(thread_id)
        
        for turn in conv["turns"]:
            agent.reply(user_id, thread_id, turn)
            
        for rq in conv.get("recall_questions", []):
            q_thread_id = str(uuid.uuid4())
            threads.add(q_thread_id)
            
            resp = agent.reply(user_id, q_thread_id, rq["question"])
            ans = resp.get("output", "")
            
            r_score = recall_points(ans, rq["expected_contains"])
            q_score = heuristic_quality(ans, rq["expected_contains"])
            
            total_recall += r_score
            total_quality += q_score
            total_questions += 1
            
    agent_tokens = sum(agent.token_usage(t) for t in threads)
    prompt_tokens = sum(agent.prompt_token_usage(t) for t in threads)
    
    total_memory_growth = agent.memory_file_size(user_id) if hasattr(agent, "memory_file_size") and user_id else 0
    total_compactions = sum(agent.compaction_count(t) for t in threads) if hasattr(agent, "compaction_count") else 0

    avg_recall = (total_recall / total_questions) if total_questions else 0.0
    avg_quality = (total_quality / total_questions) if total_questions else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=total_memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    import tabulate
    headers = ["Agent", "Agent Tokens", "Prompt Tokens", "Recall", "Quality", "Memory (B)", "Compactions"]
    table_data = [
        [r.agent_name, r.agent_tokens_only, r.prompt_tokens_processed, f"{r.recall_score:.2f}", f"{r.response_quality:.2f}", r.memory_growth_bytes, r.compactions] 
        for r in rows
    ]
    return tabulate.tabulate(table_data, headers=headers, tablefmt="pipe")


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    
    data_dir = config.data_dir
    std_convs = load_conversations(data_dir / "conversations.json")
    stress_convs = load_conversations(data_dir / "advanced_long_context.json")
    if not stress_convs:
        stress_convs = std_convs
    
    print("Running Standard Benchmark...")
    if config.state_dir.exists():
        shutil.rmtree(config.state_dir)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    baseline_std = run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), std_convs, config)
    
    if config.state_dir.exists():
        shutil.rmtree(config.state_dir)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    advanced_std = run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), std_convs, config)
    
    print(format_rows([baseline_std, advanced_std]))
    
    print("\nRunning Stress Benchmark...")
    if config.state_dir.exists():
        shutil.rmtree(config.state_dir)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    baseline_stress = run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), stress_convs, config)
    
    if config.state_dir.exists():
        shutil.rmtree(config.state_dir)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    advanced_stress = run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), stress_convs, config)
    
    print(format_rows([baseline_stress, advanced_stress]))


if __name__ == "__main__":
    main()
