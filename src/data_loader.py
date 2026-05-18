"""Load and sample LongMemEval data."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

from .config import DATA_DIR, SEED, DEV_SIZE, TEST_SIZE, TOTAL_SAMPLE


def load_raw_data(filename: str = "longmemeval_oracle.json") -> List[Dict[str, Any]]:
    path = DATA_DIR / filename
    with open(path) as f:
        return json.load(f)


def sample_data(data: List[Dict], seed: int = SEED, total: int = TOTAL_SAMPLE):
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    selected = [data[i] for i in indices[:total]]
    return selected[:DEV_SIZE], selected[DEV_SIZE:DEV_SIZE + TEST_SIZE]


def extract_sessions(item: Dict, max_turns: int = 0) -> List[Dict]:
    """Extract all turns from all sessions, preserving session_id info.
    If max_turns > 0, truncate to that many turns (keep last ones as they're most recent)."""
    all_turns = []
    for sid, session in zip(item["haystack_session_ids"], item["haystack_sessions"]):
        for turn in session:
            all_turns.append({
                "session_id": sid,
                "role": turn["role"],
                "content": turn["content"],
                "has_answer": turn.get("has_answer", False),
            })
    if max_turns > 0 and len(all_turns) > max_turns:
        all_turns = all_turns[-max_turns:]
    return all_turns


def get_gold_turns(item: Dict) -> List[str]:
    """Get content of turns marked as gold evidence."""
    gold = []
    for session in item["haystack_sessions"]:
        for turn in session:
            if turn.get("has_answer", False):
                gold.append(turn["content"])
    return gold
