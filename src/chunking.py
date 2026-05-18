"""Text chunking — split turn content into fixed-size word chunks."""

from typing import List, Dict
from .config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_turns(turns: List[Dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Split turns into word-level chunks. Each chunk keeps metadata about source."""
    chunks = []
    chunk_id = 0
    for turn_idx, turn in enumerate(turns):
        words = turn["content"].split()
        if len(words) <= chunk_size + 40:
            chunks.append({
                "chunk_id": chunk_id,
                "text": turn["content"],
                "source_turn": turn_idx,
                "session_id": turn["session_id"],
                "role": turn["role"],
                "has_answer": turn["has_answer"],
            })
            chunk_id += 1
        else:
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = " ".join(words[start:end])
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "source_turn": turn_idx,
                    "session_id": turn["session_id"],
                    "role": turn["role"],
                    "has_answer": turn["has_answer"],
                })
                chunk_id += 1
                if end >= len(words):
                    break
                start = end - overlap
    return chunks
