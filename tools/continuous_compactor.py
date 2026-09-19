#!/usr/bin/env python3
"""
Continuous Compactor Engine for Agentic Coding Workflows
Based on the Claude Code / Pi Harness Architecture.

Key Principles:
1. Incremental Chunking: Detects when context grows by delta chunks (e.g. 24,000 tokens).
2. Background Asynchronous Compaction: Offloads summarization of older tool outputs to the
   AMD XDNA 2 NPU (FastFlowLM on port 52625 @ 107 tok/s prefill, 71.5 tok/s decode, 0% GPU load).
3. Instantaneous Swap (Zero Latency): Pre-generates the rolling state summary so that when the
   compaction threshold is reached, the context roll happens with 0ms user-visible delay.
4. Preserves Peak Memory Bandwidth: Keeps active context bounded (e.g., 24k-36k tokens),
   preventing decode throughput on Strix Halo from dropping from 28.4 tok/s down to 11.4 tok/s.
"""

import sys
import time
import json
import re
import threading
import requests
from typing import List, Dict, Any, Tuple, Optional

DEFAULT_NPU_URL = "http://127.0.0.1:52625/v1/chat/completions"
DEFAULT_NPU_MODEL = "qwen3:0.6b"
DEFAULT_CHUNK_SIZE = 24000      # Tokens per compaction chunk
DEFAULT_THRESHOLD = 32000       # Total tokens before instantaneous context swap
DEFAULT_KEEP_RECENT = 6         # Number of recent messages to preserve verbatim

COMPACTOR_SYSTEM_PROMPT = """You are an expert technical context compactor for an autonomous coding agent.
Your task is to condense the provided conversation history into an ultra-dense, structured Technical State.
Focus strictly on:
- Primary Goal & Objectives
- Files Read, Modified, or Created
- Key Architectural Decisions & Resolved Bugs
- Verification Status (test results, build status)
- Current Working State & Pending Tasks

Rules:
- Eliminate all raw stdout, verbose compiler warnings, and discarded attempts.
- Format with concise markdown bullet points.
- Maximum length: 400 words.
"""

def estimate_tokens(text: str) -> int:
    """Fast token estimation (~4 chars per token for code/prose)."""
    if not text:
        return 0
    return max(1, len(text) // 4)

def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += estimate_tokens(part["text"])
        # Account for role and message overhead
        total += 4
    return total

class ContinuousCompactor:
    """
    Manages session-level continuous context compaction.
    Maintains a precomputed rolling summary in memory so compaction events are instantaneous.
    """
    def __init__(
        self,
        npu_url: str = DEFAULT_NPU_URL,
        model: str = DEFAULT_NPU_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        threshold: int = DEFAULT_THRESHOLD,
        keep_recent: int = DEFAULT_KEEP_RECENT
    ):
        self.npu_url = npu_url
        self.model = model
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.keep_recent = keep_recent

        self._lock = threading.Lock()
        self._rolling_summary: Optional[str] = None
        self._last_compacted_index: int = 0
        self._is_compacting: bool = False
        self._last_compaction_time: float = 0.0
        self._total_compaction_events: int = 0

    @property
    def rolling_summary(self) -> Optional[str]:
        with self._lock:
            return self._rolling_summary

    def should_trigger_background_compaction(self, messages: List[Dict[str, Any]]) -> bool:
        """Determines if a new 24k-token chunk is ready for background summarization."""
        with self._lock:
            if self._is_compacting:
                return False

        n_msgs = len(messages)
        if n_msgs <= self.keep_recent + 2:
            return False

        # Consider messages eligible for compaction: from last_compacted_index up to (n_msgs - keep_recent)
        eligible_msgs = messages[self._last_compacted_index : max(self._last_compacted_index, n_msgs - self.keep_recent)]
        chunk_tokens = estimate_messages_tokens(eligible_msgs)

        return chunk_tokens >= self.chunk_size

    def trigger_async_compaction(self, messages: List[Dict[str, Any]]):
        """Dispatches compaction asynchronously to the NPU without blocking the caller."""
        with self._lock:
            if self._is_compacting:
                return
            self._is_compacting = True

        t = threading.Thread(target=self._run_compaction_worker, args=(messages,), daemon=True)
        t.start()

    def _run_compaction_worker(self, messages: List[Dict[str, Any]]):
        """Worker thread executing on AMD XDNA 2 NPU."""
        t0 = time.time()
        try:
            n_msgs = len(messages)
            cutoff = max(1, n_msgs - self.keep_recent)
            msgs_to_compact = messages[self._last_compacted_index:cutoff]

            if not msgs_to_compact:
                return

            # Prepare text block to summarize
            formatted_chunks = []
            for idx, m in enumerate(msgs_to_compact):
                role = m.get("role", "unknown")
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
                # Strip excessive repetition or huge terminal logs
                if len(content) > 10000:
                    content = content[:3000] + "\n...[truncated terminal output]...\n" + content[-3000:]
                formatted_chunks.append(f"[{role.upper()}]: {content}")

            chunk_text = "\n\n".join(formatted_chunks)

            user_prompt = ""
            with self._lock:
                if self._rolling_summary:
                    user_prompt += f"PREVIOUS TECHNICAL SUMMARY:\n{self._rolling_summary}\n\n"
            user_prompt += f"NEW ACTIVITY CHUNK TO INCORPORATE:\n{chunk_text}"

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": COMPACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 600
            }

            resp = requests.post(self.npu_url, json=payload, timeout=25)
            if resp.status_code == 200:
                summary = resp.json()["choices"][0]["message"]["content"].strip()
                duration = time.time() - t0
                with self._lock:
                    self._rolling_summary = summary
                    self._last_compacted_index = cutoff
                    self._last_compaction_time = duration
                    self._total_compaction_events += 1
                print(f"[COMPACTOR] Successfully compacted chunk to NPU in {duration:.2f}s ({len(summary)} chars)")
            else:
                print(f"[COMPACTOR] NPU error {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"[COMPACTOR] Background compaction failed: {e}")
        finally:
            with self._lock:
                self._is_compacting = False

    def compact_messages(self, messages: List[Dict[str, Any]], force: bool = False) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Instantaneous context compaction event:
        If total tokens exceed threshold (or force=True) and a precomputed summary exists,
        replaces historical turns with the precomputed summary block.
        Returns (compacted_messages, did_compact).
        """
        total_tokens = estimate_messages_tokens(messages)

        # Trigger background compaction if chunk size met
        if self.should_trigger_background_compaction(messages):
            self.trigger_async_compaction(messages)

        if not force and total_tokens < self.threshold:
            return messages, False

        with self._lock:
            summary = self._rolling_summary
            cutoff = self._last_compacted_index

        if not summary or cutoff <= 1:
            # Fallback if no summary ready yet: return unmodified
            return messages, False

        # Reconstruct message history:
        # 1. Preserve original System Prompt (if present)
        new_messages = []
        start_idx = 0
        if messages and messages[0].get("role") == "system":
            new_messages.append(messages[0])
            start_idx = 1

        # 2. Inject Precomputed Rolling Summary as System Context Note
        compaction_banner = (
            f"[CONTEXT COMPACTION EVENT: Context compacted to preserve zero-latency decode and focus]\n"
            f"SUMMARY OF PREVIOUS AGENT ACTIONS & ARCHITECTURAL STATE:\n\n{summary}\n"
        )
        new_messages.append({
            "role": "system",
            "content": compaction_banner
        })

        # 3. Append Recent Turns Verbatim
        recent_cutoff = max(start_idx, len(messages) - self.keep_recent)
        for m in messages[recent_cutoff:]:
            new_messages.append(m)

        return new_messages, True

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "rolling_summary_available": bool(self._rolling_summary),
                "rolling_summary_chars": len(self._rolling_summary) if self._rolling_summary else 0,
                "is_compacting": self._is_compacting,
                "last_compaction_duration_s": self._last_compaction_time,
                "total_compaction_events": self._total_compaction_events,
                "chunk_size": self.chunk_size,
                "threshold": self.threshold
            }

if __name__ == "__main__":
    print("=== Testing Continuous Compactor against AMD XDNA 2 NPU ===")
    compactor = ContinuousCompactor(chunk_size=500, threshold=1000, keep_recent=2)

    # Generate synthetic conversation representing 15 tool turns
    mock_messages = [
        {"role": "system", "content": "You are a software engineering assistant."},
        {"role": "user", "content": "Let's investigate the test failure in test_cluster_routing.py."},
        {"role": "assistant", "content": "I will examine the test file.", "tool_calls": [{"name": "read_file"}]},
        {"role": "tool", "content": "def test_routing(): assert route('/v1/models') == 200 # FAILED with 404"},
        {"role": "assistant", "content": "The route handler in cluster-dashboard.py lacks wildcard support."},
        {"role": "tool", "content": "git diff: added wildcard regex match for /v1/models/*"},
        {"role": "assistant", "content": "Tests now pass. Next step is updating the proxy documentation."},
        {"role": "user", "content": "Awesome, please update the docs now."},
        {"role": "assistant", "content": "Updating docs/routing.md with new wildcard behavior."}
    ]

    print(f"Initial estimated tokens: {estimate_messages_tokens(mock_messages)}")
    print("Triggering compaction worker...")
    compactor._run_compaction_worker(mock_messages)

    print("\nCompacted Rolling Summary:")
    print("-" * 50)
    print(compactor.rolling_summary)
    print("-" * 50)

    compacted_msgs, did_compact = compactor.compact_messages(mock_messages, force=True)
    print(f"Did compact: {did_compact}")
    print(f"New message count: {len(compacted_msgs)} (down from {len(mock_messages)})")
    print(f"New estimated tokens: {estimate_messages_tokens(compacted_msgs)}")
    for i, m in enumerate(compacted_msgs):
        print(f"  [{i}] ({m['role']}): {m['content'][:80]}...")
