# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
KV-aware load balancer using NVIDIA Dynamo's Rust bindings.

Uses dynamo._core (compiled from Rust via PyO3/maturin) for:
- compute_block_hash_for_seq: Fast XXH3-based block hashing
- RadixTree: Efficient prefix matching with per-worker overlap scoring
- OverlapScores: Cost-based worker selection

Prerequisites:
    pip install ai-dynamo-runtime
    # or build from source:
    cd <dynamo>/lib/bindings/python && maturin develop --release

Architecture:
    The router operates within a Ray actor, using dynamo's Rust RadixTree for
    fast prefix matching. KV cache state is tracked via JSON-serialized events
    sent to the Rust tree. No NATS or dynamo service mesh required — event
    transport uses Ray actor calls.

    Workers can optionally report real KV events (stored/removed blocks) for
    accurate cache state tracking. Without worker events, the router predicts
    cache state from its own routing decisions (approximate mode).

Cost function (per worker):
    cost = overlap_score_weight * prefill_blocks + inflight_requests

Where:
    - prefill_blocks = max(0, total_input_blocks - overlap_blocks)
    - overlap_blocks = number of prefix blocks cached on this worker (from RadixTree)
    - inflight_requests = current active requests on this worker
"""

from __future__ import annotations

import json
import logging
import math
import os
import random

import ray
from cachetools import LRUCache

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _check_dynamo_available():
    """Check if dynamo Rust bindings are available."""
    try:
        from dynamo._core import RadixTree, compute_block_hash_for_seq  # noqa: F401

        return True
    except ImportError:
        return False


def _make_stored_event_json(
    event_id: int,
    block_hashes: list[int],
    parent_hash: int | None = None,
    dp_rank: int = 0,
) -> bytes:
    """Construct a JSON-serialized KvCacheEvent for a stored event.

    Matches dynamo's KvCacheEvent wire format (serde JSON):
        KvCacheEvent {
            event_id: u64,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Option<u64>,
                blocks: Vec<KvCacheStoredBlockData>,
            }),
            dp_rank: u32,
        }

    Args:
        event_id: Monotonically increasing event identifier.
        block_hashes: List of block hash values (u64) from compute_block_hash_for_seq.
        parent_hash: Hash of the parent block (for prefix chaining). None for new sequences.
        dp_rank: Data parallel rank of the worker. 0 if DP is not used.

    Returns:
        JSON bytes ready for RadixTree.apply_event().
    """
    blocks = [
        {
            "block_hash": h,
            "tokens_hash": h,
            "mm_extra_info": None,
        }
        for h in block_hashes
    ]

    event = {
        "event_id": event_id,
        "data": {
            "stored": {
                "parent_hash": parent_hash,
                "blocks": blocks,
            }
        },
        "dp_rank": dp_rank,
    }
    return json.dumps(event).encode("utf-8")


def _make_removed_event_json(
    event_id: int,
    block_hashes: list[int],
    dp_rank: int = 0,
) -> bytes:
    """Construct a JSON-serialized KvCacheEvent for a removed event.

    Args:
        event_id: Monotonically increasing event identifier.
        block_hashes: List of block hash values to remove.
        dp_rank: Data parallel rank of the worker.

    Returns:
        JSON bytes ready for RadixTree.apply_event().
    """
    event = {
        "event_id": event_id,
        "data": {
            "removed": {
                "block_hashes": block_hashes,
            }
        },
        "dp_rank": dp_rank,
    }
    return json.dumps(event).encode("utf-8")


def _make_cleared_event_json(event_id: int, dp_rank: int = 0) -> bytes:
    """Construct a JSON-serialized KvCacheEvent for a cleared event (full cache reset)."""
    event = {
        "event_id": event_id,
        "data": "cleared",
        "dp_rank": dp_rank,
    }
    return json.dumps(event).encode("utf-8")


@ray.remote
class DynamoKVRouter:
    """KV-aware load balancer using dynamo's Rust RadixTree.

    Combines:
    1. Sticky session: multi-turn requests reuse the same server
    2. KV overlap: Rust radix tree finds server with most cached prefix blocks
    3. Load balancing: inflight request count prevents overloading

    The Rust RadixTree (from dynamo._core) provides:
    - O(depth) prefix matching via radix tree traversal
    - Per-worker overlap scores in a single tree query
    - Thread-safe operation (tree runs on dedicated thread)
    - TTL-based expiration for stale cache entries

    Args:
        server_actor_ids: List of server address strings.
        block_size: Tokens per KV cache block (must match inference engine).
        overlap_score_weight: Weight for prefill cost vs inflight cost.
        temperature: Softmax temperature for worker selection (0 = deterministic).
        max_cache_size: Max entries in sticky-session LRU cache.
        ttl_secs: TTL for radix tree entries. 0 = no expiration.
    """

    def __init__(
        self,
        server_actor_ids: list[str],
        block_size: int = 16,
        overlap_score_weight: float = 1.0,
        temperature: float = 0.0,
        max_cache_size: int = 10000,
        ttl_secs: float = 120.0,
    ):
        if not server_actor_ids:
            raise ValueError("server_actor_ids must be non-empty")

        from dynamo._core import RadixTree, compute_block_hash_for_seq

        self._compute_block_hash = compute_block_hash_for_seq
        self._tree = RadixTree(
            expiration_duration_secs=ttl_secs if ttl_secs > 0 else None
        )

        self._server_ids = list(server_actor_ids)
        self._block_size = block_size
        self._overlap_score_weight = overlap_score_weight
        self._temperature = temperature

        # Map server addresses ↔ integer worker IDs (dynamo RadixTree uses u64 worker IDs)
        self._server_to_wid: dict[str, int] = {
            sid: i for i, sid in enumerate(server_actor_ids)
        }
        self._wid_to_server: dict[int, str] = {
            i: sid for i, sid in enumerate(server_actor_ids)
        }

        # Inflight request tracking
        self._inflight: dict[str, int] = {sid: 0 for sid in server_actor_ids}

        # Sticky session LRU cache
        self._sticky: LRUCache = LRUCache(maxsize=max_cache_size)

        # Monotonic event counter for KvCacheEvent.event_id
        self._event_counter: int = 0

        # Metrics
        self._total_requests: int = 0
        self._cache_hits: int = 0
        self._sticky_hits: int = 0

    def acquire_server(
        self, request_id: str, prompt_ids: list[int] | None = None
    ) -> str:
        """Acquire a server for the given request using KV-aware routing.

        Args:
            request_id: Unique request ID. Same ID across turns enables sticky session.
            prompt_ids: Token IDs for the prompt. Used for prefix matching.
                If None, falls back to least-loaded routing.

        Returns:
            Server address string for the selected server.
        """
        self._total_requests += 1

        # 1. Sticky session: multi-turn conversations reuse the same server
        if request_id in self._sticky:
            server_id = self._sticky[request_id]
            self._inflight[server_id] += 1
            self._sticky_hits += 1
            # Update tree with new prompt prefix for this server
            if prompt_ids:
                self._record_stored(server_id, prompt_ids)
            return server_id

        # 2. KV-aware routing with Rust RadixTree
        if prompt_ids and len(prompt_ids) >= self._block_size:
            server_id = self._kv_aware_select(prompt_ids)
        else:
            # Not enough tokens for block matching — least-loaded fallback
            server_id = min(self._inflight, key=self._inflight.get)

        self._sticky[request_id] = server_id
        self._inflight[server_id] += 1

        # Record prefix in tree for the selected server
        if prompt_ids:
            self._record_stored(server_id, prompt_ids)

        return server_id

    def release_server(self, server_id: str) -> None:
        """Release a server after request completion."""
        if server_id not in self._inflight:
            raise ValueError(f"Invalid server_id for release: {server_id}")
        if self._inflight[server_id] <= 0:
            raise ValueError(
                f"Release called with no inflight requests on server {server_id}"
            )
        self._inflight[server_id] -= 1

    def report_kv_stored(
        self,
        server_id: str,
        token_ids: list[int],
    ) -> None:
        """Worker reports that KV blocks for token_ids are now cached.

        Call this from workers after inference to provide accurate cache state.
        Without these reports, the router uses prediction-based tracking.
        """
        self._record_stored(server_id, token_ids)

    def report_kv_removed(
        self,
        server_id: str,
        block_hashes: list[int],
    ) -> None:
        """Worker reports that specific KV blocks were evicted.

        Args:
            server_id: Server address string.
            block_hashes: List of block hash values that were evicted.
        """
        worker_id = self._server_to_wid[server_id]
        self._event_counter += 1
        event_bytes = _make_removed_event_json(
            event_id=self._event_counter,
            block_hashes=block_hashes,
        )
        self._tree.apply_event(worker_id, event_bytes)

    def report_kv_cleared(self, server_id: str) -> None:
        """Worker reports full KV cache reset (e.g., after clear_kv_cache)."""
        worker_id = self._server_to_wid[server_id]
        self._event_counter += 1
        event_bytes = _make_cleared_event_json(event_id=self._event_counter)
        self._tree.apply_event(worker_id, event_bytes)

    def get_metrics(self) -> dict:
        """Return routing metrics for monitoring."""
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "sticky_hits": self._sticky_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._total_requests),
            "sticky_hit_rate": self._sticky_hits / max(1, self._total_requests),
            "inflight": dict(self._inflight),
        }

    # ── internal ──────────────────────────────────────────────────────────

    def _kv_aware_select(self, prompt_ids: list[int]) -> str:
        """Select best server using Rust RadixTree overlap scores + inflight load."""
        # Compute block hashes using dynamo's XXH3 hasher (Rust, fast)
        block_hashes: list[int] = self._compute_block_hash(
            prompt_ids, self._block_size
        )
        total_blocks = len(block_hashes)

        if total_blocks == 0:
            return min(self._inflight, key=self._inflight.get)

        # Query Rust RadixTree for per-worker overlap scores
        overlap_scores = self._tree.find_matches(block_hashes)
        scores: dict[tuple[int, int], int] = overlap_scores.scores
        # scores is Dict[(worker_id, dp_rank), num_matched_blocks]

        has_overlap = any(v > 0 for v in scores.values())
        if has_overlap:
            self._cache_hits += 1

        # Compute cost for each server
        costs: dict[str, float] = {}
        for sid in self._server_ids:
            wid = self._server_to_wid[sid]
            overlap = scores.get((wid, 0), 0)
            prefill_blocks = max(0, total_blocks - overlap)
            inflight = self._inflight[sid]
            cost = self._overlap_score_weight * prefill_blocks + inflight
            costs[sid] = cost

        # Select server
        if self._temperature <= 0.0:
            # Deterministic: minimum cost, break ties by least inflight then random
            min_cost = min(costs.values())
            candidates = [sid for sid, c in costs.items() if c == min_cost]
            if len(candidates) == 1:
                return candidates[0]
            min_inflight = min(self._inflight[sid] for sid in candidates)
            final = [
                sid for sid in candidates if self._inflight[sid] == min_inflight
            ]
            return random.choice(final)
        else:
            return self._softmax_select(costs)

    def _softmax_select(self, costs: dict[str, float]) -> str:
        """Select server via softmax sampling on negative costs."""
        server_ids = list(costs.keys())
        neg_costs = [-costs[sid] / self._temperature for sid in server_ids]
        max_nc = max(neg_costs)
        exp_vals = [math.exp(nc - max_nc) for nc in neg_costs]
        total = sum(exp_vals)
        probs = [e / total for e in exp_vals]

        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return server_ids[i]
        return server_ids[-1]

    def _record_stored(self, server_id: str, token_ids: list[int]) -> None:
        """Record that server_id now has the token prefix cached in the Rust tree."""
        worker_id = self._server_to_wid[server_id]
        block_hashes = self._compute_block_hash(token_ids, self._block_size)
        if not block_hashes:
            return

        self._event_counter += 1
        event_bytes = _make_stored_event_json(
            event_id=self._event_counter,
            block_hashes=block_hashes,
            parent_hash=None,
        )
        self._tree.apply_event(worker_id, event_bytes)
