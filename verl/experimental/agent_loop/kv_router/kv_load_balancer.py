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
KV-aware load balancer for verl's agent loop.

Inspired by NVIDIA Dynamo's KV-aware router, this load balancer uses a radix tree
to track prefix cache state across inference workers and routes requests to maximize
KV cache reuse.

The router operates in **prediction mode** (no worker KV events needed):
it tracks which token prefixes have been routed to each server and predicts
cache state from its own routing decisions.

Cost function (per worker):
    cost = overlap_score_weight * prefill_blocks + inflight_requests

Where:
    - prefill_blocks = max(0, total_input_blocks - overlap_blocks)
    - overlap_blocks = number of prefix blocks already cached on this worker
    - inflight_requests = current number of active requests on this worker
"""

from __future__ import annotations

import logging
import math
import os
import random
import time

import ray
from cachetools import LRUCache

from verl.experimental.agent_loop.kv_router.radix_tree import RadixTree, tokens_to_block_hashes

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
class KVAwareLoadBalancer:
    """KV-aware load balancer that maximizes prefix cache reuse across workers.

    Combines three routing strategies:
    1. **Sticky session**: Multi-turn requests go to the same server (prefix cache reuse)
    2. **KV overlap**: New requests go to the server with most matching cached prefix blocks
    3. **Load balancing**: Inflight request count prevents overloading any single server

    Args:
        server_actor_ids: List of server address strings.
        block_size: Tokens per KV cache block (must match inference engine, typically 16).
        overlap_score_weight: Weight for prefill cost vs inflight cost.
            Higher values prioritize cache reuse (better TTFT).
            Lower values prioritize even load distribution (better ITL).
        temperature: Softmax temperature for worker selection.
            0.0 = deterministic (always pick best), >0 = stochastic sampling.
        max_cache_size: Max entries in sticky-session LRU cache.
        ttl_secs: TTL for radix tree entries. Stale entries are periodically expired.
        expire_interval: Seconds between automatic expiration sweeps.
    """

    def __init__(
        self,
        server_actor_ids: list[str],
        block_size: int = 16,
        overlap_score_weight: float = 1.0,
        temperature: float = 0.0,
        max_cache_size: int = 10000,
        ttl_secs: float = 120.0,
        expire_interval: float = 30.0,
    ):
        if not server_actor_ids:
            raise ValueError("server_actor_ids must be non-empty")

        self._server_ids = list(server_actor_ids)
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in server_actor_ids}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)

        self._block_size = block_size
        self._overlap_score_weight = overlap_score_weight
        self._temperature = temperature

        self._tree = RadixTree(
            block_size=block_size,
            ttl_secs=ttl_secs,
            max_tree_size=1_000_000,
        )

        self._expire_interval = expire_interval
        self._last_expire_time = time.monotonic()

        # Metrics
        self._total_requests = 0
        self._cache_hits = 0  # requests where overlap > 0
        self._sticky_hits = 0  # requests served by sticky session

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> str:
        """Acquire a server for the given request, using KV-aware routing.

        Args:
            request_id: Unique request identifier. Multi-turn conversations should
                use the same request_id to enable sticky-session routing.
            prompt_ids: Token IDs for the prompt. Used to compute block hashes
                for prefix matching. If None, falls back to least-loaded routing.

        Returns:
            Server address string for the selected server.
        """
        self._total_requests += 1
        self._maybe_expire()

        # 1. Sticky session: multi-turn conversations reuse the same server
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            self._inflight_requests[server_id] += 1
            self._sticky_hits += 1
            # Update the tree with the new prompt prefix for this server
            if prompt_ids:
                block_hashes = tokens_to_block_hashes(prompt_ids, self._block_size)
                self._tree.insert(server_id, block_hashes)
            return server_id

        # 2. KV-aware routing: find server with best cost
        if prompt_ids and len(prompt_ids) >= self._block_size:
            server_id = self._kv_aware_select(prompt_ids)
        else:
            # Not enough tokens for block matching; fall back to least-loaded
            server_id = min(self._inflight_requests, key=self._inflight_requests.get)

        self._request_id_to_server[request_id] = server_id
        self._inflight_requests[server_id] += 1

        # Record this prefix in the tree for the selected server
        if prompt_ids:
            block_hashes = tokens_to_block_hashes(prompt_ids, self._block_size)
            self._tree.insert(server_id, block_hashes)

        return server_id

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes."""
        if server_id not in self._inflight_requests:
            raise ValueError(f"Invalid server_id for release: {server_id}")
        if self._inflight_requests[server_id] <= 0:
            raise ValueError(f"Release called with no inflight requests on server {server_id}")
        self._inflight_requests[server_id] -= 1

    def get_metrics(self) -> dict:
        """Return routing metrics for monitoring."""
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "sticky_hits": self._sticky_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._total_requests),
            "sticky_hit_rate": self._sticky_hits / max(1, self._total_requests),
            "tree_size": self._tree.size,
            "inflight": dict(self._inflight_requests),
        }

    def _kv_aware_select(self, prompt_ids: list[int]) -> str:
        """Select the best server using KV overlap scores and inflight load.

        Cost function per worker:
            cost = overlap_score_weight * prefill_blocks + inflight_requests

        Lower cost is better.
        """
        block_hashes = tokens_to_block_hashes(prompt_ids, self._block_size)
        total_blocks = len(block_hashes)

        # Get overlap scores from radix tree
        overlap_scores = self._tree.find_matches(block_hashes)

        if any(v > 0 for v in overlap_scores.values()):
            self._cache_hits += 1

        # Compute cost for each server
        costs: dict[str, float] = {}
        for sid in self._server_ids:
            overlap = overlap_scores.get(sid, 0)
            prefill_blocks = max(0, total_blocks - overlap)
            inflight = self._inflight_requests[sid]
            cost = self._overlap_score_weight * prefill_blocks + inflight
            costs[sid] = cost

        # Select server
        if self._temperature <= 0.0:
            # Deterministic: pick minimum cost, break ties by least inflight
            min_cost = min(costs.values())
            candidates = [sid for sid, c in costs.items() if c == min_cost]
            if len(candidates) == 1:
                return candidates[0]
            # Break tie by least inflight, then random
            min_inflight = min(self._inflight_requests[sid] for sid in candidates)
            final_candidates = [sid for sid in candidates if self._inflight_requests[sid] == min_inflight]
            return random.choice(final_candidates)
        else:
            # Stochastic: softmax sampling with temperature
            return self._softmax_select(costs)

    def _softmax_select(self, costs: dict[str, float]) -> str:
        """Select a server using softmax sampling on negative costs."""
        server_ids = list(costs.keys())
        # Negate costs so lower cost = higher probability
        neg_costs = [-costs[sid] / self._temperature for sid in server_ids]
        # Numerical stability: subtract max
        max_nc = max(neg_costs)
        exp_vals = [math.exp(nc - max_nc) for nc in neg_costs]
        total = sum(exp_vals)
        probs = [e / total for e in exp_vals]

        # Sample
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return server_ids[i]
        return server_ids[-1]

    def _maybe_expire(self) -> None:
        """Periodically expire stale tree entries."""
        now = time.monotonic()
        if now - self._last_expire_time >= self._expire_interval:
            self._tree.expire_stale()
            self._last_expire_time = now
