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
Radix tree for tracking KV cache block prefixes across inference workers.

Inspired by NVIDIA Dynamo's KV-aware router. Each node in the tree represents
a token block hash, and tracks which workers have that block cached.
The tree enables efficient prefix matching to compute overlap scores.
"""

from __future__ import annotations

import hashlib
import struct
import time
from collections import defaultdict
from dataclasses import dataclass, field


def _hash_token_block(tokens: tuple[int, ...]) -> int:
    """Deterministic hash for a block of token IDs."""
    data = struct.pack(f">{len(tokens)}i", *tokens)
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def tokens_to_block_hashes(token_ids: list[int], block_size: int) -> list[int]:
    """Convert a token ID sequence into a list of block hashes.

    Only complete blocks are hashed (trailing tokens that don't fill a block are ignored).
    """
    hashes = []
    for i in range(0, len(token_ids) - block_size + 1, block_size):
        block = tuple(token_ids[i : i + block_size])
        hashes.append(_hash_token_block(block))
    return hashes


@dataclass
class RadixNode:
    """A node in the radix tree.

    Each node corresponds to one block hash in a prefix sequence.
    """

    block_hash: int
    workers: set[str] = field(default_factory=set)
    children: dict[int, RadixNode] = field(default_factory=dict)
    last_access: float = field(default_factory=time.monotonic)

    def touch(self):
        self.last_access = time.monotonic()


class RadixTree:
    """Radix tree for tracking which token block prefixes are cached on which workers.

    Supports:
    - Insert: record that a worker has cached a sequence of block hashes
    - Find matches: for a given sequence of block hashes, return per-worker overlap scores
    - TTL-based expiration: prune stale entries
    - Max size pruning: limit tree growth

    Args:
        block_size: Number of tokens per KV cache block (must match the inference engine).
        ttl_secs: Time-to-live in seconds for cached entries. 0 = no expiration.
        max_tree_size: Maximum number of nodes before triggering pruning. 0 = unlimited.
        prune_target_ratio: When pruning, reduce to this fraction of max_tree_size.
    """

    def __init__(
        self,
        block_size: int = 16,
        ttl_secs: float = 120.0,
        max_tree_size: int = 1_000_000,
        prune_target_ratio: float = 0.8,
    ):
        self.block_size = block_size
        self.ttl_secs = ttl_secs
        self.max_tree_size = max_tree_size
        self.prune_target_ratio = prune_target_ratio

        # Root node has no block hash; its children are the first blocks of sequences
        self.root = RadixNode(block_hash=-1)
        self._size = 0  # total number of non-root nodes

        # Per-worker block count for tree_size metric
        self._worker_block_counts: dict[str, int] = defaultdict(int)

    @property
    def size(self) -> int:
        return self._size

    def insert(self, worker_id: str, block_hashes: list[int]) -> None:
        """Record that `worker_id` has cached the prefix represented by `block_hashes`.

        Traverses or creates nodes for each block hash, adding the worker to each node.
        """
        if not block_hashes:
            return

        node = self.root
        new_blocks = 0
        for bh in block_hashes:
            if bh not in node.children:
                node.children[bh] = RadixNode(block_hash=bh)
                self._size += 1
            child = node.children[bh]
            child.touch()
            if worker_id not in child.workers:
                child.workers.add(worker_id)
                new_blocks += 1
            node = child

        self._worker_block_counts[worker_id] += new_blocks

        # Prune if over limit
        if self.max_tree_size > 0 and self._size > self.max_tree_size:
            self._prune_lru()

    def find_matches(self, block_hashes: list[int]) -> dict[str, int]:
        """Find per-worker overlap scores for the given block hash sequence.

        Returns a dict mapping worker_id to the number of contiguous prefix blocks
        that worker has cached. Only counts the longest contiguous prefix match
        from the beginning of the sequence.

        This mirrors dynamo's approach: traverse the tree following the block hash
        sequence, tracking which workers are present at each depth level.
        """
        if not block_hashes:
            return {}

        # scores[worker_id] = number of matched prefix blocks
        scores: dict[str, int] = {}
        # active_workers tracks the current set of workers that have all blocks up to this depth
        active_workers: set[str] | None = None
        depth = 0

        node = self.root
        for bh in block_hashes:
            if bh not in node.children:
                break

            child = node.children[bh]
            child.touch()
            depth += 1

            if active_workers is None:
                # First block: all workers at this node are active
                active_workers = set(child.workers)
            else:
                # Workers that dropped out at this depth get their score recorded
                dropped = active_workers - child.workers
                for w in dropped:
                    scores[w] = depth - 1
                active_workers = active_workers & child.workers

            if not active_workers:
                break

            node = child

        # Record final scores for workers still active at the deepest matched level
        if active_workers:
            for w in active_workers:
                scores[w] = depth

        return scores

    def get_worker_tree_sizes(self) -> dict[str, int]:
        """Return per-worker total block counts in the tree."""
        return dict(self._worker_block_counts)

    def expire_stale(self) -> int:
        """Remove nodes older than TTL. Returns the number of nodes removed."""
        if self.ttl_secs <= 0:
            return 0

        cutoff = time.monotonic() - self.ttl_secs
        removed = self._expire_recursive(self.root, cutoff)
        return removed

    def remove_worker(self, worker_id: str) -> None:
        """Remove a worker from all nodes in the tree."""
        self._remove_worker_recursive(self.root, worker_id)
        self._worker_block_counts.pop(worker_id, None)

    def _remove_worker_recursive(self, node: RadixNode, worker_id: str) -> None:
        dead_children = []
        for bh, child in node.children.items():
            child.workers.discard(worker_id)
            self._remove_worker_recursive(child, worker_id)
            if not child.workers and not child.children:
                dead_children.append(bh)
        for bh in dead_children:
            del node.children[bh]
            self._size -= 1

    def _expire_recursive(self, node: RadixNode, cutoff: float) -> int:
        removed = 0
        dead_children = []
        for bh, child in node.children.items():
            removed += self._expire_recursive(child, cutoff)
            if child.last_access < cutoff and not child.children:
                # Leaf node that is stale — remove it
                for w in child.workers:
                    self._worker_block_counts[w] = max(0, self._worker_block_counts[w] - 1)
                dead_children.append(bh)
        for bh in dead_children:
            del node.children[bh]
            self._size -= 1
            removed += 1
        return removed

    def _prune_lru(self) -> None:
        """Prune least-recently-used leaf nodes until tree size is at target."""
        target = int(self.max_tree_size * self.prune_target_ratio)
        while self._size > target:
            # Collect all leaf nodes with their access times
            leaves: list[tuple[float, RadixNode, RadixNode, int]] = []
            self._collect_leaves(self.root, leaves)
            if not leaves:
                break
            # Sort by access time (oldest first) and remove
            leaves.sort(key=lambda x: x[0])
            for _, leaf, parent, bh in leaves:
                if self._size <= target:
                    break
                if bh in parent.children and parent.children[bh] is leaf:
                    for w in leaf.workers:
                        self._worker_block_counts[w] = max(0, self._worker_block_counts[w] - 1)
                    del parent.children[bh]
                    self._size -= 1

    def _collect_leaves(
        self, node: RadixNode, leaves: list[tuple[float, RadixNode, RadixNode, int]]
    ) -> None:
        for bh, child in node.children.items():
            if not child.children:
                leaves.append((child.last_access, child, node, bh))
            else:
                self._collect_leaves(child, leaves)
