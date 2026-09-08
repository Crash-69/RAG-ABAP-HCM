"""Content deduplication (V4).

ABAP/HCM repositories routinely contain copy-pasted routines (e.g. cloned
"ZHR_*" reports, near-identical FORM routines varying only by infotype
number). Naive exact-hash deduplication misses these near-duplicates, which
pollutes retrieval results with redundant chunks. This module implements
generation "V4" of the project's deduplication strategy:

* V1 (implicit, not implemented here): exact string match.
* V2: whitespace/case normalized exact match.
* V3: token-shingle Jaccard similarity.
* V4 (this module): SimHash based near-duplicate detection, which scales to
  large chunk sets (O(n) fingerprinting instead of O(n^2) shingle
  comparison) while still catching near-duplicates, combined with exact
  normalized-hash matching for perfect duplicates.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\S")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize whitespace/case for exact-duplicate comparison."""
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(normalize(text))


def simhash(text: str, hash_bits: int = 64) -> int:
    """Compute a SimHash fingerprint for near-duplicate detection.

    Splits the text into overlapping 3-token shingles, hashes each shingle,
    and combines the hashes bit-by-bit weighted by shingle frequency, which
    is the standard SimHash construction (Charikar, 2002).
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0
    shingles = [tuple(tokens[i : i + 3]) for i in range(max(1, len(tokens) - 2))]
    weights = [0] * hash_bits
    for shingle in shingles:
        digest = hashlib.md5(" ".join(shingle).encode("utf-8")).digest()
        h = int.from_bytes(digest[: (hash_bits // 8) or 8], "big")
        for bit in range(hash_bits):
            if (h >> bit) & 1:
                weights[bit] += 1
            else:
                weights[bit] -= 1
    fingerprint = 0
    for bit in range(hash_bits):
        if weights[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def similarity_from_hamming(distance: int, hash_bits: int = 64) -> float:
    return 1.0 - (distance / hash_bits)


@dataclass
class DedupResult:
    """Outcome of deduplicating a sequence of chunks."""

    kept_indices: List[int]
    duplicate_of: dict  # index -> index it duplicates
    fingerprints: List[int]


class ContentDeduplicatorV4:
    """Near-duplicate content detector for ingestion pipelines.

    Usage::

        dedup = ContentDeduplicatorV4(similarity_threshold=0.9)
        result = dedup.deduplicate(chunk_contents)
        unique_chunks = [chunks[i] for i in result.kept_indices]
    """

    def __init__(self, similarity_threshold: float = 0.90, hash_bits: int = 64):
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        self.similarity_threshold = similarity_threshold
        self.hash_bits = hash_bits

    def deduplicate(self, texts: Sequence[str]) -> DedupResult:
        kept_indices: List[int] = []
        duplicate_of: dict = {}
        fingerprints: List[int] = []
        seen_exact: dict = {}
        kept_fingerprints: List[int] = []

        for idx, text in enumerate(texts):
            fp = simhash(text, self.hash_bits)
            fingerprints.append(fp)

            exact_key = content_hash(text)
            if exact_key in seen_exact:
                duplicate_of[idx] = seen_exact[exact_key]
                continue

            duplicate_index = self._find_near_duplicate(fp, kept_indices, fingerprints)
            if duplicate_index is not None:
                duplicate_of[idx] = duplicate_index
                continue

            seen_exact[exact_key] = idx
            kept_indices.append(idx)
            kept_fingerprints.append(fp)

        return DedupResult(
            kept_indices=kept_indices,
            duplicate_of=duplicate_of,
            fingerprints=fingerprints,
        )

    def is_near_duplicate(self, text_a: str, text_b: str) -> bool:
        fp_a = simhash(text_a, self.hash_bits)
        fp_b = simhash(text_b, self.hash_bits)
        distance = hamming_distance(fp_a, fp_b)
        return similarity_from_hamming(distance, self.hash_bits) >= self.similarity_threshold

    def _find_near_duplicate(
        self, fp: int, kept_indices: Iterable[int], fingerprints: List[int]
    ) -> Optional[int]:
        for kept_idx in kept_indices:
            distance = hamming_distance(fp, fingerprints[kept_idx])
            if similarity_from_hamming(distance, self.hash_bits) >= self.similarity_threshold:
                return kept_idx
        return None
