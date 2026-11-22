"""SOP retrieval. Chunk = one numbered section (§n) of an SOP. Search = TF-IDF cosine similarity.

Why TF-IDF and not embeddings: 10 short documents, exact terminology (door, vaccine, quarantine),
no network/model download needed, deterministic, and citations map 1:1 to a chunk id such as
SOP-02#2. The index class is small enough to swap for embeddings (see README, stretch idea).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from . import config as C

SYNONYMS = {  # tiny query expansion for dispatcher vocabulary
    "fridge": "reefer refrigeration", "cooling unit": "reefer", "cooler": "reefer", "ac": "reefer",
    "thermometer": "sensor", "battery": "battery power", "shut": "closed", "ajar": "open",
    "who do i call": "escalate escalation notify", "call": "notify contact",
}


def _stem(t: str) -> str:
    """Very light suffix stripping so 'redirects/redirecting', 'cycles/cycling' match."""
    for suf in ("ies", "ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            t = t[: -len(suf)] + ("y" if suf == "ies" else "")
            break
    return t[:-1] if t.endswith("e") and len(t) > 4 else t


def analyze(text: str) -> list[str]:
    toks = [_stem(t) for t in re.findall(r"[a-z0-9°]+", text.lower())
            if t not in ENGLISH_STOP_WORDS and (len(t) > 1 or t.isdigit())]
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


@dataclass
class Chunk:
    chunk_id: str      # e.g. SOP-02#2
    doc_id: str        # e.g. SOP-02
    title: str
    section: str       # e.g. §2 Vaccines and pharmaceuticals
    text: str
    source_file: str

    @property
    def retrieval_text(self) -> str:
        return f"{self.title}. {self.section}. {self.text}"


def load_chunks(sop_dir=C.SOP_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(sop_dir.glob("SOP-*.txt")):
        lines = path.read_text(encoding="utf-8").splitlines()
        m = re.match(r"(SOP-\d+):\s*(.+)", lines[0])
        doc_id, title = m.group(1), m.group(2).strip()
        cur = None
        for ln in lines[1:]:
            sm = re.match(r"§(\d+)\s+(.*)", ln)
            if sm:
                cur = dict(num=sm.group(1), head=sm.group(2).strip(), body=[])
                chunks.append(cur)
                cur["doc"], cur["title"], cur["file"] = doc_id, title, path.name
            elif cur is not None and ln.strip():
                cur["body"].append(ln.strip())
    return [Chunk(f"{c['doc']}#{c['num']}", c["doc"], c["title"], f"§{c['num']} {c['head']}",
                  " ".join(c["body"]), c["file"]) for c in chunks]


class SopIndex:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or load_chunks()
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.vec = TfidfVectorizer(analyzer=analyze, sublinear_tf=True)
        self.mat = self.vec.fit_transform([c.retrieval_text for c in self.chunks])

    @staticmethod
    def expand(q: str) -> str:
        ql = q.lower()
        extra = " ".join(v for k, v in SYNONYMS.items() if re.search(rf"\b{re.escape(k)}\b", ql))
        return f"{q} {extra}".strip()

    def coverage(self, query: str) -> float:
        """IDF-weighted share of the query's content words that occur anywhere in the SOPs.
        Cosine alone lets a single shared generic word ("drivers", "limit") pass; coverage catches
        questions whose key terms (overtime, fuel card, customs ...) simply are not in the SOPs."""
        terms = {t for t in analyze(query) if " " not in t}
        if not terms:
            return 0.0
        voc, idf = self.vec.vocabulary_, self.vec.idf_
        wmax = float(idf.max())
        tot = sum(idf[voc[t]] if t in voc else wmax for t in terms)
        got = sum(idf[voc[t]] for t in terms if t in voc)
        return float(got / tot)

    def search(self, query: str, k: int = C.RAG_TOP_K, min_score: float = C.RAG_MIN_SCORE,
               min_coverage: float = C.RAG_MIN_COVERAGE) -> list[dict]:
        if min_coverage and self.coverage(self.expand(query)) < min_coverage:
            return []
        qv = self.vec.transform([self.expand(query)])
        sims = (self.mat @ qv.T).toarray().ravel()
        order = sims.argsort()[::-1][:k]
        return [dict(chunk_id=self.chunks[i].chunk_id, source=self.chunks[i].source_file,
                     section=self.chunks[i].section, title=self.chunks[i].title,
                     text=self.chunks[i].text, score=round(float(sims[i]), 3))
                for i in order if sims[i] >= min_score]


@lru_cache(maxsize=1)
def get_index() -> SopIndex:
    return SopIndex()
