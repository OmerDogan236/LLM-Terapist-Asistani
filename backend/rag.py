# backend/rag.py

import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss kurulu değil. requirements.txt içinde faiss-cpu olmalı.") from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise RuntimeError("sentence-transformers kurulu değil.") from e


TR_MENTAL_QUERY_TRIGGERS = [
    "kaygı", "kaygi", "anksiyete", "panik", "stres", "korku",
    "uyuyamıyorum", "uyuyamiyorum", "uykusuz", "uyku", "insomnia",
    "mutsuz", "üzgün", "uzgun", "umutsuz", "depres", "depresyon",
    "yalnız", "yalniz", "travma", "ptsd", "okb", "ocd",
    "bipolar", "dikkat dağınık", "dikkat daginik", "adhd",
    "kendime zarar", "intihar", "yaşamak istemiyorum", "yasamak istemiyorum",
]

MENTAL_FOCUS_PATTERNS = [
    r"\banxiety\b",
    r"anxiety disorders?",
    r"^depression$",
    r"postpartum depression",
    r"\binsomnia\b",
    r"\bpanic\b",
    r"\bptsd\b|post[- ]traumatic",
    r"\bbipolar\b",
    r"schizoph",
    r"obsessive-?compulsive|\bocd\b",
    r"\badhd\b|attention deficit",
    r"\bsuicid|\bself harm",
    r"mental health",
    r"mood disorders?",
    r"compulsive gambling",
    r"teen mental health",
    r"child mental health",
]

MENTAL_EXCLUDE_FOCUS = [
    "bell's palsy",
    "croup",
    "sleep apnea",
    "sleeping sickness",
    "respiratory",
    "pulmonary",
    "hernia",
    "ocular",
    "short stature",
    "traumatic brain injury",
    "distress syndrome",
    "schizenceph",
]


def _is_mental_query(text: str) -> bool:
    t = (text or "").lower().strip()
    if not t:
        return False

    # TR trigger
    if any(k in t for k in TR_MENTAL_QUERY_TRIGGERS):
        return True

    # EN / mental keywords (rewrite sonrası da yakalasın)
    en_hits = [
        "anxiety", "panic", "depression", "insomnia", "ptsd", "ocd", "adhd",
        "self harm", "suicid", "mental health", "mood", "trauma", "stress"
    ]
    if any(k in t for k in en_hits):
        return True

    return False


# main.py uyumluluğu (import bozulmasın)
def _looks_mental(text: str) -> bool:
    return _is_mental_query(text)


def _is_mental_focus(focus_area: str) -> bool:
    t = (focus_area or "").strip().lower()
    if not t:
        return False

    for bad in MENTAL_EXCLUDE_FOCUS:
        if bad in t:
            return False

    return any(re.search(p, t) for p in MENTAL_FOCUS_PATTERNS)

def _is_positive_query(q: str) -> bool:
    t = (q or "").strip().lower()
    # Türkçe + İngilizce basit pozitif marker'lar
    pos = [
        "daha iyi hissediyorum", "bugün daha iyi", "iyim", "iyiyim", "iyi gibiyim",
        "daha umutlu", "umutluyum", "keyif almaya başladım", "kendime daha iyi bakıyorum",
        "better", "feel better", "i feel better", "i'm better", "im better",
        "hopeful", "optimistic",
    ]
    return any(p in t for p in pos)

def _is_depression_focus(focus: str) -> bool:
    f = (focus or "").strip().lower()
    return "depression" in f or "depres" in f


def _dedupe_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for h in hits:
        key = (
            (h.get("source") or "").strip().lower(),
            (h.get("focus_area") or "").strip().lower(),
            (h.get("question") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _pick_diverse_topk(candidates: List[Dict[str, Any]], k: int, max_per_focus: int) -> List[Dict[str, Any]]:
    """
    Önce çeşitlilik: aynı focus_area'dan max_per_focus kadar al.
    Yetmezse kalanları skor sırasıyla tamamla (k dolsun diye).
    """
    out: List[Dict[str, Any]] = []
    count: Dict[str, int] = {}

    for h in candidates:
        f = (h.get("focus_area") or "").strip().lower()
        count[f] = count.get(f, 0) + 1
        if count[f] <= max_per_focus:
            out.append(h)
        if len(out) >= k:
            return out[:k]

    # yetmediyse çeşitlilik limitini aşarak doldur
    if len(out) < k:
        already = set((o.get("idx"), o.get("question")) for o in out)
        for h in candidates:
            key = (h.get("idx"), h.get("question"))
            if key in already:
                continue
            out.append(h)
            if len(out) >= k:
                break

    return out[:k]


class RAGStore:
    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parent.parent  # rag_therapist/
        data_dir = project_root / "data"

        self.mental_dir = data_dir / "index_mental"
        self.generic_dir = data_dir / "index"

        # embed modeli
        self.model_name = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)


        self.index_mental = None
        self.meta_mental: List[Dict[str, Any]] = []

        self.index_generic = None
        self.meta_generic: List[Dict[str, Any]] = []


        # mental index
        mental_index_path = self.mental_dir / "faiss.index"
        mental_meta_path = self.mental_dir / "meta.pkl"
        if mental_index_path.exists() and mental_meta_path.exists():
            self.index_mental = faiss.read_index(str(mental_index_path))
            with open(mental_meta_path, "rb") as f:
                self.meta_mental = pickle.load(f)

        # generic index
        generic_index_path = self.generic_dir / "faiss.index"
        generic_meta_path = self.generic_dir / "meta.pkl"
        if generic_index_path.exists() and generic_meta_path.exists():
            self.index_generic = faiss.read_index(str(generic_index_path))
            with open(generic_meta_path, "rb") as f:
                self.meta_generic = pickle.load(f)

        if self.index_mental is None and self.index_generic is None:
            raise RuntimeError("Hiç FAISS index bulunamadı. data/index veya data/index_mental kontrol et.")

        self.min_score_default = float(os.getenv("RAG_MIN_SCORE", "0.45"))
        self.max_per_focus = int(os.getenv("RAG_MAX_PER_FOCUS", "2"))


    def _embed(self, text: str) -> np.ndarray:
        v = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

    def _faiss_search(self, query: str, top_n: int, which: str) -> List[Dict[str, Any]]:
        qv = self._embed(query)

        if which == "mental":
            if self.index_mental is None:
                return []
            scores, idxs = self.index_mental.search(qv, top_n)
            meta = self.meta_mental
        else:
            if self.index_generic is None:
                return []
            scores, idxs = self.index_generic.search(qv, top_n)
            meta = self.meta_generic

        hits: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            m = meta[idx] if idx < len(meta) else {}
            hits.append(
                {
                    "idx": int(idx),
                    "question": str(m.get("question", "")),
                    "answer": str(m.get("answer", "")),
                    "source": str(m.get("source", "")),
                    "focus_area": str(m.get("focus_area", "")),
                    "score": float(score),
                    "which_index": which,
                }
            )
        return hits

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        mental_query = _is_mental_query(q)

        # geniş çek (filtre + diversity sonrası k dolsun)
        top_n = max(k * 8, 40)

        # 1) önce uygun index
        primary = "mental" if (mental_query and self.index_mental is not None) else "generic"
        if primary == "generic" and self.index_generic is None and self.index_mental is not None:
            primary = "mental"

        raw_hits = self._faiss_search(q, top_n=top_n, which=primary)
        raw_hits = _dedupe_hits(raw_hits)
        raw_hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # mental query ise focus filtresi
        if mental_query:
            candidates = [h for h in raw_hits if _is_mental_focus(h.get("focus_area", ""))]
            if len(candidates) < max(5, k):
                candidates = raw_hits
        else:
            candidates = raw_hits

        # ✅ Pozitif sorgularda "Depression" focus'unu sona at (tamamen silme yok)
        if _is_positive_query(q):
            candidates.sort(
                key=lambda h: (_is_depression_focus(h.get("focus_area", "")), -float(h.get("score", 0.0)))
            )

        thresholds = [
            max(self.min_score_default, 0.55),
            self.min_score_default,
            0.35,
            0.20,
            0.0,
        ]

        chosen: List[Dict[str, Any]] = []
        for th in thresholds:
            pool = [h for h in candidates if float(h.get("score", 0.0)) >= th]
            if not pool:
                continue

            chosen = _pick_diverse_topk(pool, k=k, max_per_focus=self.max_per_focus)
            if len(chosen) >= k:
                break

        # 2) yetmediyse diğer index'ten tamamla (fallback fill)
        if len(chosen) < k:
            secondary = "generic" if primary == "mental" else "mental"
            raw2 = self._faiss_search(q, top_n=top_n, which=secondary)
            raw2 = _dedupe_hits(raw2)
            raw2.sort(key=lambda x: x.get("score", 0.0), reverse=True)

            # chosen ile birleştir, tekrar threshold/diversity uygula
            merged = chosen + raw2
            merged = _dedupe_hits(merged)
            merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)

            # mental query ise filtreden geçir
            if mental_query:
                merged_candidates = [h for h in merged if _is_mental_focus(h.get("focus_area", ""))]
                if len(merged_candidates) < max(5, k):
                    merged_candidates = merged
            else:
                merged_candidates = merged

            for th in thresholds:
                pool = [h for h in merged_candidates if float(h.get("score", 0.0)) >= th]
                if not pool:
                    continue
                chosen = _pick_diverse_topk(pool, k=k, max_per_focus=self.max_per_focus)
                if len(chosen) >= k:
                    break

        return chosen[:k]