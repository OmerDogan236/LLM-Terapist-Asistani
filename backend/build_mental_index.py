# backend/build_mental_index.py
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss kurulu değil. requirements.txt içinde faiss-cpu olmalı.") from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise RuntimeError("sentence-transformers kurulu değil.") from e


def is_mental_focus(focus_area: str) -> bool:
    t = (focus_area or "").strip().lower()
    if not t:
        return False

    # net medical dışlamalar (false-positive engeli)
    excl = [
        "respiratory", "pulmonary", "sleep apnea", "sleeping sickness",
        "traumatic brain injury", "distress syndrome", "schizenceph",
        "ocular", "hernia", "short stature", "environmental"
    ]
    for e in excl:
        if e in t:
            return False

    # mental-focus patternleri (az ama temiz)
    patterns = [
        r"\banxiety\b",
        r"anxiety disorders",
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
    return any(re.search(p, t) for p in patterns)


def main():
    root = Path(__file__).resolve().parent
    project_root = root.parent  # rag_therapist/
    data_dir = project_root / "data"  # rag_therapist/data
    csv_path = data_dir / "raw" / "medquad.csv"
    out_dir = data_dir / "index_mental"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.fillna("")

    # mental subset seç
    mask = df["focus_area"].apply(is_mental_focus)
    sub = df[mask].copy()

    if len(sub) < 20:
        raise RuntimeError(
            f"Mental subset çok az çıktı: {len(sub)}. Filtreyi gevşetmek gerekir."
        )

    # embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # index için embedding: question (RAG soru benzerliği)
    texts = sub["question"].astype(str).tolist()
    emb = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # meta.pkl list olarak yaz (RAGStore zaten list kabul ediyor)
    meta = []
    for _, r in sub.iterrows():
        meta.append(
            {
                "question": str(r["question"]),
                "answer": str(r["answer"]),
                "source": str(r["source"]),
                "focus_area": str(r["focus_area"]),
            }
        )

    faiss.write_index(index, str(out_dir / "faiss.index"))
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"✅ mental index hazır: {out_dir}")
    print(f"✅ doc sayısı: {len(meta)}")


if __name__ == "__main__":
    main()