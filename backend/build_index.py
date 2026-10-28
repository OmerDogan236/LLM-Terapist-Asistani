import os
import pickle
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

print("START build_index")

DATA_PATH = "data/raw/medquad.csv"
INDEX_PATH = "data/index/faiss.index"
META_PATH = "data/index/meta.pkl"

os.makedirs("data/index", exist_ok=True)

df = pd.read_csv(DATA_PATH).fillna("")
print("CSV OK, rows:", len(df))

texts = (df["question"].astype(str) + "\n" + df["answer"].astype(str)).tolist()

print("Embedding başlıyor...")
emb_model = SentenceTransformer("intfloat/e5-small-v2")
emb = emb_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

faiss.write_index(index, INDEX_PATH)

meta = df[["question", "answer", "source", "focus_area"]].to_dict("records")
with open(META_PATH, "wb") as f:
    pickle.dump(meta, f)

print("OK -> index hazır. vectors:", index.ntotal)