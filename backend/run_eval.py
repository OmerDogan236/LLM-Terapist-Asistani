
import pandas as pd
from backend.rag import RAGStore

print("Evaluation başlıyor...")

# Dataset yükle
df = pd.read_csv("data/raw/medquad.csv").fillna("")

# 30 soruluk evaluation set
eval_df = df.sample(30, random_state=42)

store = RAGStore()

rows = []

for i, r in eval_df.iterrows():
    question = r["question"]
    true_source = r["source"]

    hits = store.search(question, k=3)

    top1_source = hits[0]["source"] if hits else ""
    top3_sources = [h["source"] for h in hits]

    rows.append({
        "question": question,
        "expected_source": true_source,
        "top1_source": top1_source,
        "top3_sources": "; ".join(top3_sources),
        "top1_hit": int(top1_source == true_source),
        "top3_hit": int(true_source in top3_sources)
    })

out_df = pd.DataFrame(rows)

# Sonuçları kaydet
out_df.to_csv("data/eval_results.csv", index=False)

print("Evaluation tamamlandı.")
print("Top-1 Accuracy:", out_df["top1_hit"].mean())
print("Top-3 Accuracy:", out_df["top3_hit"].mean())
print("Sonuç dosyası: data/eval_results.csv")
