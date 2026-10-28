
import pandas as pd

df = pd.read_csv("data/raw/medquad.csv").fillna("")
sample = df.sample(50, random_state=42)[["question","answer","source","focus_area"]]
sample.to_csv("data/eval_50.csv", index=False)
print("OK -> data/eval_50.csv üretildi:", len(sample))
