import json
import os
from typing import Dict, Any, List, Tuple

from backend.safety import assess_risk
from backend.rag import RAGStore
from backend.scope_rules import decide_domain_tr
from backend.llm_client import translate_tr_to_en_scope


def load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mental_top_score(store: RAGStore, query_en: str) -> float:
    try:
        hits = store._faiss_search(query_en, top_n=1, which="mental")  # type: ignore[attr-defined]
        if not hits:
            return 0.0
        s = hits[0].get("score", 0.0)
        return float(s) if s is not None else 0.0
    except Exception:
        return 0.0


# Basit cache: aynı TR metin tekrar gelirse LLM çağırma
_TRANSLATE_CACHE: Dict[str, str] = {}


def tr_to_en_cached(text_tr: str) -> str:
    if text_tr in _TRANSLATE_CACHE:
        return _TRANSLATE_CACHE[text_tr]
    en = translate_tr_to_en_scope(text_tr)
    _TRANSLATE_CACHE[text_tr] = en
    return en


def classify_scope(
    msg: str,
    store: RAGStore,
    use_llm_gate: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    t = (msg or "").strip()
    if not t:
        return "out", {"reason": "empty"}

    # 1) Risk first
    pre_risk = assess_risk(t)
    if pre_risk == "high":
        return "crisis", {"reason": "high_risk"}

    # 2) Score (TR->EN)
    q_en = tr_to_en_cached(t)
    score = mental_top_score(store, q_en)

    domain, reason, dbg = decide_domain_tr(
        msg_tr=t,
        pre_risk=pre_risk,
        score=score,
        use_llm_gate=use_llm_gate,
    )

    if domain == "in_scope":
        return "in", {"reason": reason, "score": score, "en": q_en, **(dbg or {})}

    return "out", {"reason": reason, "score": score, "en": q_en, **(dbg or {})}


def summarize(results: List[Dict[str, Any]]) -> None:
    total = len(results)
    correct = sum(1 for r in results if r["ok"])
    acc = correct / total if total else 0.0

    labels = ["in", "out", "crisis"]
    cm = {gt: {pr: 0 for pr in labels} for gt in labels}
    for r in results:
        cm[r["gt"]][r["pred"]] += 1

    print("\n=== SUMMARY ===")
    print(f"Total: {total}")
    print(f"Accuracy: {acc:.3f} ({correct}/{total})")

    missed_crisis = [r for r in results if r["gt"] == "crisis" and r["pred"] != "crisis"]
    false_out = [r for r in results if r["gt"] == "in" and r["pred"] == "out"]
    false_in = [r for r in results if r["gt"] == "out" and r["pred"] == "in"]

    print("\n=== KEY METRICS ===")
    print(f"Missed crisis: {len(missed_crisis)} (placeholder kullanıyorsan normal)")
    print(f"False OUT (in->out): {len(false_out)}")
    print(f"False IN  (out->in): {len(false_in)}")

    if false_out:
        print("\n--- FALSE OUT SAMPLE ---")
        for r in false_out[:10]:
            print(f"- {r['text']!r} score={r['debug'].get('score')} reason={r['debug'].get('reason')}")

    if false_in:
        print("\n--- FALSE IN SAMPLE ---")
        for r in false_in[:10]:
            print(f"- {r['text']!r} score={r['debug'].get('score')} reason={r['debug'].get('reason')}")


def main():
    print(">>> run_scope_eval başladı")

    cases_path = os.path.join(os.path.dirname(__file__), "eval_scope_cases.json")
    cases = load_cases(cases_path)
    print(f">>> case sayısı: {len(cases)}")

    print(">>> RAGStore yükleniyor...")
    store = RAGStore()
    print(">>> RAGStore hazır.")

    results: List[Dict[str, Any]] = []
    for i, c in enumerate(cases, start=1):
        text = c["text"]
        gt = c["label"]
        pred, debug = classify_scope(text, store=store)
        results.append({"text": text, "gt": gt, "pred": pred, "ok": pred == gt, "debug": debug})

        if i % 25 == 0:
            print(f">>> ilerleme: {i}/{len(cases)} (translate cache size={len(_TRANSLATE_CACHE)})")

    missed_crisis = [
        r for r in results
        if r["gt"] == "crisis" and r["pred"] != "crisis"
    ]
    print("\n--- MISSED CRISIS (DETAYLI) ---")
    for r in missed_crisis:
        print("TEXT :", r["text"])
        print("PRED :", r["pred"])
        print("DEBUG:", r["debug"])
        print("-" * 40)

    summarize(results)


if __name__ == "__main__":
    main()
