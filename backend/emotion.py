# emotion.py

from transformers import pipeline

_sentiment = pipeline(
    "sentiment-analysis",
    model="savasy/bert-base-turkish-sentiment-cased"
)

POS_WORDS = [
    "iyiyim", "daha iyiyim", "mutluyum", "sevinçliyim", "sevindim", "rahatladım",
    "iyi hissediyorum", "güzel", "harika", "şükür", "keyfim yerinde"
]
NEG_WORDS = [
    "mutsuzum", "kötüyüm", "berbat", "umutsuzum", "yalnızım",
    "hiçbir şey yolunda gitmiyor", "bunaldım", "bitkinim", "çaresizim"
]

# Distress / overwhelm ama "mutsuzum" gibi direkt negatif kelime geçmeyen cümleler için
DISTRESS_MARKERS = [
    "üst üste", "ust uste",
    "üst üste gel", "ust uste gel",
    "üstüme geliyor", "ustume geliyor",
    "her şey üst üste", "her sey ust uste",
    "her şey üstüme", "her sey ustume",
    "içim karışık", "icim karisik",
    "kafam karışık", "kafam karisik",
    "yetişemiyorum", "yetisemiyorum",
    "tükendim", "tukendim",
    "dayanamıyorum", "dayanamiyorum",
    "içim daralıyor", "icim daraliyor",
]

INFO_PATTERNS = [
    "nedir", "nelerdir", "belirtileri", "semptom", "tanı", "tedavi",
    "neden olur", "nasıl geçer", "ne iyi gelir", "ne yapmalıyım"
]

def _has_any(text: str, words: list[str]) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)

def is_info_question(text: str) -> bool:
    t = (text or "").lower()
    # kısa bilgi soruları genelde nötr sayılmalı
    return any(p in t for p in INFO_PATTERNS) and len(t.split()) <= 8


def analyze_emotion(text: str, scope_reason: str = "", risk_level: str = "") -> str:
    """
    Emotion etiketi döndürür: positive / neutral / negative
    - scope_reason ve risk_level opsiyonel; verilirse deterministik override yapar.
    """
    t = (text or "").strip()
    if not t:
        return "neutral"

    # 0) Bilgi soruları nötr (override'tan önce)
    if is_info_question(t):
        return "neutral"

    # 0.5) Deterministik override (senin istediğin davranış)
    # Support intent => negatif (distress)
    if (scope_reason or "").strip().lower() == "support_intent":
        return "negative"

    # Medium risk => negatif (duygulanım en az negatif olmalı)
    if (risk_level or "").strip().lower() == "medium":
        return "negative"

    # Distress marker varsa negatif (örn: "üst üste geldi" gibi)
    if _has_any(t, DISTRESS_MARKERS):
        return "negative"

    # 1) Keyword override
    if _has_any(t, POS_WORDS):
        return "positive"
    if _has_any(t, NEG_WORDS):
        return "negative"

    # 2) Model
    try:
        res = _sentiment(t[:512])[0]
        label = str(res.get("label", "")).lower()
        score = float(res.get("score", 0.0))

        if score < 0.70:
            return "neutral"

        if "negative" in label:
            return "negative"
        if "positive" in label:
            return "positive"
        return "neutral"
    except Exception:
        return "neutral"
