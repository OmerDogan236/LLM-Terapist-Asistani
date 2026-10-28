# backend/risk.py

from typing import Optional, Tuple

def analyze_emotion(msg: str, scope_reason: str = "", risk_level: str = "") -> Tuple[str, str, Optional[str]]:
    """Keyword tabanlı emotion analizi.
    Returns: (label, reason, hit)
      label: positive | neutral | negative
    """
    if not msg:
        return "neutral", "empty", None

    t = (msg or "").lower().strip()

    improve_markers = [
        "azaldı", "azaldi",
        "daha az",
        "eskisi kadar değil", "eskisi kadar degil",
        "eskisi kadar kaygılanm", "eskisi kadar kaygilanm",
        "artık eskisi gibi değil", "artik eskisi gibi degil",
        "daha iyi hissediyorum", "daha iyiyim",
        "rahatladım", "rahatladim",
        "iyileşiyorum", "iyilesiyorum",
        "toparlanıyorum", "toparlaniyorum",
        "kontrol edebiliyorum", "kontrol edebiliyorum",
        "geçiyor gibi", "geciyor gibi",
    ]
    target_topics = [
        "kayg", "anksiy", "panik", "stres", "gergin",
        "huzursuz", "kuruntu", "endiş", "endise",
    ]
    hit_improve = next((m for m in improve_markers if m in t), None)
    hit_topic = next((m for m in target_topics if m in t), None)

    if hit_improve and hit_topic:
        return "positive", "improvement_override", f"{hit_improve} + {hit_topic}"

    # ✅ 0) Scope support intent => negative
    if (scope_reason or "").lower().strip() == "support_intent":
        return "negative", "support_intent_override", None

    # ✅ 0.1) Medium risk => en az negative
    if (risk_level or "").lower().strip() == "medium":
        # (keyword bulamazsa bile distress kabul ediyoruz)
        return "negative", "risk_medium_override", None

    # ✅ 0.2) “üst üste geldi / overwhelm” distress => negative
    distress_markers = [
        "üst üste", "ust uste",
        "üstüme geliyor", "ustume geliyor",
        "her şey üst üste", "her sey ust uste",
        "yetişemiyorum", "yetisemiyorum",
        "bunaldım", "bunaldim",
        "tükendim", "tukendim",
        "içim karışık", "icim karisik",
    ]
    hit_distress = next((w for w in distress_markers if w in t), None)
    if hit_distress:
        return "negative", "distress_marker", hit_distress

    # --- mevcut keyword logic aynen devam ---
    negative_words = [
        ("mutsuz", "keyword_negative"),
        ("umutsuz", "keyword_negative"),
        ("kayg", "keyword_negative"),
        ("anksiy", "keyword_negative"),
        ("panik", "keyword_negative"),
        ("huzursuz", "keyword_negative"),
        ("gergin", "keyword_negative"),
        ("içim daral", "keyword_negative"),
        ("icim daral", "keyword_negative"),
        ("sıkıntı", "keyword_negative"),
        ("sikinti", "keyword_negative"),
        ("kork", "keyword_negative"),
        ("endiş", "keyword_negative"),
        ("endise", "keyword_negative"),
        ("yalnız", "keyword_negative"),
        ("yalniz", "keyword_negative"),
        ("değersiz", "keyword_negative"),
        ("degersiz", "keyword_negative"),
        ("yetersiz", "keyword_negative"),
        ("zevk almı", "keyword_negative"),
        ("zevk alam", "keyword_negative"),
        ("keyif almı", "keyword_negative"),
        ("keyif alam", "keyword_negative"),
        ("hiçbir şeyden", "keyword_negative"),
        ("hicbir seyden", "keyword_negative"),
        ("istemiyorum", "keyword_negative"),
        ("bıktım", "keyword_negative"),
        ("biktim", "keyword_negative"),
        ("tükendim", "keyword_negative"),
        ("tukendim", "keyword_negative"),
        ("kötüye gidiyor", "keyword_negative"),
        ("kotuye gidiyor", "keyword_negative"),
        ("anlamsız", "keyword_negative"),
        ("anlamsiz", "keyword_negative"),
    ]

    positive_words = [
        ("iyi hissediyorum", "keyword_positive"),
        ("daha iyi", "keyword_positive"),
        ("mutluyum", "keyword_positive"),
        ("keyif almaya başladım", "keyword_positive"),
        ("keyif almaya basladim", "keyword_positive"),
        ("rahatladım", "keyword_positive"),
        ("rahatladim", "keyword_positive"),
        ("şükür", "keyword_positive"),
        ("sukur", "keyword_positive"),
        ("minnettar", "keyword_positive"),
        ("umutluyum", "keyword_positive"),
        ("kendime daha fazla vakit", "keyword_positive"),
        ("kendime vakit ayırıyorum", "keyword_positive"),
        ("kendime vakit ayiriyorum", "keyword_positive"),
        ("umut", "keyword_positive"),
        ("umutlu", "keyword_positive"),
        ("daha iyi", "keyword_positive"),
        ("rahatlad", "keyword_positive"),
        ("iyi gibiyim", "keyword_positive"),
        ("iyiyim", "keyword_positive"),
        ("fena değil", "keyword_positive"),
        ("fena degil", "keyword_positive"),
        ("idare ediyorum", "keyword_positive"),
    ]

    neg_hit = next((w for w, _ in negative_words if w in t), None)
    pos_hit = next((w for w, _ in positive_words if w in t), None)

    if neg_hit and pos_hit:
        distress = any(x in t for x in ["dayanam", "katlanam", "anlamsız", "anlamsiz", "bitsin", "kendime zarar", "ölmek", "olmek"])
        if distress:
            return "negative", "mixed_distress", neg_hit
        return "neutral", "mixed", None

    if neg_hit:
        return "negative", "keyword_negative", neg_hit
    if pos_hit:
        return "positive", "keyword_positive", pos_hit
    return "neutral", "default", None

def normalize_emotion(emotion: str, risk_level: str) -> str:
    """
    Risk'e göre emotion label'ını normalize eder.
    - risk=high => critical
    - risk=medium => positive olamaz (neutral'a çek)
    - risk=low => negative olamaz (neutral'a çek)
    """
    e = (emotion or "").strip().lower()
    r = (risk_level or "").strip().lower()

    if r == "high":
        return "critical"

    if r == "medium" and e == "positive":
        return "neutral"

    if r == "low" and e == "negative":
        return "neutral"

    if e not in ("positive", "neutral", "negative", "critical"):
        return "neutral"

    return e