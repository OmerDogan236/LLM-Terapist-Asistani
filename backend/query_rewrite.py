# backend/query_rewrite.py

from typing import Optional


MENTAL_REWRITE_MAP = [
    # NEW: overwhelmed / confusion / "üst üste geldi" support intent
    (
        [
            "içim karışık", "icim karisik",
            "kafam karışık", "kafam karisik",
            "üst üste", "ust uste",
            "üstüme geliyor", "ustume geliyor",
            "her şey üstüme geliyor", "hersey ustume geliyor",
            "bunaldım", "bunaldim",
            "yetişemiyorum", "yetisemiyorum",
            "çok yoruldum", "cok yoruldum",
            "tükendim", "tukendim",
            "bıktım", "biktim",
            "dayanamıyorum", "dayanamiyorum",
            "içim daralıyor", "icim daraliyor",
        ],
        "feeling overwhelmed emotional confusion stress coping strategies mental health support",
    ),

    (["kaygı", "kaygi", "anksiy", "endişe", "endis"],
     "anxiety symptoms and coping strategies mental health"),

    (["uyuyamıyorum", "uyuyamiyorum", "uykusuz", "uyku"],
     "insomnia and sleep problems related to mental health"),

    (["umutsuz", "tüken", "tuken", "bıktım", "biktim"],
     "hopelessness and emotional exhaustion mental health"),

    (["mutsuz", "keyifsiz", "moralsiz"],
     "persistent low mood depression mental health"),

    (["panik", "panik atak"],
     "panic attacks symptoms and coping"),

    (["stres", "gergin"],
     "stress management and mental wellbeing"),
]


def rewrite_query_tr_to_en(message: str, memory_summary: Optional[str] = None) -> str:
    """
    Türkçe mental mesajı, retrieval için kısa ve net İngilizce query'ye çevirir.
    LLM KULLANMAZ (deterministik).
    """
    t = (message or "").lower()

    for keys, rewrite in MENTAL_REWRITE_MAP:
        if any(k in t for k in keys):
            return rewrite

    # fallback (gene mental bağlamlı)
    base = "mental health emotional support"

    # memory varsa biraz güçlendir
    if memory_summary:
        return base + " ongoing emotional distress"

    return base
