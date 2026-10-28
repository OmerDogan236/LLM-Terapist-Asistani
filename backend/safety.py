# backend/safety.py
import re
from typing import List

# ❗ Bilgi / akademik / tanı soruları (intihar nedir? gibi)
INFO_TRIGGERS = [
    "nedir",
    "nelerdir",
    "belirtileri",
    "belirtiler",
    "semptom",
    "tanı",
    "tani",
    "tedavi",
    "nasıl",
    "nasil",
    "ne demek",
    "açıkla",
    "acikla",
    "bilgi",
]

# 💤 Uyku / yorgunluk tek başına LOW olsun (ama medium/high varsa onlar baskın)
SLEEP_LOW_TRIGGERS = [
    "uyuyamıyorum", "uyuyamiyorum",
    "uykusuzum", "uykusuz",
    "uykuluyum",
    "insomnia",
    "uyku problemim var",
]

def _has_any(text: str, triggers: List[str]) -> bool:
    return any(k in text for k in triggers)


def _looks_like_info_query(text: str) -> bool:
    """
    'intihar nedir?' gibi bilgi sorularını LOW'a çekmek için.
    Ama açık niyet/eylem varsa ASLA bilgi sayılmaz.
    """
    t = (text or "").lower()
    if not _has_any(t, INFO_TRIGGERS):
        return False

    intent_markers = [
        "istiyorum", "isterim",
        "edeceğim", "edecegim", "edicem",
        "yapacağım", "yapacagim", "yapicam",
        "etmek istiyorum", "yapmak istiyorum",
        "planlıyorum", "planliyorum",
        "hazırladım", "hazirladim",
    ]
    first_person = ["ben ", "kendim", "kendime", "kendimi"]

    if any(x in t for x in intent_markers):
        return False
    if any(x in t for x in first_person):
        return False

    return True


# 🔴 HIGH RISK – niyet/eylem köklerini yakalayan regexler
# Not: "intihar" kelimesi tek başına HIGH yapmıyor; mutlaka niyet/eylem lazım.
_HARD_HIGH_PATTERNS = [
    r"\b(kafamı|kafama|kafasına|kafasını)\b.*\b(vurmak|kesmek|sıkmak)\w*\b",
    r"\b(kendimi|kendime|kendine)\s*(öldür|oldur|katlet|gebert|bitir)\w*\b",
    r"\b(kendimi|kendime)\s*zara?r\s*ver\w*\b",
    r"\b(can(ı|i)ma|kendime)\s*k(ı|i)y\w*\b",
    r"\bintihar\s*(et|edece|edecegim|edicem|ederim|etmek)\w*\b",
    r"\b(ölmek|olmek)\s*istiyor\w*\b",
    r"\b(yaşamak|yasamak)\s*istemiyor\w*\b",
    r"\b(kendimi|kendime)\s*(öldür|oldur)\w*\s*(ecegim|acağım|acagim|cam|cem)\b",
    '\\b(bitsin|bitmesini)\\s*istiyor\\w*\\b',
    '\\b(her\\s*şey\\s*)?bitsin\\s*istiyor\\w*\\b',
    '\\b(dayanacak|katlanacak)\\s*güc(ü|u)m\\s*kalmad\\w*\\b',
    '\\b(dayanacak|katlanacak)\\s*güc(ü|u)m\\s*kalma(dı|di)\\b',
    '\\bartık\\s*dayanam\\w*\\b',
    '\\bartık\\s*katlanam\\w*\\b',

]


def _match_high_risk(text: str) -> bool:
    t = (text or "").lower()
    for p in _HARD_HIGH_PATTERNS:
        if re.search(p, t):
            return True
    return False


# 🟡 MEDIUM RISK – yoğun duygu/panik vb.
MEDIUM_RISK_TRIGGERS = [
    "çok mutsuzum",
    "cok mutsuzum",
    "mutsuzum",
    "çok umutsuzum",
    "cok umutsuzum",
    "umutsuzum",
    "hiç umudum yok",
    "hic umudum yok",
    "tükenmiş",
    "tukenmis",
    "kötü hissediyorum",
    "kotu hissediyorum",
    "iyi hissetmiyorum",
    "canım sıkkın",
    "cani sikkin",
    "keyifsizim",
    "moralim bozuk",
    "çok kaygılıyım",
    "cok kaygiliyim",
    "kaygılıyım",
    "kaygiliyim",
    "kaygım arttı",
    "kaygim artti",
    "yoğun kaygı yaşıyorum",
    "yogun kaygi yasiyorum",
    "panik",
    "panik atak",
    "anksiyete",
    "depresyon",
    "depres",
]
VAGUE_LOW_TRIGGERS = [
    "boşlukta hissediyorum", "boslukta hissediyorum",
    "garip hissediyorum",
    "kendim gibi hissetmiyorum",
    "yolunda değil gibi",
    "emin değilim",
    "iyi gibiyim ama",
    "iyiyim ama",
    "fena değil ama",
    "idare ediyorum ama"
    # Türkçe büyük İ lower() problemi için:
    "i\u0307yi gibiyim ama",  # "i̇yi gibiyim ama"
    "i\u0307yiyim ama",  # "i̇yiyim ama"
    "her şey yolunda gibi ama",
    "her sey yolunda gibi ama",
    "nedense keyfim yok",
    "nedense keyfim yok.",
    "keyfim yok",
]


# LLM risk sınıflandırıcı

try:
    from backend.llm_client import classify_risk_tr  # type: ignore
except Exception:
    classify_risk_tr = None


def assess_risk(msg: str) -> str:
    """
    Genel kural: en yüksek seviye kazanır. (high > medium > low)

    Adımlar:
    1) Regex ile açık HIGH niyet/eylem -> HIGH
    2) MEDIUM tetikleri -> MEDIUM
    3) Bilgi sorusu (nedir/nasıl...) -> LOW
    4) Sadece uyku/yorgunluk -> LOW
    5) LLM classifier (opsiyonel): SADECE yukarı çeker, aşağı indirmez.
    """
    if not msg:
        return "low"

    text = msg.lower()

    # 1) HIGH
    rule_high = _match_high_risk(text)
    if rule_high:
        base = "high"
    else:
        # 2) MEDIUM
        rule_medium = _has_any(text, MEDIUM_RISK_TRIGGERS)
        if rule_medium:
            base = "medium"
        else:
            # 3) INFO -> LOW
            if _looks_like_info_query(text) or _has_any(text, INFO_TRIGGERS):
                base = "low"
            else:
                # 4) SLEEP ONLY -> LOW
                if _has_any(text, SLEEP_LOW_TRIGGERS):
                    base = "low"
                else:
                    base = "low"

    if base == "low" and _has_any(text, VAGUE_LOW_TRIGGERS):
        return base

    # 5) LLM classifier: sadece yükselt
    if classify_risk_tr is not None:
        try:
            llm_label = classify_risk_tr(msg)
        except Exception:
            llm_label = None

        if llm_label == "high":
            return "high"
        if llm_label == "medium" and base == "low":
            return "medium"
        # llm low gelirse base'i düşürme

    return base


def gate_high_risk(current_msg: str, risk_level: str) -> str:
    """
    Geçmiş yüzünden high yapışmasını engeller.
    'high' sadece mevcut mesajda açık high niyet varsa kalsın.
    """
    if risk_level != "high":
        return risk_level

    if not _match_high_risk(current_msg or ""):
        return "medium"

    return "high"


def is_crisis(msg: str) -> bool:
    return assess_risk(msg) == "high"


def crisis_message_tr() -> str:
    return (
        "Şu an çok zorlandığını hissediyorum ve bununla yalnız başına başa çıkmak zorunda değilsin.\n\n"
        "Eğer kendine zarar verme düşüncelerin varsa, lütfen **hemen** yerel acil destek hattına başvur.\n\n"
        "🇹🇷 Türkiye için: **112 Acil**\n\n"
        "Ayrıca güvendiğin bir yakınla **şimdi** iletişime geçmeni öneririm."
    )