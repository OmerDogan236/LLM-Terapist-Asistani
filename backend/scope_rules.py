# backend/scope_rules.py
from __future__ import annotations

import re
from typing import Optional, Tuple, Dict, Any


# -------------------------------------------------------------------
# Scope routing rules (single source of truth)
#
# Goal:
# - Mental-health supportive content => IN
# - Diagnosis / medication / dosage / symptom-checking => ALWAYS OUT
# - Support-intent sentences must be IN even if retrieval score is low
# - High-risk (self-harm / suicide intent) => IN (handled by safety layer)
# -------------------------------------------------------------------

# Anything here is ALWAYS OUT (non-mental domain or medical/diagnosis/treatment)
HARD_OUT_MARKERS = [
    # file/office utilities
    "pdf", "word", "excel", "powerpoint", "ppt",
    "hangi bölüm", "hangi poliklinik", "randevu", "psikiyatri randevusu", "hangi doktora",
    "hangi branş", "MHRS", "randevu al", "fark nedir", "belirtisi nedir", "tanısı nasıl konur",
    "python", "java", "javascript", "kod", "program", "error", "404",
    "modem", "wifi", "driver", "windows", "format",
    "DSM", "kriter", "test", "hangi ilaç",
    # finance / gambling / sports
    "borsa", "bitcoin", "dolar", "altın", "altin", "kredi",
    "maç", "mac", "skor", "bahis", "hisse senedi",

    # food/recipes
    "tarif", "yemek",

    # medical diagnosis / treatment / medication / dosage (ALWAYS OUT for this project)
    "tanı", "tani", "teşhis", "teshis","terapi mi psikiyatri",
    "belirti", "belirtiler", "semptom", "kriter", "dsm", "psikiyatri mi psikolog",
    "tedavi", "ilaç", "ilac", "doz", "mg", "ml", "reçete", "recete",
    "antidepresan", "ssri", "snri", "benzodiazepin", "psikiyatrist mi psikolog",
    "prozac", "zoloft", "sertralin", "fluoksetin", "paroksetin", "hangi uzmana gitmeliyim",
    "escitalopram", "citalopram", "venlafaksin", "duloksetin", "hangi bölüme gitmeliyim",
    "alprazolam", "xanax", "diazepam", "mhrs",
    "antibiyotik", "krem", "eczane", "tahlil", "randevu için hangi bölüm"
]

# Support intent: must be IN even with low retrieval score
SUPPORT_INTENT_MARKERS = [
    # distress / burnout
    "bunalm", "bunal", "tüken", "tuken", "yorul", "yorgun",
    "bittim", "yapacak halim yok", "yapacak gücüm yok", "yapacak gucum yok",
    "içim karışık", "kafam karışık", "üst üste geldi", "her şey üstüme geliyor",
    "bunaldım", "tükendim", "yetişemiyorum", "içim daralıyor", "sıkışıyorum",
    "hiçbir şey yapmak istem", "hicbir sey yapmak istem",
    "hiçbir şey istemiyorum", "hicbir sey istemiyorum",
    "üst üste", "ust uste", "her şey üst üste", "her sey ust uste",
    "üst üste gel", "ust uste gel", "her şey üstüme", "her sey ustume",
    "üstüme gel", "ustume gel", "üstüme geliyor", "ustume geliyor",
    "kendime karşı", "kendime karsi", "kendime nazik", "kendime daha nazik",
    "kendime şefkat", "kendime sefkat", "kendime iyi davran", "kendime iyi davranam",
    "mutsuz", "umutsuz", "yalnız", "yalniz", "değersiz", "degersiz",
    "yetersiz", "suçlu", "suclu", "pişman", "pisman",
    "kendime acımasız", "kendime acimasiz",
    # anxiety / stress
    "kork", "endiş", "endise", "kayg", "anksiy", "panik", "stres",
    "odaklanam", "dalgın", "dalgin", "keyif alamıyorum", "keyif alamiyorum",
    "hiçbir şeyden keyif", "hicbir seyden eskisi keyif", "hiçbir şeyden zevk",
    "eskisi gibi keyif alamıyorum", "eskisi gibi keyif alamiyorum", "hicbir seyden zevk",
    # somatic anxiety / tightness
    "içim daral", "icim daral", "sıkıntı", "sikinti",
    "rahatlayam", "huzursuz", "gergin",
    "don't enjoy", "dont enjoy", "enjoy anything anymore",
    "i don't enjoy anything", "i dont enjoy anything", "i don’t enjoy anything",
    # sleep
    "uyuyam", "uykusuz", "kabus", "i don't enjoy anything","i dont enjoy anything",
    "i dont enjoy anything anymore", "i don’t enjoy anything",
    # compulsive control
    "takıntı", "takinti", "kontrol edem",

    # crying
    "ağlamak", "aglamak",

    # relationships / communication
    "ilişki", "iliski", "yanlış anlaş", "yanlis anlas", "iletişim", "iletisim",
    "sağlıklı konuş", "saglikli konus",

    # boundaries / saying no
    "sınır", "sinir", "hayır", "hayir",

    # self-criticism / compassion
    "içsel eleştir", "icsel elestir", "azarl", "kendimi affedem",
    "öz-şefkat", "oz-sefkat", "oz sefkat", "öz şefkat",

    # rumination / thinking traps
    "düşünc", "dusunc", "susturam", "tekrar tekrar", "senaryo",
    "en kötü ihtimal", "en kotu ihtimal",
    "kontrol etmezsem", "kötü sonuçlan", "kotu sonuclan",

    # suppression / anger
    "bastır", "bastir", "patlıyorum", "patliyorum",

    # social anxiety symptoms (roots)
    "utan", "rezil", "kızar", "kizar", "kaçmak", "kacmak", "kalbim",
]

POSITIVE_MENTAL_MARKERS = [
    "keyif almaya başladım",
    "keyif alıyorum",
    "kendimi daha iyi hissediyorum",
    "daha iyiyim",
    "iyi hissediyorum",
    "rahatladım",
    "umutluyum", "daha umutlu", "kaygım azaldı", "kaygim azaldi",
    "küçük şeylerden keyif", "daha az kaygılanıyorum",
    "kendimi toparlıyorum", "eskisi kadar kaygılanmadığımı",
    "umutlu hissediyorum", "eskisi kadar kaygılanmıyorum",
]

MENTAL_TOPIC_MARKERS = [
    "anksiy", "panik", "depres", "depresyon",
    "travma", "tssb", "okb", "obses", "fobi", "bipolar",
    "yas", "kayıp", "kayip",
    "uyku", "uykusuz",
]
VAGUE_DISTRESS_MARKERS = [
  "yolunda değil", "yolunda degil",
  "emin değilim", "emin degilim",
  "bir şeyler ters", "bir seyler ters",
  "garip hissediyorum",
  "kendim gibi değilim", "kendim gibi degilim",
]


# Thresholds (tuned for your current FAISS score scale)
SCOPE_TOPIC_THRESH = 0.12
SCOPE_IN_STRICT = 0.30
SCOPE_OOS_STRICT = 0.22

# Dosage pattern (ALWAYS OUT)
_DOSAGE_RE = re.compile(r"\b(\d{1,4})\s*(mg|ml)\b", re.IGNORECASE)


def _first_hit(tl: str, markers) -> Optional[str]:
    return next((m for m in markers if m in tl), None)


def _has_dosage(tl: str) -> bool:
    # e.g. "50 mg", "2ml", "kaç mg"
    if _DOSAGE_RE.search(tl):
        return True
    if "kaç mg" in tl or "kac mg" in tl or "kaç ml" in tl or "kac ml" in tl:
        return True
    return False


def scope_llm_gate_tr(text_tr: str) -> str:
    """Optional LLM gate; returns 'IN' or 'OUT'."""
    # Local import to avoid circulars
    from backend.llm_client import scope_llm_gate_tr as _gate  # type: ignore

    try:
        out = (_gate(text_tr) or "").strip().upper()
        if out.startswith("IN"):
            return "IN"
        return "OUT"
    except Exception:
        return "OUT"


def decide_domain_tr(
    *,
    msg_tr: str,
    pre_risk: str,
    score: Optional[float],
    use_llm_gate: bool = True,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Returns: (domain, reason, debug)
    domain: 'in_scope' | 'out_of_scope'
    """
    t = (msg_tr or "").strip()
    tl = t.lower()

    # 0) High-risk exception: always allow to reach safety layer
    if (pre_risk or "").lower().strip() == "high":
        return "in_scope", "high_risk_exception", {"score": score}

    # 1) Hard OUT (includes diagnosis/meds/dosage/symptom-checking)
    hit = _first_hit(tl, HARD_OUT_MARKERS)
    if hit:
        return "out_of_scope", "hard_out_marker", {"hit": hit, "score": score}
    if _has_dosage(tl):
        return "out_of_scope", "dosage_pattern", {"score": score}
    if _first_hit(tl, POSITIVE_MENTAL_MARKERS):
        return "in_scope", "positive_mental_state", {"score": score}



    # 2) Support intent override (always IN)
    sup = _first_hit(tl, SUPPORT_INTENT_MARKERS)
    if sup:
        return "in_scope", "support_intent", {"hit": sup, "score": score}

    # 3) Score-based routing (with a small 'mental topic' assist)
    s = float(score) if (score is not None) else 0.0

    has_mental_topic = _first_hit(tl, MENTAL_TOPIC_MARKERS) is not None
    if has_mental_topic and s >= SCOPE_TOPIC_THRESH:
        return "in_scope", "mental_topic+score", {"score": s}

    if s >= SCOPE_IN_STRICT:
        return "in_scope", "high_score_only_strict", {"score": s}
    if _first_hit(tl, VAGUE_DISTRESS_MARKERS):
        # llm gate açıksa: gate’e bırak (IN gelirse IN)
        return "in_scope", "llm_gate_borderline", {"score": score}

    if s < SCOPE_OOS_STRICT:
        return "out_of_scope", "low_score_strict", {"score": s}

    # 4) Borderline band -> optional LLM gate
    if use_llm_gate:
        gate = scope_llm_gate_tr(t)
        if gate == "IN":
            return "in_scope", "llm_gate=IN", {"score": s}
        return "out_of_scope", "llm_gate=OUT", {"score": s}

    return "out_of_scope", "borderline_no_gate", {"score": s}
