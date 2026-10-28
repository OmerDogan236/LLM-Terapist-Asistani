# backend/source_why.py

import re
from typing import Dict, Any, List, Tuple


TR_KEYWORDS = [
    "kaygı", "kaygi", "anksiyete", "panik", "stres", "korku",
    "uyku", "uyuyamıyorum", "uyuyamiyorum", "uykusuz",
    "mutsuz", "umutsuz", "depres", "depresyon",
    "yalnız", "yalniz", "travma", "okb", "ocd", "bipolar",
    "nefes", "nefes darlığı", "nefes darligi", "çarpıntı", "carpinti",
    "atak", "kriz", "takıntı", "takinti",

]
TR_STOPWORDS = [
    "ve","veya","ama","fakat","ancak","çünkü","cunku","gibi",
    "bir","bu","şu","su","o","ben","sen","biz","siz","onlar",
    "de","da","ki","mi","mı","mu","mü","icin","için","ile",
    "son","zaman","zamanlarda","nedense","daha","çok","cok",
    "kendi", "kendim", "kendimi", "kendine", "kendini"
]
# TR -> EN basit eşleme (kaynaklar çoğunlukla İngilizce olduğu için)
TR_EN_MAP = {
    "panik": ["panic", "panic attack", "panic disorder"],
    "atak": ["attack", "episode"],
    "kaygı": ["anxiety", "anxious", "anxiety disorder"],
    "kaygi": ["anxiety", "anxious", "anxiety disorder"],
    "anksiyete": ["anxiety", "anxious", "anxiety disorder"],
    "stres": ["stress", "stressed", "stress management"],
    "korku": ["fear", "scared", "phobia", "phobic"],
    "travma": ["trauma", "ptsd", "post-traumatic"],
    "depresyon": ["depression", "depressive", "mood disorder"],
    "depres": ["depression", "depressive"],
    "uyku": ["sleep", "insomnia", "sleep hygiene"],
    "uykusuz": ["insomnia", "sleepless", "sleep deprivation"],
    "uyuyamıyorum": ["insomnia", "can't sleep", "difficulty sleeping"],
    "uyuyamiyorum": ["insomnia", "can't sleep", "difficulty sleeping"],
    "nefes": ["breath", "breathing", "shortness of breath"],
    "nefes darlığı": ["shortness of breath", "breathing difficulty"],
    "nefes darligi": ["shortness of breath", "breathing difficulty"],
    "çarpıntı": ["palpitations", "rapid heartbeat", "heart racing"],
    "carpinti": ["palpitations", "rapid heartbeat", "heart racing"],
    "okb": ["ocd", "obsessive compulsive"],
    "ocd": ["ocd", "obsessive compulsive"],
    "bipolar": ["bipolar", "bipolar disorder"],
    "takıntı": ["obsession", "obsessive", "compulsion"],
    "takinti": ["obsession", "obsessive", "compulsion"],
    "yalnız": ["lonely", "loneliness", "social support"],
    "yalniz": ["lonely", "loneliness", "social support"],
    "mutsuz": ["sad", "sadness", "low mood"],
    "umutsuz": ["hopeless", "hopelessness"],
}


def _tokenize_tr(text: str) -> List[str]:
    t = (text or "").lower()
    return re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", t)


def _pick_keywords(msg: str, max_k: int = 5) -> List[str]:
    t = (msg or "").lower()
    found: List[str] = []

    for k in TR_KEYWORDS:
        if k in t:
            found.append(k)
        if len(found) >= max_k:
            break

    if not found:
        words = _tokenize_tr(t)
        filtered = []
        for w in words:
            w2 = (w or "").strip().lower()
            if len(w2) < 4:
                continue
            if w2 in TR_STOPWORDS:
                continue
            filtered.append(w2)
            if len(filtered) >= 3:
                break
        found = filtered or words[:3]  # hiçbir şey kalmazsa eski davranışa düş

    # unique
    out = []
    seen = set()
    for x in found:
        x = (x or "").strip()
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)

    return out[:max_k]


def _build_needles(keys: List[str]) -> List[Tuple[str, str]]:
    """
    Return list of (display, needle) pairs.
    display: UI'da gösterilecek metin
    needle: haystack içinde aranacak ifade (TR veya EN)
    """
    pairs: List[Tuple[str, str]] = []
    seen = set()

    for k in keys:
        kk = (k or "").strip().lower()
        if not kk:
            continue

        # TR needle
        disp_tr = kk
        needle_tr = kk
        if (disp_tr, needle_tr) not in seen:
            pairs.append((disp_tr, needle_tr))
            seen.add((disp_tr, needle_tr))

        # EN needles
        en_list = TR_EN_MAP.get(kk, [])
        for en in en_list:
            en2 = (en or "").strip().lower()
            if not en2:
                continue
            disp = f"{kk} ({en2})"  # kanıt daha anlaşılır
            needle = en2
            if (disp, needle) not in seen:
                pairs.append((disp, needle))
                seen.add((disp, needle))

    return pairs


def _find_matches(pairs: List[Tuple[str, str]], haystack: str, max_m: int = 3) -> List[str]:
    h = (haystack or "").lower()
    matches: List[str] = []
    for disp, needle in pairs:
        if needle and needle in h:
            matches.append(disp)
        if len(matches) >= max_m:
            break
    return matches


def explain_sources_tr(user_msg: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = _pick_keywords(user_msg, max_k=5)
    pairs = _build_needles(keys)

    out: List[Dict[str, Any]] = []

    for s in (sources or []):
        q = (s.get("question") or "")
        a = (s.get("answer") or "")
        fa = (s.get("focus_area") or "")
        score = s.get("score", None)

        hay = f"{q}\n{a}\n{fa}"
        matches = _find_matches(pairs, haystack=hay, max_m=3)

        if matches:
            if len(matches) == 1:
                why = f"Bu kaynak, mesajındaki “{matches[0]}” ifadesiyle örtüştüğü için getirildi."
            else:
                joined = "”, “".join(matches)
                why = f"Bu kaynak, mesajındaki “{joined}” ifadeleriyle örtüştüğü için getirildi."
        else:
            if isinstance(score, (int, float)):
                why = f"Bu kaynak, soruna semantik olarak benzer bulunduğu için getirildi (skor: {score:.2f})."
            else:
                why = "Bu kaynak, soruna semantik olarak benzer bulunduğu için getirildi."

        ns = dict(s)
        ns["why"] = why
        out.append(ns)

    return out