# backend/llm_client.py
import os
import json
import re
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv

load_dotenv()

# OpenAI (new SDK)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def translate_tr_to_en_scope(text_tr: str) -> str:
    """
    Out_of_scope / in_scope skorlaması için TR -> EN kısa çeviri.
    Amaç: retrieval'a uygun, kısa ve nötr bir İngilizce sorgu üretmek (rewrite değil).
    OpenAI yoksa/fail olursa fallback: orijinal metni döndürür.
    """
    t = (text_tr or "").strip()
    if not t:
        return ""

    # OpenAI yoksa fallback
    if OpenAI is None:
        return t

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the user's Turkish message to concise English for mental-health scope checking. "
                        "Return ONLY the English translation. Keep it short. No extra commentary."
                    ),
                },
                {"role": "user", "content": t},
            ],
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or t
    except Exception:
        return t



def _get_client():
    if OpenAI is None:
        raise RuntimeError("openai paketi bulunamadı. `pip install openai` kur.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY bulunamadı. .env içine ekle.")
    return OpenAI(api_key=api_key)


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        if not text:
            return {}
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
        return {}


# ------------------------------------------------------------
# DOMAIN ROUTER (OUT OF SCOPE)  ✅ Hybrid: Rule + LLM gate
# ------------------------------------------------------------

def _domain_llm_json(user_text: str) -> Dict[str, Any]:
    """
    LLM'i sadece router olarak kullanır:
    JSON döndürür: {"domain":"in_scope|out_of_scope", "reason":"..."}
    """
    client = _get_client()

    sys = (
        "Sen bir DOMAIN ROUTER modülüsün.\n"
        "Görevin: Kullanıcının mesajı 'mental sağlık & iyi oluş desteği' kapsamına giriyor mu karar vermek.\n\n"
        "KAPSAM (in_scope):\n"
        "- Kullanıcı kendi duygu durumu / stres / kaygı / panik / uyku / motivasyon / ilişki stresi gibi "
        "konularda destek istiyor, başa çıkma önerisi istiyor.\n\n"
        "KAPSAM DIŞI (out_of_scope):\n"
        "- Akademik bilgi: 'X nedir', 'teşhis nasıl konur', 'tedavi nedir' gibi sadece bilgi\n"
        "- Maaş/fiyat/para\n"
        "- Teknik istek: kod yaz, programlama, donanım, kripto, spor, hava, yol tarifi\n"
        "- Yaratıcı görev: şiir/destan/roman/hikaye/senaryo yaz\n"
        "- Nasıl yapılır / tarif / tutorial\n\n"
        "ÖNEMLİ:\n"
        "- Mesajda 'depresyon/kaygı/stres' kelimesi geçmesi tek başına in_scope yapmaz.\n"
        "- Kullanıcının kendisiyle ilgili destek niyeti yoksa out_of_scope seç.\n\n"
        "Sadece JSON döndür:\n"
        "{\"domain\":\"in_scope|out_of_scope\",\"reason\":\"kısa gerekçe\"}\n"
    )

    r = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f"Mesaj: {user_text}"},
        ],
        temperature=0,
        max_tokens=120,
    )

    content = (r.choices[0].message.content or "").strip()
    return _safe_json_loads(content)


def classify_domain_llm_gate_tr(msg: str) -> Tuple[str, str]:
    """
    Gri bölgede LLM gate çalıştırır.
    Asla uzun cevap üretmez, sadece label verir.
    """
    if not msg:
        return "out_of_scope", "Boş mesaj."

    try:
        data = _domain_llm_json(msg)
    except Exception:
        # LLM gate çökerse güvenli tarafta kal
        return "out_of_scope", "Sınıflandırma yapılamadı (fallback)."

    dom = (data.get("domain") or "").strip().lower()
    reason = (data.get("reason") or "").strip()

    if dom not in {"in_scope", "out_of_scope"}:
        dom = "out_of_scope"
    if not reason:
        reason = "Belirsiz."

    return dom, reason


def classify_domain_tr(msg: str) -> Tuple[str, str]:
    """
    Hybrid router:
    1) Rule-based kesin durumlar (hızlı, deterministik)
    2) Gri bölgede LLM gate (sadece label)
    """
    t = (msg or "").strip().lower()
    if not t:
        return "out_of_scope", "Boş mesaj."

    # 1) Konu (mental topic) var mı?
    topic_patterns = [
        r"\bkayg(ı|i)\b", r"\banksiyet(e|i)\b", r"\bpanik\b", r"\bstres\b",
        r"\bdepres(yon)?\b", r"\bumutsuz\w*\b", r"\bmutsuz\w*\b",
        r"\btravma\b", r"\bokb\b", r"\bocd\b",
        r"\buyku\b", r"\buykusuz\w*\b", r"\buyuyam\w*\b", r"\binsomnia\b",
        r"\bduygu\b", r"\bhis(sediyorum|setmiyorum|)\b", r"\biyi hissediyorum\b",
        r"\bmutlu(yum|)\b", r"\bkeyif(liyim|sizim|)\b", r"\bmoralim\b",
        r"\bterapi\b", r"\bpsikolog\b", r"\bpsikiyatri\b", r"\bdanışman\b", r"\bdanisman\b",
    ]
    has_topic = any(re.search(p, t) for p in topic_patterns)
    if not has_topic:
        return "out_of_scope", "Mental sağlık/duygu/uyku/stres teması yok."

    # 2) Kullanıcı kendisiyle ilgili destek niyeti var mı?
    self_markers = [
        "ben", "bende", "bana", "beni", "benim", "kendimi", "kendime",
        "hissediyorum", "hissetmiyorum", "yaşıyorum", "yasiyorum",
        "zorlanıyorum", "zorlaniyorum", "baş edemiyorum", "bas edemiyorum",
        "ne yapabilirim", "ne yapmalıyım", "ne yapmaliyim",
        "nasıl başa çıkarım", "nasil basa cikarim",
        "yardım", "yardim",
    ]
    is_self = any(m in t for m in self_markers)

    # 3) Bariz akademik/iş/teknik/yaratıcı görev intent (kesin out_of_scope)
    info_intent = [
        "nedir", "tanım", "tanim", "belirtileri", "belirtiler", "semptom",
        "teşhis", "teshis", "prosedür", "prosedur",
        "tedavi", "ilaç", "ilac", "doz",
        "maaş", "maas", "fiyat", "kaç tl", "kac tl", "ne kadar",
        "kaç yıl", "kac yil",
    ]
    creative_intent = [
        "şiir", "siir", "destan", "roman", "hikaye", "öykü", "oyku",
        "senaryo", "kompozisyon", "metin yaz", "paragraf yaz",
    ]
    coding_intent = [
        "kod", "code", "python", "java", "javascript", "html", "css", "sql",
        "fonksiyon", "algoritma", "bug", "hata", "debug",
    ]
    how_to_intent = [
        "nasıl yapılır", "nasil yapilir", "yapımı", "yapimi", "tarif", "tutorial",
    ]

    looks_task = (
        any(x in t for x in info_intent)
        or any(x in t for x in creative_intent)
        or any(x in t for x in coding_intent)
        or any(x in t for x in how_to_intent)
    )

    # ✅ Bariz görev/bilgi isteği -> out_of_scope (LLM'e bile gitme)
    if looks_task and not is_self:
        return "out_of_scope", "Konu geçse de mesaj bilgi/iş isteği gibi görünüyor."

    # ✅ Kişisel destek isteği -> in_scope
    if is_self and not looks_task:
        return "in_scope", "Kullanıcının kendi iyi oluş/duygu durumu ile ilgili destek isteği algılandı."

    # 4) Gri bölge (ör: tek kelime 'kaygı', ya da karışık ifadeler)
    # Burada LLM gate devreye girer.
    return classify_domain_llm_gate_tr(msg)


# ------------------------------------------------------------
# RISK CLASSIFIER
# ------------------------------------------------------------

def classify_risk_tr(user_msg: str) -> str:
    """
    low | medium | high
    Bu classifier safety.py'de opsiyonel kullanılıyor.
    Asla low demek için "bilgi sorusu" gibi net bir durum olmalı.
    """
    msg = (user_msg or "").strip()
    if not msg:
        return "low"

    client = _get_client()

    sys = (
        "Görev: Kullanıcı mesajındaki kendine zarar / intihar riski seviyesini sınıflandır.\n"
        "Etiketler:\n"
        "- high: açık niyet/plan/eylem (örn: 'intihar edeceğim', 'kendimi öldüreceğim', 'kendime zarar vereceğim').\n"
        "- medium: yoğun umutsuzluk, kendine zarar düşüncesi ima, 'yaşamak istemiyorum' gibi.\n"
        "- low: bilgi/akademik soru (örn: 'intihar nedir?') veya genel sohbet.\n"
        "Sadece JSON döndür: {\"risk\":\"low|medium|high\"}\n"
        "Not: Emin değilsen low verme; medium seç.\n"
    )
    user = f"Mesaj: {msg}"

    r = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )

    content = (r.choices[0].message.content or "").strip()
    data = _safe_json_loads(content)
    risk = (data.get("risk") or "").strip().lower()

    if risk in {"low", "medium", "high"}:
        return risk
    return "medium"


# ------------------------------------------------------------
# ANSWER GENERATION (RAG)
# ------------------------------------------------------------

def generate_answer_tr(
    user_msg: str,
    sources: List[Dict[str, Any]],
    emotion: str = "neutral",
    risk_level: str = "low",
    memory_summary: str = "",
    profile_context: str = "",
) -> str:
    """
    Türkçe cevap üretir (RAG + safety + profile prompt).
    Not: Teşhis/ilaç/doz vermez.
    """

    msg = (user_msg or "").strip()
    if not msg:
        return "Bir mesaj yazabilir misin?"

    # Kaynakları kısa metne çevir (LLM'e verilecek context)
    src_lines: List[str] = []
    for i, s in enumerate(sources or [], start=1):
        focus = s.get("focus_area") or ""
        q = s.get("question") or ""
        a = s.get("answer") or ""
        why = s.get("why") or ""
        score = s.get("score")
        score_txt = f"{score:.2f}" if isinstance(score, (int, float)) else ""
        src_lines.append(
            f"[{i}] focus: {focus} | score: {score_txt}\n"
            f"Q: {q}\n"
            f"A: {a}\n"
            f"WHY: {why}".strip()
        )

    rag_context = "\n\n".join(src_lines).strip()
    if not rag_context:
        rag_context = "(Kaynak bulunamadı)"

    profile_block = ""
    if profile_context:
        profile_block = (
            "\n\nAşağıda kullanıcı hakkında uzun vadeli özet bağlam bulunmaktadır.\n"
            "Bu bilgiyi cevabının TONUNU ve YAKLAŞIMINI ayarlamak için kullan.\n"
            "Bu bilgiyi kullanıcıya aynen tekrar etme veya görünür şekilde 'profil' diye bahsetme.\n\n"
            f"{profile_context}\n"
        )

    memory_block = ""
    if memory_summary:
        memory_block = (
            "\n\nKISA KONUŞMA ÖZETİ (son mesajlardan):\n"
            f"{memory_summary}\n"
        )

    system_prompt = f"""
Sen güvenli, destekleyici bir mental sağlık & genel iyi oluş asistanısın.
Dil: Türkçe.

KURALLAR:
- Teşhis koyma. (örn: "sende X var" deme)
- İlaç/Doz önerme. Reçeteleme yapma.
- Kendine zarar / intihar riski yüksekse: kısa, güvenli, acil yardım yönlendirmesi yap.
- Yargılayıcı olma, sakin ve net ol.
- Kullanıcı sorusu mental-health dışındaysa: kısa şekilde kapsam dışı olduğunu söyle ve yönlendir.
- Kaynakları "adı / focus" olarak kullan; birebir alıntı yapman şart değil.

RİSK: {risk_level}
DUYGU: {emotion}
{profile_block}
{memory_block}

RAG KAYNAKLARI:
{rag_context}

ÇIKTI FORMATIN:
1) Kısa Özet (2-3 madde)
2) Ne yapabilirsin? (3-6 madde, pratik)
3) Ne zaman uzmana başvurmalısın? (2-4 madde)
""".strip()

    client = _get_client()

    r = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": msg},
        ],
        temperature=0.4,
    )

    answer = (r.choices[0].message.content or "").strip()
    if not answer:
        return "Şu an teknik bir sorun oluştu. Lütfen tekrar dener misin?"

    # ✅ FORMAT KONTROL
    need_headers = ["Kısa Özet", "Ne yapabilirsin?", "Ne zaman uzmana başvurmalısın?"]
    missing = any(h.lower() not in answer.lower() for h in need_headers)

    if missing:
        reform_sys = """
Sen bir editörsün. Görev: Verilen metni SADECE yeniden biçimlendir.
Yeni bilgi ekleme, anlamı değiştirme, tıbbi öneri/ilaç/doz ekleme.
ÇIKTI formatı şu 3 başlıkla ve madde madde olacak:

1. Kısa Özet:
- ...
- ...

2. Ne yapabilirsin?
- ...
- ...
- ...

3. Ne zaman uzmana başvurmalısın?
- ...
- ...
""".strip()

        rr = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": reform_sys},
                {"role": "user", "content": f"Metin:\n{answer}"},
            ],
            temperature=0,
        )
        reformatted = (rr.choices[0].message.content or "").strip()
        if reformatted:
            return reformatted

    return answer

def scope_llm_gate_tr(message_tr: str) -> str:
    """
    Borderline durumlarda in_scope/out_of_scope kararı için LLM gate.
    Sadece 'IN' veya 'OUT' döndürür.
    """
    t = (message_tr or "").strip()
    if not t:
        return "OUT"

    # OpenAI yoksa güvenli tarafta kal
    if OpenAI is None:
        return "OUT"

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict scope classifier for a Turkish mental-health support assistant.\n"
                        "Decide if the user message is IN scope (mental well-being support: stress, anxiety, panic, "
                        "mood, burnout, relationships/communication, self-criticism, rumination, sleep, coping) "
                        "or OUT of scope (weather, tech support, finance/investing, gaming, food recipes, legal advice, "
                        "medical diagnosis/treatment, medication/dosage, converting files like PDF->Word, etc.).\n"
                        "Return ONLY one token: IN or OUT.\n"
                        "Rules:\n"
                        "- If the user asks for medication, dosage, 'hangi ilaç', treatment advice for physical symptoms -> OUT.\n"
                        "- If it's about emotions, coping, anxiety symptoms, relationships/communication -> IN.\n"
                        "- If unsure, return OUT."
                    ),
                },
                {"role": "user", "content": t},
            ],
        )
        out = (resp.choices[0].message.content or "").strip().upper()
        if out.startswith("IN"):
            return "IN"
        if out.startswith("OUT"):
            return "OUT"
        return "OUT"
    except Exception:
        return "OUT"


def scope_gate_llm(prompt: str) -> str:

    if OpenAI is None:
        raise RuntimeError("openai paketi bulunamadı. `pip install openai` kur.")
    client = _get_client()
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "Return ONLY IN or OUT."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=5,
    )
    return (resp.choices[0].message.content or "").strip()


def scope_llm_gate_tr_v2(message_tr: str) -> str:
    return scope_llm_gate_tr(message_tr)
