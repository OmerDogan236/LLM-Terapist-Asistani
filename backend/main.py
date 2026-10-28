# backend/main.py
from __future__ import annotations

import json
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from backend.rag import RAGStore
from backend.llm_client import translate_tr_to_en_scope, generate_answer_tr
from backend.scope_rules import decide_domain_tr
from backend.safety import assess_risk, crisis_message_tr, gate_high_risk
from backend.risk import analyze_emotion, normalize_emotion
from backend.memory import build_memory_summary_tr, risk_escalation_from_trend
from backend.query_rewrite import rewrite_query_tr_to_en
from backend.source_why import explain_sources_tr
from backend.user_profile import update_user_profile, get_user_profile

app = FastAPI(title="RAG Therapist Backend")

# Load indexes once
store = RAGStore()

OOS_MESSAGE_TR = (
    "Üzgünüm, bu isteğe yanıt veremiyorum.\n\n"
    "Bu asistan yalnızca mental sağlık ve iyi oluş (stres, kaygı, duygu yönetimi, "
    "uyku düzeni, ilişki/iletişim, öz-şefkat gibi) konularında destek vermek için tasarlandı."
)


class ChatRequest(BaseModel):
    message: str
    k: int = 5
    history: List[Dict[str, Any]] = []


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    emotion: Optional[str] = None
    risk_level: Optional[str] = None
    memory_used: bool = False
    query_en: Optional[str] = None
    domain: Optional[str] = None
    why: Optional[Dict[str, Any]] = None
    debug: Optional[str] = None  # MUST be string (FastAPI response model)


def _mental_top_score(query_en: str) -> float:

    try:
        hits = store._faiss_search(query_en, top_n=1, which="mental")  # type: ignore[attr-defined]
        if not hits:
            return 0.0
        s = hits[0].get("score", 0.0)
        return float(s) if s is not None else 0.0
    except Exception:
        return 0.0


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    msg = (req.message or "").strip()
    k = int(req.k or 5)

    try:
        # 0) Pre-risk (fast)
        pre_risk = assess_risk(msg)

        # 1) Scope score (TR->EN short translate) then decide domain
        query_en_scope = translate_tr_to_en_scope(msg)
        score = _mental_top_score(query_en_scope)

        domain, reason, scope_dbg = decide_domain_tr(
            msg_tr=msg,
            pre_risk=pre_risk,
            score=score,
            use_llm_gate=True,
        )

        why: Dict[str, Any] = {
            "scope": {
                "domain": domain,
                "reason": reason,
                "score": score,
                "en": query_en_scope,
            }
        }

        # OUT OF SCOPE -> immediate return
        if domain == "out_of_scope":
            return ChatResponse(
                answer=OOS_MESSAGE_TR,
                sources=[],
                emotion=None,
                risk_level=None,
                memory_used=False,
                query_en=None,
                domain=domain,
                why=why,
                debug=json.dumps({"scope": scope_dbg}, ensure_ascii=False),
            )

        # 2) Safety: if current message is high -> bypass everything
        if pre_risk == "high":
            why["risk"] = {"pre": "high", "final": "high", "trend_escalated": False, "gated": False}
            why["emotion"] = {"label": "critical", "reason": "risk=high", "hit": None}
            return ChatResponse(
                answer=crisis_message_tr(),
                sources=[],
                emotion="critical",
                risk_level="high",
                memory_used=False,
                query_en=None,
                domain=domain,
                why=why,
                debug=json.dumps({"risk": "high"}, ensure_ascii=False),
            )

        # 3) Memory summary + trend escalation
        history = req.history or []
        mem_summary = build_memory_summary_tr(history, msg)

        last_risks: List[str] = []
        for t in history:
            if t.get("role") == "assistant":
                meta = t.get("meta") or {}
                rl = meta.get("risk_level") or meta.get("risk")
                if isinstance(rl, str):
                    last_risks.append(rl)

        final_risk, escalated = risk_escalation_from_trend(last_risks[-3:], pre_risk)

        # Gate: prevent trend-only 'high' from sticking
        gated_risk = gate_high_risk(msg, final_risk)
        gated = (gated_risk != final_risk)
        final_risk = gated_risk

        why["risk"] = {"pre": pre_risk, "final": final_risk, "trend_escalated": escalated, "gated": gated}
        why["memory"] = {"used": bool(mem_summary), "turns_considered": min(3, len([t for t in history if t.get("role") == "user"]))}

        # If after memory+gate we are still high, treat as crisis (only if current msg matches high patterns)
        if final_risk == "high":
            why["emotion"] = {"label": "critical", "reason": "risk=high_after_gate", "hit": None}
            return ChatResponse(
                answer=crisis_message_tr(),
                sources=[],
                emotion="critical",
                risk_level="high",
                memory_used=bool(mem_summary),
                query_en=None,
                domain=domain,
                why=why,
                debug=json.dumps({"risk": "high_after_gate"}, ensure_ascii=False),
            )

        # 4) Emotion
        emo_label, emo_reason, emo_hit = analyze_emotion(
            msg,
            scope_reason=reason,
            risk_level=final_risk,
        )
        emo_label = normalize_emotion(emo_label, final_risk)
        why["emotion"] = {"label": emo_label, "reason": emo_reason, "hit": emo_hit}

        # 5) RAG
        query_en = rewrite_query_tr_to_en(msg)
        hits = store.search(query_en, k=k)

        # explainability per source
        hits = explain_sources_tr(msg, hits)
        why["rag"] = {"k": k, "returned": len(hits), "used_rewrite": True}

        # 6) User profile (minimal)
        user_id = "demo_user"
        update_user_profile(user_id, msg, detected_topics=[], risk_level=final_risk)
        profile_ctx = get_user_profile(user_id)

        # 7) LLM answer
        answer = generate_answer_tr(
            user_msg=msg,
            sources=hits,
            emotion=emo_label,
            risk_level=final_risk,
            memory_summary=mem_summary,
            profile_context=profile_ctx,
        )

        return ChatResponse(
            answer=answer,
            sources=hits,
            emotion=emo_label,
            risk_level=final_risk,
            memory_used=bool(mem_summary),
            query_en=query_en,
            domain=domain,
            why=why,
            debug=json.dumps({"scope": scope_dbg, "score": score}, ensure_ascii=False),
        )

    except Exception as e:
        print("BACKEND_EXCEPTION:", repr(e))
        traceback.print_exc()
        return ChatResponse(
            answer="Şu an teknik bir sorun oluştu. Lütfen biraz sonra tekrar dene.",
            sources=[],
            emotion=None,
            risk_level=None,
            memory_used=False,
            query_en=None,
            domain=None,
            why={"error": repr(e)},
            debug=str(e),
        )
