# backend/memory.py

from typing import List, Dict, Any, Tuple


def _last_user_messages(history: List[Dict[str, Any]], n: int = 3) -> List[str]:

    users = [t.get("content", "") for t in (history or []) if t.get("role") == "user"]
    return [u for u in users if u][-n:]


def build_memory_summary_tr(history: List[Dict[str, Any]], current_msg: str) -> str:
    """
    LLM'e verilecek kısa geçmiş özeti (çok kısa, güvenli).
    """
    last3 = _last_user_messages(history, n=3)
    if not last3:
        return ""



    joined = " | ".join(s.strip()[:120] for s in last3 if s.strip())
    return f"Son kullanıcı mesajları (özet): {joined}"


def risk_escalation_from_trend(
    last_risks: List[str],
    current_risk: str
) -> Tuple[str, bool]:
    """
    Koruma belleği:
    - current low ise ama son 3 içinde 2+ medium varsa -> medium’a yükselt
    - current already medium/high ise dokunma
    Returns: (new_risk, escalated_bool)
    """
    if current_risk in ("medium", "high"):
        return current_risk, False

    if not last_risks:
        return current_risk, False

    medium_count = sum(1 for r in last_risks if r == "medium")
    high_count = sum(1 for r in last_risks if r == "high")

    # geçmişte high göründüyse, current low olsa bile temkinli ol:

    if high_count >= 1:
        return "high", True

    if medium_count >= 2:
        return "medium", True

    return current_risk, False