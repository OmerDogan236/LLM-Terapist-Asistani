# backend/user_profile.py
from typing import Dict, List
from datetime import datetime
from collections import Counter

# Bellek RAM içi
_USER_PROFILE_STORE: Dict[str, Dict] = {}


def _init_profile() -> Dict:
    return {
        "preferred_language": "tr",
        "frequent_topics": [],
        "risk_history": [],
        "message_count": 0,
        "last_updated": None,
    }


def update_user_profile(
    user_id: str,
    user_message: str,
    detected_topics: List[str],
    risk_level: str,
):
    """
    Ham mesaj saklanmaz.
    Sadece özet davranış bilgisi tutulur.
    High-risk içerik KALICI YAZILMAZ.
    """
    if user_id not in _USER_PROFILE_STORE:
        _USER_PROFILE_STORE[user_id] = _init_profile()

    profile = _USER_PROFILE_STORE[user_id]

    # mesaj sayısı
    profile["message_count"] += 1

    # risk geçmişi (high KALICI YAZILMAZ)
    if risk_level in {"low", "medium"}:
        profile["risk_history"].append(risk_level)
        profile["risk_history"] = profile["risk_history"][-10:]

    # konu sıklığı (etiket bazlı)
    if detected_topics:
        all_topics = profile["frequent_topics"] + detected_topics
        counter = Counter(all_topics)
        profile["frequent_topics"] = [
            k for k, _ in counter.most_common(5)
        ]

    profile["last_updated"] = datetime.utcnow().isoformat()


def get_user_profile(user_id: str) -> Dict:
    return _USER_PROFILE_STORE.get(user_id, _init_profile())
