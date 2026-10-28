# ui/app.py
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import requests
import streamlit as st

# Optional: OpenAI TTS (uses same OPENAI_API_KEY as LLM)
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

from pathlib import Path
import uuid
import pandas as pd
import hashlib
import os
from dotenv import load_dotenv


def _load_env_from_root():
    """.env yükleyici: proje kök dizinini (rag_therapist) ve üst dizinleri arar."""
    try:
        here = Path(__file__).resolve().parent
        for p in [here, here.parent, here.parent.parent]:
            env_path = p / '.env'
            if env_path.exists():
                load_dotenv(env_path, override=False)
                break
        load_dotenv(override=False)  # fallback
    except Exception:
        pass


_load_env_from_root()
import base64

API_URL = "http://127.0.0.1:8000/chat"


def _project_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


ROOT = _project_root()
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

USERS_PATH = DATA_DIR / "users.json"
DIARY_PATH = DATA_DIR / "diary.json"


def _load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _save_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash_password(password: str, salt_b64: str | None = None) -> dict:
    if salt_b64 is None:
        salt = uuid.uuid4().bytes
        salt_b64 = base64.b64encode(salt).decode("utf-8")
    else:
        salt = base64.b64decode(salt_b64.encode("utf-8"))
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return {
        "salt_b64": salt_b64,
        "hash_b64": base64.b64encode(dk).decode("utf-8"),
        "algo": "pbkdf2_hmac_sha256",
        "iters": 200_000,
    }


def _verify_password(password: str, stored: dict) -> bool:
    try:
        salt_b64 = stored.get("salt_b64")
        target = stored.get("hash_b64")
        recalced = _hash_password(password, salt_b64=salt_b64)["hash_b64"]
        return recalced == target
    except Exception:
        return False


def _load_users() -> dict:
    return _load_json(USERS_PATH, {})


def _save_users(users: dict):
    _save_json(USERS_PATH, users)


def _load_diary() -> dict:
    return _load_json(DIARY_PATH, {})


def _save_diary(d: dict):
    _save_json(DIARY_PATH, d)


def _parse_entry_dt(val):
    """Parses diary entry datetime (dd.mm.yyyy HH:MM:SS or ISO) -> datetime (naive, local)."""
    if not val:
        return None
    if isinstance(val, str):
        s = val.strip()
        # dd.mm.yyyy HH:MM:SS
        try:
            return datetime.strptime(s, "%d.%m.%Y %H:%M:%S")
        except Exception:
            pass
        # ISO
        try:
            s2 = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s2)
            if dt.tzinfo is not None:
                dt = dt.astimezone().replace(tzinfo=None)
            return dt.replace(microsecond=0)
        except Exception:
            return None
    return None


def _format_any_datetime(val) -> str:
    """Accepts dd.mm.yyyy HH:MM:SS or ISO timestamps; returns dd.mm.yyyy HH:MM:SS."""
    if not val:
        return "-"
    if isinstance(val, str):
        s = val.strip()
        # Already in desired format?
        try:
            datetime.strptime(s, "%d.%m.%Y %H:%M:%S")
            return s
        except Exception:
            pass
        # Try ISO formats
        try:
            s2 = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s2)
            # If timezone-aware, convert to local time then drop tzinfo
            if dt.tzinfo is not None:
                dt = dt.astimezone().replace(tzinfo=None)
            return dt.replace(microsecond=0).strftime("%d.%m.%Y %H:%M:%S")
        except Exception:
            return s
    return str(val)


def _gender_icon(gender: str) -> str:
    g = (gender or "").lower()
    if g == "female":
        return "👩"
    if g == "male":
        return "👨"
    return "👤"


def _gender_select(label: str, current: str) -> str:
    g_map = {"Erkek": "male", "Kadın": "female", "Diğer": "other"}
    keys = list(g_map.keys())
    vals = list(g_map.values())
    idx = vals.index(current) if current in vals else 0
    sel = st.selectbox(label, keys, index=idx)
    return g_map.get(sel, "male")


def _ensure_profile_shape(profile: dict) -> dict:
    p = (profile or {}).copy()
    if "first_name" not in p and "last_name" not in p:
        full = (p.get("full_name") or "").strip()
        if full:
            parts = full.split()
            if len(parts) == 1:
                p["first_name"] = parts[0]
                p["last_name"] = ""
            else:
                p["first_name"] = parts[0]
                p["last_name"] = " ".join(parts[1:])
        else:
            p["first_name"] = ""
            p["last_name"] = ""
    p.pop("full_name", None)
    p.setdefault("age", None)
    p.setdefault("height_cm", None)
    p.setdefault("weight_kg", None)
    p.setdefault("gender", "male")
    return p


def _register_user(username: str, password: str) -> tuple[bool, str]:
    username = (username or "").strip().lower()
    if len(username) < 3:
        return False, "Kullanıcı adı en az 3 karakter olmalı."
    if len(password) < 6:
        return False, "Şifre en az 6 karakter olmalı."

    users = _load_users()
    if username in users:
        return False, "Bu kullanıcı adı zaten var."

    users[username] = {
        "user_id": str(uuid.uuid4()),
        "pass": _hash_password(password),
        "profile": _ensure_profile_shape({
            "first_name": "",
            "last_name": "",
            "age": None,
            "height_cm": None,
            "weight_kg": None,
            "gender": "male",
        }),
        "created_at": datetime.now(timezone.utc).isoformat(),
        # login times
        "last_login_local": None,  # most recent successful login (string)
        "prev_login_local": None,  # previous login before last (string)
        "last_logout_local": None,  # last time user clicked logout (string)
        "last_session_seconds": None,  # session duration in seconds (int)
    }
    _save_users(users)
    return True, "Kayıt başarılı. Şimdi profil bilgilerini doldur."


def _update_login_times(username: str) -> tuple[str, str | None]:
    """
    Returns (current_login, previous_login)
    Persisted to users.json under:
      - last_login_local (current)
      - prev_login_local (previous)
    """
    username = (username or "").strip().lower()
    users = _load_users()
    u = users.get(username)
    now_str = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    if not u:
        return now_str, None

    prev = u.get("last_login_local")  # shift current to previous
    u["prev_login_local"] = prev
    u["last_login_local"] = now_str
    users[username] = u
    _save_users(users)
    return now_str, prev


def _update_logout_time(username: str) -> str:
    username = (username or "").strip().lower()
    users = _load_users()
    u = users.get(username)
    now_dt = datetime.now()
    now_str = now_dt.strftime("%d.%m.%Y %H:%M:%S")
    if not u:
        return now_str

    # Persist logout time
    u["last_logout_local"] = now_str

    # Compute session duration from last_login_local if possible
    try:
        login_str = u.get("last_login_local")
        if login_str:
            login_dt = datetime.strptime(login_str, "%d.%m.%Y %H:%M:%S")
            dur = int((now_dt - login_dt).total_seconds())
            if dur < 0:
                dur = 0
            u["last_session_seconds"] = dur
    except Exception:
        # keep as-is if parsing fails
        pass

    users[username] = u
    _save_users(users)
    return now_str


def _login_user(username: str, password: str) -> tuple[bool, str, dict | None]:
    username = (username or "").strip().lower()
    users = _load_users()
    u = users.get(username)
    if not u:
        return False, "Kullanıcı bulunamadı.", None
    if not _verify_password(password, u.get("pass") or {}):
        return False, "Şifre yanlış.", None

    prof = _ensure_profile_shape(u.get("profile") or {})
    u["profile"] = prof
    users[username] = u
    _save_users(users)

    return True, "Giriş başarılı.", {
        "username": username,
        "user_id": u["user_id"],
        "gender": (prof.get("gender") or "male"),
    }


def _verify_user(username: str, password: str) -> bool:
    username = (username or "").strip().lower()
    users = _load_users()
    u = users.get(username)
    if not u:
        return False
    return _verify_password(password, u.get("pass") or {})


def _change_password(username: str, old_password: str, new_password: str) -> tuple[bool, str]:
    username = (username or "").strip().lower()
    if len(new_password) < 6:
        return False, "Yeni şifre en az 6 karakter olmalı."
    users = _load_users()
    u = users.get(username)
    if not u:
        return False, "Kullanıcı bulunamadı."
    if not _verify_password(old_password, u.get("pass") or {}):
        return False, "Eski şifre yanlış."
    u["pass"] = _hash_password(new_password)
    users[username] = u
    _save_users(users)
    return True, "Şifre güncellendi ✅"


def _change_username(old_username: str, new_username: str, password: str) -> tuple[bool, str]:
    old_username = (old_username or "").strip().lower()
    new_username = (new_username or "").strip().lower()

    if len(new_username) < 3:
        return False, "Yeni kullanıcı adı en az 3 karakter olmalı."
    if old_username == new_username:
        return False, "Farklı kullanıcı adı girin."

    users = _load_users()
    if old_username not in users:
        return False, "Eski kullanıcı adı bulunamadı."
    if new_username in users:
        return False, "Farklı kullanıcı adı girin."

    u = users.get(old_username)
    if not _verify_password(password, (u or {}).get("pass") or {}):
        return False, "Şifre yanlış."

    users[new_username] = u
    del users[old_username]
    _save_users(users)

    if st.session_state.get("auth", {}).get("username") == old_username:
        st.session_state.auth["username"] = new_username

    return True, "Kullanıcı adı güncellendi ✅"


def _get_profile(username: str) -> dict:
    username = (username or "").strip().lower()
    users = _load_users()
    u = users.get(username) or {}
    return _ensure_profile_shape(u.get("profile") or {})


def _set_profile(username: str, profile: dict) -> None:
    username = (username or "").strip().lower()
    users = _load_users()
    if username in users:
        users[username]["profile"] = _ensure_profile_shape(profile)
        _save_users(users)
        if st.session_state.get("auth", {}).get("username") == username:
            st.session_state.auth["gender"] = users[username]["profile"].get("gender") or "male"


def _delete_user(username: str, password: str) -> tuple[bool, str]:
    username = (username or "").strip().lower()
    users = _load_users()
    u = users.get(username)
    if not u:
        return False, "Kullanıcı bulunamadı."
    if not _verify_password(password, u.get("pass") or {}):
        return False, "Şifre yanlış."

    user_id = (u or {}).get("user_id") if isinstance(u, dict) else None

    del users[username]
    _save_users(users)

    # Diary is stored per-username; also try legacy per-user_id cleanup
    diary = _load_diary()
    if username in diary:
        del diary[username]
    if user_id and user_id in diary:
        del diary[user_id]
    _save_diary(diary)

    if st.session_state.get("auth", {}).get("username") == username:
        st.session_state.auth = {"is_authed": False, "username": None, "user_id": None, "gender": "male"}

    return True, "Hesabınız silindi ✅"


st.set_page_config(page_title="RAG Terapist Asistanı (MVP)", layout="centered")

st.markdown(
    """
    <style>
      .block-container { padding-top: 3.2rem; padding-bottom: 1.5rem; }
      section[data-testid="stSidebar"] > div { padding-top: 0.6rem; }
      div[data-testid="stVerticalBlock"] { gap: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("RAG Tabanlı Güvenli Terapötik LLM Asistanı")

if "auth" not in st.session_state:
    st.session_state.auth = {"is_authed": False, "username": None, "user_id": None, "gender": "male"}

if "pending_profile_setup" not in st.session_state:
    st.session_state.pending_profile_setup = None

if "login_flash" not in st.session_state:
    st.session_state.login_flash = ""

if "login_flash_kind" not in st.session_state:
    st.session_state.login_flash_kind = "success"

if "diary_last_saved_by_user" not in st.session_state:
    st.session_state.diary_last_saved_by_user = {}  # username -> last user message saved

if "diary_view" not in st.session_state:
    st.session_state.diary_view = "none"  # none | records | analytics

if "profile_saved_flash_for" not in st.session_state:
    st.session_state.profile_saved_flash_for = ""  # e.g. "profile_edit"

# login time state (for sidebar display)
if "current_login_time" not in st.session_state:
    st.session_state.current_login_time = ""
if "previous_login_time" not in st.session_state:
    st.session_state.previous_login_time = ""

LOGIN_KEYS = [
    "login_user", "login_pass",
    "reg_user", "reg_pass",
    "pw_user", "pw_old", "pw_new",
    "uc_old", "uc_new", "uc_pw",
    "pe_user", "pe_pass",
    "du_user", "du_pass", "du_confirm",
]


def _clear_login_inputs():
    for k in LOGIN_KEYS:
        if k in st.session_state:
            if isinstance(st.session_state[k], bool):
                st.session_state[k] = False
            else:
                st.session_state[k] = ""


def _on_section_change():
    _clear_login_inputs()
    st.session_state.pop("profile_edit_user", None)
    st.session_state.profile_saved_flash_for = ""


SECTION_META = {
    "Giriş": ("Giriş", "🔑"),
    "Kayıt Ol": ("Kayıt Ol", "📝"),
    "Şifre Değiştir": ("Şifreyi Değiştir", "🔒"),
    "Kullanıcı Adı Değiştir": ("Kullanıcı Adı Değiştir", "🆔"),
    "Profil Düzenle": ("Profil Düzenle", "👤"),
    "Hesap Sil": ("Hesabı Sil", "🗑️"),
}


def _set_login_flash(msg: str, kind: str = "success"):
    st.session_state.login_flash = msg
    st.session_state.login_flash_kind = kind


def _show_login_flash():
    msg = (st.session_state.get("login_flash") or "").strip()
    if not msg:
        return
    kind = st.session_state.get("login_flash_kind", "success")
    if kind == "success":
        st.success(msg)
    elif kind == "info":
        st.info(msg)
    else:
        st.warning(msg)
    st.session_state.login_flash = ""


def _parse_int_field(val: str) -> int | None:
    s = (val or "").strip()
    if not s:
        return None
    try:
        return int(float(s.replace(",", ".")))
    except Exception:
        return None


def render_profile_setup(username: str, mode: str = "setup"):
    if mode == "edit":
        st.subheader("Kullanıcı Profili")
    else:
        st.subheader("👤 Kullanıcı Profili")

    prof = _get_profile(username)

    with st.form(f"profile_form_{mode}", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            first_name = st.text_input("İsim", value=prof.get("first_name") or "", key=f"pf_first_{mode}")
        with c2:
            last_name = st.text_input("Soyisim", value=prof.get("last_name") or "", key=f"pf_last_{mode}")

        age_s = st.text_input("Yaş", value=str(prof.get("age") or ""), key=f"pf_age_{mode}", placeholder="Örn: 23")
        height_s = st.text_input("Boy (cm)", value=str(prof.get("height_cm") or ""), key=f"pf_h_{mode}",
                                 placeholder="Örn: 178")
        weight_s = st.text_input("Kilo (kg)", value=str(prof.get("weight_kg") or ""), key=f"pf_w_{mode}",
                                 placeholder="Örn: 72")

        gender = _gender_select("Cinsiyet", prof.get("gender") or "male")
        submitted = st.form_submit_button("Profili Kaydet", width='stretch')

    if submitted:
        age = _parse_int_field(age_s)
        height = _parse_int_field(height_s)
        weight = _parse_int_field(weight_s)

        if age_s.strip() and age is None:
            st.error("Yaş sayısal olmalı.")
            return
        if height_s.strip() and height is None:
            st.error("Boy (cm) sayısal olmalı.")
            return
        if weight_s.strip() and weight is None:
            st.error("Kilo (kg) sayısal olmalı.")
            return

        new_prof = {
            "first_name": (first_name or "").strip(),
            "last_name": (last_name or "").strip(),
            "age": age,
            "height_cm": height,
            "weight_kg": weight,
            "gender": gender,
        }
        _set_profile(username, new_prof)

        if mode == "setup":
            st.success("Profiliniz kaydedildi ✅")
            st.info("Ana sayfaya dönülüyor.")
            time.sleep(3)
            st.session_state.pending_profile_setup = None
            st.session_state.login_section = "Giriş"
            st.rerun()
        else:
            st.session_state.pop("profile_edit_user", None)
            st.session_state.profile_saved_flash_for = "profile_edit"
            st.rerun()


def _render_saved_users_list():
    """Ana ekranda kayıtlı kullanıcıları listeler."""
    st.markdown("---")
    st.markdown("#### 👥 Kayıtlı Kullanıcılar")
    users = _load_users()
    if not users:
        st.caption("Henüz kayıtlı kullanıcı yok.")
        return

    for uname in sorted(users.keys()):
        urec = users.get(uname) or {}
        prof = _ensure_profile_shape((urec.get("profile") or {}))
        icon = _gender_icon(prof.get("gender") or "male")

        created_at = _format_any_datetime(urec.get("created_at_local") or urec.get("created_at"))
        last_logout = _format_any_datetime(urec.get("last_logout_local"))

        sess = urec.get("last_session_seconds")
        dur_txt = "-"
        if isinstance(sess, int):
            mm = sess // 60
            ss = sess % 60
            if mm > 0:
                dur_txt = f"{mm} dk {ss} sn"
            else:
                dur_txt = f"{ss} sn"

        # Tek kolon: buton + altına bilgiler
        with st.container():
            if st.button(f"{icon} {uname}", key=f"pick_{uname}", width='stretch'):
                # Context-aware prefill based on current section
                sec = st.session_state.get("login_section", "Giriş")
                if sec == "Giriş":
                    st.session_state._prefill_field = "login_user"
                    st.session_state._focus_login_top = True
                elif sec == "Şifre Değiştir":
                    st.session_state._prefill_field = "pw_user"
                elif sec == "Kullanıcı Adı Değiştir":
                    st.session_state._prefill_field = "uc_old"
                elif sec == "Profil Düzenle":
                    st.session_state._prefill_field = "pe_user"
                elif sec == "Hesap Sil":
                    st.session_state._prefill_field = "du_user"
                else:
                    st.session_state._prefill_field = "login_user"

                st.session_state._prefill_value = uname
                st.rerun()

            st.caption(f"Oluşturulma Zamanı: {created_at}")
            st.caption(f"Son Çıkış Zamanı: {last_logout}")
            st.caption(f"İşlem Süresi: {dur_txt}")


def render_login():
    # Apply deferred navigation/prefill BEFORE the login_section widget is created
    if st.session_state.get("_nav_login_section"):
        st.session_state.login_section = st.session_state.pop("_nav_login_section")
    if st.session_state.get("_prefill_login_user") is not None:
        st.session_state["login_user"] = st.session_state.pop("_prefill_login_user")
        st.session_state["login_pass"] = ""

    # Apply deferred field prefill (must happen BEFORE widgets instantiate)
    if st.session_state.get("_prefill_field") and st.session_state.get("_prefill_value") is not None:
        try:
            st.session_state[st.session_state["_prefill_field"]] = st.session_state["_prefill_value"]
        except Exception:
            pass
        st.session_state._prefill_field = None
        st.session_state._prefill_value = None
    if "login_section" not in st.session_state:
        st.session_state.login_section = "Giriş"

    section = st.radio(
        "Menü",
        ["Giriş", "Kayıt Ol", "Şifre Değiştir", "Kullanıcı Adı Değiştir", "Profil Düzenle", "Hesap Sil"],
        key="login_section",
        horizontal=True,
        on_change=_on_section_change,
        label_visibility="collapsed",
    )

    title, icon = SECTION_META.get(section, ("Giriş", "🔑"))
    st.subheader(f"{icon} {title}")

    _show_login_flash()

    if section == "Giriş":
        with st.form("login_form", clear_on_submit=True):
            u = st.text_input("Kullanıcı adı", key="login_user")
            p = st.text_input("Şifre", type="password", key="login_pass")
            submitted = st.form_submit_button("Giriş Yap", type="primary", width='stretch')
        if submitted:
            ok, msg, user = _login_user(u, p)
            if ok and user:
                # Update persisted login times:
                cur, prev = _update_login_times(user["username"])
                st.session_state.current_login_time = cur
                st.session_state.previous_login_time = prev or ""

                st.session_state.auth = {"is_authed": True, "username": user["username"], "user_id": user["user_id"],
                                         "gender": user.get("gender", "male")}
                st.success("Başarılı bir şekilde giriş yaptınız.✅")
                st.info("Chat ekranına yönlendiriliyor...")
                time.sleep(3)
                st.rerun()
            else:
                st.error(msg)

    elif section == "Kayıt Ol":
        with st.form("register_form", clear_on_submit=True):
            u = st.text_input("Yeni kullanıcı adı", key="reg_user")
            p = st.text_input("Yeni şifre", type="password", key="reg_pass")
            submitted = st.form_submit_button("Kayıt Ol", width='stretch')
        if submitted:
            ok, msg = _register_user(u, p)
            if ok:
                st.session_state.pending_profile_setup = (u or "").strip().lower()
                st.rerun()
            else:
                st.error(msg)

    elif section == "Şifre Değiştir":
        with st.form("pw_change_form", clear_on_submit=True):
            u = st.text_input("Kullanıcı adı", key="pw_user")
            oldp = st.text_input("Eski şifre", type="password", key="pw_old")
            newp = st.text_input("Yeni şifre", type="password", key="pw_new")
            submitted = st.form_submit_button("Şifreyi Değiştir", width='stretch')

        # Mesajı butonun hemen altında göster
        if submitted:
            ok, msg = _change_password(u, oldp, newp)
            if ok:
                st.session_state.pw_change_flash = "Şifreniz güncellendi ✅"
            else:
                st.error(msg)

        if st.session_state.get("pw_change_flash"):
            st.success(st.session_state.pw_change_flash)
            st.session_state.pw_change_flash = ""

    elif section == "Kullanıcı Adı Değiştir":
        with st.form("username_change_form", clear_on_submit=True):
            old_u = st.text_input("Eski kullanıcı adı", key="uc_old")
            new_u = st.text_input("Yeni kullanıcı adı", key="uc_new")
            pw = st.text_input("Şifre", type="password", key="uc_pw")
            submitted_uc = st.form_submit_button("Kullanıcı Adını Değiştir", width='stretch')
        if submitted_uc:
            ok, msg = _change_username(old_u, new_u, pw)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    elif section == "Profil Düzenle":
        with st.form("profile_edit_auth", clear_on_submit=True):
            u = st.text_input("Kullanıcı adı", key="pe_user")
            p = st.text_input("Şifre", type="password", key="pe_pass")
            submitted = st.form_submit_button("Devam Et", width='stretch')

        if st.session_state.profile_saved_flash_for == "profile_edit":
            st.success("Profiliniz kaydedildi ✅")
            st.session_state.profile_saved_flash_for = ""

        if submitted:
            if _verify_user(u, p):
                st.session_state["profile_edit_user"] = (u or "").strip().lower()
                st.success("Doğrulandı ✅ Aşağıdan profilini güncelle.")
            else:
                st.error("Kullanıcı adı / şifre hatalı.")

        edit_user = st.session_state.get("profile_edit_user")
        if edit_user:
            st.markdown("---")
            render_profile_setup(edit_user, mode="edit")

    elif section == "Hesap Sil":
        st.warning("Bu işlem geri alınamaz.")
        with st.form("delete_user_form", clear_on_submit=True):
            u = st.text_input("Kullanıcı adı", key="du_user")
            p = st.text_input("Şifre", type="password", key="du_pass")
            confirm = st.checkbox("Evet, hesabımı silmek istiyorum.", key="du_confirm")
            submitted = st.form_submit_button("Hesabı Sil", width='stretch')
        if submitted:
            if not confirm:
                st.error("Onay kutusunu işaretle.")
            else:
                ok, msg = _delete_user(u, p)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    # ---------------------------------
    # Ana ekran: Kayıtlı kullanıcılar listesi
    # ---------------------------------

    # Quick login card shown near the saved users list (no scrolling needed)
    if st.session_state.get("_focus_login_top") and section == "Giriş":
        st.info("Seçtiğiniz kullanıcı için şifreyi girip giriş yapabilirsiniz 👇")
        st.session_state._focus_login_top = False
    if section != "Kayıt Ol":
        _render_saved_users_list()


if st.session_state.pending_profile_setup:
    render_profile_setup(st.session_state.pending_profile_setup, mode="setup")
    st.stop()

if not st.session_state.auth.get("is_authed"):
    render_login()
    st.stop()


# -----------------------------
# Main screen (after login)
# -----------------------------

# -----------------------------
# Diary full-screen views (records / analytics)
# -----------------------------

def _render_diary_records_fullscreen():
    st.markdown("## 📜 Günlük Kayıtları")
    user_key = (st.session_state.auth.get("username") or "").strip().lower()
    diary = _load_diary()
    entries = diary.get(user_key, []) if user_key else []
    if not entries:
        st.info("Henüz kayıt yok.")
    else:
        show = entries[::-1]  # newest first
        for e in show:
            eid = e.get("id") or uuid.uuid4().hex
            title = f"{e.get('date', '-')} • {e.get('mood', '-')}"
            with st.expander(title, expanded=False):
                st.write(f"**Etiketler:** {', '.join(e.get('tags', []) or []) or '-'}")
                pv = e.get("problem_var_mi")
                if pv:
                    st.write(f"**Problem var mı?** {pv}")
                    if pv == "Evet":
                        st.write(f"**Problem çözüldü mü?** {e.get('problem_cozuldu_mu') or '-'}")
                if e.get("chat_preview"):
                    st.write(e.get("chat_preview"))
                if e.get("chat_full"):
                    with st.expander("Detay (chat_full)", expanded=False):
                        st.code(e.get("chat_full") or "")
                if st.button("🗑️ Bu kaydı sil", key=f"records_del_{eid}", width='stretch'):
                    if e.get("id"):
                        new_entries = [x for x in entries if (x.get("id") or "") != (e.get("id") or "")]
                    else:
                        # fallback: drop last
                        new_entries = entries[:-1]
                    diary[user_key] = new_entries
                    _save_diary(diary)
                    st.success("Kayıt silindi ✅")
                    st.rerun()

    st.markdown("---")
    if st.button("⬅️ Chat ekranına geri dön", width='stretch'):
        st.session_state.diary_view = "none"
        st.rerun()


def _render_diary_analytics_fullscreen():
    st.markdown("## 📊 Günlük Analizi")
    user_key = (st.session_state.auth.get("username") or "").strip().lower()
    diary = _load_diary()
    entries = diary.get(user_key, []) if user_key else []
    if not entries:
        st.info("Analiz için önce en az 1 günlük kaydı ekle.")
    else:
        rows = []
        for e in entries:
            dt = _parse_entry_dt(e.get("date"))
            rows.append({
                "dt": dt,
                "date": e.get("date"),
                "mood": e.get("mood"),
                "tags": e.get("tags") or [],
                "problem_var_mi": e.get("problem_var_mi"),
                "problem_cozuldu_mu": e.get("problem_cozuldu_mu"),
            })
        df = pd.DataFrame(rows)
        df = df[df["dt"].notna()].copy()
        if df.empty:
            st.warning("Tarih formatı parse edilemedi. (Eski kayıt formatları olabilir.)")
        else:
            df["day"] = df["dt"].dt.date.astype(str)
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                window = st.selectbox("Dönem", ["Son 7 gün", "Son 30 gün", "Tümü"], index=1)
            with c2:
                mood_filter = st.multiselect("Mood", sorted([m for m in df["mood"].dropna().unique().tolist() if m]))
            with c3:
                tag_opts = sorted({t for ts in df["tags"].tolist() for t in (ts or [])})
                tag_filter = st.multiselect("Etiket", tag_opts)

            if window != "Tümü":
                days = 7 if window == "Son 7 gün" else 30
                cutoff = datetime.now() - pd.Timedelta(days=days)
                dfw = df[df["dt"] >= cutoff]
            else:
                dfw = df

            if mood_filter:
                dfw = dfw[dfw["mood"].isin(mood_filter)]
            if tag_filter:
                dfw = dfw[dfw["tags"].apply(lambda xs: any(t in (xs or []) for t in tag_filter))]

            if dfw.empty:
                st.warning("Bu filtrelerle kayıt bulunamadı.")
            else:
                total = len(dfw)
                prob_yes = int((dfw["problem_var_mi"] == "Evet").sum())
                prob_no = int((dfw["problem_var_mi"] == "Hayır").sum())
                solved_yes = int(((dfw["problem_var_mi"] == "Evet") & (dfw["problem_cozuldu_mu"] == "Evet")).sum())
                resolve_rate = (solved_yes / prob_yes * 100.0) if prob_yes else 0.0

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Kayıt", total)
                k2.metric("Problem: Evet", prob_yes)
                k3.metric("Problem: Hayır", prob_no)
                k4.metric("Çözülme Oranı", f"{resolve_rate:.0f}%")

                by_day = dfw.groupby("day").size().reset_index(name="count").sort_values("day")
                st.line_chart(by_day.set_index("day")["count"], height=220)

                mood_counts = dfw["mood"].fillna("-").value_counts().reset_index()
                mood_counts.columns = ["mood", "count"]
                st.bar_chart(mood_counts.set_index("mood")["count"], height=220)

                tag_counts = {}
                for ts in dfw["tags"].tolist():
                    for t in (ts or []):
                        tag_counts[t] = tag_counts.get(t, 0) + 1
                if tag_counts:
                    top = pd.DataFrame(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
                                       columns=["tag", "count"])
                    st.bar_chart(top.set_index("tag")["count"], height=220)
                else:
                    st.caption("Etiket yok.")

    st.markdown("---")
    if st.button("⬅️ Chat ekranına geri dön", width='stretch'):
        st.session_state.diary_view = "none"
        st.rerun()


# Full-screen routing: when diary_view is records/analytics, hide chat completely.
if st.session_state.get("diary_view") == "records":
    _render_diary_records_fullscreen()
    st.stop()

if st.session_state.get("diary_view") == "analytics":
    _render_diary_analytics_fullscreen()
    st.stop()

# Chat state

# Ensure chat history is isolated per-user (clear on account switch)
current_user = (st.session_state.auth.get("username") or "").strip().lower()
if st.session_state.get("_history_owner") != current_user:
    st.session_state.history = []
    st.session_state["_history_owner"] = current_user
    st.session_state["_clear_msg_input"] = True

if "history" not in st.session_state:
    st.session_state.history = []

# --- UI Settings (persist in session) ---
st.session_state.setdefault('stream_like', False)
st.session_state.setdefault('stream_speed', 0.01)
st.session_state.setdefault('tts_enabled', False)
st.session_state.setdefault('tts_voice', 'alloy')
st.session_state.setdefault('tts_model', 'gpt-4o-mini-tts')
st.session_state.setdefault('last_tts_error', '')
st.session_state.setdefault('stt_enabled', False)
st.session_state.setdefault('stt_model', 'whisper-1')
st.session_state.setdefault('stt_language', 'tr')
st.session_state.setdefault('stt_auto_send', True)
st.session_state.setdefault('last_stt_error', '')

if st.session_state.get("_clear_msg_input", False):
    st.session_state["_clear_msg_input"] = False
    st.session_state["msg_input"] = ""


def _pair_turns(history: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    i = 0
    while i < len(history):
        if history[i].get("role") == "user":
            user_turn = history[i]
            asst_turn = {}
            if i + 1 < len(history) and history[i + 1].get("role") == "assistant":
                asst_turn = history[i + 1]
                i += 2
            else:
                i += 1
            pairs.append((user_turn, asst_turn))
        else:
            i += 1
    return pairs


def build_chat_transcript_plain(history: List[Dict[str, Any]], last_n: int = 0) -> str:
    """Returns plain chat transcript (no export header) for diary storage."""
    pairs = _pair_turns(history)
    if last_n and last_n > 0:
        pairs = pairs[-last_n:]

    out: List[str] = []
    for idx, (u, a) in enumerate(pairs, start=1):
        u_text = (u.get("content") or "").strip()
        a_text = (a.get("content") or "").strip() if a else ""
        if u_text:
            out.append(f"Kullanıcı: {u_text}")
        if a_text:
            out.append(f"Asistan: {a_text}")
        out.append("")  # spacer
    return "\n".join(out).strip()


def build_export_txt(history: List[Dict[str, Any]], last_n: int = 0) -> str:
    pairs = _pair_turns(history)
    if last_n and last_n > 0:
        pairs = pairs[-last_n:]

    lines: List[str] = []
    lines.append("STREAMLIT TEST EXPORT")
    lines.append(f"Export time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total turns (paired): {len(pairs)}")
    lines.append("=" * 60)

    for idx, (u, a) in enumerate(pairs, start=1):
        lines.append(f"[{idx}] USER: {u.get('content', '')}")
        if not a:
            lines.append("ASSISTANT: (missing)")
            lines.append("-" * 60)
            continue

        meta = a.get("meta") or {}
        risk = meta.get("risk_level") or a.get("risk_level")
        emo = meta.get("emotion") or a.get("emotion")

        if risk:
            lines.append(f"RISK: {risk}")
        if emo:
            lines.append(f"EMOTION: {emo}")

        answer = a.get("content") or a.get("answer") or ""
        lines.append("ASSISTANT:")
        lines.append(answer.strip())

        why = a.get("why") or {}
        if why:
            lines.append("WHY (debug):")
            lines.append(json.dumps(why, ensure_ascii=False, indent=2))

        srcs = a.get("sources") or []
        if srcs:
            lines.append("SOURCES:")
            for s in srcs[:50]:
                lines.append(
                    f"- {s.get('source', '')} | score={s.get('score', '')} | focus={s.get('focus_area', '')} | why={s.get('why', '')}")

        lines.append("-" * 60)

    return "\n".join(lines)


with st.sidebar:
    # Login time display (only current)
    username = st.session_state.auth.get("username")
    users = _load_users()
    urec = users.get((username or "").strip().lower(), {}) if username else {}
    cur = st.session_state.current_login_time or urec.get("last_login_local") or ""
    if cur:
        st.caption(f"**Güncel Giriş Zamanı:** {cur}")
    prof = _get_profile(username)
    gender_icon = _gender_icon(prof.get("gender") or st.session_state.auth.get("gender", "male"))

    st.markdown(f"**Kullanıcı :** {gender_icon} **{username}**")

    with st.expander("Kullanıcı Profili", expanded=False):
        st.write(f"**İsim:** {prof.get('first_name') or '-'}")
        st.write(f"**Soyisim:** {prof.get('last_name') or '-'}")
        st.write(f"**Yaş:** {prof.get('age') if prof.get('age') is not None else '-'}")
        st.write(f"**Boy (cm):** {prof.get('height_cm') if prof.get('height_cm') is not None else '-'}")
        st.write(f"**Kilo (kg):** {prof.get('weight_kg') if prof.get('weight_kg') is not None else '-'}")
        g = prof.get("gender") or "male"
        st.write(f"**Cinsiyet:** {'Kadın' if g == 'female' else 'Erkek' if g == 'male' else 'Diğer'}")

    if st.button("Çıkış Yap", width='stretch'):
        try:
            _update_logout_time(st.session_state.auth.get("username"))
        except Exception:
            pass
        # Clear chat state on logout so messages do not leak across accounts
        st.session_state.history = []
        st.session_state["_history_owner"] = ""
        st.session_state["_clear_msg_input"] = True
        st.session_state.auth = {"is_authed": False, "username": None, "user_id": None, "gender": "male"}
        st.rerun()

    st.markdown("### 📔 Destek Günlüğü")
    with st.expander("Günlük ekle / geçmişi gör", expanded=False):
        user_key = (st.session_state.auth.get("username") or "").strip().lower()
        diary = _load_diary()
        entries = diary.get(user_key, []) if user_key else []
        mood = st.selectbox("Bugün nasılsın?", ["🙂 iyi", "😐 orta", "😟 kötü", "🚨 kriz"], index=1)
        tags = st.multiselect("Etiketler (opsiyonel)",
                              ["uyku", "ilişki", "öz-şefkat", "motivasyon", "yorgunluk", "panik"])

        st.markdown("**Problem var mı?**")
        prob = st.radio(" ", ["Hayır", "Evet"], horizontal=True, key="diary_prob")
        solved = None
        if prob == "Evet":
            st.markdown("**Problem çözüldü mü?**")
            solved = st.radio("  ", ["Evet", "Hayır"], horizontal=True, key="diary_solved")

        # Yeni mesaj gelmeden tekrar tekrar kaydı engelle
        user_msgs = [m.get("content", "").strip() for m in st.session_state.history if
                     m.get("role") == "user" and (m.get("content") or "").strip()]
        latest_user_msg = user_msgs[-1] if user_msgs else ""
        last_saved = st.session_state.diary_last_saved_by_user.get(user_key, "") if user_key else ""
        can_save = bool(user_key and latest_user_msg and latest_user_msg != last_saved)

        if st.button("Günlüğe Kaydet", width='stretch', disabled=not can_save):
            if not user_key:
                st.error("Kullanıcı bulunamadı.")
            elif not latest_user_msg:
                st.warning("Önce sohbet kısmına yeni bir mesaj yaz.")
            elif latest_user_msg == last_saved:
                st.info("Yeni bir mesaj yazdıktan sonra tekrar kaydedebilirsin.")
            else:
                chat_full = "\n".join(user_msgs[-10:])  # son 10 kullanıcı mesajı (sadece sen)
                chat_preview = (("Sen: " + latest_user_msg).replace("\n", " ")[:160] + "…")
                entries.append({
                    "id": uuid.uuid4().hex,
                    "date": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                    "mood": mood,
                    "tags": tags,
                    "problem_var_mi": prob,
                    "problem_cozuldu_mu": (solved if prob == "Evet" else None),
                    "chat_preview": chat_preview,
                    "chat_full": chat_full,
                })
                diary[user_key] = entries[-60:]
                _save_diary(diary)
                st.session_state.diary_last_saved_by_user[user_key] = latest_user_msg
                st.success("Kaydedildi ✅")
                st.rerun()

        # 📊 Günlük analizi

        cA, cB = st.columns(2)
        with cA:
            if st.button("📜 Günlük Kayıtları", width='stretch'):
                st.session_state.diary_view = "none" if st.session_state.get("diary_view") == "records" else "records"
                st.rerun()
        with cB:
            if st.button("📊 Günlük Analizi", width='stretch'):
                st.session_state.diary_view = "none" if st.session_state.get(
                    "diary_view") == "analytics" else "analytics"
                st.rerun()



    st.markdown("---")
    st.subheader("Ayarlar")

    # Kaynak sayısı (RAG)
    st.session_state.setdefault("k_sources", 5)
    st.session_state["k_sources"] = st.slider(
        "Kaynak sayısı",
        min_value=1, max_value=9,
        value=int(st.session_state.get("k_sources", 5)),
        step=1,
    )


    # Yanıt akışı
    st.session_state["stream_like"] = st.checkbox(
        "Yanıtı tane tane yazdır",
        value=bool(st.session_state.get("stream_like", False)),
    )
    st.session_state["stream_speed"] = st.slider(
        "Akış hızı (yazım gecikmesi)",
        min_value=0.0, max_value=0.05,
        value=float(st.session_state.get("stream_speed", 0.01)),
        step=0.001,
    )

    st.markdown("---")

    # TTS
    st.subheader("🔊 Sesli Yanıt")
    st.session_state["tts_enabled"] = st.checkbox(
        "Asistan yanıtını sesli okut",
        value=bool(st.session_state.get("tts_enabled", False)),
    )
    voices = ["alloy", "verse", "coral", "sage", "ash", "ballad"]
    current_voice = st.session_state.get("tts_voice", "alloy")
    voice_index = voices.index(current_voice) if current_voice in voices else 0
    st.session_state["tts_voice"] = st.selectbox(
        "Ses (voice)",
        voices,
        index=voice_index,
    )
    st.session_state["tts_autoplay_try"] = st.checkbox(
        "Autoplay'i dene",
        value=bool(st.session_state.get("tts_autoplay_try", False)),
    )

    st.session_state["tts_model"] = st.selectbox(
        "TTS Model",
        ["gpt-4o-mini-tts", "gpt-4o-realtime-preview-tts"],
        index=0,
    )

    # STT
    st.subheader("🎙️ Sesle Mesaj")
    st.session_state["stt_enabled"] = st.checkbox(
        "Mikrofonla konuş (ses → yazı)",
        value=bool(st.session_state.get("stt_enabled", False)),
    )
    st.session_state["stt_auto_send"] = st.checkbox(
        "Ses çevrilince otomatik gönder",
        value=bool(st.session_state.get("stt_auto_send", True)),
    )
    st.session_state["stt_language"] = st.selectbox(
        "Dil",
        ["tr", "en", "auto"],
        index=0,
    )
    st.session_state["stt_model"] = st.selectbox(
        "STT Model",
        ["whisper-1"],
        index=0,
    )

    st.markdown("---")

    with st.expander("🧪 Debug & Export", expanded=False):
        show_raw = st.checkbox("Debug: backend raw response göster", value=False)
        export_last_n = st.number_input("Export: son kaç tur? (0 = hepsi)", min_value=0, max_value=500, value=0, step=1)
    export_text = build_export_txt(st.session_state.history, last_n=int(export_last_n))
    st.download_button("📄 Test export indir (.txt)", data=export_text,
                       file_name=f"streamlit_test_export_{int(time.time())}.txt", mime="text/plain", width='stretch')
    st.markdown("---")


def risk_banner(risk: str):
    if risk == "high":
        st.error("‼️ Risk Durumu: HIGH", icon="❗")
    elif risk == "medium":
        st.warning("‼️ Risk Durumu: MEDIUM", icon="❗")
    elif risk == "low":
        st.success("‼️ Risk Durumu: LOW", icon="❗")


def emotion_banner(emotion: str):
    if emotion == "negative":
        st.warning("🧠 Duygu Analizi (Emotion): NEGATİF", icon="🧠")
    elif emotion == "neutral":
        st.info("🧠 Duygu Analizi (Emotion): NÖTR", icon="🧠")
    elif emotion == "positive":
        st.success("🧠 Duygu Analizi (Emotion): POZİTİF", icon="🧠")
    elif emotion == "critical":
        st.error("🧠 Duygu Analizi (Emotion): KRİTİK", icon="🧠")


def render_sources(srcs: list, k_val: int):
    if not srcs:
        return
    st.markdown("### 📚 Kullanılan Kaynaklar")
    for s in srcs[: int(k_val)]:
        src = s.get("source", "")
        score = s.get("score", "")
        focus = s.get("focus_area", "")
        st.write(f"- {src} | score: {score} | focus: {focus}")
        if s.get("why"):
            st.caption(f"👉 {s.get('why')}")


def render_explainability(why: dict):
    if not why:
        return
    with st.expander("🧾 Karar Açıklaması (Neden bu sonuç çıktı?)", expanded=False):
        scope = why.get("scope", {}) or {}
        if scope:
            st.markdown("**🧭 Scope (in/out kararı)**")
            st.write(f"- domain: {scope.get('domain')}")
            st.write(f"- reason: {scope.get('reason')}")
            if scope.get("score") is not None:
                try:
                    st.write(f"- score: {float(scope.get('score')):.3f}")
                except Exception:
                    st.write(f"- score: {scope.get('score')}")
        risk = why.get("risk", {}) or {}
        if risk:
            st.markdown("**🛡️ Risk (safety)**")
            st.write(f"- pre: {risk.get('pre')}")
            if risk.get("final") is not None:
                st.write(f"- final: {risk.get('final')}")
            if risk.get("trend_escalated") is not None:
                st.write(f"- trend escalation: {risk.get('trend_escalated')}")
        emo = why.get("emotion", {}) or {}
        if emo:
            st.markdown("**🙂 Duygu Analizi (Emotion)**")
            st.write(f"- label: {emo.get('label')} (reason: {emo.get('reason')}, hit: {emo.get('hit')})")
        mem = why.get("memory", {}) or {}
        if mem:
            st.markdown("**🧠 Hafıza (Memory)**")
            st.write(f"- used: {mem.get('used')} (turns_considered: {mem.get('turns_considered')})")
        rag = why.get("rag", {}) or {}
        if rag:
            st.markdown("**📚 RAG**")
            st.write(f"- k: {rag.get('k')} | returned: {rag.get('returned')} | used_rewrite: {rag.get('used_rewrite')}")


def _safe_parse_json(resp: requests.Response):
    if resp is None:
        return None, "No response object"
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text[:1200]}"
    try:
        return resp.json(), None
    except Exception:
        body = (resp.text or "").strip()
        return None, f"Backend JSON dönmedi. Body (ilk 1200 char): {body[:1200]}"


def _stt_transcribe(audio_bytes: bytes, *, model: str = "whisper-1", language: str = "tr") -> str:
    """
    Speech-to-text using OpenAI. Returns transcribed text (may be empty).
    Expects audio bytes (wav/mp3/m4a/webm etc.).
    """
    if OpenAI is None:
        raise RuntimeError("openai paketi bulunamadı. `pip install openai` kurulu olmalı.")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY bulunamadı (.env).")

    if not audio_bytes:
        return ""

    client = OpenAI(api_key=api_key)

    # Best-effort: provide a filename so the SDK can infer format
    import io
    file_obj = io.BytesIO(audio_bytes)
    file_obj.name = "speech.wav"  # type: ignore[attr-defined]

    kwargs = {}
    lang = (language or "").strip().lower()
    if lang and lang != "auto":
        kwargs["language"] = lang

    # New SDK: audio.transcriptions.create
    resp = client.audio.transcriptions.create(
        model=model,
        file=file_obj,
        **kwargs,
    )
    # SDK returns an object with .text
    text_out = getattr(resp, "text", None)
    if isinstance(text_out, str):
        return text_out.strip()
    # Fallback: try dict-like
    try:
        return (resp.get("text") or "").strip()  # type: ignore[attr-defined]
    except Exception:
        return ""


# If STT asked to fill the message box, do it BEFORE the widget is created.
if st.session_state.get("_stt_fill_msg_input") is not None:
    try:
        st.session_state["msg_input"] = st.session_state.pop("_stt_fill_msg_input") or ""
    except Exception:
        # If something goes wrong, just drop it
        st.session_state.pop("_stt_fill_msg_input", None)

# Clear message input safely before widget creation
if st.session_state.get("_clear_msg_input"):
    st.session_state["msg_input"] = ""
    st.session_state["_clear_msg_input"] = False

with st.form("chat_form", clear_on_submit=False):
    msg = st.text_input("Mesajın", key="msg_input", placeholder="Buraya yaz…")
    col1, col2 = st.columns([1, 1])
    with col1:
        submitted = st.form_submit_button("Gönder", width='stretch')
    with col2:
        clear = st.form_submit_button("Temizle", width='stretch')

# -----------------------------
# STT UI (Speech -> Text)
# -----------------------------
if bool(st.session_state.get("stt_enabled")):
    with st.expander("🎙️ Sesle mesaj yaz", expanded=False):
        audio_bytes_in = b""

        if hasattr(st, "audio_input"):
            a = st.audio_input("Mikrofondan kayıt al", key="stt_audio_input")
            if a is not None:
                try:
                    audio_bytes_in = a.getvalue()  # type: ignore[attr-defined]
                except Exception:
                    try:
                        audio_bytes_in = a.read()
                    except Exception:
                        audio_bytes_in = b""
        else:
            up = st.file_uploader(
                "Mikrofon özelliği yoksa ses dosyası yükle (wav/mp3/m4a/webm)",
                type=["wav", "mp3", "m4a", "webm", "mp4"],
                key="stt_audio_upload",
            )
            if up is not None:
                audio_bytes_in = up.getvalue()

        cstt1, cstt2 = st.columns([1, 1])
        with cstt1:
            do_transcribe = st.button("📝 Yazıya çevir", width='stretch')
        with cstt2:
            do_send = st.button("📨 Çevir ve gönder", width='stretch', disabled=not bool(audio_bytes_in))

        if do_transcribe or do_send:
            st.session_state["last_stt_error"] = ""
            try:
                tr_text = _stt_transcribe(
                    audio_bytes_in,
                    model=str(st.session_state.get("stt_model") or "whisper-1"),
                    language=str(st.session_state.get("stt_language") or "tr"),
                )
                if not tr_text:
                    st.warning("Ses çözümlenmedi (boş çıktı).")
                else:
                    st.session_state["_stt_fill_msg_input"] = tr_text
                    if do_send or bool(st.session_state.get("stt_auto_send")):
                        st.session_state["_stt_send_now"] = tr_text
                        st.rerun()
                    else:
                        st.success("Mesaj kutusuna yazıldı ✅ (Gönder'e basabilirsin)")
                        st.rerun()
            except Exception as e:
                st.session_state["last_stt_error"] = str(e)
                st.error(f"STT hata: {e}")

    if st.session_state.get("last_stt_error"):
        st.caption(f"🎙️ STT hata: {st.session_state.get('last_stt_error')}")

if clear:
    st.session_state.history = []
    st.session_state["_clear_msg_input"] = True
    st.rerun()


def _tts_speak_mp3(text: str, *, model: str, voice: str) -> bytes:
    """Return MP3 bytes for given text. Raises Exception on failure."""
    if OpenAI is None:
        raise RuntimeError("openai paketi bulunamadı. `pip install openai` kurulu olmalı.")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY bulunamadı (.env).")
    client = OpenAI(api_key=api_key)

    # Keep it short & safe for audio
    t = (text or "").strip()
    if not t:
        return b""
    if len(t) > 4000:
        t = t[:4000] + "..."

    # New SDK: audio.speech.create returns a response with bytes
    resp = client.audio.speech.create(
        model=model,
        voice=voice,
        input=t,
        response_format="mp3",
    )
    # Depending on SDK version, the bytes may be in resp.content or resp.read()
    data = getattr(resp, "content", None)
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if hasattr(resp, "read"):
        return resp.read()
    # Fallback: try to cast
    try:
        return bytes(resp)  # type: ignore
    except Exception:
        raise RuntimeError("TTS çıktısı okunamadı (OpenAI SDK response).")


def send_message(user_msg: str):
    st.session_state.history.append({"role": "user", "content": user_msg})
    payload = {
        "message": user_msg,
        "k": int(st.session_state.get("k_sources", 5)),
        "history": [{"role": t["role"], "content": t["content"], "meta": t.get("meta")} for t in
                    st.session_state.history[-20:]],
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=120)
    except Exception as e:
        st.error(f"Backend'e bağlanamadım: {e}")
        st.session_state.history.append(
            {"role": "assistant", "content": "Backend'e bağlanamadım. Lütfen backend terminalini kontrol et.",
             "meta": {}, "sources": [], "why": {}})
        st.session_state["_clear_msg_input"] = True
        st.rerun()
        return

    if show_raw:
        st.caption(f"RAW status={r.status_code}")
        st.code((r.text or "")[:4000])

    data, err = _safe_parse_json(r)
    if err:
        st.error(err)
        st.session_state.history.append(
            {"role": "assistant", "content": "Backend hata verdi. Lütfen backend terminal loglarına bak.", "meta": {},
             "sources": [], "why": {"ui_parse_error": err}})
        st.session_state["_clear_msg_input"] = True
        st.rerun()
        return

    audio_bytes = b""
    st.session_state["last_tts_error"] = ""
    if bool(st.session_state.get("tts_enabled")):
        try:
            audio_bytes = _tts_speak_mp3(
                data.get("answer") or "",
                model=str(st.session_state.get("tts_model") or "gpt-4o-mini-tts"),
                voice=str(st.session_state.get("tts_voice") or "alloy"),
            )
        except Exception as e:
            st.session_state["last_tts_error"] = str(e)
            audio_bytes = b""

    msg_id = str(uuid.uuid4())

    st.session_state.history.append({
        "id": msg_id,
        "role": "assistant",
        "content": data.get("answer") or "",
        "meta": {"risk_level": data.get("risk_level"), "emotion": data.get("emotion")},
        "sources": data.get("sources") or [],
        "why": data.get("why") or {},
        "domain": data.get("domain"),
        "risk_level": data.get("risk_level"),
        "emotion": data.get("emotion"),
        "audio": audio_bytes,
        "stream_done": False,
    })

    # Autoplay should trigger ONLY for the newest assistant audio once.
    if audio_bytes and bool(st.session_state.get("tts_autoplay_try")):
        st.session_state["tts_autoplay_target_id"] = msg_id
        st.session_state["tts_autoplay_played_id"] = None

    st.session_state["_clear_msg_input"] = True
    st.rerun()


# If a voice transcription requested an immediate send, send it now.
if st.session_state.get("_stt_send_now"):
    _voice_text = st.session_state.pop("_stt_send_now")
    if isinstance(_voice_text, str) and _voice_text.strip():
        send_message(_voice_text.strip())

if submitted and msg.strip():
    send_message(msg.strip())

st.divider()

user_icon = _gender_icon(
    _get_profile(st.session_state.auth.get("username")).get("gender") or st.session_state.auth.get("gender", "male"))

for i in range(len(st.session_state.history) - 1, -1, -1):
    turn = st.session_state.history[i]
    if turn.get("role") == "user":
        st.markdown(f"### {user_icon} Sen\n{turn.get('content', '')}")
        if i + 1 < len(st.session_state.history):
            nxt = st.session_state.history[i + 1]
            if nxt.get("role") == "assistant":
                meta = nxt.get("meta") or {}

                # Risk + Emotion
                if meta.get("risk_level"):
                    risk_banner(meta.get("risk_level"))
                if meta.get("emotion"):
                    emotion_banner(meta.get("emotion"))

                # 🔊 Audio (risk/emotion altında, cevap üstünde)
                msg_id = nxt.get("id")
                audio = nxt.get("audio", b"")

                if audio:
                    autoplay_enabled = bool(st.session_state.get("tts_autoplay_try"))
                    target_id = st.session_state.get("tts_autoplay_target_id")
                    played_id = st.session_state.get("tts_autoplay_played_id")

                    should_autoplay = (
                            autoplay_enabled
                            and msg_id is not None
                            and msg_id == target_id
                            and played_id != msg_id
                    )

                    if should_autoplay:
                        try:
                            import base64
                            import streamlit.components.v1 as components

                            b64 = base64.b64encode(audio).decode("utf-8")
                            components.html(
                                f"""
                                <audio autoplay controls style="width:100%">
                                    <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
                                </audio>
                                """,
                                height=60,
                            )
                            st.session_state["tts_autoplay_played_id"] = msg_id
                        except Exception:
                            # Fallback: no autoplay, but still playable
                            st.audio(audio, format="audio/mp3")
                            st.session_state["tts_autoplay_played_id"] = msg_id
                    else:
                        st.audio(audio, format="audio/mp3")

                elif bool(st.session_state.get("tts_enabled")) and st.session_state.get("last_tts_error"):
                    st.caption(f"🔇 TTS hata: {st.session_state.get('last_tts_error')}")

                # Answer
                st.markdown("### 🤖 Asistan")

                content = nxt.get("content", "") or ""
                use_stream = bool(st.session_state.get("stream_like")) and (not nxt.get("stream_done", False))
                if use_stream:
                    ph = st.empty()
                    shown = ""
                    delay = float(st.session_state.get("stream_speed", 0.01))
                    for ch in content:
                        shown += ch
                        ph.markdown(shown)
                        time.sleep(delay)
                    # mark done so next rerun doesn't re-stream
                    nxt["stream_done"] = True
                else:
                    st.markdown(content)

                # Explainability + sources
                if nxt.get("why"):
                    render_explainability(nxt.get("why") or {})
                if nxt.get("sources"):
                    render_sources(nxt.get("sources") or [], k_val=int(st.session_state.get("k_sources", 5)))
