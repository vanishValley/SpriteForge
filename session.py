import uuid
import time
import hashlib
from typing import Optional
from config import config


class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.assets: list[dict] = []
        self.style_ref_bytes: Optional[bytes] = None
        self.style_keywords: str = "pixel art, game asset"


class SessionManager:
    def __init__(self, ttl_seconds: Optional[int] = None):
        self._sessions: dict[str, Session] = {}
        self._cache: dict[str, tuple[bytes, float]] = {}
        self._ttl = ttl_seconds or config.session_ttl_seconds

    def create_session(self) -> str:
        sid = uuid.uuid4().hex
        self._sessions[sid] = Session(sid)
        return sid

    def get_session(self, session_id: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if time.time() - session.created_at > self._ttl:
            del self._sessions[session_id]
            return None
        return session

    def add_asset(
        self,
        session_id: str,
        job_id: str,
        asset_type: str,
        description: str,
        frame_count: int,
    ):
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found or expired")
        session.assets.append({
            "job_id": job_id,
            "asset_type": asset_type,
            "description": description,
            "created_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "frame_count": frame_count,
            "cached": False,
        })

    def get_assets(self, session_id: str) -> list[dict]:
        session = self.get_session(session_id)
        return session.assets if session else []

    def set_style_ref(self, session_id: str, image_bytes: bytes):
        session = self.get_session(session_id)
        if session:
            session.style_ref_bytes = image_bytes

    def get_style_ref(self, session_id: str) -> Optional[bytes]:
        session = self.get_session(session_id)
        return session.style_ref_bytes if session else None

    def set_style_keywords(self, session_id: str, keywords: str):
        session = self.get_session(session_id)
        if session:
            session.style_keywords = keywords

    def get_style_keywords(self, session_id: str) -> str:
        session = self.get_session(session_id)
        return session.style_keywords if session else "pixel art, game asset"

    def make_cache_key(
        self,
        session_id: str,
        asset_type: str,
        description: str,
        style_keywords: str,
        size: int,
        frames: int,
        view: str,
        grid_size: int,
    ) -> str:
        raw = (
            f"{asset_type}|{description}|{style_keywords}|"
            f"{size}|{frames}|{view}|{grid_size}"
        )
        ref = self.get_style_ref(session_id)
        if ref:
            raw += f"|ref:{hashlib.sha256(ref).hexdigest()[:16]}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_cache(self, cache_key: str) -> Optional[bytes]:
        entry = self._cache.get(cache_key)
        if entry is None:
            return None
        data, created_at = entry
        if time.time() - created_at > self._ttl:
            del self._cache[cache_key]
            return None
        return data

    def set_cache(self, cache_key: str, data: bytes):
        self._cache[cache_key] = (data, time.time())

    def cleanup_expired(self):
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.created_at > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]
        expired_keys = [
            k for k, (_, t) in self._cache.items()
            if now - t > self._ttl
        ]
        for k in expired_keys:
            del self._cache[k]
