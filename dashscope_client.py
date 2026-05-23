import base64
import time
from typing import Optional
import requests
from config import config

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.trust_env = False
    return _session


def _api_headers(async_mode: bool = True) -> dict:
    h = {
        "Authorization": f"Bearer {config.dashscope_api_key}",
        "Content-Type": "application/json",
    }
    if async_mode:
        h["X-DashScope-Async"] = "enable"
    return h


def _poll_task(task_id: str, max_wait: int = 120) -> dict:
    """Poll a DashScope async task until completion."""
    url = f"{config.dashscope_base_url}/tasks/{task_id}"
    s = _get_session()
    start = time.time()
    while time.time() - start < max_wait:
        resp = s.get(
            url,
            headers={"Authorization": f"Bearer {config.dashscope_api_key}"},
            timeout=10,
        )
        data = resp.json()
        status = data.get("output", {}).get("task_status", "UNKNOWN")
        if status == "SUCCEEDED":
            return data
        elif status == "FAILED":
            print(
                f"Task {task_id} failed: "
                f"{data.get('output', {}).get('message', '')}"
            )
            return data
        time.sleep(2)
    return {"error": "timeout"}


def _download_bytes(url: str) -> Optional[bytes]:
    """Download from a URL, return bytes or None."""
    if not url:
        return None
    try:
        resp = _get_session().get(url, timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"Download failed: {e}")
        return None


# ============================================================
# Image Generation (wanx-v1 txt2img)
# ============================================================

def generate_txt2img(
    prompt: str,
    size: str = "1024*1024",
    seed: int = 42,
) -> Optional[bytes]:
    """Generate a single image from a text prompt. Returns PNG bytes or None."""
    url = f"{config.dashscope_base_url}/services/aigc/text2image/image-synthesis"
    body = {
        "model": "wanx-v1",
        "input": {"prompt": prompt},
        "parameters": {"size": size, "n": 1, "seed": seed},
    }
    resp = _get_session().post(
        url, headers=_api_headers(True), json=body, timeout=30
    )
    data = resp.json()
    task_id = data.get("output", {}).get("task_id", "")
    if not task_id:
        print(f"txt2img task creation failed: {data}")
        return None

    result = _poll_task(task_id, max_wait=60)
    results = result.get("output", {}).get("results", [])
    if results:
        return _download_bytes(results[0].get("url", ""))
    return None


# ============================================================
# Video Generation
# ============================================================

def generate_t2v(
    prompt: str,
    resolution: str = "480P",
    duration: int = 5,
) -> Optional[bytes]:
    """Text-to-Video. Takes prompt, returns MP4 bytes or None."""
    url = f"{config.dashscope_base_url}/services/aigc/video-generation/video-synthesis"
    body = {
        "model": "wanx2.1-t2v-turbo",
        "input": {"prompt": prompt},
        "parameters": {
            "duration": duration,
            "resolution": resolution,
        },
    }
    resp = _get_session().post(
        url, headers=_api_headers(True), json=body, timeout=30
    )
    data = resp.json()
    task_id = data.get("output", {}).get("task_id", "")
    if not task_id:
        print(f"t2v task creation failed: {data}")
        return None

    result = _poll_task(task_id, max_wait=90)
    video_url = result.get("output", {}).get("video_url", "")
    if video_url:
        return _download_bytes(video_url)
    # Fallback: check results array
    results = result.get("output", {}).get("results", [])
    if results:
        return _download_bytes(results[0].get("url", ""))
    return None


def generate_i2v(
    prompt: str,
    image_bytes: bytes,
    resolution: str = "480P",
    duration: int = 5,
) -> Optional[bytes]:
    """Image-to-Video. Takes base image bytes (base64-encoded) + prompt, returns MP4 bytes."""
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    img_data_uri = f"data:image/png;base64,{img_b64}"

    url = f"{config.dashscope_base_url}/services/aigc/video-generation/video-synthesis"
    body = {
        "model": "wanx2.1-i2v-turbo",
        "input": {
            "prompt": prompt,
            "img_url": img_data_uri,
        },
        "parameters": {
            "duration": duration,
            "resolution": resolution,
        },
    }
    resp = _get_session().post(
        url, headers=_api_headers(True), json=body, timeout=30
    )
    data = resp.json()
    task_id = data.get("output", {}).get("task_id", "")
    if not task_id:
        print(f"i2v task creation failed: {data}")
        return None

    result = _poll_task(task_id, max_wait=200)
    video_url = result.get("output", {}).get("video_url", "")
    if video_url:
        return _download_bytes(video_url)
    results = result.get("output", {}).get("results", [])
    if results:
        return _download_bytes(results[0].get("url", ""))
    return None
