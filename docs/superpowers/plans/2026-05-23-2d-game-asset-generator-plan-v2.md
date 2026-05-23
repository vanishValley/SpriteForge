# 2D Game Asset Generator — Implementation Plan v2 (API-Verified)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a web tool that generates 2D game assets (character sprites, tilemap, UI, props, VFX) from text descriptions, with optional style reference image. Single-frame assets use wanx-v1 txt2img. Multi-frame animation uses video generation (wanx2.1-t2v/i2v) + frame extraction + spritesheet packing. Outputs PNG+JSON compatible with Unity/Godot/GameMaker.

**Architecture:** FastAPI backend handles session/cache, prompt construction (5 asset types), DeepSeek rewriting, DashScope image generation (wanx-v1 txt2img) and video generation (wanx2.1-t2v-turbo for text-driven, wanx2.1-i2v-turbo for image-anchored), OpenCV-based frame extraction, rembg/crop/spritesheet-packing post-processing. Single-page vanilla HTML/CSS/JS frontend.

**Tech Stack:** Python 3.9+, FastAPI, opencv-python, rembg, Pillow, requests (no dashscope SDK needed — all API calls use raw HTTP).

**Verified API facts built into this plan:**
- wanx-v1: async only, 4 fixed sizes (1024*1024, 720*1280, 1280*720, 768*1152), ~27s
- wanx2.1-t2v-turbo: text→video, ~30s, 480P/720P
- wanx2.1-i2v-turbo: image→video, ~160s, `img_url` in `input`, 480P/720P
- wanx-style-repaint-v1: works with style_index=-1, but only transfers style, not content
- No true img2img exists — keyword-driven txt2img is the primary path
- Same seed + same prompt = similar output (style consistency via seed)

**Cuts for 2-day timeline:** auto-tiling rule metadata, onion skinning in animation player, nine-slice JSON. Video frame extraction adds complexity but is core to the animation feature.

---

## File Structure

```
project/
├── main.py                # FastAPI app, lifespan, all route handlers
├── config.py              # Env vars, settings
├── models.py              # Pydantic request/response schemas
├── prompts.py             # Prompt templates (5 types), preset data
├── deepseek.py            # DeepSeek prompt rewriting client
├── dashscope_client.py    # Wanx image + video generation (raw HTTP)
├── frame_extractor.py     # OpenCV video → evenly-sampled frames
├── postprocess.py         # rembg, crop, grid-slice, spritesheet pack, tile edge-wrap
├── session.py             # Session + generation cache (in-memory, TTL cleanup)
├── static/
│   └── index.html         # Complete single-page UI
├── requirements.txt
├── .env
└── tests/
    ├── test_prompts.py
    ├── test_session.py
    ├── test_postprocess.py
    └── test_frame_extractor.py
```

---

## Day 1: Backend Pipeline (8 tasks)

### Task 1: Project Scaffold & Configuration

**Files:**
- Create: `requirements.txt`
- Create: `.env`
- Create: `config.py`

- [ ] **Step 1: Write requirements.txt**

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
Pillow>=10.0.0
rembg>=2.0.50
opencv-python>=4.9.0
requests>=2.27.0
python-multipart>=0.0.9
aiofiles>=23.0.0
```

- [ ] **Step 2: Write .env with real keys**

```
DASHSCOPE_API_KEY=sk-e6cf63c5746546d7ade299dfeed6d629
DEEPSEEK_API_KEY=sk-948d04fb35e54330beeb69977773de4b
```

- [ ] **Step 3: Write config.py**

```python
import os
from dataclasses import dataclass, field

@dataclass
class Config:
    dashscope_api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    session_ttl_seconds: int = 1800
    temp_dir: str = "/tmp/asset_generator"
    valid_image_sizes: tuple = ("1024*1024", "720*1280", "1280*720", "768*1152")
    default_image_size: str = "1024*1024"
    video_resolutions: dict = field(default_factory=lambda: {"480P": "480P", "720P": "720P"})
    default_video_resolution: str = "480P"
    default_video_duration: int = 5

config = Config()
```

- [ ] **Step 4: Verify imports**

```bash
python -c "from config import config; print('Config OK:', config.dashscope_base_url)"
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .env config.py
git commit -m "feat: project scaffold with verified config and API keys"
```

---

### Task 2: Pydantic Models

**Files:**
- Create: `models.py`

Backend task — write all Pydantic models (same as v1 plan, no changes needed). Models cover all 5 asset types, job status polling, session management.

- [ ] **Step 1: Write models.py** — same content as v1 plan Task 2
- [ ] **Step 2: Verify import**
- [ ] **Step 3: Commit**

---

### Task 3: Prompt Templates & Presets

**Files:**
- Create: `prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write tests/test_prompts.py**

```python
from prompts import build_prompt, build_video_prompt, PROMPT_TEMPLATES, PRESETS, AssetType

def test_build_character_prompt():
    result = build_prompt(AssetType.CHARACTER, "a knight in armor", "pixel art", "front", 1, False)
    assert "2D game character sprite" in result
    assert "a knight in armor" in result
    assert "pixel art" in result
    assert "full body" in result

def test_build_vfx_prompt_with_frames():
    result = build_prompt(AssetType.VFX, "fire explosion", "cartoon", "front", 8, False)
    assert "8 frames" in result
    assert "fire explosion" in result
    assert "flipbook" in result.lower()

def test_build_video_prompt_character():
    result = build_video_prompt(AssetType.CHARACTER, "warrior slash attack", "pixel art", mode="i2v")
    assert "warrior performs" in result.lower()
    assert "smooth motion" in result.lower()
    assert "pixel art" in result

def test_build_video_prompt_vfx():
    result = build_video_prompt(AssetType.VFX, "fire explosion", "cartoon", mode="t2v")
    assert "fire explosion" in result.lower()
    assert "animation" in result.lower()

def test_all_asset_types_have_template():
    for at in AssetType:
        assert at in PROMPT_TEMPLATES, f"Missing template for {at}"

def test_presets_per_type():
    for at in AssetType:
        assert at in PRESETS, f"Missing presets for {at}"
        assert len(PRESETS[at]) >= 3, f"Need >=3 presets for {at}"
```

- [ ] **Step 2: Write prompts.py**

```python
from enum import Enum


class AssetType(str, Enum):
    CHARACTER = "character"
    TILEMAP = "tilemap"
    UI = "ui"
    PROPS = "props"
    VFX = "vfx"


PROMPT_TEMPLATES: dict[AssetType, dict[str, str]] = {
    AssetType.CHARACTER: {
        "prefix": "2D game character sprite, ",
        "suffix": (
            ", full body, standing pose, clean silhouette, "
            "isolated on transparent background, game-ready asset, "
            "{style_keywords}"
        ),
    },
    AssetType.TILEMAP: {
        "prefix": "2D game environment tile, ",
        "suffix": (
            ", {view_angle}, single tile unit, seamlessly tileable, "
            "flat composition, game environment art, {style_keywords}"
        ),
    },
    AssetType.UI: {
        "prefix": "2D game UI element, ",
        "suffix": (
            ", clean vector-like, minimal design, "
            "game interface asset, {style_keywords}"
        ),
    },
    AssetType.PROPS: {
        "prefix": "2D game item/prop, ",
        "suffix": (
            ", centered, isolated object, clean edges, transparent background, "
            "game inventory asset, {view_angle}, {style_keywords}"
        ),
    },
    AssetType.VFX: {
        "prefix": "2D game visual effect sprite sheet, ",
        "suffix": (
            ", flipbook animation frames, {frames} frames arranged in sequence, "
            "transparent background, game VFX asset, {style_keywords}"
        ),
    },
}

QUALITY_PREFIX = "masterpiece, best quality, highly detailed, game-ready asset, clean edges, "

VIEW_LABELS = {
    "front": "front view",
    "side": "side view",
    "top_down": "top-down view",
}

# Video prompt templates for animation
VIDEO_PROMPTS: dict[AssetType, dict[str, str]] = {
    AssetType.CHARACTER: {
        "t2v": (
            "2D pixel art game character, {description}, "
            "full body, smooth motion animation, clean white background, "
            "game sprite animation, {style_keywords}, consistent character design"
        ),
        "i2v": (
            "the character performs {description}, "
            "smooth fluid motion, consistent character appearance, "
            "pixel art game sprite animation, clean white background, {style_keywords}"
        ),
    },
    AssetType.VFX: {
        "t2v": (
            "2D game visual effect animation, {description}, "
            "smooth motion, transparent-black background, "
            "game VFX sprite sheet source, {style_keywords}"
        ),
        "i2v": (
            "the visual effect animates with {description}, "
            "smooth fluid motion, consistent style, "
            "game VFX animation, black background, {style_keywords}"
        ),
    },
}

PRESETS: dict[AssetType, list[dict]] = {
    AssetType.CHARACTER: [
        {"label": "Knight sword slash (4 frames)", "description": "knight swinging a sword in a horizontal slash attack", "frames": 4, "mode": "i2v"},
        {"label": "Slime monster bounce (4 frames)", "description": "cute slime monster bouncing up and down", "frames": 4, "mode": "t2v"},
        {"label": "NPC villager idle", "description": "medieval villager NPC, casual clothes, standing idle", "frames": 1},
        {"label": "Pixel cat run cycle (4 frames)", "description": "pixel art cat running cycle for platformer game", "frames": 4, "mode": "t2v"},
    ],
    AssetType.TILEMAP: [
        {"label": "Grass field top-down RPG", "description": "green grass field tile for top-down RPG", "grid_size": 32, "view": "top_down"},
        {"label": "Dungeon stone floor", "description": "dark stone floor tile for dungeon", "grid_size": 32, "view": "top_down"},
        {"label": "Space platform side-view", "description": "sci-fi metal platform tile side-view", "grid_size": 64, "view": "side"},
        {"label": "Forest path with flowers", "description": "forest ground tile with small flowers", "grid_size": 32, "view": "top_down"},
    ],
    AssetType.UI: [
        {"label": "Fantasy health bar", "description": "fantasy themed health bar with gem decorations", "view": "front"},
        {"label": "Sci-fi button panel", "description": "sci-fi holographic button panel", "view": "front"},
        {"label": "Wooden inventory slot", "description": "wooden framed inventory slot medieval style", "view": "front"},
        {"label": "Gold coin icon", "description": "gold coin icon for in-game currency", "view": "front"},
    ],
    AssetType.PROPS: [
        {"label": "Iron longsword with glow", "description": "iron longsword with magical blue glow", "view": "front"},
        {"label": "Red health potion", "description": "red health potion in round glass bottle", "view": "front"},
        {"label": "Golden treasure chest", "description": "golden treasure chest with ornate details", "view": "front"},
        {"label": "Magic staff with crystal", "description": "wooden magic staff with glowing purple crystal", "view": "front"},
    ],
    AssetType.VFX: [
        {"label": "Fire explosion (8 frames)", "description": "fire explosion effect expanding outward", "frames": 8, "mode": "t2v"},
        {"label": "Sword slash arc (4 frames)", "description": "sword slash arc trail effect", "frames": 4, "mode": "t2v"},
        {"label": "Magic circle summon (6 frames)", "description": "magic circle appearing with glowing runes", "frames": 6, "mode": "t2v"},
        {"label": "Smoke puff (4 frames)", "description": "small cartoon smoke puff dissipating", "frames": 4, "mode": "t2v"},
    ],
}


def build_prompt(
    asset_type: AssetType,
    description: str,
    style_keywords: str,
    view: str,
    frames: int,
    has_reference: bool,
) -> str:
    template = PROMPT_TEMPLATES[asset_type]
    view_label = VIEW_LABELS.get(view, "front view")
    prefix = QUALITY_PREFIX + template["prefix"]
    desc = description[:400]
    suffix = template["suffix"].format(
        style_keywords=style_keywords,
        view_angle=view_label,
        frames=frames,
    )
    return (prefix + desc + suffix)[:1500]


def build_video_prompt(
    asset_type: AssetType,
    description: str,
    style_keywords: str,
    mode: str = "t2v",
) -> str:
    """Build a prompt for video generation (t2v or i2v)."""
    templates = VIDEO_PROMPTS.get(asset_type)
    if templates is None:
        # Fallback: use Character template
        templates = VIDEO_PROMPTS[AssetType.CHARACTER]
    template = templates.get(mode, templates["t2v"])
    return template.format(description=description, style_keywords=style_keywords)[:800]
```

- [ ] **Step 3: Run tests** → all 6 pass
- [ ] **Step 4: Commit**

---

### Task 4: DeepSeek Prompt Rewriting

**Files:**
- Create: `deepseek.py`

Same as v1 plan Task 4 but using the real key from config. Graceful fallback when key is empty.

```python
from openai import OpenAI
from config import config

SYSTEM_PROMPT = """You are a game asset prompt engineer. Rewrite a user's brief description into a detailed, professional prompt for AI image generation.

Rules:
1. Expand with specific visual details: pose, perspective, art style, color palette, composition.
2. Use standard game-art terminology appropriate to the asset type.
3. Keep output to 2-3 sentences, under 300 characters.
4. Do NOT add phrases like "create an image of..." — output the description directly.
5. Preserve all key nouns and concepts from the original description.
6. Output ONLY the rewritten description, nothing else."""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.deepseek_api_key, base_url=config.deepseek_base_url)
    return _client


def rewrite_prompt(raw_description: str, asset_type: str) -> str:
    if not config.deepseek_api_key:
        return raw_description
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Asset type: {asset_type}\nUser description: {raw_description}\n\nRewrite into a detailed game-asset prompt."},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else raw_description
    except Exception:
        return raw_description
```

- [ ] **Step 1: Verify**

```bash
python -c "from deepseek import rewrite_prompt; r = rewrite_prompt('a cool knight', 'character'); print('Rewritten:', r[:100])"
```

Expected: Expanded description from DeepSeek.

- [ ] **Step 2: Commit**

---

### Task 5: DashScope Client (Image + Video Generation)

**Files:**
- Create: `dashscope_client.py`

This is the core engine. Handles:
- wanx-v1 txt2img (create task → poll → download image bytes)
- wanx2.1-t2v-turbo (create task → poll → download video bytes)
- wanx2.1-i2v-turbo (create task with base64 image → poll → download video bytes)

```python
import io
import time
import base64
import requests
from config import config
from typing import Optional

_session = None


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
    """Poll a DashScope async task until completion. Returns the full task result dict."""
    url = f"{config.dashscope_base_url}/tasks/{task_id}"
    s = _get_session()
    start = time.time()
    while time.time() - start < max_wait:
        resp = s.get(url, headers={"Authorization": f"Bearer {config.dashscope_api_key}"}, timeout=10)
        data = resp.json()
        status = data.get("output", {}).get("task_status", "UNKNOWN")
        if status == "SUCCEEDED":
            return data
        elif status == "FAILED":
            print(f"Task {task_id} failed: {data.get('output', {}).get('message', '')}")
            return data
        time.sleep(2)
    return {"error": "timeout"}


def _download_bytes(url: str) -> Optional[bytes]:
    """Download from a URL, return bytes."""
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

def generate_txt2img(prompt: str, size: str = "1024*1024", seed: int = 42) -> Optional[bytes]:
    """Generate a single image from text prompt. Returns PNG bytes or None."""
    url = f"{config.dashscope_base_url}/services/aigc/text2image/image-synthesis"
    body = {
        "model": "wanx-v1",
        "input": {"prompt": prompt},
        "parameters": {"size": size, "n": 1, "seed": seed},
    }
    resp = _get_session().post(url, headers=_api_headers(True), json=body, timeout=30)
    data = resp.json()
    task_id = data.get("output", {}).get("task_id", "")
    if not task_id:
        return None

    result = _poll_task(task_id, max_wait=60)
    results = result.get("output", {}).get("results", [])
    if results:
        return _download_bytes(results[0].get("url", ""))
    return None


# ============================================================
# Video Generation (wanx2.1-t2v-turbo / wanx2.1-i2v-turbo)
# ============================================================

def generate_t2v(prompt: str, resolution: str = "480P", duration: int = 5) -> Optional[bytes]:
    """Text-to-Video. Returns MP4 bytes or None."""
    url = f"{config.dashscope_base_url}/services/aigc/video-generation/video-synthesis"
    body = {
        "model": "wanx2.1-t2v-turbo",
        "input": {"prompt": prompt},
        "parameters": {"duration": duration, "resolution": resolution},
    }
    resp = _get_session().post(url, headers=_api_headers(True), json=body, timeout=30)
    data = resp.json()
    task_id = data.get("output", {}).get("task_id", "")
    if not task_id:
        return None

    result = _poll_task(task_id, max_wait=90)
    video_url = result.get("output", {}).get("video_url", "")
    if video_url:
        return _download_bytes(video_url)
    return None


def generate_i2v(prompt: str, image_bytes: bytes, resolution: str = "480P", duration: int = 5) -> Optional[bytes]:
    """Image-to-Video. Takes base image bytes, encodes as base64, returns MP4 bytes or None."""
    # Method A: base64 encode the image for reliable transfer
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    img_data_uri = f"data:image/png;base64,{img_b64}"

    url = f"{config.dashscope_base_url}/services/aigc/video-generation/video-synthesis"
    body = {
        "model": "wanx2.1-i2v-turbo",
        "input": {
            "prompt": prompt,
            "img_url": img_data_uri,
        },
        "parameters": {"duration": duration, "resolution": resolution},
    }
    resp = _get_session().post(url, headers=_api_headers(True), json=body, timeout=30)
    data = resp.json()
    task_id = data.get("output", {}).get("task_id", "")
    if not task_id:
        return None

    result = _poll_task(task_id, max_wait=200)
    video_url = result.get("output", {}).get("video_url", "")
    if video_url:
        return _download_bytes(video_url)
    return None
```

- [ ] **Step 1: Verify image gen works**

```bash
python -c "
from dashscope_client import generate_txt2img
img = generate_txt2img('a cute cat, pixel art', size='1024*1024')
print('Image bytes:', len(img) if img else 'FAILED')
"
```

- [ ] **Step 2: Commit**

---

### Task 6: Frame Extractor (OpenCV)

**Files:**
- Create: `frame_extractor.py`
- Create: `tests/test_frame_extractor.py`

- [ ] **Step 1: Write tests/test_frame_extractor.py**

```python
import io
import pytest
from frame_extractor import extract_frames, frames_to_spritesheet, get_consistent_bbox
from PIL import Image


def _make_test_video_bytes(num_frames=30, w=100, h=100):
    """Create a minimal test video as bytes (using a simple approach for testing)."""
    import cv2
    import tempfile
    import numpy as np

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp.name, fourcc, 10.0, (w, h), isColor=True)
    for i in range(num_frames):
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    with open(tmp.name, "rb") as f:
        data = f.read()
    import os
    os.unlink(tmp.name)
    return data


def test_extract_frames_count():
    video_bytes = _make_test_video_bytes(30, 100, 100)
    frames = extract_frames(video_bytes, num_frames=8)
    assert len(frames) == 8
    assert all(isinstance(f, bytes) for f in frames)


def test_extract_frames_returns_png():
    video_bytes = _make_test_video_bytes(10, 64, 64)
    frames = extract_frames(video_bytes, num_frames=4)
    for f in frames:
        img = Image.open(io.BytesIO(f))
        assert img.size == (64, 64)


def test_frames_to_spritesheet():
    video_bytes = _make_test_video_bytes(16, 32, 32)
    frames = extract_frames(video_bytes, num_frames=4)
    sheet_bytes, coords = frames_to_spritesheet(frames)
    assert isinstance(sheet_bytes, bytes)
    assert len(coords) == 4
    sheet_img = Image.open(io.BytesIO(sheet_bytes))
    assert sheet_img.width == 128  # 4 * 32


def test_get_consistent_bbox():
    # Create 4 frames with content in slightly different positions
    frames = []
    for offset in range(4):
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        img.paste(Image.new("RGBA", (20, 30), (255, 0, 0, 255)), (10 + offset, 20))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        frames.append(buf.getvalue())
    bbox = get_consistent_bbox(frames)
    assert bbox is not None
    # Union bbox should cover all offsets
    assert bbox[0] <= 10
    assert bbox[2] >= 30 + 20  # max x = 13+20 = 33
```

- [ ] **Step 2: Write frame_extractor.py**

```python
import io
import tempfile
import os
import cv2
import numpy as np
from PIL import Image


def extract_frames(video_bytes: bytes, num_frames: int = 8) -> list[bytes]:
    """Extract evenly-spaced frames from video bytes. Returns list of PNG bytes."""
    # Write video bytes to temp file (cv2.VideoCapture needs a file path)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(video_bytes)
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []

        frames: list[bytes] = []
        for i in range(num_frames):
            frame_idx = int(i * total / num_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                frames.append(buf.getvalue())
        cap.release()
        return frames
    finally:
        os.unlink(tmp.name)


def get_consistent_bbox(frames_png: list[bytes]) -> tuple[int, int, int, int] | None:
    """Find the union bounding box of all non-transparent content across frames.
    Used after rembg to get a consistent crop region for all animation frames.
    """
    union_bbox = None
    for fbytes in frames_png:
        img = Image.open(io.BytesIO(fbytes)).convert("RGBA")
        bbox = img.getbbox()
        if bbox is None:
            continue
        if union_bbox is None:
            union_bbox = list(bbox)
        else:
            union_bbox[0] = min(union_bbox[0], bbox[0])
            union_bbox[1] = min(union_bbox[1], bbox[1])
            union_bbox[2] = max(union_bbox[2], bbox[2])
            union_bbox[3] = max(union_bbox[3], bbox[3])
    return tuple(union_bbox) if union_bbox else None


def frames_to_spritesheet(frames_png: list[bytes], layout: str = "horizontal") -> tuple[bytes, list[dict]]:
    """Pack PNG frame bytes into a spritesheet. Returns (spritesheet_png_bytes, frame_coords_list)."""
    if not frames_png:
        return b"", []

    pil_frames = [Image.open(io.BytesIO(f)).convert("RGBA") for f in frames_png]
    frame_w, frame_h = pil_frames[0].size
    coords: list[dict] = []

    if layout == "horizontal":
        total_w = frame_w * len(pil_frames)
        sheet = Image.new("RGBA", (total_w, frame_h), (0, 0, 0, 0))
        for i, frame in enumerate(pil_frames):
            x = i * frame_w
            sheet.paste(frame, (x, 0))
            coords.append({
                "name": f"frame_{i:02d}",
                "rect": {"x": x, "y": 0, "w": frame_w, "h": frame_h},
            })
    else:
        cols = min(len(pil_frames), 4)
        rows = (len(pil_frames) + cols - 1) // cols
        sheet = Image.new("RGBA", (cols * frame_w, rows * frame_h), (0, 0, 0, 0))
        for i, frame in enumerate(pil_frames):
            col = i % cols
            row = i // cols
            x = col * frame_w
            y = row * frame_h
            sheet.paste(frame, (x, y))
            coords.append({
                "name": f"frame_{i:02d}",
                "rect": {"x": x, "y": y, "w": frame_w, "h": frame_h},
            })

    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue(), coords
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_frame_extractor.py -v
```

- [ ] **Step 4: Commit**

---

### Task 7: Post-Processing + Session + Cache (combined for efficiency)

**Files:**
- Create: `postprocess.py`
- Create: `session.py`
- Create: `tests/test_postprocess.py`
- Create: `tests/test_session.py`

All three modules are independent. Write them in parallel.

**postprocess.py** — same as v1 plan Task 6 content (remove_bg, crop_to_content, grid_slice, pack_spritesheet, edge_wrap_tile).

**session.py** — same as v1 plan Task 7 content (SessionManager with cache, TTL, style ref/keyword storage).

- [ ] **Step 1: Write all files**
- [ ] **Step 2: Run all tests**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 3: Commit**

---

### Task 8: FastAPI Application (All Endpoints)

**Files:**
- Create: `main.py`

This integrates everything. Key differences from v1:
- Generation pipeline uses `dashscope_client.generate_txt2img` (not SDK)
- Animation pipeline uses `dashscope_client.generate_t2v` / `generate_i2v` → `frame_extractor.extract_frames` → `postprocess.remove_bg` on each frame → `frame_extractor.frames_to_spritesheet`
- Consistent bbox cropping via `frame_extractor.get_consistent_bbox`
- All sizes use the 4 valid wanx-v1 sizes
- Video resolution uses 480P/720P enums

```python
import io
import json
import uuid
import zipfile
import base64
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from models import (
    AssetType, GenerateResponse, JobStatus,
    JobStatusResponse, SessionCreateResponse, SessionAssetsResponse,
)
from prompts import build_prompt, build_video_prompt, PRESETS, AssetType as PromptAssetType
from deepseek import rewrite_prompt
from dashscope_client import generate_txt2img, generate_t2v, generate_i2v
from frame_extractor import extract_frames, frames_to_spritesheet, get_consistent_bbox
from postprocess import remove_bg, crop_to_content, grid_slice, pack_spritesheet, edge_wrap_tile
from session import SessionManager
from PIL import Image

session_manager = SessionManager()
jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_cleanup_loop())
    yield


async def _cleanup_loop():
    while True:
        await asyncio.sleep(300)
        session_manager.cleanup_expired()


app = FastAPI(title="2D Game Asset Generator", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _img_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --- Routes ---

@app.post("/api/session", response_model=SessionCreateResponse)
async def create_session():
    sid = session_manager.create_session()
    return SessionCreateResponse(session_id=sid)


@app.post("/api/style-ref")
async def upload_style_ref(session_id: str = Form(...), file: UploadFile = File(...)):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found or expired")
    content = await file.read()
    session_manager.set_style_ref(session_id, content)
    return {"status": "ok", "filename": file.filename}


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(
    session_id: str = Form(...),
    asset_type: str = Form(...),
    description: str = Form(...),
    style_keywords: str = Form(default="pixel art, game asset"),
    size: str = Form(default="1024*1024"),
    frames: int = Form(default=1),
    view: str = Form(default="front"),
    grid_size: int = Form(default=32),
    use_ai_rewrite: bool = Form(default=True),
    animation_mode: str = Form(default="t2v"),
):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    at = AssetType(asset_type)
    desc = rewrite_prompt(description, at.value) if use_ai_rewrite else description

    # Check cache
    cache_key = session_manager.make_cache_key(
        session_id, at.value, desc, style_keywords,
        int(size.split("*")[0]), frames, view, grid_size,
    )
    cached = session_manager.get_cache(cache_key)

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "status": JobStatus.PROCESSING,
        "asset_type": at,
        "description": desc,
        "cached": cached is not None,
        "cache_key": cache_key,
        "session_id": session_id,
        "style_keywords": style_keywords,
        "size": size,
        "frames": frames,
        "view": view,
        "grid_size": grid_size,
        "animation_mode": animation_mode,
    }
    asyncio.create_task(_run_generation(job_id))
    return GenerateResponse(job_id=job_id, status=JobStatus.PROCESSING)


async def _run_generation(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        return

    try:
        at = job["asset_type"]
        at_enum = PromptAssetType(at.value)
        ref_bytes = session_manager.get_style_ref(job["session_id"])

        if job["cached"]:
            data = session_manager.get_cache(job["cache_key"])
            if data:
                job["result_image"] = data
                job["status"] = JobStatus.DONE
                return

        needs_animation = job["frames"] > 1

        if needs_animation:
            # === Video-based animation pipeline ===
            mode = job.get("animation_mode", "t2v")

            if mode == "i2v" and ref_bytes:
                # i2v: generate base character first, then animate it
                base_prompt = build_prompt(at_enum, job["description"], job["style_keywords"], job["view"], 1, bool(ref_bytes))
                base_img = generate_txt2img(base_prompt, size=job["size"])
                if base_img is None:
                    job["status"] = JobStatus.FAILED
                    job["error"] = "Base image generation failed"
                    return
                base_img = remove_bg(base_img)
                video_prompt = build_video_prompt(at_enum, job["description"], job["style_keywords"], "i2v")
                video_bytes = generate_i2v(video_prompt, base_img)
            else:
                # t2v: text directly to video
                video_prompt = build_video_prompt(at_enum, job["description"], job["style_keywords"], "t2v")
                video_bytes = generate_t2v(video_prompt)

            if video_bytes is None:
                job["status"] = JobStatus.FAILED
                job["error"] = "Video generation failed"
                return

            # Extract frames
            frame_pngs = extract_frames(video_bytes, num_frames=job["frames"])

            # rembg each frame (skip for VFX — rembg eats translucent pixels)
            if at != AssetType.VFX:
                frame_pngs = [remove_bg(f) for f in frame_pngs]

            # Consistent crop across all frames
            bbox = get_consistent_bbox(frame_pngs)
            if bbox:
                cropped = []
                for f in frame_pngs:
                    img = Image.open(io.BytesIO(f)).convert("RGBA")
                    img = img.crop(bbox)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    cropped.append(buf.getvalue())
                frame_pngs = cropped

            # Pack into spritesheet
            sheet_bytes, frame_coords = frames_to_spritesheet(frame_pngs, layout="horizontal")

        else:
            # === Single image pipeline ===
            prompt = build_prompt(at_enum, job["description"], job["style_keywords"], job["view"], 1, bool(ref_bytes))
            img_bytes = generate_txt2img(prompt, size=job["size"])

            if img_bytes is None:
                job["status"] = JobStatus.FAILED
                job["error"] = "Image generation failed"
                return

            # Post-process based on asset type
            if at != AssetType.TILEMAP:
                img_bytes = remove_bg(img_bytes)

            img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

            if at == AssetType.TILEMAP:
                img = edge_wrap_tile(img)
            elif at in (AssetType.CHARACTER, AssetType.PROPS):
                img = crop_to_content(img)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            sheet_bytes = buf.getvalue()
            frame_coords = [{
                "name": "frame_00",
                "rect": {"x": 0, "y": 0, "w": img.width, "h": img.height},
            }]

        # Save result
        job["result_image"] = sheet_bytes
        job["result_json"] = json.dumps({
            "version": "1.0",
            "meta": {
                "type": "spritesheet" if len(frame_coords) > 1 else "single",
                "frame_count": len(frame_coords),
                "generated_at": "",
            },
            "frames": frame_coords,
        }, indent=2)

        session_manager.set_cache(job["cache_key"], sheet_bytes)
        session_manager.add_asset(
            job["session_id"], job_id,
            at.value, job["description"], job["frames"],
        )
        job["status"] = JobStatus.DONE

    except Exception as e:
        job["status"] = JobStatus.FAILED
        job["error"] = str(e)


@app.get("/api/generate/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    resp = JobStatusResponse(job_id=job_id, status=job["status"], error=job.get("error"))
    if job["status"] == JobStatus.DONE:
        resp.image_base64 = base64.b64encode(job.get("result_image", b"")).decode("utf-8")
        resp.json_metadata = job.get("result_json")
    return resp


@app.get("/api/download/{job_id}")
async def download_asset(job_id: str):
    job = jobs.get(job_id)
    if job is None or job["status"] != JobStatus.DONE:
        raise HTTPException(404, "Asset not found or not ready")
    base_name = f"{job['asset_type'].value}_{job_id}"
    files = {
        f"{base_name}.png": job["result_image"],
        f"{base_name}.json": job.get("result_json", "{}").encode("utf-8"),
    }
    zip_data = _make_zip(files)
    return StreamingResponse(io.BytesIO(zip_data), media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename={base_name}.zip"})


@app.get("/api/session/assets", response_model=SessionAssetsResponse)
async def list_assets(session_id: str):
    assets = session_manager.get_assets(session_id)
    return SessionAssetsResponse(session_id=session_id, assets=assets)


@app.post("/api/session/download-all")
async def download_all(session_id: str = Form(...)):
    assets = session_manager.get_assets(session_id)
    if not assets:
        raise HTTPException(404, "No assets found")
    all_files: dict[str, bytes] = {}
    for asset in assets:
        job = jobs.get(asset["job_id"])
        if job and job["status"] == JobStatus.DONE:
            base = f"{asset['asset_type']}_{asset['job_id']}"
            all_files[f"{base}.png"] = job["result_image"]
            all_files[f"{base}.json"] = job.get("result_json", "{}").encode("utf-8")
    if not all_files:
        raise HTTPException(404, "No completed assets")
    zip_data = _make_zip(all_files)
    return StreamingResponse(io.BytesIO(zip_data), media_type="application/zip",
                             headers={"Content-Disposition": "attachment; filename=all_assets.zip"})


@app.get("/api/presets")
async def get_presets():
    result = {}
    for at, preset_list in PRESETS.items():
        result[at.value] = preset_list
    return result


app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

- [ ] **Step 1: Verify app loads**

```bash
python -c "from main import app; print('App loaded:', app.title)"
```

- [ ] **Step 2: Commit**

---

### Task 9: Backend Integration Smoke Test

- [ ] **Step 1: Start server**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 &
sleep 2
```

- [ ] **Step 2: Test endpoints**

```bash
# Session
SID=$(curl -s -X POST http://localhost:8000/api/session | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "Session: $SID"

# Presets
curl -s http://localhost:8000/api/presets | python -c "import sys,json; d=json.load(sys.stdin); print(f'Presets: {sum(len(v) for v in d.values())} total')"

# Single image generation
curl -s -X POST "http://localhost:8000/api/generate" \
  -F "session_id=$SID" \
  -F "asset_type=props" \
  -F "description=a golden sword" \
  -F "frames=1" \
  -F "size=1024*1024" | python -m json.tool
```

- [ ] **Step 3: Poll and verify the result downloads**
- [ ] **Step 4: Stop server, commit any fixes**

---

## Day 2: Frontend + Final Integration (3 tasks)

### Task 10: Single-Page HTML/CSS/JS Frontend

**Files:**
- Create: `static/index.html`

Same structure as v1 plan Task 10, with additions:
- Animation mode selector (t2v / i2v) visible when frames > 1
- Flipbook player for multi-frame results
- Tile tiling preview grid
- Preset dropdown grouped by type
- Cache badge indicators
- Download per-asset and download-all buttons

No canon skinning toggle (cut for time).

- [ ] **Step 1: Write complete index.html** (~400 lines, same structure as v1 plan)
- [ ] **Step 2: Verify page loads at http://localhost:8000**
- [ ] **Step 3: Commit**

---

### Task 11: End-to-End Verification

- [ ] **Step 1: Start server, generate one of each asset type**
- [ ] **Step 2: Generate a multi-frame character animation (i2v mode)**
- [ ] **Step 3: Generate a VFX animation (t2v mode)**
- [ ] **Step 4: Verify all downloads produce valid PNG + JSON zip files**
- [ ] **Step 5: Commit final polish**

---

## Implementation Notes

- **Start server:** `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
- **API keys:** already configured in `.env`
- **Run tests:** `python -m pytest tests/ -v`
- **opencv-python install:** `pip install opencv-python` (already in requirements.txt)
- **rembg first run** downloads the ONNX model (~176MB), allow time on first execution
- **Video generation:** t2v ~30s, i2v ~160s. Frontend polling interval should accommodate this.
