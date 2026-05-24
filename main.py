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
from PIL import Image

from models import (
    AssetType,
    GenerateResponse,
    JobStatus,
    JobStatusResponse,
    SessionCreateResponse,
    SessionAssetsResponse,
)
from prompts import build_prompt, build_video_prompt, PRESETS
from prompts import AssetType as PromptAssetType
from deepseek import rewrite_prompt
from dashscope_client import generate_txt2img, generate_t2v, generate_i2v
from frame_extractor import extract_frames, frames_to_spritesheet, get_consistent_bbox
from postprocess import remove_bg, crop_to_content, edge_wrap_tile, resize_for_game, GAME_SIZES
from session import SessionManager

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


# ============================================================
# Routes
# ============================================================

@app.post("/api/session", response_model=SessionCreateResponse)
async def create_session():
    sid = session_manager.create_session()
    return SessionCreateResponse(session_id=sid)


@app.post("/api/style-ref")
async def upload_style_ref(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
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
        raise HTTPException(404, "Session not found or expired")

    at = AssetType(asset_type)
    desc = rewrite_prompt(description, at.value) if use_ai_rewrite else description

    size_px = int(size.split("*")[0])
    cache_key = session_manager.make_cache_key(
        session_id, at.value, desc, style_keywords,
        size_px, frames, view, grid_size,
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
        has_ref = ref_bytes is not None

        if job["cached"]:
            data = session_manager.get_cache(job["cache_key"])
            if data:
                job["result_image"] = data
                job["status"] = JobStatus.DONE
                return

        if job["frames"] > 1:
            # === Video-based animation pipeline ===
            mode = job.get("animation_mode", "t2v")
            video_prompt = build_video_prompt(
                at_enum, job["description"], job["style_keywords"], mode
            )

            if mode == "i2v":
                # Generate base sprite first, then animate it
                base_prompt = build_prompt(
                    at_enum, job["description"],
                    job["style_keywords"], job["view"], 1, has_ref
                )
                base_img = generate_txt2img(base_prompt, size=job["size"])
                if base_img is None:
                    job["status"] = JobStatus.FAILED
                    job["error"] = "Base image generation failed"
                    return
                base_img = remove_bg(base_img)
                video_bytes = generate_i2v(video_prompt, base_img)
            else:
                video_bytes = generate_t2v(video_prompt)

            if video_bytes is None:
                job["status"] = JobStatus.FAILED
                job["error"] = "Video generation failed"
                return

            frame_pngs = extract_frames(video_bytes, num_frames=job["frames"])

            if at != AssetType.VFX:
                frame_pngs = [remove_bg(f) for f in frame_pngs]

            bbox = get_consistent_bbox(frame_pngs)
            if bbox:
                cropped_frames = []
                for f in frame_pngs:
                    img = Image.open(io.BytesIO(f)).convert("RGBA")
                    img = img.crop(bbox)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    cropped_frames.append(buf.getvalue())
                frame_pngs = cropped_frames

            sheet_bytes, frame_coords = frames_to_spritesheet(
                frame_pngs, layout="horizontal"
            )
        else:
            # === Single image pipeline ===
            prompt = build_prompt(
                at_enum, job["description"],
                job["style_keywords"], job["view"], 1, has_ref
            )
            img_bytes = generate_txt2img(prompt, size=job["size"])

            if img_bytes is None:
                job["status"] = JobStatus.FAILED
                job["error"] = "Image generation failed"
                return

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

        job["result_image"] = sheet_bytes
        is_spritesheet = len(frame_coords) > 1
        job["result_json"] = json.dumps({
            "version": "1.0",
            "meta": {
                "type": "spritesheet" if is_spritesheet else "single",
                "frame_count": len(frame_coords),
                "fps": 12 if is_spritesheet else 0,
                "generated_at": "",
                "available_sizes": list(GAME_SIZES),
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

    resp = JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        error=job.get("error"),
    )
    if job["status"] == JobStatus.DONE:
        resp.image_base64 = base64.b64encode(
            job.get("result_image", b"")
        ).decode("utf-8")
        resp.json_metadata = job.get("result_json")
    return resp


@app.get("/api/download/{job_id}")
async def download_asset(job_id: str):
    job = jobs.get(job_id)
    if job is None or job["status"] != JobStatus.DONE:
        raise HTTPException(404, "Asset not found or not ready")

    safe_desc = "".join(c for c in job.get("description", "asset")[:20] if c.isalnum() or c in " _-").strip().replace(" ", "_")
    base_name = f"{job['asset_type'].value}_{safe_desc}_{job_id}"
    original_img = Image.open(io.BytesIO(job["result_image"]))
    files: dict[str, bytes] = {}
    # Original
    files[f"{base_name}.png"] = job["result_image"]
    files[f"{base_name}.json"] = job.get("result_json", "{}").encode("utf-8")
    # Game-size variants
    variants = resize_for_game(original_img)
    for size, data in variants.items():
        files[f"{base_name}_{size}px.png"] = data

    zip_data = _make_zip(files)
    return StreamingResponse(
        io.BytesIO(zip_data),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={base_name}.zip"
        },
    )


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
            safe_desc = "".join(c for c in job.get("description", "asset")[:20] if c.isalnum() or c in " _-").strip().replace(" ", "_")
            base = f"{asset['asset_type']}_{safe_desc}_{asset['job_id']}"
            all_files[f"{base}.png"] = job["result_image"]
            all_files[f"{base}.json"] = (
                job.get("result_json", "{}").encode("utf-8")
            )
            # Game-size variants
            orig_img = Image.open(io.BytesIO(job["result_image"]))
            for size, data in resize_for_game(orig_img).items():
                all_files[f"{base}_{size}px.png"] = data

    if not all_files:
        raise HTTPException(404, "No completed assets to download")

    zip_data = _make_zip(all_files)
    return StreamingResponse(
        io.BytesIO(zip_data),
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=all_assets.zip"
        },
    )


@app.get("/api/presets")
async def get_presets():
    result = {}
    for at, preset_list in PRESETS.items():
        result[at.value] = preset_list
    return result


app.mount("/", StaticFiles(directory="static", html=True), name="static")
