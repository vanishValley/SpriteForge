import io
import os
import tempfile
from typing import Optional
import cv2
from PIL import Image


def extract_frames(video_bytes: bytes, num_frames: int = 8) -> list[bytes]:
    """Extract evenly-spaced frames from video bytes. Returns list of PNG bytes."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(video_bytes)
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        frames: list[bytes] = []
        for i in range(num_frames):
            frame_idx = int(i * total / num_frames)
            if frame_idx >= total:
                frame_idx = total - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                frames.append(buf.getvalue())

        cap.release()
        return frames
    finally:
        os.unlink(tmp.name)


def get_consistent_bbox(frames_png: list[bytes]) -> Optional[tuple[int, int, int, int]]:
    """Find the union bounding box of non-transparent content across all frames.

    Used after rembg to get a consistent crop region for all animation frames,
    so the character stays in the same position across the spritesheet.
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


def frames_to_spritesheet(
    frames_png: list[bytes], layout: str = "horizontal"
) -> tuple[bytes, list[dict]]:
    """Pack PNG frame bytes into a spritesheet.

    Returns (spritesheet_png_bytes, frame_coords_list).
    """
    if not frames_png:
        return b"", []

    pil_frames = [
        Image.open(io.BytesIO(f)).convert("RGBA") for f in frames_png
    ]
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
        sheet = Image.new(
            "RGBA", (cols * frame_w, rows * frame_h), (0, 0, 0, 0)
        )
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
