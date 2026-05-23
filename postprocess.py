import io
import os
from PIL import Image
from rembg import remove

# Path to the onnx model that rembg needs
_u2net_home = os.path.expanduser(os.environ.get(
    "U2NET_HOME", "~/.u2net"
))
_u2net_path = os.path.join(_u2net_home, "u2net.onnx")


def remove_bg(image_bytes: bytes) -> bytes:
    """Remove background using rembg. Returns PNG bytes.
    Falls back to original image if rembg model not available.
    """
    if not os.path.exists(_u2net_path):
        return image_bytes
    try:
        return remove(image_bytes)
    except Exception:
        return image_bytes


def crop_to_content(image: Image.Image) -> Image.Image:
    """Crop image to non-transparent content bounds."""
    bbox = image.getbbox()
    if bbox is None:
        return image
    return image.crop(bbox)


def grid_slice(image: Image.Image, tile_w: int, tile_h: int) -> list[Image.Image]:
    """Slice an image into a grid of tiles. Partial edge tiles are dropped."""
    tiles: list[Image.Image] = []
    for y in range(0, image.height - tile_h + 1, tile_h):
        for x in range(0, image.width - tile_w + 1, tile_w):
            tile = image.crop((x, y, x + tile_w, y + tile_h))
            tiles.append(tile)
    return tiles


def pack_spritesheet(
    frames: list[Image.Image], layout: str = "horizontal"
) -> tuple[Image.Image, list[dict]]:
    """Pack frames into a single spritesheet image.

    Returns (spritesheet_image, frame_metadata_list).
    """
    if not frames:
        return Image.new("RGBA", (1, 1)), []

    frame_w, frame_h = frames[0].size
    coords: list[dict] = []

    if layout == "horizontal":
        total_w = frame_w * len(frames)
        sheet = Image.new("RGBA", (total_w, frame_h), (0, 0, 0, 0))
        for i, frame in enumerate(frames):
            x = i * frame_w
            sheet.paste(frame, (x, 0))
            coords.append({
                "name": f"frame_{i:02d}",
                "rect": {"x": x, "y": 0, "w": frame_w, "h": frame_h},
            })
    else:
        cols = min(len(frames), 4)
        rows = (len(frames) + cols - 1) // cols
        sheet = Image.new(
            "RGBA", (cols * frame_w, rows * frame_h), (0, 0, 0, 0)
        )
        for i, frame in enumerate(frames):
            col = i % cols
            row = i // cols
            x = col * frame_w
            y = row * frame_h
            sheet.paste(frame, (x, y))
            coords.append({
                "name": f"frame_{i:02d}",
                "rect": {"x": x, "y": y, "w": frame_w, "h": frame_h},
            })

    return sheet, coords


def edge_wrap_tile(image: Image.Image) -> Image.Image:
    """Apply mirror-edge wrapping to reduce visible seams when tiling.

    Creates a 3x3 mirrored grid of the tile and blends the center back,
    producing a tile that tiles more seamlessly.
    """
    w, h = image.size
    big = Image.new("RGBA", (w * 3, h * 3))
    big.paste(image, (w, h))
    big.paste(image.transpose(Image.FLIP_LEFT_RIGHT), (0, h))
    big.paste(image.transpose(Image.FLIP_LEFT_RIGHT), (w * 2, h))
    top_row = big.crop((0, h, w * 3, h * 2))
    big.paste(top_row.transpose(Image.FLIP_TOP_BOTTOM), (0, 0))
    big.paste(top_row.transpose(Image.FLIP_TOP_BOTTOM), (0, h * 2))
    result = big.crop((w, h, w * 2, h * 2))
    result = Image.blend(image, result, 0.5)
    return result
