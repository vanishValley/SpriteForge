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

QUALITY_PREFIX = (
    "masterpiece, best quality, highly detailed, "
    "game-ready asset, clean edges, "
)

VIEW_LABELS = {
    "front": "front view",
    "side": "side view",
    "top_down": "top-down view",
}

# Video prompt templates for animation generation
VIDEO_PROMPTS: dict[AssetType, dict[str, str]] = {
    AssetType.CHARACTER: {
        "t2v": (
            "2D pixel art game character animation, {description}, "
            "full body, smooth motion, clean white background, "
            "game sprite animation sequence, {style_keywords}, "
            "consistent character design throughout"
        ),
        "i2v": (
            "the character performs {description}, "
            "smooth fluid motion, consistent character appearance, "
            "pixel art game sprite animation, clean white background, "
            "{style_keywords}"
        ),
    },
    AssetType.VFX: {
        "t2v": (
            "2D game visual effect animation, {description}, "
            "smooth motion sequence, dark background, "
            "game VFX sprite sheet source, {style_keywords}"
        ),
        "i2v": (
            "the effect animates with {description}, "
            "smooth fluid motion, consistent style, "
            "game VFX animation, dark background, {style_keywords}"
        ),
    },
}

PRESETS: dict[AssetType, list[dict]] = {
    AssetType.CHARACTER: [
        {
            "label": "Knight sword slash (4 frames)",
            "description": "knight swinging a sword in a horizontal slash attack",
            "frames": 4,
            "animation_mode": "i2v",
        },
        {
            "label": "Slime monster bounce (4 frames)",
            "description": "cute slime monster bouncing up and down",
            "frames": 4,
            "animation_mode": "t2v",
        },
        {
            "label": "NPC villager idle",
            "description": "medieval villager NPC, casual clothes, standing idle pose",
            "frames": 1,
        },
        {
            "label": "Pixel cat run cycle (4 frames)",
            "description": "pixel art cat running cycle for platformer game",
            "frames": 4,
            "animation_mode": "t2v",
        },
    ],
    AssetType.TILEMAP: [
        {
            "label": "Grass field top-down RPG",
            "description": "green grass field tile for top-down RPG",
            "grid_size": 32,
            "view": "top_down",
        },
        {
            "label": "Dungeon stone floor",
            "description": "dark stone floor tile for dungeon",
            "grid_size": 32,
            "view": "top_down",
        },
        {
            "label": "Space platform side-view",
            "description": "sci-fi metal platform tile side-view",
            "grid_size": 64,
            "view": "side",
        },
        {
            "label": "Forest path with flowers",
            "description": "forest ground tile with small flowers",
            "grid_size": 32,
            "view": "top_down",
        },
    ],
    AssetType.UI: [
        {
            "label": "Fantasy health bar with gem",
            "description": "fantasy themed health bar with red gem decorations",
            "view": "front",
        },
        {
            "label": "Sci-fi button panel",
            "description": "sci-fi holographic button panel, blue glow",
            "view": "front",
        },
        {
            "label": "Wooden inventory slot",
            "description": "wooden framed inventory slot, medieval style",
            "view": "front",
        },
        {
            "label": "Gold coin icon",
            "description": "gold coin icon for in-game currency",
            "view": "front",
        },
    ],
    AssetType.PROPS: [
        {
            "label": "Iron longsword with glow",
            "description": "iron longsword with magical blue glow",
            "view": "front",
        },
        {
            "label": "Red health potion bottle",
            "description": "red health potion in a round glass bottle",
            "view": "front",
        },
        {
            "label": "Golden treasure chest",
            "description": "golden treasure chest with ornate details",
            "view": "front",
        },
        {
            "label": "Magic staff with crystal",
            "description": "wooden magic staff with glowing purple crystal on top",
            "view": "front",
        },
    ],
    AssetType.VFX: [
        {
            "label": "Fire explosion (8 frames)",
            "description": "fire explosion effect expanding outward",
            "frames": 8,
            "animation_mode": "t2v",
        },
        {
            "label": "Sword slash arc (4 frames)",
            "description": "sword slash arc trail effect",
            "frames": 4,
            "animation_mode": "t2v",
        },
        {
            "label": "Magic circle summon (6 frames)",
            "description": "magic circle appearing with glowing runes",
            "frames": 6,
            "animation_mode": "t2v",
        },
        {
            "label": "Smoke puff (4 frames)",
            "description": "small cartoon smoke puff dissipating",
            "frames": 4,
            "animation_mode": "t2v",
        },
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
    """Build a prompt for single-frame image generation (wanx-v1 txt2img)."""
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
    """Build a prompt for video generation (t2v or i2v mode)."""
    templates = VIDEO_PROMPTS.get(asset_type, VIDEO_PROMPTS[AssetType.CHARACTER])
    template = templates.get(mode, templates["t2v"])
    return template.format(
        description=description,
        style_keywords=style_keywords,
    )[:800]
