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
            "label": "骑士挥砍攻击（4帧）",
            "description": "身穿铠甲的骑士双手挥剑横斩",
            "frames": 4,
            "animation_mode": "t2v",
        },
        {
            "label": "史莱姆弹跳（4帧）",
            "description": "可爱的Q版史莱姆上下弹跳",
            "frames": 4,
            "animation_mode": "t2v",
        },
        {
            "label": "中世纪村民NPC",
            "description": "中世纪风格的村民NPC，休闲服装，站立姿势",
            "frames": 1,
        },
        {
            "label": "像素猫跑动（4帧）",
            "description": "像素艺术风格的猫咪横版跑动循环",
            "frames": 4,
            "animation_mode": "t2v",
        },
    ],
    AssetType.TILEMAP: [
        {
            "label": "草地地表（俯视RPG）",
            "description": "绿色草地地表瓦片，俯视视角RPG风格",
            "grid_size": 32,
            "view": "top_down",
        },
        {
            "label": "地牢石板地面",
            "description": "深色石板地面瓦片，阴暗地牢风格",
            "grid_size": 32,
            "view": "top_down",
        },
        {
            "label": "太空平台（侧视图）",
            "description": "科幻金属平台瓦片，侧视图",
            "grid_size": 64,
            "view": "side",
        },
        {
            "label": "森林花丛小径",
            "description": "森林地面瓦片，点缀小花",
            "grid_size": 32,
            "view": "top_down",
        },
    ],
    AssetType.UI: [
        {
            "label": "奇幻风格血条",
            "description": "奇幻风格的血条，镶嵌红色宝石装饰",
            "view": "front",
        },
        {
            "label": "科幻按钮面板",
            "description": "科幻风格的蓝色全息按钮面板",
            "view": "front",
        },
        {
            "label": "木质物品栏格子",
            "description": "中世纪风格的木质物品栏格子",
            "view": "front",
        },
        {
            "label": "金币图标",
            "description": "游戏内金币货币图标",
            "view": "front",
        },
    ],
    AssetType.PROPS: [
        {
            "label": "蓝光铁长剑",
            "description": "铁质长剑，散发魔法蓝光",
            "view": "front",
        },
        {
            "label": "红色生命药水",
            "description": "圆形玻璃瓶装的红色生命恢复药水",
            "view": "front",
        },
        {
            "label": "金色宝箱",
            "description": "带华丽纹饰的黄金宝箱",
            "view": "front",
        },
        {
            "label": "紫晶魔法杖",
            "description": "木制法杖，顶端镶嵌发光紫色水晶",
            "view": "front",
        },
    ],
    AssetType.VFX: [
        {
            "label": "火焰爆炸（8帧）",
            "description": "向外扩散的火焰爆炸特效",
            "frames": 8,
            "animation_mode": "t2v",
        },
        {
            "label": "剑气斩击弧（4帧）",
            "description": "刀剑挥砍的弧形斩击轨迹",
            "frames": 4,
            "animation_mode": "t2v",
        },
        {
            "label": "魔法阵召唤（6帧）",
            "description": "发光的符文魔法阵逐渐显现",
            "frames": 6,
            "animation_mode": "t2v",
        },
        {
            "label": "烟雾消散（4帧）",
            "description": "卡通风格的小团烟雾逐渐消散",
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
