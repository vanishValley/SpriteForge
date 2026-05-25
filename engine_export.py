"""Engine-specific export helpers.

Generates resource files that can be directly imported into:
- Godot 4.x (SpriteFrames .tres)
- Unity (sprite slice data)
- GameMaker (sprite import notes)
"""

import json


def generate_godot_tres(
    png_filename: str,
    frame_coords: list[dict],
    fps: int = 12,
) -> str:
    """Generate a Godot 4.x SpriteFrames .tres resource file.

    Usage: Place .tres next to the .png in your Godot project.
    Godot auto-imports the texture; the .tres references it as an
    AnimatedSprite2D-ready SpriteFrames resource.
    """
    if len(frame_coords) <= 1:
        return ""

    frames_block = []
    for fc in frame_coords:
        r = fc["rect"]
        frames_block.append(
            '{\n'
            f'    "duration": {1.0 / fps:.3f},\n'
            f'    "region": Rect2({r["x"]}, {r["y"]}, {r["w"]}, {r["h"]})\n'
            '  }'
        )
    frames_str = ", ".join(frames_block)

    return (
        f'[gd_resource type="SpriteFrames" load_steps=2 format=3]\n'
        f'\n'
        f'[ext_resource type="Texture2D" path="res://{png_filename}" id="1_tex"]\n'
        f'\n'
        f'[resource]\n'
        f'animations = [{{\n'
        f'"frames": [{frames_str}],\n'
        f'"loop": true,\n'
        f'"name": &"default",\n'
        f'"speed": {float(fps):.1f}\n'
        f'}}]\n'
    )


def generate_unity_sprite_json(
    frame_coords: list[dict],
    image_width: int,
    image_height: int,
) -> str:
    """Generate sprite slice metadata for Unity Sprite Editor.

    Unity can import sprite sheets via Sprite Editor's Grid By Cell Count
    or auto-slice. This JSON documents the exact frame rects for manual
    setup instructions.
    """
    slices = []
    for fc in frame_coords:
        r = fc["rect"]
        slices.append({
            "name": fc["name"],
            "x": r["x"],
            "y": r["y"],
            "width": r["w"],
            "height": r["h"],
        })

    return json.dumps({
        "unity_sprite_sheet": {
            "image_width": image_width,
            "image_height": image_height,
            "pixels_per_unit": 32,
            "filter_mode": "Point (no filter)",
            "slices": slices,
            "usage": (
                "1. Import this PNG into Unity (drag to Assets)\n"
                "2. Select the texture, set Sprite Mode = Multiple\n"
                "3. Open Sprite Editor, use Slice > Grid By Cell Count\n"
                "4. Set column count to match the frame count\n"
                "5. Click Apply"
            ),
        },
    }, indent=2)


def generate_engine_readme(
    frame_count: int,
    fps: int,
) -> str:
    """Generate engine import instructions."""
    if frame_count <= 1:
        return (
            "单个精灵图片，可直接拖入任意引擎使用。\n\n"
            "=== Unity ===\n"
            "拖入 Assets 文件夹 → Texture Type 改为 Sprite (2D and UI)\n\n"
            "=== Godot ===\n"
            "拖入文件系统 → 右键图片选 New SpriteFrames\n\n"
            "=== GameMaker ===\n"
            "右键 Sprites → Create Sprite → Import 选择图片"
        )

    return (
        f"多帧精灵表 ({frame_count} 帧, {fps} FPS)\n\n"
        f"=== Unity ===\n"
        f"1. 拖 PNG 到 Assets 文件夹\n"
        f"2. 选中纹理 → Sprite Mode 改为 Multiple\n"
        f"3. Sprite Editor → Slice → Grid By Cell Count\n"
        f"4. 拖入 Scene → 自动创建 Animation Clip\n"
        f"提示: 查看 unity_sprite_slices.json 获取切图坐标\n\n"
        f"=== Godot 4.x ===\n"
        f"1. 把 PNG 和 .tres 文件一起拖入 Godot 项目\n"
        f"2. 创建 AnimatedSprite2D 节点\n"
        f"3. 在 Inspector 中把 .tres 拖到 SpriteFrames 属性\n"
        f"4. AnimationPlayer 自动创建 default 动画\n\n"
        f"=== GameMaker ===\n"
        f"1. 右键 Sprites → Create Sprite → Edit Image\n"
        f"2. Import Strip Image → 选择 PNG\n"
        f"3. 设置 Number of Frames = {frame_count}, Frames Per Row = {frame_count}"
    )
