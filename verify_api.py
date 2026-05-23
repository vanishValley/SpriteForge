"""
API 可行性验证脚本
验证通义万象 DashScope API 的实际能力，确定方案哪些可行哪些不可行。

验证项目:
1. txt2img 基础生成 (wanx-v1)
2. 相同 seed 的风格一致性
3. style-repaint 是否可用于通用 img2img
4. 确认可用的模型列表
"""

import os
import sys
import time
import json
import base64
from typing import Optional
import requests
from pathlib import Path

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = "https://dashscope.aliyuncs.com/api/v1"

# wanx-v1 仅支持这 4 种尺寸
VALID_SIZES = ["1024*1024", "720*1280", "1280*720", "768*1152"]
DEFAULT_SIZE = "1024*1024"


def _session():
    """Create a requests session that bypasses system proxy."""
    s = requests.Session()
    s.trust_env = False
    return s

if not API_KEY:
    print("=" * 60)
    print("FAIL: 请先设置 DASHSCOPE_API_KEY 环境变量")
    print("=" * 60)
    print("\n方式 1 (临时):")
    print("  export DASHSCOPE_API_KEY=sk-xxx")
    print("\n方式 2 (创建 .env 文件):")
    print("  在项目根目录创建 .env 文件，内容:")
    print("  DASHSCOPE_API_KEY=sk-xxx")
    print("\n然后重新运行: python verify_api.py")
    sys.exit(0)


def create_task(model: str, payload: dict, async_mode: bool = True) -> dict:
    """创建异步任务"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    if async_mode:
        headers["X-DashScope-Async"] = "enable"

    url = f"{BASE_URL}/services/aigc/text2image/image-synthesis"

    body = {
        "model": model,
        "input": payload.get("input", {}),
        "parameters": payload.get("parameters", {}),
    }

    print(f"  → POST {url}")
    print(f"  → Model: {model}")
    print(f"  → Prompt: {payload.get('input', {}).get('prompt', '')[:80]}...")

    resp = _session().post(url, headers=headers, json=body, timeout=30)
    data = resp.json()

    if resp.status_code != 200:
        print(f"  ← HTTP {resp.status_code}: {data}")
        return {"error": data}

    task_id = data.get("output", {}).get("task_id", "")
    print(f"  ← Task ID: {task_id}")
    return {"task_id": task_id, "raw": data}


def poll_task(task_id: str, max_wait: int = 60) -> dict:
    """轮询任务结果"""
    url = f"{BASE_URL}/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    start = time.time()
    while time.time() - start < max_wait:
        resp = _session().get(url, headers=headers, timeout=10)
        data = resp.json()
        status = data.get("output", {}).get("task_status", "UNKNOWN")
        print(f"  [.] Status: {status} ({time.time() - start:.0f}s)")

        if status == "SUCCEEDED":
            results = data.get("output", {}).get("results", [])
            if results:
                print(f"  OK: Done! Image URL: {results[0].get('url', 'N/A')[:80]}...")
            return data
        elif status == "FAILED":
            print(f"  FAIL: Failed: {data}")
            return data

        time.sleep(2)

    print(f"  TIMEOUT: Timeout after {max_wait}s")
    return {"error": "timeout", "task_status": "UNKNOWN"}


def download_image(url: str) -> Optional[bytes]:
    """下载生成的图片"""
    if not url:
        return None
    try:
        resp = _session().get(url, timeout=30)
        return resp.content
    except Exception as e:
        print(f"  WARN: Download failed: {e}")
        return None


def save_image(data: bytes, filename: str):
    """保存图片到 output 目录"""
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / filename
    path.write_bytes(data)
    print(f"  SAVED: Saved: {path}")


# ============================================================
# Test 1: txt2img 基础生成
# ============================================================
def test_txt2img():
    print("\n" + "=" * 60)
    print("Test 1: txt2img 基础生成 (wanx-v1)")
    print("=" * 60)

    result = create_task("wanx-v1", {
        "input": {
            "prompt": "2D game character sprite, a brave knight in silver armor, "
                      "full body, standing pose, clean silhouette, pixel art style, "
                      "game-ready asset",
        },
        "parameters": {
            "size": DEFAULT_SIZE,
            "n": 1,
        },
    })

    if "error" in result:
        return None

    task_result = poll_task(result["task_id"])
    results = task_result.get("output", {}).get("results", [])
    if results:
        img = download_image(results[0].get("url"))
        if img:
            save_image(img, "test1_txt2img_knight.png")
            return img
    return None


# ============================================================
# Test 2: Seed 一致性 — 相同 seed + 相同 prompt
# ============================================================
def test_seed_consistency():
    print("\n" + "=" * 60)
    print("Test 2: Seed 一致性 — 相同 seed 生成 2 张图")
    print("=" * 60)

    prompt = "2D game item, red health potion, pixel art, game inventory asset"
    seed = 42

    for i in range(2):
        print(f"\n  --- Generation {i+1} ---")
        result = create_task("wanx-v1", {
            "input": {"prompt": prompt},
            "parameters": {"size": DEFAULT_SIZE, "n": 1, "seed": seed},
        })
        if "error" in result:
            continue
        task_result = poll_task(result["task_id"])
        results = task_result.get("output", {}).get("results", [])
        if results:
            img = download_image(results[0].get("url"))
            if img:
                save_image(img, f"test2_seed_{i+1}.png")
                print(f"  SIZE: Size: {len(img)} bytes")


# ============================================================
# Test 3: 不同 seed 生成多个 sprite 帧
# ============================================================
def test_multi_frame_seed():
    print("\n" + "=" * 60)
    print("Test 3: 多帧生成 — 相同 prompt，不同 seed")
    print("=" * 60)

    base_prompt = ("2D game character sprite, warrior attack animation, "
                   "pixel art, game-ready asset, full body")

    for i in range(4):
        seed = 100 + i
        print(f"\n  --- Frame {i+1} (seed={seed}) ---")
        result = create_task("wanx-v1", {
            "input": {"prompt": base_prompt},
            "parameters": {"size": DEFAULT_SIZE, "n": 1, "seed": seed},
        })
        if "error" in result:
            continue
        task_result = poll_task(result["task_id"])
        results = task_result.get("output", {}).get("results", [])
        if results:
            img = download_image(results[0].get("url"))
            if img:
                save_image(img, f"test3_frame_{i+1}.png")


# ============================================================
# Test 4: txt2img 不同素材类型
# ============================================================
def test_asset_types():
    print("\n" + "=" * 60)
    print("Test 4: 不同素材类型生成")
    print("=" * 60)

    tests = [
        ("Tilemap", "2D game environment tile, grass field top-down RPG, "
                    "pixel art, seamlessly tileable, flat composition, game environment art"),
        ("UI", "2D game UI element, fantasy health bar with red gem, "
               "clean vector-like, minimal design, game interface asset"),
        ("VFX", "2D game visual effect, fire explosion sprite sheet, "
                "4 frames arranged in sequence, pixel art, game VFX asset"),
    ]

    for label, prompt in tests:
        print(f"\n  --- {label} ---")
        result = create_task("wanx-v1", {
            "input": {"prompt": prompt},
            "parameters": {"size": DEFAULT_SIZE, "n": 1},
        })
        if "error" in result:
            continue
        task_result = poll_task(result["task_id"])
        results = task_result.get("output", {}).get("results", [])
        if results:
            img = download_image(results[0].get("url"))
            if img:
                save_image(img, f"test4_{label.lower()}.png")


# ============================================================
# Test 5: 测试 style-repaint 是否可用于通用 img2img
# ============================================================
def test_style_repaint():
    print("\n" + "=" * 60)
    print("Test 5: style-repaint-v1 — 能否做通用 img2img？")
    print("=" * 60)

    # wanx-style-repaint-v1 的 endpoint 不同
    url = f"{BASE_URL}/services/aigc/image-generation/generation"

    # 先用 txt2img 生成一个 base image 作为 "输入图"
    print("\n  步骤1: 生成 base image (txt2img)...")
    result = create_task("wanx-v1", {
        "input": {"prompt": "a simple cartoon character, full body, white background"},
        "parameters": {"size": DEFAULT_SIZE, "n": 1},
    })
    if "error" in result:
        print("  FAIL: 无法生成 base image，跳过此测试")
        return

    task_result = poll_task(result["task_id"])
    results = task_result.get("output", {}).get("results", [])
    if not results:
        return
    base_url = results[0].get("url", "")

    # 步骤2: 用 style-repaint，以 base image 作为输入，尝试改风格
    # 关键问题: style-repaint 输入图的要求
    #   1. 必须是人像图? 还是任意图?
    #   2. style_ref_url 是风格参考, input image 是原图
    #   3. 如果是人像专用的, 非人像图会怎样?
    print("\n  步骤2: style-repaint 重绘 (人像模式)...")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }

    # Test 5a: 人像重绘 (预期能工作)
    body_a = {
        "model": "wanx-style-repaint-v1",
        "input": {
            "image_url": base_url,
            "style_index": "anime",  # 预设风格
        },
        "parameters": {"n": 1},
    }
    print("  → Test 5a: 预设风格 (anime)")
    resp = _session().post(url, headers=headers, json=body_a, timeout=30)
    data_a = resp.json()
    task_id_a = data_a.get("output", {}).get("task_id", "")
    if task_id_a:
        poll_task(task_id_a)
    else:
        print(f"  ← Response: {json.dumps(data_a, indent=2, ensure_ascii=False)[:500]}")

    # Test 5b: 自定义风格参考图 (核心验证)
    # 用 txt2img 再生成一张不同风格的图作为 style_ref
    print("\n  步骤3: 生成 style reference image...")
    result_ref = create_task("wanx-v1", {
        "input": {"prompt": "pixel art style character, 16-bit retro game, full body"},
        "parameters": {"size": DEFAULT_SIZE, "n": 1},
    })
    if "error" in result_ref:
        return

    task_ref = poll_task(result_ref["task_id"])
    ref_results = task_ref.get("output", {}).get("results", [])
    if not ref_results:
        return
    style_url = ref_results[0].get("url", "")

    print("\n  步骤4: style-repaint with custom style reference...")
    body_b = {
        "model": "wanx-style-repaint-v1",
        "input": {
            "image_url": base_url,
            "style_ref_url": style_url,
            "style_index": -1,  # 自定义风格参考
        },
        "parameters": {"n": 1},
    }
    resp_b = _session().post(url, headers=headers, json=body_b, timeout=30)
    data_b = resp_b.json()
    task_id_b = data_b.get("output", {}).get("task_id", "")
    if task_id_b:
        poll_task(task_id_b)
        # 如果有结果, 下载查看
        final = _session().get(f"{BASE_URL}/tasks/{task_id_b}", headers=headers).json()
        final_results = final.get("output", {}).get("results", [])
        if final_results:
            img = download_image(final_results[0].get("url"))
            if img:
                save_image(img, "test5_style_repaint.png")
    else:
        print(f"  ← Response: {json.dumps(data_b, indent=2, ensure_ascii=False)[:500]}")

    # Test 5c: 尝试用 style-repaint 做 chain-anchoring
    # 输入图 = 帧1 (base image), 尝试重绘成不同姿态
    print("\n  步骤5: 验证 chain-anchoring 可行性...")
    body_c = {
        "model": "wanx-style-repaint-v1",
        "input": {
            "image_url": base_url,
            "style_ref_url": style_url,
            "style_index": -1,
        },
        "parameters": {
            "n": 1,
            # 这个 prompt 参数能加吗? 文档未明确
        },
    }
    print("  WARN: style-repaint 可能不支持 prompt 参数引导内容变化")
    print("  → 这意味着它只能改变风格, 不能改变姿态/动作")


# ============================================================
# Test 6: 检查 wanx2.1-imageedit 是否更适合
# ============================================================
def test_image_edit():
    print("\n" + "=" * 60)
    print("Test 6: wanx2.1-imageedit — 是否支持风格锚定？")
    print("=" * 60)

    url = f"{BASE_URL}/services/aigc/image-generation/generation"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }

    # 先准备 base image
    print("\n  准备 base image...")
    result = create_task("wanx-v1", {
        "input": {"prompt": "a warrior character, game art, white background"},
        "parameters": {"size": DEFAULT_SIZE, "n": 1},
    })
    if "error" in result:
        print("  FAIL: 跳过")
        return

    task_result = poll_task(result["task_id"])
    results = task_result.get("output", {}).get("results", [])
    if not results:
        return
    base_url = results[0].get("url", "")

    # Test stylization_all
    print("\n  Test: wanx2.1-imageedit stylization_all...")
    body = {
        "model": "wanx2.1-imageedit",
        "input": {
            "image_url": base_url,
            "function": "stylization_all",
            "prompt": "pixel art 16-bit retro game style",
            "strength": 0.8,
        },
        "parameters": {},
    }
    resp = _session().post(url, headers=headers, json=body, timeout=30)
    data = resp.json()
    task_id = data.get("output", {}).get("task_id", "")
    if task_id:
        poll_task(task_id)
        final = _session().get(f"{BASE_URL}/tasks/{task_id}", headers=headers).json()
        final_results = final.get("output", {}).get("results", [])
        if final_results:
            img = download_image(final_results[0].get("url"))
            if img:
                save_image(img, "test6_image_edit.png")
    else:
        print(f"  ← {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  2D Game Asset Generator — API 可行性验证")
    print("=" * 60)
    print(f"  API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    print(f"  Base URL: {BASE_URL}")

    test_txt2img()
    test_seed_consistency()
    test_multi_frame_seed()
    test_asset_types()
    test_style_repaint()
    test_image_edit()

    print("\n" + "=" * 60)
    print("  验证完成！")
    print("  请查看 output/ 目录下的生成图片，评估质量")
    print("=" * 60)
