# 2D Game Asset Generator — Design Spec

## Overview

A web-based tool that generates 2D game assets from text descriptions and simple parameters. Users describe what they want via text, optionally upload a style reference image, and select asset type. The tool generates PNG + JSON metadata compatible with mainstream 2D game engines (Unity, Godot, GameMaker).

**Two generation modes:**
- **Text-only mode:** pure txt2img, style controlled by style keywords (e.g. "pixel art", "cartoon", "dark fantasy")
- **Reference-anchored mode:** upload a style reference image → img2img, better style consistency across multiple generations

**Target users:** both game developers and non-technical creators.
**Hard constraint:** 3-day development timeline.

---

## Architecture

```
User Browser (Web UI)
       │
       ▼
FastAPI Backend
       │
       ├── Prompt Rewriter (DeepSeek API, optional toggle)
       ├── Prompt Template Engine (per asset type)
       ├── Style Manager (keyword-driven txt2img, optional style-repaint)
       ├── Generation Scheduler (DashScope API, with cache layer)
       │     ├── Image gen: wanx-v1 (txt2img, ~27s)
       │     └── Video gen: wanx2.1-t2v-turbo (~30s) / wanx2.1-i2v-turbo (~160s)
       ├── Frame Extractor (cv2, video → evenly-sampled frames)
       ├── Post-processor (rembg, crop, grid-slice, spritesheet packing, tile edge-wrapping)
       └── Packager (PNG + JSON metadata, zip download)
```

**Tech stack:**
- Backend: Python 3.11+ / FastAPI
- Image generation: Tongyi Wanxiang wanx-v1 (DashScope)
- Video generation: wanx2.1-t2v-turbo / wanx2.1-i2v-turbo (DashScope)
- Frame extraction: OpenCV (cv2)
- Prompt rewriting: DeepSeek chat/completions API
- Background removal: rembg (local ONNX model, offline)
- Frontend: vanilla HTML/CSS/JS, single page
- Storage: none — in-memory dict per session, temp files in `/tmp`, 30-minute TTL auto-cleanup

---

## Asset Types & Prompt Templates

### 1. Character Sprites
```
{[Style reference image] + } "2D game character sprite, {rewritten description},
full body, standing pose, clean silhouette,
isolated on transparent background, game-ready asset, {style keywords}"
```
**Post-processing:** rembg background removal → crop to content bounds → center.
**Output:** single sprite or multi-frame spritesheet PNG + frame coordinate JSON.

### 2. Scene / Tilemap
```
{[Style reference image] + } "2D game environment tile/background, {rewritten description},
{top-down / side-view}, seamlessly tileable, flat composition,
game environment art, {style keywords}"
```
**Post-processing:** seamless tiling preview → grid slicing (16×16 / 32×32 / user-defined).
**Output:** individual tile PNGs + tile index JSON.

### 3. UI Elements
```
{[Style reference image] + } "2D game UI element, {rewritten description},
clean vector-like, minimal design, game interface asset, {style keywords}"
```
**Post-processing:** minimal (optional rembg).
**Output:** single PNG, optional nine-slice metadata in JSON.

### 4. Props / Items (new)
```
{[Style reference image] + } "2D game item/prop, {rewritten description},
centered, isolated object, clean edges, transparent background,
game inventory asset, top-down/front-view, {style keywords}"
```
**Post-processing:** rembg background removal → crop to content bounds → center.
**Output:** single PNG per item; batch mode: spritesheet PNG + frame coordinate JSON.
**Use cases:** weapons, potions, treasure chests, keys, equipment pieces, collectibles.

### 5. Effects / VFX (new)
```
{[Style reference image] + } "2D game visual effect sprite sheet, {rewritten description},
flipbook animation frames, {frame_count} frames arranged in sequence,
transparent background, game VFX asset, {style keywords}"
```
**Post-processing:** rembg → grid-slice into frames → pack into spritesheet.
**Output:** spritesheet PNG (horizontal strip or grid) + frame coordinate JSON.
**Use cases:** explosions, slash/swing trails, magic glow, smoke puffs, impact sparks.

---

## Style Consistency Strategy

**Verified API capability:** No true img2img exists. wanx-style-repaint-v1 transfers artistic style onto an existing image but cannot change pose/content. Therefore:

### Path A: Keyword-driven txt2img (primary, always available)
1. User selects/enters style keywords (e.g. "16-bit pixel art", "cartoon cel-shaded", "dark gothic").
2. Keywords injected into every prompt template.
3. System prepends quality anchor prefix: `"masterpiece, best quality, game-ready asset, {style_keywords}, ..."`.
4. **Same seed across related generations** improves consistency (verified: same seed = similar output).
5. Best for: most use cases, always works.

### Path B: Style-repaint post-processing (optional, after txt2img)
1. Generate base image via txt2img.
2. Use wanx-style-repaint-v1 with `style_index=-1` + `style_ref_url` to transfer the reference style.
3. **Limitation:** only changes artistic style (color palette, rendering technique), does NOT change pose or composition.
4. Best for: unifying the visual style of separately-generated assets.

---

## Multi-Frame Animation Strategy (video-based)

**Why not chain-anchoring with img2img:** wanx-v1 has no true img2img capability (wanx-style-repaint-v1 only transfers artistic style, cannot change pose/content). Independent txt2img frames with different seeds show severe character inconsistency (verified).

**Solution: Text/Image-to-Video → Extract Frames → Spritesheet**

```
User prompt (+ optional base image) 
       │
       ▼
wanx2.1-t2v-turbo (text→video, ~30s)  OR  wanx2.1-i2v-turbo (image→video, ~160s)
       │
       ▼
5-second video @ 480P (624×624) or 720P (720×720), 30fps, 150+ frames
       │
       ▼
Frame Extractor: evenly sample N frames (user-specified, default 8)
       │
       ▼
rembg each frame → resize to target → pack into horizontal spritesheet
       │
       ▼
PNG + frame coordinate JSON
```

**Two modes:**
| Mode | Model | Input | Speed | Best for |
|------|-------|-------|-------|----------|
| Fast | `wanx2.1-t2v-turbo` | Text only | ~30s | Quick animation sketch, VFX |
| Quality | `wanx2.1-i2v-turbo` | Base image + text | ~160s | Character animation anchored to generated sprite |

**Verified consistency:** Histogram correlation between adjacent video frames reaches 0.97-0.98, vs random correlation for independent txt2img frames. The video model's temporal attention layer ensures character appearance stays consistent throughout the motion.

**Resolution note:** Video output is fixed (480P=624×624, 720P=720×720). Post-processing rescales to user's target size.

### Frame Animation Preview

When a spritesheet has multiple frames, the center preview area shows a simple flipbook player:
- **Play/Pause** toggle with adjustable FPS (default: 12 fps)
- **Frame scrubber** — drag to inspect individual frames
- **Onion skinning toggle** — overlay previous/next frame at 30% opacity to check frame-to-frame consistency
- **Grid overlay** — toggle a grid matching the frame size to verify alignment

This is critical for evaluating sprite animation quality before downloading.

---

## Tilemap Seamless Tiling Strategy (revised)

Generating a large image then slicing into tiles rarely produces seamless results. Instead:

1. **Generate a single tile unit** (e.g. 32×32 or 64×64) — this is small enough for the AI to produce clean, coherent content.
2. **Post-process:** apply edge-wrapping (mirror/blend the tile edges) to reduce visible seams.
3. **Tiling preview:** render the tile in a 4×4 or 5×5 grid in the browser so the user can visually verify seamlessness.
4. **Variation tiles:** optionally generate multiple variant tiles with the same prompt (each API call produces slight variations) — user picks the best or combines them.
5. **Auto-tiling support:** for tilesets that need borders/transitions (e.g. grass-to-dirt edges), generate a set of labeled tiles with rule-based metadata for engines like Godot's TileMap or Unity's Rule Tile.

This approach is much more reliable than the "generate big → slice" approach.

---

## Preset Templates & User Guidance

To lower the barrier for non-technical users, each asset type has preset examples:

| Asset Type | Preset Examples |
|---|---|
| Character | "Knight in plate armor with sword", "Slime monster, bouncy", "NPC villager, medieval", "Pixel cat, platformer hero" |
| Tilemap | "Grass field, top-down RPG", "Dungeon stone floor", "Space platform, side-view", "Forest path with flowers" |
| UI | "Fantasy health bar with gem", "Sci-fi button panel", "Wooden inventory slot", "Gold coin icon" |
| Props | "Iron longsword with glow", "Red health potion bottle", "Treasure chest, golden", "Magic staff with crystal" |
| VFX | "Fire explosion, 8 frames", "Sword slash arc, 4 frames", "Magic circle summon, 6 frames", "Smoke puff, 4 frames" |

- Click a preset → auto-fills the description field
- Each preset has pre-configured recommended parameters (asset type, view angle, frame count, tile size)
- Users can modify the filled description before generating

---

## Generation Cache

To avoid redundant API calls and costs:

- **Cache key:** SHA256 hash of (asset_type + description + style_keywords + size + frames + view + other params). Reference image hashed by its content.
- **Scope:** per-session, in-memory dict.
- **Behavior:** before calling the generation API, check cache. If hit → return cached result instantly (no API cost, sub-second response).
- **User-facing:** cache hits are indicated in the UI ("Instant (cached)"), user can force regenerate to bypass cache.
- **TTL:** matches session TTL (30 min).

---

## Prompt Rewriting (DeepSeek)

- **Default:** enabled
- **Toggle:** off for power users who write their own prompts
- **Why keep it:** non-technical users often write vague descriptions like "a cool warrior." DeepSeek expands these into detailed game-asset prompts with proper terminology (pose, perspective, art style, composition, color palette), significantly improving generation quality. The cost is negligible (~1c per 10 rewrites) and latency is sub-second — well worth the quality gain.
- **Flow:** user raw text → DeepSeek chat/completions with system prompt for game-asset-specific rewriting → rewritten description injected into prompt template
- **System prompt:** instructs DeepSeek to expand brief descriptions into detailed game-asset prompts with standard terminology (pose, perspective, art style, composition)

---

## Output Format

### Single sprite
```
<name>.png
```

### Spritesheet / tile set
```
<name>.png              ← the packed image
<name>.json             ← frame/tile coordinates
```

**JSON schema (compatible with TexturePacker/Godot/Unity Sprite Atlas):**
```json
{
  "version": "1.0",
  "meta": {
    "type": "spritesheet|tileset",
    "tile_size": {"w": 32, "h": 32},
    "frame_count": 4,
    "generated_at": "2026-05-23T..."
  },
  "frames": [
    {"name": "warrior_idle_01", "rect": {"x": 0, "y": 0, "w": 32, "h": 32}},
    {"name": "warrior_idle_02", "rect": {"x": 32, "y": 0, "w": 32, "h": 32}}
  ]
}
```

---

## Web UI Layout

```
┌──────────────────────────────────────────────────────────┐
│  2D Game Asset Generator                                 │
├──────────────┬──────────────────┬────────────────────────┤
│  Left Panel  │  Center          │  Right Panel           │
│  (Input)     │  (Preview)       │  (Session Assets)      │
│              │                  │                        │
│ ①Style Ref   │  Large preview   │  Thumbnail list        │
│   [Upload]   │  of generated    │  of current            │
│   (optional) │  image           │  session results       │
│              │                  │                        │
│ ②Asset Type  │  ┌────────────┐  │  Click to preview      │
│   ○Char Sprite│  │ Animation  │  │  /download/delete      │
│   ○Tilemap   │  │ ▶ ⏸ ⏪ ⏩  │  │  (cached) badge if    │
│   ○UI Element│  │ FPS:12[▁▃▅]│  │  served from cache     │
│   ○Props     │  │ Onion ▣ Off│  │                        │
│   ○VFX       │  └────────────┘  │  [Download All]        │
│              │                  │                        │
│ ③Presets     │  ← Previous      │                        │
│   [Dropdown] │  Next →          │                        │
│   Click→fill │                  │                        │
│              │                  │                        │
│ ④Description │                  │                        │
│   [________] │                  │                        │
│              │                  │                        │
│ ⑤Parameters  │                  │                        │
│   Size: 512  │                  │                        │
│   Frames: 4  │                  │                        │
│   View:Front │                  │                        │
│   Grid:32×32 │                  │                        │
│              │                  │                        │
│ [✓] AI Rewrite│                 │                        │
│              │                  │                        │
│ ⑥[Generate] │                  │                        │
│  ⓘ cached → │                  │                        │
│  instant     │                  │                        │
└──────────────┴──────────────────┴────────────────────────┘
```

**Key UI behaviors:**
- Style ref upload: labeled "(optional)" with a subtle info tooltip explaining text-only mode
- Presets dropdown: grouped by asset type, selecting a preset auto-switches asset type + fills description + sets recommended params
- Animation player: only visible when asset has ≥2 frames; auto-hides for single images
- Cache indicator: toast/badge on generate button when result is served from cache
- Force regenerate: small "⟳" button next to cached items in the right panel

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/session` | Create session, return session_id |
| POST | `/api/style-ref` | Upload style reference image |
| POST | `/api/generate` | Submit generation request (returns job_id) |
| GET | `/api/generate/{job_id}` | Poll generation status & result |
| GET | `/api/download/{job_id}` | Download PNG + JSON as zip |
| GET | `/api/session/assets` | List current session assets |
| POST | `/api/session/download-all` | Download all session assets as zip |

---

## 2-Day Plan

### Day 1: API Integration + Backend Pipeline
- FastAPI project scaffold, config, Pydantic models
- Prompt template engine (**5 asset types:** character, tilemap, UI, props, VFX) + presets
- DeepSeek API integration for prompt rewriting
- **Image generation:** DashScope wanx-v1 txt2img (verified: ~27s, 4 fixed sizes)
- **Video generation:** wanx2.1-t2v-turbo (~30s) + wanx2.1-i2v-turbo (~160s) for animation
- **Frame extraction:** OpenCV video → evenly-sampled frames → spritesheet packing
- Style manager (keyword-driven primary + optional style-repaint)
- Session management + generation cache layer
- Post-processing: rembg, crop, grid-slice, spritesheet packing, tile edge-wrapping
- All API endpoints + temp file cleanup scheduler (30min TTL)

### Day 2: Frontend + Integration
- Single-page HTML/CSS/JS UI
- File upload (optional style ref), parameter forms, preset selector
- Preview area: flipbook animation player, tile tiling preview
- Session asset list with download/delete, cache indicators
- End-to-end testing and polish

---

## Environment Variables

```
DASHSCOPE_API_KEY=sk-xxx    # Required: Tongyi Wanxiang (image + video gen)
DEEPSEEK_API_KEY=sk-xxx     # Optional: prompt rewriting (gracefully skipped if not set)
```

---

## Constraints & Trade-offs

- **No persistent storage** — refresh loses history. Trade-off for simplicity.
- **Image resolution:** wanx-v1 only supports 4 fixed sizes (1024*1024, 720*1280, 1280*720, 768*1152). Rescaling to target size happens in post-processing.
- **Video resolution:** 480P (624×624) or 720P (720×720). Frames rescaled to sprite size in post-processing.
- **Multi-frame animation:** achieved via video generation + frame extraction, NOT chain-anchoring (which was verified infeasible). Video-based approach has verified frame-to-frame consistency (histogram correlation 0.97+).
- **Style consistency:** keyword-driven txt2img is the primary path. wanx-style-repaint-v1 can transfer artistic style but cannot change content — used as optional post-processing, not as generation anchor.
- **No user accounts / auth** — single-user local tool.
- **API cost:** wanx-v1 images = ~0.16 CNY/image. Video generation costs more (varies by model and duration). DeepSeek usage negligible.
