# SpriteForge · 2D 游戏素材生成器

基于 AI 的 2D 游戏素材在线生成工具，用户通过输入文本描述即可生成角色、场景、道具、UI、特效等游戏素材，支持多帧动画输出。

## 功能特性

- **5 种素材类型**：角色精灵、场景瓦片、UI 元素、道具物品、视觉特效
- **文本生图**：基于阿里云通义万象 wanx-v1，输入描述即可生成游戏素材
- **多帧动画**：基于通义万象视频生成模型，文本→视频→抽帧→精灵表
- **AI 提示词优化**：集成 DeepSeek，自动将简单描述扩展为专业游戏美术提示词
- **背景移除**：本地 rembg 模型自动去背（需下载 ONNX 模型）
- **精灵表输出**：多帧动画自动打包为水平条带精灵表 + JSON 坐标元数据
- **适配主流引擎**：JSON 格式兼容 TexturePacker / Unity Sprite Atlas / Godot TileMap

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | Python 3.9+ / FastAPI |
| 图像生成 | 阿里云 DashScope 通义万象 wanx-v1 |
| 视频生成（动画） | 阿里云 DashScope wanx2.1-t2v-turbo / wanx2.1-i2v-turbo |
| 提示词优化 | DeepSeek Chat API (OpenAI SDK 兼容调用) |
| 背景移除 | rembg (u2net ONNX 本地模型) |
| 视频抽帧 | OpenCV (cv2) |
| 图像处理 | Pillow |
| 前端 | 原生 HTML/CSS/JS (零框架依赖) |

## 依赖库

```
fastapi
uvicorn
Pillow
rembg
opencv-python
requests
openai
python-multipart
python-dotenv
aiofiles
```

完整依赖见 [requirements.txt](requirements.txt)。

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入 API Key：

```
DASHSCOPE_API_KEY=sk-xxx    # 阿里云百炼 DashScope（必填）
DEEPSEEK_API_KEY=sk-xxx     # DeepSeek（可选，无 Key 则跳过改写）
```

### 3. 下载去背模型（可选）

rembg 首次运行会自动下载 u2net.onnx 模型（约 176MB）。也可手动下载：

```bash
# 手动下载到 ~/.u2net/u2net.onnx
# 下载地址: https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
```

未下载时去背功能自动降级，不影响其他功能。

### 4. 启动服务

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

浏览器打开 http://localhost:8000 即可使用。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/session` | 创建会话 |
| POST | `/api/style-ref` | 上传风格参考图（可选） |
| POST | `/api/generate` | 提交生成请求 |
| GET | `/api/generate/{job_id}` | 查询任务状态与结果 |
| GET | `/api/download/{job_id}` | 下载素材 zip |
| GET | `/api/session/assets` | 列出会话资产 |
| POST | `/api/session/download-all` | 打包下载全部 |
| GET | `/api/presets` | 获取预设模板 |

## 输出格式

### 单帧素材
```
<name>.png
```

### 多帧动画
```
<name>.png    ← 水平拼接精灵表
<name>.json   ← 帧坐标元数据
```

JSON 兼容 TexturePacker / Unity Sprite Atlas / Godot TileMap 格式。

## 设计文档

详见 [docs/superpowers/specs/](docs/superpowers/specs/) 目录下的设计规格与实现计划。

## 原创功能说明

本项目全部代码为原创实现，核心原创点：
1. 视频生成→抽帧→精灵表的动画管道（区别于常规逐帧生成方案）
2. 基于 API 验证结果的风格一致性策略（keyword-driven + seed 锚定）
3. 5 种素材类型的统一 prompt 模板引擎 + 预设系统
4. 去背模型未就绪时的优雅降级机制
