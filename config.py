import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    dashscope_api_key: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", "")
    )
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    session_ttl_seconds: int = 1800
    temp_dir: str = "/tmp/asset_generator"
    valid_image_sizes: tuple = (
        "1024*1024",
        "720*1280",
        "1280*720",
        "768*1152",
    )
    default_image_size: str = "1024*1024"
    default_video_resolution: str = "480P"
    default_video_duration: int = 5


config = Config()
