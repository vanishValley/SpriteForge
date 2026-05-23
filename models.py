from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class AssetType(str, Enum):
    CHARACTER = "character"
    TILEMAP = "tilemap"
    UI = "ui"
    PROPS = "props"
    VFX = "vfx"


class ViewAngle(str, Enum):
    FRONT = "front"
    SIDE = "side"
    TOP_DOWN = "top_down"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class GenerateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    image_base64: Optional[str] = None
    json_metadata: Optional[str] = None
    error: Optional[str] = None


class SessionCreateResponse(BaseModel):
    session_id: str


class AssetInfo(BaseModel):
    job_id: str
    asset_type: str
    description: str
    created_at: str
    frame_count: int
    cached: bool = False


class SessionAssetsResponse(BaseModel):
    session_id: str
    assets: list[AssetInfo]
