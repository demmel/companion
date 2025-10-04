from pydantic import BaseModel


class ResetResponse(BaseModel):
    message: str
    timestamp: str


class AutoWakeupStatusResponse(BaseModel):
    enabled: bool
    delay_seconds: int


class AutoWakeupSetRequest(BaseModel):
    enabled: bool


class AutoWakeupSetResponse(BaseModel):
    enabled: bool
    message: str
    timestamp: str


class ImageUploadResponse(BaseModel):
    id: str
    size: int
    url: str
