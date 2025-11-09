from pydantic import BaseModel
from typing import Dict


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


class ModelConfigResponse(BaseModel):
    """Response containing current model configuration"""

    state_initialization_model: str
    action_planning_model: str
    situational_analysis_model: str
    memory_retrieval_model: str
    memory_formation_model: str
    trigger_compression_model: str
    think_action_model: str
    speak_action_model: str
    visual_action_model: str
    fetch_url_action_model: str
    evaluate_priorities_action_model: str


class ModelConfigUpdateRequest(BaseModel):
    """Request to update model configuration"""

    state_initialization_model: str
    action_planning_model: str
    situational_analysis_model: str
    memory_retrieval_model: str
    memory_formation_model: str
    trigger_compression_model: str
    think_action_model: str
    speak_action_model: str
    visual_action_model: str
    fetch_url_action_model: str
    evaluate_priorities_action_model: str


class ModelConfigUpdateResponse(BaseModel):
    """Response after updating model configuration"""

    message: str
    timestamp: str
    config: ModelConfigResponse


class SupportedModelsResponse(BaseModel):
    """Response containing list of all supported models"""

    models: list[str]
