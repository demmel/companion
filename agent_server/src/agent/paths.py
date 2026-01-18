from pathlib import Path
from typing import Optional

from agent.config import Config


class AgentPaths:
    """Centralized path management for the agent system"""

    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            # Default to project root - navigate from this file location
            # paths.py is in src/agent/, so go up 3 levels to reach project root
            base_path = Path(__file__).parent.parent.parent

        # Core project structure
        self.base_path = base_path
        self.src_dir = base_path / "src"
        self.models_dir = base_path / "models"
        self.logs_dir = base_path / "logs"

        # Client build paths
        self.client_dir = base_path / ".." / "client"
        self.client_dist_dir = self.client_dir / "dist"
        self.client_assets_dir = self.client_dist_dir / "assets"
        self.client_index_html = self.client_dist_dir / "index.html"

        # Generated content directories
        self.generated_images_dir = base_path / "generated_images"
        self.uploaded_images_dir = base_path / "uploaded_images"
        self.generated_audio_dir = base_path / "generated_audio"

        # TTS reference audio (from config, or None if not configured)
        self._tts_reference_audio: Optional[Path] = None
        tts_audio_path = Config.tts_reference_audio()
        if tts_audio_path:
            path = Path(tts_audio_path)
            # Support both absolute and relative paths
            if path.is_absolute():
                self._tts_reference_audio = path
            else:
                self._tts_reference_audio = base_path / path

        # Test and development paths
        self.tests_dir = base_path / "tests"

        # Ensure necessary directories exist
        self.generated_images_dir.mkdir(exist_ok=True)
        self.uploaded_images_dir.mkdir(exist_ok=True)
        self.generated_audio_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

    def get_base_path(self) -> Path:
        """Get project base path"""
        return self.base_path

    def get_src_dir(self) -> Path:
        """Get src directory path"""
        return self.src_dir

    def get_models_dir(self) -> Path:
        """Get models directory path"""
        return self.models_dir

    def get_logs_dir(self) -> Path:
        """Get logs directory path"""
        return self.logs_dir

    def get_client_dir(self) -> Path:
        """Get client directory path"""
        return self.client_dir

    def get_client_dist_dir(self) -> Path:
        """Get client dist directory path"""
        return self.client_dist_dir

    def get_client_assets_dir(self) -> Path:
        """Get client assets directory path"""
        return self.client_assets_dir

    def get_client_index_html(self) -> Path:
        """Get client index.html path"""
        return self.client_index_html

    def get_generated_images_dir(self) -> Path:
        """Get generated images directory path"""
        return self.generated_images_dir

    def get_uploaded_images_dir(self) -> Path:
        """Get uploaded images directory path"""
        return self.uploaded_images_dir

    def get_generated_audio_dir(self) -> Path:
        """Get generated audio directory path"""
        return self.generated_audio_dir

    def get_tts_reference_audio(self) -> Optional[Path]:
        """Get TTS reference audio path, or None if not configured"""
        return self._tts_reference_audio

    def get_tests_dir(self) -> Path:
        """Get tests directory path"""
        return self.tests_dir

    def get_relative_to_base(self, path: Path) -> Path:
        """Get path relative to project base"""
        return path.relative_to(self.base_path)


# Global instance
agent_paths = AgentPaths()
