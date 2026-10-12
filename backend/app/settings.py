from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    frontend_origin: str = "http://localhost:5173,http://127.0.0.1:5173"
    model_path: Path = Path("models/material-yolo.pt")
    classifier_path: Path = Path("models/rion-classifier.pt")
    model_s3_bucket: str = ""
    model_s3_key: str = "models/material-yolo.pt"
    classifier_s3_key: str = "models/rion-classifier.pt"
    model_sha256: str = ""
    classifier_sha256: str = ""
    classifier_threshold: float = 0.50
    detector_threshold: float = 0.55
    classifier_fallback_threshold: float = 0.50
    database_url: str = ""
    runtime_secret_arn: str = ""
    max_upload_mb: int = 10
    max_image_pixels: int = 16_000_000
    allowed_hosts: str = "127.0.0.1,localhost,testserver,rion-resources-intelligance-and.onrender.com"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def host_allowlist(self) -> list[str]:
        return [host.strip() for host in self.allowed_hosts.split(",") if host.strip()]

    @property
    def frontend_origins(self) -> list[str]:
        return [origin.strip() for origin in self.frontend_origin.split(",") if origin.strip()]


settings = Settings()

