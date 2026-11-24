import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)
    studio_timezone: str = "Asia/Shanghai"

    @field_validator("studio_timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("STUDIO_TIMEZONE must be an IANA timezone") from exc
        return value

    database_url: str = "sqlite:///./data/studio.db"
    data_dir: Path = Path("./data")
    site_url: str = "http://localhost:8788"
    studio_password: str = ""
    encryption_key: str = ""
    session_secret: str = ""
    environment: str = "development"
    max_upload_bytes: int = 100 * 1024 * 1024
    allowed_hosts: list[str] = []
    studio_api_token: str = ""
    creation_comfy_url: str = ""
    native_comfy_url: str = ""
    native_comfy_backend_url: str = ""
    native_comfy_idle_minutes: int = Field(default=120, ge=1)
    native_comfy_max_runtime_minutes: int = Field(default=720, ge=1)
    native_comfy_cache_max_age_minutes: int = Field(default=60, ge=1, le=153722867)
    native_comfy_cache_gb: int = Field(default=20, ge=1, le=2**31 - 1)
    native_comfy_max_upload_mb: int = Field(default=2048, ge=1)
    gpu_auto_recovery_enabled: bool = False
    gpu_auto_recovery_max_rounds: int = Field(default=3, ge=1, le=10)
    gpu_auto_recovery_min_remaining_minutes: int = Field(default=15, ge=1)
    gpu_auto_recovery_max_job_retries: int = Field(default=1, ge=0, le=3)
    gpu_auto_recovery_backoff_base_seconds: int = Field(default=30, ge=10, le=300)
    gpu_auto_recovery_round_timeout_seconds: int = Field(default=600, ge=60, le=3600)
    gpu_auto_recovery_startup_grace_seconds: int = Field(default=30, ge=0, le=300)
    gpu_auto_recovery_max_vps_outage_minutes: int = Field(default=18, ge=5, le=60)
    gpu_active_keepalive: bool = True
    gpu_in_vm_self_heal: bool = True


if os.getenv("STUDIO_OFFLINE_TEST") == "1":
    from .offline import install
    install()
    settings = Settings(_env_file=None)
else:
    settings = Settings(_env_file=None) if os.getenv("ENVIRONMENT") == "offline-test" else Settings()
