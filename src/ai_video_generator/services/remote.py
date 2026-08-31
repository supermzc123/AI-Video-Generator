from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import Field, model_validator

from ai_video_generator.domain import WorkerCapabilities
from ai_video_generator.domain.chain import SHA256_PATTERN, FrozenModel


class WorkerConnectionState(StrEnum):
    ONLINE = "online"
    DRAINING = "draining"
    OFFLINE = "offline"


class WorkerRegistration(FrozenModel):
    capabilities: WorkerCapabilities
    connection_state: WorkerConnectionState = WorkerConnectionState.ONLINE
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkerHeartbeat(FrozenModel):
    worker_id: str = Field(min_length=1)
    sent_at: datetime
    running_task_ids: tuple[str, ...] = ()
    available_gpu_slots: int = Field(ge=0)


class ArtifactTransfer(FrozenModel):
    artifact_id: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_size: int = Field(ge=0)
    media_type: str = Field(min_length=1)
    upload_path: str = Field(min_length=1)


class WorkerResultStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class TaskResultReport(FrozenModel):
    report_id: str = Field(pattern=SHA256_PATTERN)
    task_id: str = Field(min_length=1)
    worker_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    status: WorkerResultStatus
    artifact_sha256_values: tuple[str, ...] = ()
    error_code: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> TaskResultReport:
        if any(not re.fullmatch(SHA256_PATTERN, value) for value in self.artifact_sha256_values):
            raise ValueError("artifact SHA-256 values must be lowercase hexadecimal")
        if self.status == WorkerResultStatus.FAILED:
            if not self.error_code:
                raise ValueError("failed results require an error code")
        elif self.error_code is not None or self.error_message is not None:
            raise ValueError("only failed results may include error details")
        return self


class TaskResultReceipt(FrozenModel):
    report_id: str = Field(pattern=SHA256_PATTERN)
    task_id: str = Field(min_length=1)
    state: str = Field(min_length=1)
    accepted_at: datetime
