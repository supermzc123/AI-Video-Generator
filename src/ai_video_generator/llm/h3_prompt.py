from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from contextvars import Token
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from ai_video_generator.domain.chain import GenerationMode
from ai_video_generator.domain.h3_prompt import (
    H3AssetInput,
    H3AssetKind,
    H3AssetPromptRole,
    H3CreativeBrief,
    H3DirectorDecision,
    H3MultishotPlan,
    H3PromptCandidate,
    H3PromptRequest,
    H3PromptResult,
    H3ReviewerDecision,
    H3ShotBeat,
    H3ShotStrategy,
    H3StageTrace,
)
from ai_video_generator.domain.orchestration import (
    H3HarnessManifest,
    HarnessDocumentSnapshot,
)
from ai_video_generator.services.harness_sources import validate_h3_harness_source

from .budget import claim_business_repair, llm_operation
from .client import (
    ChatMessage,
    ImageURL,
    ImageURLContentPart,
    TextContentPart,
    VideoURL,
    VideoURLContentPart,
    llm_delta_callback,
)
from .harness_files import load_harness

MAX_H3_REPAIR_PASSES = 1
MAX_H3_SCHEMA_REPAIRS = 1
MIN_BASE_DESCRIPTION_WORDS = 60
MIN_REF_DESCRIPTION_WORDS = 300

_MULTISHOT_MARKERS = (
    "[shot 2]",
    "多镜头",
    "蒙太奇",
    "镜头切换",
    "切到",
    "切至",
    "cut to",
    "match cut",
    "montage",
)


class H3PromptHarnessError(RuntimeError):
    def __init__(self, message: str, errors: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.errors = errors


class H3JSONCompletionClient(Protocol):
    async def complete_json(self, messages: Sequence[ChatMessage]) -> str: ...


@dataclass(frozen=True)
class H3CallTelemetry:
    stage: str
    attempt: int
    total_seconds: float
    input_characters: int
    output_characters: int
    image_count: int
    succeeded: bool
    error_type: str | None = None


@dataclass(frozen=True)
class H3HarnessLibrary:
    official_skill: str
    official_base: str
    official_reference: str
    community_director: str
    community_planner: str
    community_text_writer: str
    community_keyframe_writer: str
    community_reference_writer: str
    community_reviewer: str
    community_review_checklist: str

    _DOCUMENTS = {
        "official_skill": ("official", "skills/h3-prompt-writing/SKILL.md", ("all",)),
        "official_base": (
            "official",
            "skills/h3-prompt-writing/references/base-en.txt",
            ("text_writer", "keyframe_writer", "reviewer"),
        ),
        "official_reference": (
            "official",
            "skills/h3-prompt-writing/references/ref-en.txt",
            ("reference_writer", "reviewer"),
        ),
        "community_director": (
            "community",
            "skills/minimax-h3-creative-director/SKILL.md",
            ("preflight", "director"),
        ),
        "community_planner": (
            "community",
            "skills/minimax-h3-multishot-planner/SKILL.md",
            ("planner",),
        ),
        "community_text_writer": (
            "community",
            "skills/minimax-h3-text-video-prompt/SKILL.md",
            ("text_writer",),
        ),
        "community_keyframe_writer": (
            "community",
            "skills/minimax-h3-keyframe-video-prompt/SKILL.md",
            ("keyframe_writer",),
        ),
        "community_reference_writer": (
            "community",
            "skills/minimax-h3-reference-video-prompt/SKILL.md",
            ("reference_writer",),
        ),
        "community_reviewer": (
            "community",
            "skills/minimax-h3-prompt-reviewer/SKILL.md",
            ("reviewer",),
        ),
        "community_review_checklist": (
            "community",
            "skills/minimax-h3-prompt-reviewer/references/validation-checklist.md",
            ("validator", "reviewer"),
        ),
    }

    @classmethod
    def load(
        cls,
        *,
        official_root: str | Path,
        community_root: str | Path,
    ) -> H3HarnessLibrary:
        official = Path(official_root)
        community = Path(community_root)
        validate_h3_harness_source(official, source_id="official")
        validate_h3_harness_source(community, source_id="community")
        official_skill_root = official / "skills" / "h3-prompt-writing"
        community_skills = community / "skills"
        return cls(
            official_skill=_read(official_skill_root / "SKILL.md"),
            official_base=_read(official_skill_root / "references" / "base-en.txt"),
            official_reference=_read(official_skill_root / "references" / "ref-en.txt"),
            community_director=_read(
                community_skills / "minimax-h3-creative-director" / "SKILL.md"
            ),
            community_planner=_read(community_skills / "minimax-h3-multishot-planner" / "SKILL.md"),
            community_text_writer=_read(
                community_skills / "minimax-h3-text-video-prompt" / "SKILL.md"
            ),
            community_keyframe_writer=_read(
                community_skills / "minimax-h3-keyframe-video-prompt" / "SKILL.md"
            ),
            community_reference_writer=_read(
                community_skills / "minimax-h3-reference-video-prompt" / "SKILL.md"
            ),
            community_reviewer=_read(community_skills / "minimax-h3-prompt-reviewer" / "SKILL.md"),
            community_review_checklist=_read(
                community_skills
                / "minimax-h3-prompt-reviewer"
                / "references"
                / "validation-checklist.md"
            ),
        )

    def to_manifest(
        self,
        *,
        official_commit: str,
        community_commit: str,
    ) -> H3HarnessManifest:
        commits = {"official": official_commit, "community": community_commit}
        documents = tuple(
            HarnessDocumentSnapshot(
                source_id=source_id,
                source_commit=commits[source_id],
                path=path,
                sha256=hashlib.sha256(getattr(self, field).encode("utf-8")).hexdigest(),
                content=getattr(self, field),
                stages=stages,
            )
            for field, (source_id, path, stages) in self._DOCUMENTS.items()
        )
        stage_documents: dict[str, tuple[str, ...]] = {}
        for document in documents:
            for stage in document.stages:
                stage_documents[stage] = (*stage_documents.get(stage, ()), document.path)
        return H3HarnessManifest(
            documents=documents,
            stage_documents=stage_documents,
            policy={
                "interaction": "fully_automatic",
                "max_schema_repairs": MAX_H3_SCHEMA_REPAIRS,
                "max_semantic_repairs": MAX_H3_REPAIR_PASSES,
                "max_prompt_characters": 7000,
                "min_base_description_words": MIN_BASE_DESCRIPTION_WORDS,
                "min_ref_description_words": MIN_REF_DESCRIPTION_WORDS,
                "motion_context": {
                    "segmenting_owner": "storyboard_harness",
                    "required_fields": ["id", "durationSeconds", "summary"],
                    "first_segment_seconds": {"min": 4, "max": 15},
                    "continuation_segment_seconds": {"min": 4, "max": 12},
                    "duration_sum": "must equal shot duration",
                    "summary": (
                        "current action plus the visual/audio end state inherited "
                        "by the next segment"
                    ),
                },
            },
        )

    @classmethod
    def from_manifest(cls, manifest: H3HarnessManifest) -> H3HarnessLibrary:
        by_path = {item.path: item for item in manifest.documents}
        values: dict[str, str] = {}
        for field, (_, path, _) in cls._DOCUMENTS.items():
            document = by_path.get(path)
            if document is None:
                raise H3PromptHarnessError(f"H3 manifest is missing {path}")
            digest = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
            if digest != document.sha256:
                raise H3PromptHarnessError(f"H3 manifest document hash mismatch: {path}")
            values[field] = document.content
        return cls(**values)


class H3PromptHarness:
    """Runs routing, writing, deterministic validation, and targeted repair.

    Prompt authoring is intentionally free of a second LLM reviewer.  The
    deterministic validator is the fast quality gate; an optional repair call
    is made only when that gate reports a concrete error.
    """

    def __init__(
        self,
        client: H3JSONCompletionClient,
        library: H3HarnessLibrary,
        *,
        telemetry_sink: Callable[[H3CallTelemetry], None] | None = None,
        manifest_sha256: str | None = None,
    ) -> None:
        self._client = client
        self._library = library
        self._telemetry_sink = telemetry_sink
        self._manifest_sha256 = manifest_sha256

    async def generate(
        self,
        request: H3PromptRequest,
        *,
        asset_image_urls: tuple[str, ...] = (),
        asset_media_urls: tuple[str | None, ...] | None = None,
    ) -> H3PromptResult:
        async with llm_operation():
            return await self._generate_bounded(
                request, asset_image_urls=asset_image_urls, asset_media_urls=asset_media_urls
            )

    async def _generate_bounded(
        self,
        request: H3PromptRequest,
        *,
        asset_image_urls: tuple[str, ...] = (),
        asset_media_urls: tuple[str | None, ...] | None = None,
    ) -> H3PromptResult:
        if asset_media_urls is None:
            asset_media_urls = tuple(asset_image_urls)
        if asset_media_urls and len(asset_media_urls) != len(request.assets):
            raise ValueError("asset image URLs must align with H3 request assets")
        creative_brief, preflight_assumptions = compile_h3_creative_brief(request)
        director = deterministic_director_decision(request)
        director = director.model_copy(
            update={"assumptions": (*director.assumptions, *preflight_assumptions)}
        )
        trace: list[H3StageTrace] = [H3StageTrace(stage="preflight", status="succeeded")]
        trace.append(
            H3StageTrace(
                stage="director",
                status="skipped" if not preflight_assumptions else "defaulted",
                findings=preflight_assumptions,
            )
        )

        plan: H3MultishotPlan | None = None
        if director.use_multishot:
            plan = await self._complete(
                H3MultishotPlan,
                self._messages(
                    "multishot_planner",
                    self._library.community_planner,
                    {
                        "request": _request_without_memory(request),
                        "director": director,
                    },
                    H3MultishotPlan,
                    highest_instruction=request.highest_instruction,
                ),
                stage="plan",
            )
            plan_errors = validate_timeline(request, plan.shots)
            if plan_errors:
                raise H3PromptHarnessError("H3 planner returned an invalid timeline", plan_errors)
            trace.append(H3StageTrace(stage="planner", status="succeeded"))
        else:
            trace.append(H3StageTrace(stage="planner", status="skipped"))

        candidate = await self._write(
            request,
            director,
            plan,
            repair_context=None,
            asset_image_urls=asset_image_urls,
            asset_media_urls=asset_media_urls,
        )
        # Do not spend a second provider round-trip on a reviewer.  This is a
        # deliberate product contract: prompt writing must stay responsive and
        # deterministic checks remain the source of truth for structural issues.
        review_history: list[H3ReviewerDecision] = []
        last_deterministic_errors: tuple[str, ...] = ()
        for repair_pass in range(MAX_H3_REPAIR_PASSES + 1):
            deterministic_errors = validate_h3_candidate(request, director, plan, candidate)
            rendered_length = len(render_h3_prompt(candidate))
            if rendered_length > request.runtime_limits.max_prompt_characters:
                deterministic_errors = (
                    *deterministic_errors,
                    "execution prompt exceeds runtime character limit",
                )
            last_deterministic_errors = deterministic_errors
            trace.append(
                H3StageTrace(
                    stage="validator",
                    status="failed" if deterministic_errors else "succeeded",
                    attempt=repair_pass,
                    findings=deterministic_errors,
                )
            )
            trace.append(
                H3StageTrace(
                    stage="reviewer",
                    status="skipped",
                    attempt=repair_pass,
                    findings=("LLM reviewer disabled for prompt authoring",),
                )
            )
            if not deterministic_errors:
                execution_prompt = render_h3_prompt(candidate)
                trace.append(H3StageTrace(stage="finalize", status="succeeded"))
                return H3PromptResult(
                    request=request,
                    director=director,
                    multishot_plan=plan,
                    candidate=candidate,
                    review_history=tuple(review_history),
                    repair_passes=repair_pass,
                    execution_prompt=execution_prompt,
                    creative_brief=creative_brief,
                    assumptions=director.assumptions,
                    validator_findings=last_deterministic_errors,
                    stage_trace=tuple(trace),
                    harness_manifest_sha256=self._manifest_sha256,
                )
            if repair_pass == MAX_H3_REPAIR_PASSES or not claim_business_repair():
                break
            candidate = await self._write(
                request,
                director,
                plan,
                repair_context={
                    "candidate": candidate,
                    "deterministic_errors": tuple(dict.fromkeys(deterministic_errors)),
                },
                asset_image_urls=asset_image_urls,
                asset_media_urls=asset_media_urls,
            )
            trace.append(
                H3StageTrace(stage="repair_writer", status="succeeded", attempt=repair_pass + 1)
            )
        raise H3PromptHarnessError(
            "H3 prompt did not pass deterministic validation after one repair pass",
            last_deterministic_errors or deterministic_errors,
        )

    async def _write(
        self,
        request: H3PromptRequest,
        director: H3DirectorDecision,
        plan: H3MultishotPlan | None,
        repair_context: object | None,
        asset_image_urls: tuple[str, ...],
        asset_media_urls: tuple[str | None, ...],
    ) -> H3PromptCandidate:
        if director.mode == GenerationMode.REF2VA:
            guide = self._library.official_reference
            skill = self._library.community_reference_writer
        elif director.mode == GenerationMode.T2VA:
            guide = self._library.official_base
            skill = self._library.community_text_writer
        else:
            guide = self._library.official_base
            skill = self._library.community_keyframe_writer
        return await self._complete(
            H3PromptCandidate,
            self._messages(
                "repair_writer" if repair_context else "mode_writer",
                skill,
                {
                    # Project memory is already compiled into the structured brief
                    # and continuity fields. Avoid resending the full transcript on
                    # every writer/repair call.
                    "request": _request_without_memory(request),
                    "director": director,
                    "multishot_plan": plan,
                    "repair_context": repair_context,
                },
                H3PromptCandidate,
                highest_instruction=request.highest_instruction,
                official_guide=guide,
                asset_image_urls=asset_image_urls,
                asset_media=tuple(zip(request.assets, asset_media_urls, strict=True)),
            ),
            stage="repair" if repair_context else "write",
        )

    async def _complete(
        self,
        model: type[BaseModel],
        messages: tuple[ChatMessage, ...],
        *,
        stage: str,
    ):
        active_messages = messages
        failures: list[str] = []
        for attempt in range(MAX_H3_SCHEMA_REPAIRS + 1):
            started = time.perf_counter()
            try:
                # H3 generation always uses provider streaming. httpx applies
                # the configured timeout between bytes, so once the first
                # token arrives a slow generation remains alive as long as the
                # provider continues making progress.
                callback_token: Token[Callable[[str], None] | None] | None = None
                if llm_delta_callback.get() is None:
                    callback_token = llm_delta_callback.set(lambda _delta: None)
                try:
                    raw = await self._client.complete_json(active_messages)
                finally:
                    if callback_token is not None:
                        llm_delta_callback.reset(callback_token)
            except Exception as exc:
                self._record_telemetry(
                    stage,
                    attempt,
                    active_messages,
                    "",
                    time.perf_counter() - started,
                    succeeded=False,
                    error_type=type(exc).__name__,
                )
                raise
            elapsed = time.perf_counter() - started
            try:
                result = model.model_validate(_decode_json(raw))
            except (ValueError, ValidationError) as exc:
                self._record_telemetry(
                    stage,
                    attempt,
                    active_messages,
                    raw,
                    elapsed,
                    succeeded=False,
                    error_type=type(exc).__name__,
                )
                failures.append(str(exc))
                if attempt == MAX_H3_SCHEMA_REPAIRS or not claim_business_repair():
                    raise H3PromptHarnessError(
                        f"H3 {model.__name__} response did not match its schema "
                        "within the shared one-repair budget",
                        tuple(failures),
                    ) from exc
                repair = json.dumps(
                    {
                        "task": "repair_invalid_h3_stage_json",
                        "target_schema": model.model_json_schema(),
                        "validation_error": str(exc),
                        "instruction": (
                            "修复 JSON 语法和字段结构，仅返回一个完整 JSON 对象。"
                            "保留有效内容，不要缩写提示词。执行描述继续遵循官方英文规范；"
                            "只有对白、歌词和画面内文字保留其原语言。"
                        ),
                    },
                    ensure_ascii=False,
                )
                active_messages = (
                    *messages,
                    ChatMessage(role="assistant", content=raw),
                    ChatMessage(role="user", content=repair),
                )
            else:
                self._record_telemetry(
                    stage,
                    attempt,
                    active_messages,
                    raw,
                    elapsed,
                    succeeded=True,
                )
                return result
        raise AssertionError("H3 schema repair loop must return or raise")

    def _record_telemetry(
        self,
        stage: str,
        attempt: int,
        messages: tuple[ChatMessage, ...],
        output: str,
        total_seconds: float,
        *,
        succeeded: bool,
        error_type: str | None = None,
    ) -> None:
        if self._telemetry_sink is None:
            return
        self._telemetry_sink(
            H3CallTelemetry(
                stage=stage,
                attempt=attempt,
                total_seconds=total_seconds,
                input_characters=sum(_message_character_count(item) for item in messages),
                output_characters=len(output),
                image_count=sum(_message_image_count(item) for item in messages),
                succeeded=succeeded,
                error_type=error_type,
            )
        )

    def _messages(
        self,
        stage: str,
        community_skill: str,
        context: object,
        response_model: type[BaseModel],
        *,
        highest_instruction: str = "",
        official_guide: str | None = None,
        asset_image_urls: tuple[str, ...] = (),
        asset_media: tuple[tuple[H3AssetInput, str | None], ...] = (),
    ) -> tuple[ChatMessage, ...]:
        system = (
            load_harness("h3-staged-system.md")
            + f"\n\n阶段：{stage}\n\n当前阶段 Skill：\n{community_skill}"
        )
        highest = highest_instruction.strip()
        if highest:
            system = (
                "PROJECT HIGHEST INSTRUCTION\n"
                f"{highest}\n\n"
                "The following Harness documents define the task-specific rules, output "
                "format, and safety boundaries.\n\n"
                f"{system}"
            )
        if stage == "reviewer":
            system += "\n\n" + load_harness("h3-reviewer-visual.md")
        if official_guide is not None:
            system += f"\n\n当前模式的官方规范：\n{official_guide}"
        payload = {
            "stage": stage,
            "response_schema": response_model.model_json_schema(),
            "context": _jsonable(context),
        }
        user_text = json.dumps(payload, ensure_ascii=False)
        user_content = user_text
        if asset_media:
            parts = [TextContentPart(text=user_text)]
            for asset, url in asset_media:
                if not url:
                    continue
                if asset.kind == H3AssetKind.IMAGE:
                    parts.append(ImageURLContentPart(image_url=ImageURL(url=url, detail="high")))
                elif asset.kind == H3AssetKind.VIDEO:
                    parts.append(VideoURLContentPart(video_url=VideoURL(url=url)))
            user_content = tuple(parts)
        return (
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user_content),
        )


def route_h3_mode(request: H3PromptRequest) -> GenerationMode:
    if not request.assets:
        return GenerationMode.T2VA
    if any(asset.kind.value != "image" for asset in request.assets):
        return GenerationMode.REF2VA
    boundary_roles = {H3AssetPromptRole.FIRST_FRAME, H3AssetPromptRole.LAST_FRAME}
    if any(asset.role not in boundary_roles for asset in request.assets):
        return GenerationMode.REF2VA
    roles = {asset.role for asset in request.assets}
    if roles == {H3AssetPromptRole.FIRST_FRAME} and len(request.assets) == 1:
        return GenerationMode.I2VA
    if roles == {H3AssetPromptRole.LAST_FRAME} and len(request.assets) == 1:
        return GenerationMode.L2VA
    if roles == boundary_roles and len(request.assets) == 2:
        return GenerationMode.FL2VA
    return GenerationMode.REF2VA


def deterministic_director_decision(request: H3PromptRequest) -> H3DirectorDecision:
    mode = route_h3_mode(request)
    use_multishot = _requires_multishot_plan(request)
    assumptions: tuple[str, ...] = ()
    if request.shot_strategy == H3ShotStrategy.AUTO:
        assumptions = ("AUTO 仅在创作说明明确包含剪切、蒙太奇或多个镜头标记时启用多镜头规划。",)
    return H3DirectorDecision(
        operation_id=request.operation_id,
        mode=mode,
        use_multishot=use_multishot,
        rationale=(
            f"依据素材职责确定为 {mode.value}；"
            + ("创作说明明确要求多镜头。" if use_multishot else "当前片段按单镜头执行。")
        ),
        assumptions=assumptions,
    )


def compile_h3_creative_brief(request: H3PromptRequest) -> tuple[H3CreativeBrief, tuple[str, ...]]:
    if request.creative is not None and request.creative.complete:
        return request.creative, ()
    source = request.creative or H3CreativeBrief()
    defaults = {
        "visual_style": "cinematic treatment consistent with the project brief",
        "action_arc": request.creative_brief,
        "composition": "clear subject staging with readable foreground and background separation",
        "camera_strategy": "restrained camera movement motivated by the visible action",
        "sound_plan": "synchronized physical Foley and spatial ambience",
        "desired_end_state": request.prior_continuity_state
        or "a stable readable state that completes the segment action",
    }
    values = source.model_dump()
    assumptions: list[str] = []
    for field, default in defaults.items():
        if not str(values[field]).strip():
            values[field] = default
            assumptions.append(f"Director automatically supplied {field}: {default}")
    return H3CreativeBrief.model_validate(values), tuple(assumptions)


def _requires_multishot_plan(request: H3PromptRequest) -> bool:
    if request.shot_strategy == H3ShotStrategy.MULTI:
        return True
    if request.shot_strategy == H3ShotStrategy.SINGLE:
        return False
    text = "\n".join((request.creative_brief, *request.constraints)).casefold()
    return any(marker in text for marker in _MULTISHOT_MARKERS)


def _request_without_memory(request: H3PromptRequest) -> H3PromptRequest:
    return request.model_copy(update={"project_memory": ""})


def _review_request_context(request: H3PromptRequest) -> dict[str, object]:
    return {
        "operation_id": request.operation_id,
        "segment_id": request.segment_id,
        "creative_brief": request.creative_brief,
        "duration_seconds": request.duration_seconds,
        "fps": request.fps,
        "assets": [
            {
                "asset_id": asset.asset_id,
                "label": asset.label,
                "companion_audio_label": asset.companion_audio_label,
                "kind": asset.kind.value,
                "role": asset.role.value,
                "preservation": asset.preservation,
            }
            for asset in request.assets
        ],
        "constraints": request.constraints,
    }


def validate_timeline(request: H3PromptRequest, shots: tuple[H3ShotBeat, ...]) -> tuple[str, ...]:
    errors: list[str] = []
    if [shot.shot_number for shot in shots] != list(range(1, len(shots) + 1)):
        errors.append("shot numbers must be consecutive and ordered")
    if shots and abs(shots[0].start_seconds) > 0.01:
        errors.append("timeline must begin at 0.00 seconds")
    for previous, current in zip(shots, shots[1:], strict=False):
        if abs(previous.end_seconds - current.start_seconds) > 0.01:
            errors.append("timeline contains a gap or overlap")
    if shots and abs(shots[-1].end_seconds - request.duration_seconds) > 0.01:
        errors.append("timeline must cover the full segment duration")
    return tuple(errors)


def validate_h3_candidate(
    request: H3PromptRequest,
    director: H3DirectorDecision,
    plan: H3MultishotPlan | None,
    candidate: H3PromptCandidate,
) -> tuple[str, ...]:
    errors: list[str] = []
    if candidate.operation_id != request.operation_id:
        errors.append("candidate operation_id does not match request")
    if candidate.mode != director.mode:
        errors.append("candidate mode does not match the director route")
    errors.extend(validate_timeline(request, candidate.timeline))
    if director.use_multishot and len(candidate.timeline) < 2:
        errors.append("multishot prompt must contain at least two shots")
    if not director.use_multishot and len(candidate.timeline) != 1:
        errors.append("single-shot prompt must contain exactly one shot")
    if plan is not None and candidate.timeline != plan.shots:
        errors.append("writer changed the approved multishot plan")

    if candidate.mode == GenerationMode.REF2VA:
        required = {
            "subject_definitions": candidate.subject_definitions or "",
            "summary": candidate.summary or "",
            "retention_analysis": candidate.retention_analysis or "",
            "detailed_description": candidate.detailed_description or "",
        }
        minimum_words = {
            "subject_definitions": 12,
            "summary": 15,
            "retention_analysis": 15,
            "detailed_description": MIN_REF_DESCRIPTION_WORDS,
        }
        for name, value in required.items():
            if not _uses_official_english(value):
                errors.append(
                    f"{name} must use English descriptive content; only dialogue, "
                    "lyrics, and visible text may retain their original language"
                )
            if _english_word_count(value) < minimum_words[name]:
                errors.append(
                    f"{name} is too short to be executable "
                    f"(minimum {minimum_words[name]} English words)"
                )
    else:
        description = candidate.integrated_multimodal_description or ""
        if not _uses_official_english(description):
            errors.append(
                "integrated_multimodal_description must use English descriptive content; "
                "only dialogue, lyrics, and visible text may retain their original language"
            )
        if _english_word_count(description) < MIN_BASE_DESCRIPTION_WORDS:
            errors.append(
                "integrated_multimodal_description is too short to be executable "
                f"(minimum {MIN_BASE_DESCRIPTION_WORDS} English words)"
            )
    if _english_word_count(candidate.overall_soundscape) < 6:
        errors.append("overall_soundscape must describe the native audio plan")
    elif not _uses_official_english(candidate.overall_soundscape):
        errors.append("overall_soundscape must use English descriptive content")
    if candidate.non_diegetic_music.strip().upper() != "N/A" and not _uses_official_english(
        candidate.non_diegetic_music
    ):
        errors.append("non_diegetic_music must use English descriptive content or N/A")

    rendered = render_h3_prompt(candidate)
    for shot in candidate.timeline:
        if f"[Shot {shot.shot_number}]" not in rendered:
            errors.append(f"prompt is missing [Shot {shot.shot_number}]")
    for asset in request.assets:
        if asset.label not in rendered:
            errors.append(f"prompt does not assign a role to {asset.label}")
        if asset.companion_audio_label and asset.companion_audio_label not in rendered:
            errors.append(f"prompt does not assign a role to {asset.companion_audio_label}")
        if asset.forbidden_propagation_targets and not any(
            target.casefold() in rendered.casefold()
            for target in asset.forbidden_propagation_targets
        ):
            errors.append(f"prompt does not state propagation exclusions for {asset.label}")
    if len(rendered) > request.runtime_limits.max_prompt_characters:
        errors.append("execution prompt exceeds runtime character limit")
    lower = rendered.casefold()
    if not director.use_multishot and any(marker in lower for marker in ("[shot 2]", "cut to")):
        errors.append("single-shot prompt contradicts itself with an explicit cut")
    if candidate.non_diegetic_music.strip().upper() == "N/A" and re.search(
        r"\b(score|soundtrack|background music|non-diegetic music)\b", lower
    ):
        errors.append("prompt requests music while non_diegetic_music is N/A")
    dialogue_words = sum(
        _english_word_count(match.group(0)) for match in _DIALOGUE_PATTERN.finditer(rendered)
    )
    if dialogue_words > request.duration_seconds * 3.5:
        errors.append("dialogue density exceeds the available segment duration")
    return tuple(dict.fromkeys(errors))


def render_h3_prompt(candidate: H3PromptCandidate) -> str:
    if candidate.mode == GenerationMode.REF2VA:
        fields = (
            ("subject_definitions", candidate.subject_definitions),
            ("summary", candidate.summary),
            ("retention_analysis", candidate.retention_analysis),
            ("detailed_description", candidate.detailed_description),
            ("overall_soundscape", candidate.overall_soundscape),
            ("non_diegetic_music", candidate.non_diegetic_music),
        )
    else:
        fields = (
            ("integrated_multimodal_description", candidate.integrated_multimodal_description),
            ("overall_soundscape", candidate.overall_soundscape),
            ("non_diegetic_music", candidate.non_diegetic_music),
        )
    return "\n\n".join(f"{name}: {value}" for name, value in fields)


T = TypeVar("T")


def _decode_json(content: str) -> object:
    value = content.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            value = "\n".join(lines[1:-1])
    return json.loads(value)


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _message_character_count(message: ChatMessage) -> int:
    if isinstance(message.content, str):
        return len(message.content)
    return sum(
        len(part.text) if isinstance(part, TextContentPart) else 0 for part in message.content
    )


def _message_image_count(message: ChatMessage) -> int:
    if isinstance(message.content, str):
        return 0
    return sum(isinstance(part, ImageURLContentPart) for part in message.content)


def _read(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise H3PromptHarnessError(f"required H3 harness document is missing: {path}") from exc
    if not value.strip():
        raise H3PromptHarnessError(f"required H3 harness document is empty: {path}")
    return value


_DIALOGUE_PATTERN = re.compile(r"<d>.*?</d>", flags=re.DOTALL | re.IGNORECASE)
_ENGLISH_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")


def _descriptive_text(value: str) -> str:
    return _DIALOGUE_PATTERN.sub("", value)


def _english_word_count(value: str) -> int:
    return len(_ENGLISH_WORD_PATTERN.findall(_descriptive_text(value)))


def _uses_official_english(value: str) -> bool:
    descriptive = _descriptive_text(value)
    latin_count = sum(character.isascii() and character.isalpha() for character in descriptive)
    cjk_count = sum("\u4e00" <= character <= "\u9fff" for character in descriptive)
    # A small amount of original-language visible text is valid. Descriptive prose is not.
    return latin_count >= 12 and cjk_count <= max(8, latin_count // 20)
