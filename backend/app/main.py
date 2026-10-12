import io
import hmac
import logging
import uuid
import warnings
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .catalog import NOTICE, POLICY_VERSION, get_guidance, get_policy_version, list_guidance
from .aws_runtime_config import get_runtime_config
from .inference import ModelUnavailableError, YoloInference
from .schemas import ClassificationResponse, Detection
from .settings import settings

logger = logging.getLogger(__name__)
import os
Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
engine = YoloInference(
    settings.model_path,
    settings.classifier_path,
    s3_bucket=settings.model_s3_bucket or None,
    model_s3_key=settings.model_s3_key,
    classifier_s3_key=settings.classifier_s3_key,
    model_sha256=settings.model_sha256,
    classifier_sha256=settings.classifier_sha256,
    classifier_threshold=settings.classifier_threshold,
    detector_threshold=settings.detector_threshold,
    classifier_fallback_threshold=settings.classifier_fallback_threshold,
)

logger.warning(
    "RION MODEL DIAGNOSTICS | cwd=%s | detector_path=%s | detector_exists=%s | classifier_path=%s | classifier_exists=%s",
    os.getcwd(),
    engine.model_path.resolve(),
    engine.model_path.is_file(),
    engine.classifier_path.resolve() if engine.classifier_path else None,
    engine.classifier_path.is_file() if engine.classifier_path else False,
)

app = FastAPI(
    title="RION — Resource Intelligence & Optimization Network API",
    description="Image-based material estimates with explicitly illustrative route guidance.",
    version="0.2.0",
    docs_url="/docs" if settings.app_env == "development" else None,
    redoc_url=None,
)


class HealthCheckTrustedHostMiddleware(TrustedHostMiddleware):
    """ALB health checks use a target's private IP as Host, not its DNS name."""

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http" and scope.get("path") == "/health":
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)


app.add_middleware(
    HealthCheckTrustedHostMiddleware,
    allowed_hosts=settings.host_allowlist,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def security_headers(request, call_next):
    if settings.runtime_secret_arn and request.url.path != "/health":
        try:
            token = get_runtime_config().get("origin_verify_token", "")
        except Exception:
            logger.exception("RION runtime secret could not be read")
            return JSONResponse(status_code=503, content={"detail": "Service configuration is unavailable."})
        supplied = request.headers.get("x-rion-origin-token", "")
        if not token or not hmac.compare_digest(supplied, token):
            return JSONResponse(status_code=403, content={"detail": "Forbidden"})
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(HTTPException)
async def http_error_response(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_available": engine.available,
        "classifier_available": engine.classifier_available,
        "model_version": engine.version if engine.available else None,
        "policy_mode": "illustrative",
        "policy_version": get_policy_version(),
    }


@app.get("/api/v1/materials")
def materials() -> dict[str, object]:
    return {
        "policy_mode": "illustrative",
        "policy_version": POLICY_VERSION,
        "notice": NOTICE,
        "materials": list_guidance(),
    }


def decode_image(raw: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as probe:
                if probe.format not in {"JPEG", "PNG", "WEBP"}:
                    raise HTTPException(status_code=415, detail="Use a JPEG, PNG, or WebP image.")
                if probe.width * probe.height > settings.max_image_pixels:
                    raise HTTPException(status_code=413, detail="Image dimensions exceed the configured pixel limit.")
                probe.verify()
            with Image.open(io.BytesIO(raw)) as decoded:
                decoded.load()
                return decoded.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid supported image.") from exc


@app.post("/api/v1/classifications", response_model=ClassificationResponse)
@app.post("/api/v1/live-frame", response_model=ClassificationResponse)
async def classify_image(
    image: Annotated[UploadFile, File(description="JPEG, PNG, or WebP image")],
    facility_id: Annotated[str, Form()] = "example_facility",
) -> ClassificationResponse:
    # The original filename and claimed content type are deliberately ignored.
    raw = await image.read(settings.max_upload_bytes + 1)
    await image.close()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {settings.max_upload_mb} MB upload limit.")

    decoded = decode_image(raw)
    try:
        raw_detections = await run_in_threadpool(engine.predict, decoded)
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="No trained material model is configured. Train a model and set MODEL_PATH.",
        ) from exc
    except Exception as exc:
        # Avoid exposing model internals or local paths in public error responses.
        logger.exception("Material model inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed. Check the backend log for details.") from exc

    detections: list[Detection] = []
    for item in raw_detections:
        guidance = get_guidance(item["material"], facility_id)
        detections.append(
            Detection(
                box=item["box"],
                object_label=str(item["object_label"]),
                material=item["material"],
                confidence=float(item["confidence"]),
                route=guidance["route"],
                reason=str(guidance["reason"]),
                needs_review=bool(guidance["needs_review"]),
                policy_source=guidance["policy_source"],
                possible_outputs=guidance["possible_outputs"],
                reuse_ideas=guidance["reuse_ideas"],
            )
        )

    return ClassificationResponse(
        request_id=str(uuid.uuid4()),
        model_version=engine.version,
        policy_version=get_policy_version(facility_id),
        policy_mode="illustrative",
        facility_id=facility_id[:80],
        detections=detections,
        notice=NOTICE,
    )

