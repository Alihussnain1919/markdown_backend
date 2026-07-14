import io
import ipaddress
import json
import os
import re
import shutil
import socket
import tempfile
import threading
import time
import zipfile
import mimetypes
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from markitdown import MarkItDown
from pydantic import BaseModel

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig

app = FastAPI(title="Markdown Converter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

md = MarkItDown()

DOC_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".txt", ".csv", ".json"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".opus", ".webm", ".flac"}
ALLOWED_EXT = DOC_EXT | IMAGE_EXT | AUDIO_EXT

# --- File size caps -----------------------------------------------------
# Every upload path used to have NO size limit at all -- a single large
# upload could exhaust Render's free-tier disk/RAM. These are enforced
# while streaming the upload to disk, not after the fact.
MAX_DOC_BYTES = 10 * 1024 * 1024      # 10MB
MAX_IMAGE_BYTES = 1 * 1024 * 1024     # matches OCR.space free-tier cap
MAX_AUDIO_BYTES = 15 * 1024 * 1024    # generous for a short voice note

OCR_SPACE_API_KEY = os.environ.get("OCR_SPACE_API_KEY")
OCR_SPACE_URL = "https://api.ocr.space/parse/image"
OCR_DAILY_LIMIT = 500        # per IP, per OCR.space's own free-tier docs
OCR_MONTHLY_LIMIT = 25000

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_WHISPER_URL = os.environ.get("WHISPER_URL")
UPSTASH_URL = os.environ.get("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN")


class UrlRequest(BaseModel):
    url: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/health")
def health_check():
    return {"status": "awake", "message": "Server is ready"}


# ---------------------------------------------------------------------------
# Text stats (character / word / rough token count)
# ---------------------------------------------------------------------------

def _compute_stats(text: str) -> dict:
    chars = len(text)
    words = len(text.split())
    # ~4 characters/token is the standard rough estimate used across most
    # tokenizers for English text -- good enough for a ballpark, not exact.
    approx_tokens = max(0, round(chars / 4))
    return {"characters": chars, "words": words, "approx_tokens": approx_tokens}


# ---------------------------------------------------------------------------
# OCR usage tracking (persisted to disk so it survives Render's sleep/wake)
# ---------------------------------------------------------------------------

# USAGE_FILE = os.path.join(tempfile.gettempdir(), "ocr_usage.json")
# _usage_lock = threading.Lock()


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _month_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _redis_available() -> bool:
    return bool(UPSTASH_URL and UPSTASH_TOKEN)
 
 
def _redis_cmd(*args):
    """Run one Redis command via Upstash's REST API."""
    if not _redis_available():
        raise RuntimeError(
            "Database isn't configured. Set UPSTASH_REDIS_REST_URL and "
            "UPSTASH_REDIS_REST_TOKEN in Render's Environment tab "
            "(free database at upstash.com, no credit card needed)."
        )
    response = requests.post(
        UPSTASH_URL,
        headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
        json=list(args),
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"Database error: {data['error']}")
    return data.get("result")
 
 
# --- OCR usage tracking (real-time counters, survive sleep/wake AND redeploys) ---
 
def get_ocr_usage() -> dict:
    if not _redis_available():
        # Degrade gracefully instead of crashing the whole /convert endpoint
        # if the database isn't configured yet.
        return {
            "daily": {"used": 0, "limit": OCR_DAILY_LIMIT, "remaining": OCR_DAILY_LIMIT},
            "monthly": {"used": 0, "limit": OCR_MONTHLY_LIMIT, "remaining": OCR_MONTHLY_LIMIT},
        }
 
    daily_key = f"ocr:daily:{_today_str()}"
    monthly_key = f"ocr:monthly:{_month_str()}"
    daily_used = int(_redis_cmd("GET", daily_key) or 0)
    monthly_used = int(_redis_cmd("GET", monthly_key) or 0)
 
    return {
        "daily": {
            "used": daily_used,
            "limit": OCR_DAILY_LIMIT,
            "remaining": max(0, OCR_DAILY_LIMIT - daily_used),
        },
        "monthly": {
            "used": monthly_used,
            "limit": OCR_MONTHLY_LIMIT,
            "remaining": max(0, OCR_MONTHLY_LIMIT - monthly_used),
        },
    }
 
 
def _record_ocr_call() -> None:
    if not _redis_available():
        return  # usage tracking is best-effort; never block OCR itself
    daily_key = f"ocr:daily:{_today_str()}"
    monthly_key = f"ocr:monthly:{_month_str()}"
    _redis_cmd("INCR", daily_key)
    _redis_cmd("EXPIRE", daily_key, 172800)          # 2-day safety TTL
    _redis_cmd("INCR", monthly_key)
    _redis_cmd("EXPIRE", monthly_key, 60 * 60 * 24 * 40)  # ~40-day safety TTL
 
def _check_ocr_quota() -> None:
    # This calls the NEW get_ocr_usage() which talks to Upstash Redis!
    usage = get_ocr_usage()
    
    if usage["daily"]["remaining"] <= 0:
        raise RuntimeError(
            "Daily free OCR quota (500 requests) is used up for today. It resets at midnight UTC."
        )
    if usage["monthly"]["remaining"] <= 0:
        raise RuntimeError(
            "Monthly free OCR quota (25,000 requests) is used up. It resets at the start of next month."
        )
     
@app.get("/usage")
def usage():
    return get_ocr_usage()
 


# ---------------------------------------------------------------------------
# Very lightweight per-IP rate limiting (in-memory sliding window)
#
# Without this, a single visitor could hammer /convert or /convert-url and
# burn through the shared free OCR quota or the Render free compute minutes
# on their own. This is not a replacement for a real rate limiter under
# heavy traffic, but it stops accidental/careless abuse on a small project.
# ---------------------------------------------------------------------------

_rate_lock = threading.Lock()
_rate_buckets: dict[str, list[float]] = {}
RATE_LIMIT_WINDOW_SECONDS = 600  # 10 minutes
RATE_LIMIT_MAX_REQUESTS = 30


def _enforce_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets.setdefault(ip, [])
        bucket[:] = [t for t in bucket if now - t < RATE_LIMIT_WINDOW_SECONDS]
        if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail="Too many requests from this IP. Please wait a few minutes and try again.",
            )
        bucket.append(now)


# ---------------------------------------------------------------------------
# OCR (image -> text) via OCR.space free API
# ---------------------------------------------------------------------------

def _ocr_image(path: str) -> str:
    if not OCR_SPACE_API_KEY:
        raise RuntimeError(
            "OCR_SPACE_API_KEY is not set. Get a free key at "
            "https://ocr.space/ocrapi and add it in Render's Environment tab."
        )

    _check_ocr_quota()

    with open(path, "rb") as f:
        response = requests.post(
            OCR_SPACE_URL,
            files={"file": f},
            data={"apikey": OCR_SPACE_API_KEY, "language": "eng", "OCREngine": 2},
            timeout=30,
        )
    _record_ocr_call()

    data = response.json()

    if data.get("IsErroredOnProcessing"):
        error_msg = data.get("ErrorMessage", ["Unknown OCR error"])
        raise RuntimeError(error_msg[0] if isinstance(error_msg, list) else str(error_msg))

    parsed_results = data.get("ParsedResults") or []
    if not parsed_results:
        return "*No text was detected in this image.*"

    text = parsed_results[0].get("ParsedText", "").strip()
    return text if text else "*No text was detected in this image.*"


# ---------------------------------------------------------------------------
# Audio / voice message transcription via Hugging Face free Inference API
# ---------------------------------------------------------------------------

def _transcribe_audio(path: str) -> str:
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. Get a free token at https://huggingface.co/settings/tokens "
            "and add it in Render's Environment tab."
        )

    content_type, _ = mimetypes.guess_type(path)

    if not content_type:
        content_type = "audio/mpeg"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": content_type,
        "Accept": "application/json"
    }

    with open(path, "rb") as f:
        audio_bytes = f.read()

    # The model may need to "cold start" on Hugging Face's shared free
    # infrastructure -- it returns 503 + estimated_time while loading.
    # Retry a few times rather than failing on the very first call.
    max_attempts = 4
    last_error = None
    for attempt in range(max_attempts):
        response = requests.post(HF_WHISPER_URL, headers=headers, data=audio_bytes, timeout=60)

        if response.status_code == 200:
            result = response.json()
            text = (result.get("text") or "").strip()
            return text if text else "*No speech was detected in this audio.*"

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if response.status_code == 503 and "estimated_time" in payload:
            wait = min(float(payload["estimated_time"]), 20)
            last_error = payload.get("error", "Model is loading")
            time.sleep(wait)
            continue

        last_error = (
        payload.get("error")
        or response.text[:500]
        or f"HTTP {response.status_code}"
    )
        break

    raise RuntimeError(f"Transcription failed: {last_error}")


# ---------------------------------------------------------------------------
# Shared per-file conversion logic (used by both /convert and /convert-batch)
# ---------------------------------------------------------------------------

def _max_bytes_for(ext: str) -> int:
    if ext in IMAGE_EXT:
        return MAX_IMAGE_BYTES
    if ext in AUDIO_EXT:
        return MAX_AUDIO_BYTES
    return MAX_DOC_BYTES


def _save_upload_with_limit(file: UploadFile, ext: str) -> tuple[str, int]:
    """Stream the upload to a temp file, aborting early if it exceeds the cap
    for its file type. Returns (temp_path, bytes_written)."""
    limit = _max_bytes_for(ext)
    written = 0
    chunk_size = 1024 * 1024
 
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = file.file.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large for {ext} (max {limit // (1024 * 1024)}MB on the free tier).",
                    )
                out.write(chunk)
    except Exception:
        os.remove(tmp_path)
        raise
    return tmp_path, written
 
 
def _convert_single(file: UploadFile) -> dict:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
 
    tmp_path, original_bytes = _save_upload_with_limit(file, ext)
 
    try:
        if ext in IMAGE_EXT:
            extracted = _ocr_image(tmp_path)
            markdown_output = (
                f"# OCR Result: {file.filename}\n\n"
                f"*Text extracted with OCR.space*\n\n---\n\n{extracted}"
            )
        elif ext in AUDIO_EXT:
            extracted = _transcribe_audio(tmp_path)
            markdown_output = (
                f"# Voice Transcript: {file.filename}\n\n"
                f"*Transcribed with Whisper (Hugging Face)*\n\n---\n\n{extracted}"
            )
        else:
            result = md.convert(tmp_path)
            markdown_output = result.text_content
 
        stats = _compute_stats(markdown_output)
        stats["original_bytes"] = original_bytes
        stats["output_bytes"] = len(markdown_output.encode("utf-8"))
 
        return {
            "filename": file.filename,
            "markdown": markdown_output,
            "stats": stats,
        }
    finally:
        os.remove(tmp_path)


@app.post("/convert")
async def convert_file(request: Request, file: UploadFile = File(...)):
    _enforce_rate_limit(request)
    try:
        return _convert_single(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


MAX_BATCH_FILES = 4
MAX_BATCH_SIZE_BYTES = 40 * 1024 * 1024  # 40 MB

@app.post("/convert-batch")
async def convert_batch(request: Request, files: List[UploadFile] = File(...), as_zip: bool = False):
    """Convert multiple files in one request. Each file succeeds or fails
    independently -- one bad file in the batch won't sink the rest."""
    _enforce_rate_limit(request)

    # 1. Reject if more than 4 files are submitted
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch limit is {MAX_BATCH_FILES} files at a time."
        )

    # 2. Reject if the total combined file size is over 40MB
    total_batch_size = 0
    for f in files:
        if f.size:
            total_batch_size += f.size

    if total_batch_size > MAX_BATCH_SIZE_BYTES:
        raise HTTPException(
            status_code=400, 
            detail="The combined size of your files exceeds the 40 MB batch limit."
        )

    results = []
    for f in files:
        try:
            results.append(_convert_single(f))
        except HTTPException as e:
            results.append({"filename": f.filename, "error": e.detail})
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    if not as_zip:
        return {"results": results}

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            if "markdown" in r:
                base = os.path.splitext(r["filename"])[0]
                zf.writestr(f"{base}.md", r["markdown"])
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=converted.zip"},
    )


# ---------------------------------------------------------------------------
# SSRF protection for URL-based conversion
#
# md.convert(url) makes an outbound HTTP request FROM THE SERVER. Without
# validation, a user could pass internal addresses (localhost, the cloud
# metadata IP 169.254.169.254, a private 10.x/192.168.x address, etc.) and
# use the server as a proxy to probe or read internal-only resources. This
# blocks anything that doesn't resolve to a public IP.
# ---------------------------------------------------------------------------

def _validate_public_url(url_str: str) -> None:
    parsed = urlparse(url_str)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http:// and https:// URLs are allowed.")
    if not parsed.hostname:
        raise ValueError("Could not determine the host from that URL.")

    try:
        resolved_ip = socket.gethostbyname(parsed.hostname)
    except socket.gaierror:
        raise ValueError("Could not resolve that hostname.")

    ip_obj = ipaddress.ip_address(resolved_ip)
    if (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_reserved
        or ip_obj.is_multicast
        or ip_obj.is_unspecified
    ):
        raise ValueError("That URL points to a private or internal address and can't be fetched.")


# ---------------------------------------------------------------------------
# YouTube transcript fetching
#
# Root cause of the 429s: YouTube blocks /api/timedtext requests based on the
# IP/ASN they come from. Cloud hosts (Render, AWS, GCP, etc.) are commonly
# flagged, so this can fail on the very first request regardless of which
# provider you deploy to. There is no code-only trick that changes this --
# only changing *what IP/identity the request carries* helps:
#
#   1. youtube_transcript_api with a residential proxy (paid, most reliable)
#   2. yt-dlp using cookies from a real logged-in YouTube session (free,
#      works because the request now looks like an authenticated browser
#      session instead of anonymous datacenter traffic -- not a bypass of
#      IP blocking, a different mitigation: authenticated requests get more
#      leeway than anonymous ones)
#
# We try (1) first since it's fast and needs no external file, then fall
# back to (2) if a cookies file has been configured.
# ---------------------------------------------------------------------------

def _get_proxy_config():
    ws_user = os.environ.get("WEBSHARE_PROXY_USERNAME")
    ws_pass = os.environ.get("WEBSHARE_PROXY_PASSWORD")
    if ws_user and ws_pass:
        return WebshareProxyConfig(proxy_username=ws_user, proxy_password=ws_pass)

    generic_url = os.environ.get("GENERIC_PROXY_URL")
    if generic_url:
        https_url = os.environ.get("HTTPS_PROXY_URL", generic_url)
        return GenericProxyConfig(http_url=generic_url, https_url=https_url)

    return None


def _extract_video_id(url_str: str) -> str:
    match = re.search(r"(?:v=|\/v\/|youtu\.be\/|\/embed\/)([a-zA-Z0-9_-]{11})", url_str)
    if not match:
        raise ValueError("Could not parse a valid YouTube video ID from the link.")
    return match.group(1)


def _is_blocklike_error(err_str: str) -> bool:
    markers = ["429", "Too Many Requests", "IpBlocked", "RequestBlocked", "blocked it"]
    return any(m in err_str for m in markers)


def _fetch_via_transcript_api(video_id: str):
    proxy_config = _get_proxy_config()
    ytt_api = YouTubeTranscriptApi(proxy_config=proxy_config)
    transcript_list = ytt_api.list(video_id)

    primary_transcript = next(iter(transcript_list), None)
    if primary_transcript is None:
        raise RuntimeError("No transcript tracks are available for this video.")

    if primary_transcript.is_translatable:
        try:
            primary_transcript = primary_transcript.translate("en")
        except Exception:
            pass

    transcript_data = primary_transcript.fetch()
    text = " ".join(snippet.text for snippet in transcript_data)
    return text, primary_transcript.language, proxy_config is not None


def _clean_vtt(raw: str) -> str:
    lines = raw.splitlines()
    out = []
    seen_last = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        if "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        if line and line != seen_last:
            out.append(line)
            seen_last = line
    return " ".join(out)


def _fetch_via_ytdlp(video_id: str):
    import yt_dlp

    cookies_file = os.environ.get("YTDLP_COOKIES_FILE")
    if not cookies_file or not os.path.exists(cookies_file):
        raise RuntimeError(
            "yt-dlp fallback is not configured: no cookies file found. "
            "Set YTDLP_COOKIES_FILE to a Netscape-format cookies.txt exported "
            "from a real logged-in YouTube session."
        )

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-orig"],
        "quiet": True,
        "no_warnings": True,
        "cookiefile": cookies_file,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

    subs = info.get("requested_subtitles") or {}
    if not subs:
        raise RuntimeError("yt-dlp found no subtitle tracks for this video.")

    lang, sub_info = next(iter(subs.items()))
    resp = requests.get(sub_info["url"], timeout=20)
    resp.raise_for_status()
    text = _clean_vtt(resp.text) if "vtt" in sub_info.get("ext", "vtt") else resp.text
    return text, lang

@app.post("/convert-url")
async def convert_url(request: Request, body: UrlRequest):
    _enforce_rate_limit(request)
    url_str = body.url.strip()
    is_youtube = any(domain in url_str.lower() for domain in ["youtube.com", "youtu.be"])

    if not is_youtube:
        try:
            _validate_public_url(url_str)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        try:
            result = md.convert(url_str)
            return {
                "url": url_str,
                "markdown": result.text_content,
                "stats": _compute_stats(result.text_content),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # --- YouTube: layered fallback. Each attempt raises its OWN real error
    # instead of silently returning a metadata-only page (that's what
    # md.convert() alone was doing -- it never told you the transcript step
    # had failed, it just skipped it). ---
    try:
        video_id = _extract_video_id(url_str)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    primary_error = None

    # Attempt 1: youtube_transcript_api (with proxy if WEBSHARE_* / GENERIC_PROXY_URL is set)
    try:
        text, lang, _ = _fetch_via_transcript_api(video_id)
        markdown_output = (
            f"# YouTube Video Transcript\n\n**Source URL:** {url_str}\n"
            f"**Language:** {lang}\n\n---\n\n{text}"
        )
        return {"url": url_str, "markdown": markdown_output, "stats": _compute_stats(markdown_output)}
    except Exception as e:
        primary_error = str(e)
        if not _is_blocklike_error(primary_error):
            raise HTTPException(status_code=500, detail=primary_error)

    # Attempt 2: yt-dlp with cookies (only worth trying if attempt 1 looked IP-blocked)
    try:
        text, lang = _fetch_via_ytdlp(video_id)
        markdown_output = (
            f"# YouTube Video Transcript\n\n**Source URL:** {url_str}\n"
            f"**Language:** {lang} (via yt-dlp fallback)\n\n---\n\n{text}"
        )
        return {"url": url_str, "markdown": markdown_output, "stats": _compute_stats(markdown_output)}
    except Exception as fallback_error:
        detail = (
            "This video's transcript couldn't be retrieved automatically -- YouTube is blocking "
            "requests from this server's IP address, which is common on free cloud hosts and isn't "
            "fixed by redeploying elsewhere. Use 'Paste transcript manually' instead: open the video "
            "on YouTube, click the '...' menu below the video, choose 'Show transcript', copy the text, "
            "and paste it in. "
            f"(Primary error: {primary_error} | yt-dlp error: {fallback_error})"
        )
        raise HTTPException(status_code=429, detail=detail)
    
# =============================================================================
# PATCH 2: Free reviews system (persisted to disk, same pattern as usage
# tracking -- survives Render's sleep/wake, resets on a fresh redeploy).
# No sign-up, no accounts -- just a name, star rating, and short comment.
# =============================================================================

# --- Reviews (persisted list, newest first, capped at MAX_REVIEWS_STORED) ---
 
MAX_REVIEWS_STORED = 200
MAX_NAME_LEN = 60
MAX_COMMENT_LEN = 500
 
 
class ReviewRequest(BaseModel):
    name: str
    rating: int
    comment: str
 
 
def get_reviews_from_db() -> list:
    if not _redis_available():
        return []
    raw_items = _redis_cmd("LRANGE", "reviews", 0, MAX_REVIEWS_STORED - 1) or []
    reviews = []
    for item in raw_items:
        try:
            reviews.append(json.loads(item))
        except (json.JSONDecodeError, TypeError):
            continue
    return reviews
 
 
def add_review_to_db(review: dict) -> None:
    _redis_cmd("LPUSH", "reviews", json.dumps(review))
    _redis_cmd("LTRIM", "reviews", 0, MAX_REVIEWS_STORED - 1)
 
 
@app.get("/reviews")
def get_reviews():
    return {"reviews": get_reviews_from_db()}
 
 
@app.post("/reviews")
async def post_review(request: Request, body: ReviewRequest):
    # 1. First, run your general, fast rate-limit check
    _enforce_rate_limit(request)
 
    # 2. Get the user's IP address
    ip = request.client.host if request.client else "unknown"
    
    # 3. Clean and Validate input data BEFORE touching the database
    name = body.name.strip()[:MAX_NAME_LEN]
    comment = body.comment.strip()[:MAX_COMMENT_LEN]
 
    if not name:
        raise HTTPException(status_code=400, detail="Please enter a name (or a nickname).")
    if not comment:
        raise HTTPException(status_code=400, detail="Please enter a comment.")
    if body.rating < 1 or body.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5.")

    # 4. NOW we check and update Upstash Redis
    if _redis_available() and ip != "unknown":
        ip_review_key = f"review:limit:{ip}"
        
        # Get how many reviews this IP has submitted today
        reviews_today = int(_redis_cmd("GET", ip_review_key) or 0)
        
        if reviews_today >= 5:
            raise HTTPException(
                status_code=429,
                detail="You have reached the maximum limit of 5 reviews per day. Thank you!"
            )
            
        # If they are under the limit, increment their count
        _redis_cmd("INCR", ip_review_key)
        # Set it to expire in 24 hours (86,400 seconds) so it resets tomorrow
        _redis_cmd("EXPIRE", ip_review_key, 86400)

    # 5. Save the valid review to the database
    review = {
        "name": name,
        "rating": body.rating,
        "comment": comment,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
 
    try:
        add_review_to_db(review)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
 
    return {"status": "ok", "review": review}
    
@app.get("/health")
def health():
    return {
        "status": "awake",
        "message": "Server is ready"
    }