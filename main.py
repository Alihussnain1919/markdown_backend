import os
import re
import shutil
import tempfile

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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

ALLOWED_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".txt", ".csv", ".json"}


class UrlRequest(BaseModel):
    url: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/convert")
async def convert_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = md.convert(tmp_path)
        return {"filename": file.filename, "markdown": result.text_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)


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
        line = re.sub(r"<[^>]+>", "", line)  # strip inline vtt tags like <00:00:01.000>
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
async def convert_url(request: UrlRequest):
    url_str = request.url.strip()
    is_youtube = any(domain in url_str.lower() for domain in ["youtube.com", "youtu.be"])

    if not is_youtube:
        try:
            result = md.convert(url_str)
            return {"url": url_str, "markdown": result.text_content}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    try:
        video_id = _extract_video_id(url_str)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    primary_error = None

    # Attempt 1: youtube_transcript_api (with proxy if configured)
    try:
        text, lang, _ = _fetch_via_transcript_api(video_id)
        markdown_output = (
            f"# YouTube Video Transcript\n\n**Source URL:** {url_str}\n"
            f"**Language:** {lang}\n\n---\n\n{text}"
        )
        return {"url": url_str, "markdown": markdown_output}
    except Exception as e:
        primary_error = str(e)
        if not _is_blocklike_error(primary_error):
            raise HTTPException(status_code=500, detail=primary_error)

    # Attempt 2: yt-dlp with cookies, only makes sense if attempt 1 was IP-blocked
    try:
        text, lang = _fetch_via_ytdlp(video_id)
        markdown_output = (
            f"# YouTube Video Transcript\n\n**Source URL:** {url_str}\n"
            f"**Language:** {lang} (via yt-dlp fallback)\n\n---\n\n{text}"
        )
        return {"url": url_str, "markdown": markdown_output}
    except Exception as fallback_error:
        detail = (
            "This video's transcript couldn't be retrieved. YouTube blocked the request "
            "based on this server's IP address, which happens on any cloud host "
            "(Render included) and isn't fixed by redeploying elsewhere. "
            f"Primary method error: {primary_error} | Fallback (yt-dlp) error: {fallback_error}"
        )
        raise HTTPException(status_code=429, detail=detail)