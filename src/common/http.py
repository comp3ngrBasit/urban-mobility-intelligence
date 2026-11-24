"""HTTP session with exponential-backoff retries, timeouts and streaming downloads."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.common.config import get_settings
from src.common.logging import get_logger

log = get_logger(__name__)


class SourceUnavailable(RuntimeError):
    """Raised when an external source cannot be reached after all retries."""


def build_session(extra_headers: dict[str, str] | None = None) -> requests.Session:
    s = get_settings()
    retry = Retry(
        total=s.http_max_retries,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST", "HEAD"),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "urban-mobility-intelligence/1.0 (+research)"})
    if extra_headers:
        session.headers.update(extra_headers)
    return session


def get_json(session: requests.Session, url: str, params: dict[str, Any] | None = None) -> Any:
    try:
        resp = session.get(url, params=params, timeout=get_settings().http_timeout_s)
    except requests.RequestException as exc:  # network-level failure after retries
        raise SourceUnavailable(f"{url}: {exc}") from exc
    if resp.status_code >= 400:
        raise SourceUnavailable(f"{url}: HTTP {resp.status_code} {resp.text[:200]}")
    return resp.json()


def download_file(session: requests.Session, url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Stream to a temp file then atomically move into place (no partial files on failure)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with session.get(url, stream=True, timeout=get_settings().http_timeout_s) as resp:
            if resp.status_code == 404:
                raise FileNotFoundError(url)
            if resp.status_code >= 400:
                raise SourceUnavailable(f"{url}: HTTP {resp.status_code}")
            with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent, suffix=".part") as tmp:
                for block in resp.iter_content(chunk_size=chunk):
                    tmp.write(block)
                tmp_path = Path(tmp.name)
    except requests.RequestException as exc:
        raise SourceUnavailable(f"{url}: {exc}") from exc
    shutil.move(tmp_path, dest)
    log.info("downloaded %s -> %s (%.1f MB)", url, dest, dest.stat().st_size / 1e6)
    return dest
