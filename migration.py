# requirements: pip install requests
from __future__ import annotations
import argparse
import json
import os
import random
import re
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Union
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from requests import HTTPError

DEFAULT_CONFIG_PATH = Path(os.getenv("MIGRATION_CONFIG", "config.json"))

CONFIG: Optional[MigrationConfig] = None

MAYAN_BASE = ""
MAYAN_API_BASE = ""
MAYAN_USER = ""
MAYAN_PASS = ""
MAYAN_MOVED_TAG_LABEL = ""

PAPERLESS_BASE = ""
MAPPINGS: Dict[str, Dict[str, str]] = {}

DEBUG_ENABLED = False
COUNT_ONLY_MODE = False
DOWNLOAD_FIRST_MODE = False
EXPORT_POLL_INTERVAL = 1.0
EXPORT_POLL_TIMEOUT = 60.0
EMPTY_DOWNLOAD_RETRY_LIMIT = 5


def debug(message: str) -> None:
    if DEBUG_ENABLED:
        print(f"[DEBUG] {message}")

sess_mayan = requests.Session()
sess_mayan.auth = (MAYAN_USER, MAYAN_PASS)

sess_paperless = requests.Session()

_MAYAN_TAG_CACHE: Dict[str, Optional[int]] = {}
_MAYAN_DOCUMENT_TYPE_CACHE: Dict[str, Optional[int]] = {}
_MAYAN_DOC_TAG_CACHE: Dict[int, List[Dict[str, Any]]] = {}
_MAYAN_DOC_DETAIL_CACHE: Dict[int, Dict[str, Any]] = {}

QueryPrimitive = Union[str, bytes, int, float]
QueryValue = Union[QueryPrimitive, Iterable[QueryPrimitive], None]
QueryParams = Dict[str, QueryValue]


@dataclass
class MigrationConfig:
    mayan_base: str
    paperless_base: str
    mayan_user: str
    mayan_pass: str
    mayan_moved_tag: str
    mappings: Dict[str, Dict[str, str]]
    export_poll_interval: float
    export_poll_timeout: float
    download_retry_limit: int
    download_backoff_min: float
    download_backoff_max: float
    download_backoff_multiplier: float


class ExponentialBackoff:
    def __init__(
        self,
        initial: float,
        maximum: float,
        multiplier: float = 2.0,
    ) -> None:
        self.initial = initial
        self.maximum = maximum
        self.multiplier = multiplier
        self.current = initial

    def next_delay(self) -> float:
        delay = self.current
        self.current = min(self.current * self.multiplier, self.maximum)
        return delay + random.uniform(0, delay * 0.1)

    def reset(self) -> None:
        self.current = self.initial


def apply_runtime_config(cfg: MigrationConfig) -> None:
    global CONFIG, MAYAN_BASE, MAYAN_API_BASE, PAPERLESS_BASE
    global EXPORT_POLL_INTERVAL, EXPORT_POLL_TIMEOUT, EMPTY_DOWNLOAD_RETRY_LIMIT
    global MAYAN_USER, MAYAN_PASS, MAYAN_MOVED_TAG_LABEL, MAPPINGS

    CONFIG = cfg
    MAYAN_BASE = cfg.mayan_base.rstrip("/")
    MAYAN_API_BASE = f"{MAYAN_BASE}/api/v4".rstrip("/")
    PAPERLESS_BASE = cfg.paperless_base.rstrip("/")
    EXPORT_POLL_INTERVAL = cfg.export_poll_interval
    EXPORT_POLL_TIMEOUT = cfg.export_poll_timeout
    EMPTY_DOWNLOAD_RETRY_LIMIT = cfg.download_retry_limit
    MAYAN_USER = cfg.mayan_user
    MAYAN_PASS = cfg.mayan_pass
    MAYAN_MOVED_TAG_LABEL = cfg.mayan_moved_tag
    MAPPINGS = cfg.mappings


def reset_mayan_caches() -> None:
    _MAYAN_TAG_CACHE.clear()
    _MAYAN_DOCUMENT_TYPE_CACHE.clear()
    _MAYAN_DOC_TAG_CACHE.clear()
    _MAYAN_DOC_DETAIL_CACHE.clear()

SCRIPT_REDIRECT_PATTERNS = [
    re.compile(r"window\.(?:location(?:\.href)?)\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE),
    re.compile(r"window\.open\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE),
]

DOWNLOAD_URL_PATTERNS = [
    re.compile(r"/api/v4/documents/(?P<doc>\d+)/versions/(?P<ver>\d+)/download/?"),
    re.compile(r"/documents/(?P<doc>\d+)/versions/(?P<ver>\d+)/download/?"),
    re.compile(r"/api/v4/documents/document_versions/(?P<ver>\d+)/download/?"),
    re.compile(r"/documents/document_versions/(?P<ver>\d+)/download/?"),
]


def paperless_get_or_create_tag(token: str, name: str) -> int:
    r = sess_paperless.get(
        f"{PAPERLESS_BASE}/api/tags/?name__iexact={name}",
        headers={"Authorization": f"Token {token}"},
    )
    r.raise_for_status()
    data = r.json()
    if data["count"] > 0:
        return data["results"][0]["id"]
    r = sess_paperless.post(
        f"{PAPERLESS_BASE}/api/tags/",
        headers={"Authorization": f"Token {token}"},
        json={"name": name},
    )
    r.raise_for_status()
    return r.json()["id"]


def mayan_find_tag_id(tag_label: str) -> Optional[int]:
    if tag_label in _MAYAN_TAG_CACHE:
        return _MAYAN_TAG_CACHE[tag_label]
    url = f"{MAYAN_API_BASE}/tags/"
    params: Optional[QueryParams] = {"page_size": 100, "search": tag_label}
    normalized_label = tag_label.lower()
    debug(f"Looking up Mayan tag '{tag_label}'")
    while url:
        debug_params = params.copy() if params else None
        debug(f"GET {url} params={debug_params}" if debug_params else f"GET {url}")
        r = _mayan_get(url, params=params)
        try:
            r.raise_for_status()
        except HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                _MAYAN_TAG_CACHE[tag_label] = None
                debug(f"Tag endpoint not found (404) while searching for '{tag_label}'")
                return None
            raise
        data = r.json()
        results = data.get("results", [])
        sample_labels = ", ".join(t.get("label", "") for t in results[:5])
        debug(
            f"Page returned {len(results)} tags; sample labels: {sample_labels or 'n/a'}"
        )
        for tag in results:
            if tag.get("label", "").lower() == normalized_label:
                tag_id = tag.get("id")
                _MAYAN_TAG_CACHE[tag_label] = tag_id
                debug(f"Found tag '{tag_label}' with id {tag_id}")
                return tag_id
        url = data.get("next")
        params = None
    debug(f"Finished searching; tag '{tag_label}' not found")
    _MAYAN_TAG_CACHE[tag_label] = None
    return None


def mayan_find_document_type_id(type_label: str) -> Optional[int]:
    if type_label in _MAYAN_DOCUMENT_TYPE_CACHE:
        return _MAYAN_DOCUMENT_TYPE_CACHE[type_label]

    url = f"{MAYAN_API_BASE}/document_types/"
    params: Optional[QueryParams] = {"page_size": 100, "search": type_label}
    normalized_label = type_label.lower()
    debug(f"Looking up Mayan document type '{type_label}'")
    while url:
        debug_params = params.copy() if params else None
        debug(f"GET {url} params={debug_params}" if debug_params else f"GET {url}")
        r = _mayan_get(url, params=params)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        sample_labels = ", ".join(dt.get("label", "") for dt in results[:5])
        debug(
            f"Doc type page returned {len(results)} items; sample labels: {sample_labels or 'n/a'}"
        )
        for doc_type in results:
            if doc_type.get("label", "").lower() == normalized_label:
                doc_type_id = doc_type.get("id")
                _MAYAN_DOCUMENT_TYPE_CACHE[type_label] = doc_type_id
                debug(f"Found doc type '{type_label}' with id {doc_type_id}")
                return doc_type_id
        url = data.get("next")
        params = None
    debug(f"Finished searching; document type '{type_label}' not found")
    _MAYAN_DOCUMENT_TYPE_CACHE[type_label] = None
    return None


def mayan_list_docs_for_document_type(type_label: str) -> List[int]:
    doc_type_id = mayan_find_document_type_id(type_label)
    if doc_type_id is None:
        raise RuntimeError(f"Document type '{type_label}' not found in Mayan.")

    doc_ids: List[int] = []
    url = f"{MAYAN_API_BASE}/document_types/{doc_type_id}/documents/?page_size=100"
    debug(f"Listing documents for doc type '{type_label}' (id={doc_type_id})")
    while url:
        debug(f"GET {url}")
        rr = _mayan_get(url)
        rr.raise_for_status()
        jj = rr.json()
        doc_ids.extend(d["id"] for d in jj.get("results", []))
        url = jj.get("next")
    debug(f"Found {len(doc_ids)} documents for doc type '{type_label}'")
    return doc_ids


def mayan_doc_detail(doc_id: int) -> Dict[str, Any]:
    if doc_id in _MAYAN_DOC_DETAIL_CACHE:
        return _MAYAN_DOC_DETAIL_CACHE[doc_id]
    r = _mayan_get(f"{MAYAN_API_BASE}/documents/{doc_id}/")
    r.raise_for_status()
    data = r.json()
    _MAYAN_DOC_DETAIL_CACHE[doc_id] = data
    if "tags" in data:
        _MAYAN_DOC_TAG_CACHE[doc_id] = data.get("tags", []) or []
    return data


def mayan_document_tags(doc_id: int) -> List[Dict[str, Any]]:
    if doc_id in _MAYAN_DOC_TAG_CACHE:
        return _MAYAN_DOC_TAG_CACHE[doc_id]
    detail = _MAYAN_DOC_DETAIL_CACHE.get(doc_id)
    if detail and detail.get("tags") is not None:
        tags = detail.get("tags") or []
        _MAYAN_DOC_TAG_CACHE[doc_id] = tags
        return tags
    url = f"{MAYAN_API_BASE}/documents/{doc_id}/tags/"
    tag_list: List[Dict[str, Any]] = []
    debug(f"Fetching tags for document {doc_id}")
    while url:
        resp = _mayan_get(url)
        resp.raise_for_status()
        data = resp.json()
        tag_list.extend(data.get("results", []))
        url = data.get("next")
    _MAYAN_DOC_TAG_CACHE[doc_id] = tag_list
    return tag_list


def mayan_latest_download_url(doc_id: int) -> str:
    # Either use latest_version in the doc detail, or list versions and pick the latest:
    d = mayan_doc_detail(doc_id)
    file_url = _download_url_from_file_info(d.get("file_latest"), doc_id)
    if file_url:
        debug(f"Using latest file download URL for document {doc_id}")
        return file_url

    file_list_url = d.get("file_list_url") or f"{MAYAN_API_BASE}/documents/{doc_id}/files/"
    debug(f"Fetching file list for document {doc_id} via {file_list_url}")
    rv_files = _mayan_get(file_list_url)
    rv_files.raise_for_status()
    file_results = rv_files.json().get("results", [])
    if file_results:
        file_results.sort(key=lambda f: f.get("timestamp") or "", reverse=True)
        file_url = _download_url_from_file_info(file_results[0], doc_id)
        if file_url:
            debug(f"Using file list download URL for document {doc_id}")
            return file_url

    # fallback: resolve versions endpoint explicitly and pick most recent result
    versions_url = d.get("versions_url") or d.get("versions")
    if isinstance(versions_url, dict):
        versions_url = versions_url.get("url")
    if not versions_url:
        versions_url = f"{MAYAN_API_BASE}/documents/{doc_id}/versions/"
    debug(f"Fetching versions for document {doc_id} via {versions_url}")
    rv = _mayan_get(versions_url)
    rv.raise_for_status()
    data = rv.json()
    results = data.get("results")
    if not results:
        raise RuntimeError(f"No versions found for Mayan document {doc_id}")
    latest = results[-1]
    download_url = latest.get("download_url")
    if not download_url:
        detail_url = latest.get("url")
        if detail_url:
            debug(
                "Latest version missing download_url; fetching detail "
                f"from {detail_url}"
            )
            detail_resp = _mayan_get(detail_url)
            detail_resp.raise_for_status()
            detail = detail_resp.json()
            for key in (
                "download_url",
                "download_api_url",
                "download_api_view_url",
                "download_url_api_view",
            ):
                if detail.get(key):
                    download_url = detail[key]
                    debug(f"Found download URL via detail key '{key}'")
                    break
    if not download_url:
        version_id = latest.get("id")
        file_info = latest.get("file") or {}
        file_url = _download_url_from_file_info(file_info, doc_id)
        if file_url:
            debug("Using file info from latest version")
            return file_url
        if not version_id:
            raise RuntimeError(
                f"Latest version for document {doc_id} missing download_url and id"
            )
        return mayan_export_version_download(doc_id, version_id, d.get("label"))
    return download_url


def _download_url_from_file_info(
    file_info: Optional[Dict[str, Any]], default_doc_id: int
) -> Optional[str]:
    if not file_info:
        return None
    file_id = file_info.get("id")
    document_id = file_info.get("document_id") or default_doc_id
    if not file_id or not document_id:
        return None
    return f"{MAYAN_API_BASE}/documents/{document_id}/files/{file_id}/download/"


def mayan_export_version_download(doc_id: int, version_id: int, doc_label: Optional[str]) -> str:
    initial_downloads = _list_recent_downloads()
    existing_ids = _current_download_ids(initial_downloads)
    url = f"{MAYAN_API_BASE}/documents/{doc_id}/versions/{version_id}/export/"
    debug(
        f"Triggering export for document {doc_id} version {version_id} via {url}"
    )
    resp = _mayan_post(url, json={})
    if resp.status_code not in (200, 201, 202):
        resp.raise_for_status()
    deadline = time.time() + EXPORT_POLL_TIMEOUT
    label_lower = (doc_label or "").lower()
    attempts_without_label_match = 0
    poll_backoff = ExponentialBackoff(
        EXPORT_POLL_INTERVAL,
        min(EXPORT_POLL_TIMEOUT, EXPORT_POLL_INTERVAL * 10),
        1.5,
    )
    while time.time() < deadline:
        recent = _list_recent_downloads()
        saw_new_download = False
        for download in recent:
            dl_id = download.get("id")
            if not isinstance(dl_id, int) or dl_id in existing_ids:
                continue
            existing_ids.add(dl_id)
            saw_new_download = True
            filename = (download.get("filename") or "").lower()
            if label_lower and label_lower not in filename:
                continue
            download_url = download.get("download_url") or download.get("url")
            if not download_url:
                continue
            resolved_url = _resolve_mayan_url(download_url)
            debug(
                "Using exported download %s for document %s version %s"
                % (dl_id, doc_id, version_id)
            )
            return resolved_url
        if saw_new_download:
            poll_backoff.reset()
        if saw_new_download and label_lower:
            attempts_without_label_match += 1
            if attempts_without_label_match >= 3:
                debug(
                    "No download matched label for document %s; accepting first new download"
                    % doc_id
                )
                label_lower = ""
        delay = poll_backoff.next_delay()
        debug(f"Waiting {delay:.2f}s for export to finish")
        time.sleep(delay)
    raise RuntimeError(
        f"Timed out waiting for exported version {version_id} of document {doc_id}"
    )


def _current_download_ids(
    downloads: Optional[List[Dict[str, Any]]] = None,
) -> set[int]:
    if downloads is None:
        downloads = _list_recent_downloads()
    result: set[int] = set()
    for download in downloads:
        download_id = download.get("id")
        if isinstance(download_id, int):
            result.add(download_id)
    return result


def _list_recent_downloads() -> List[Dict[str, Any]]:
    resp = _mayan_get(
        f"{MAYAN_API_BASE}/downloads/?ordering=-datetime&page_size=20"
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def _ensure_download_param(url: str) -> str:
    parsed = urlparse(url)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if not any(k == "download" for k, _ in query_pairs):
        query_pairs.append(("download", "1"))
    new_query = urlencode(query_pairs)
    return urlunparse(parsed._replace(query=new_query))


def _resolve_mayan_url(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return urljoin(f"{MAYAN_BASE}/", url.lstrip("/"))


def _extract_script_redirect(text: str) -> Optional[str]:
    for pattern in SCRIPT_REDIRECT_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def mayan_download_document_file(download_url: str) -> bytes:
    if CONFIG is None:
        raise RuntimeError("Configuration must be loaded before downloading documents.")
    backoff = ExponentialBackoff(
        CONFIG.download_backoff_min,
        CONFIG.download_backoff_max,
        CONFIG.download_backoff_multiplier,
    )
    candidates: Deque[str] = deque()
    normalized_url = _resolve_mayan_url(download_url)
    candidates.append(normalized_url)
    download_param_url = _ensure_download_param(normalized_url)
    if download_param_url not in candidates:
        candidates.append(download_param_url)
    visited: set[str] = set()
    empty_attempts: Dict[str, int] = {}
    while candidates:
        current = candidates.popleft()
        if current in visited:
            continue
        debug(f"Downloading Mayan document via {current}")
        try:
            response = _mayan_get(current)
            response.raise_for_status()
        except HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                for alt in _alternate_download_urls(current):
                    resolved_alt = _resolve_mayan_url(alt)
                    if resolved_alt not in visited and resolved_alt not in candidates:
                        debug(
                            "Download returned 404; trying alternative "
                            f"URL {resolved_alt}"
                        )
                        candidates.append(resolved_alt)
                continue
            raise
        content_type = (response.headers.get("Content-Type") or "").lower()
        if any(ct in content_type for ct in ("application/javascript", "text/html")):
            redirect_target = _extract_script_redirect(response.text)
            if redirect_target:
                resolved_redirect = _resolve_mayan_url(redirect_target)
                debug(
                    "Download returned script/html; following redirect to "
                    f"{resolved_redirect}"
                )
                if resolved_redirect not in visited:
                    candidates.append(resolved_redirect)
                continue
            visited.add(current)
            raise RuntimeError(
                "Received non-binary response when downloading document from "
                f"{current}"
            )
        if not response.content:
            attempt = empty_attempts.get(current, 0)
            if attempt < EMPTY_DOWNLOAD_RETRY_LIMIT:
                empty_attempts[current] = attempt + 1
                debug(
                    f"Empty download from {current}; retrying (attempt {attempt + 1}/"
                    f"{EMPTY_DOWNLOAD_RETRY_LIMIT})"
                )
                delay = backoff.next_delay()
                debug(f" Waiting {delay:.2f}s before retrying download")
                time.sleep(delay)
                candidates.append(current)
                continue
            raise RuntimeError(
                f"Repeated empty downloads from {current}; giving up"
            )
        backoff.reset()
        visited.add(current)
        return response.content
    raise RuntimeError(
        f"Unable to retrieve document bytes after following redirects for {download_url}"
    )


def _alternate_download_urls(url: str) -> List[str]:
    parsed = urlparse(url)
    path = parsed.path
    alts: List[str] = []
    doc_id = None
    ver_id = None
    for pattern in DOWNLOAD_URL_PATTERNS:
        match = pattern.search(path)
        if match:
            doc_id = match.groupdict().get("doc") or doc_id
            ver_id = match.groupdict().get("ver") or ver_id
            break
    if not ver_id:
        return alts
    if doc_id:
        alts.append(f"{MAYAN_API_BASE}/documents/{doc_id}/versions/{ver_id}/download/")
        alts.append(f"{MAYAN_BASE}/documents/{doc_id}/versions/{ver_id}/download/")
    alts.append(f"{MAYAN_API_BASE}/documents/document_versions/{ver_id}/download/")
    alts.append(f"{MAYAN_BASE}/documents/document_versions/{ver_id}/download/")
    return alts


def mayan_tag_document(doc_id: int, tag_id: int) -> None:
    url = f"{MAYAN_API_BASE}/documents/{doc_id}/tags/attach/"
    payload = {"tag": str(tag_id)}
    debug(f"Attaching tag {tag_id} to document {doc_id} via {url}")
    r = _mayan_post(url, json=payload)
    r.raise_for_status()
    _MAYAN_DOC_TAG_CACHE.pop(doc_id, None)
    _MAYAN_DOC_DETAIL_CACHE.pop(doc_id, None)


def paperless_upload(
    paperless_token: str,
    file_bytes: bytes,
    filename: str,
    created_iso: Optional[str],
    tag_id: int,
) -> Dict[str, Any]:
    files = {"document": (filename, file_bytes)}
    data = {
        "title": filename,
        "created": created_iso,
        "tags": str(tag_id)  # can repeat field to add more tags
    }
    r = sess_paperless.post(
        f"{PAPERLESS_BASE}/api/documents/post_document/",
        headers={"Authorization": f"Token {paperless_token}"},
        files=files,
        data=data,
    )
    try:
        r.raise_for_status()
    except HTTPError as exc:
        try:
            error_detail = r.json()
        except ValueError:
            error_detail = r.text
        raise RuntimeError(
            f"Paperless upload failed for '{filename}': {exc}. "
            f"Response detail: {error_detail}"
        ) from exc
    # We get a task UUID; we can optionally poll /api/tasks/?task_id=...
    return r.json()


def migrate_one_document_type(type_label: str, cfg) -> bool:
    tag_id = paperless_get_or_create_tag(
        cfg["paperless_token"],
        cfg["paperless_tag_name"],
    )
    moved_tag_id = mayan_find_tag_id(MAYAN_MOVED_TAG_LABEL)
    if moved_tag_id is None:
        raise RuntimeError(f"Required Mayan tag '{MAYAN_MOVED_TAG_LABEL}' not found.")
    moved_label_lower = MAYAN_MOVED_TAG_LABEL.lower()
    ids = mayan_list_docs_for_document_type(type_label)
    operation = "counting" if COUNT_ONLY_MODE else "migrating"
    print(f"{type_label}: {operation} {len(ids)} docs")
    eligible_count = 0
    failure_count = 0
    for i, doc_id in enumerate(ids, 1):
        try:
            det = mayan_doc_detail(doc_id)
            tags = det.get("tags")
            if tags is None:
                tags = mayan_document_tags(doc_id)
            already_moved = any(
                t.get("label", "").lower() == moved_label_lower
                for t in tags
            )
            if already_moved:
                continue
            eligible_count += 1
            if COUNT_ONLY_MODE:
                if i % 20 == 0:
                    print(f"  ...inspected {i} / {len(ids)}")
                continue
            created = det.get("datetime_created") or det.get("date_added") or None
            fname = det.get("label") or f"mayan_doc_{doc_id}.pdf"
            download_url = mayan_latest_download_url(doc_id)
            file_bytes = mayan_download_document_file(download_url)
            if DOWNLOAD_FIRST_MODE:
                safe_name = Path(fname).name or f"mayan_doc_{doc_id}.bin"
                sample_name = f"sample_{doc_id}_{safe_name}"
                sample_path = Path.cwd() / sample_name
                sample_path.write_bytes(file_bytes)
                print(
                    "Download-first mode: saved sample document to "
                    f"{sample_path}. No uploads were performed."
                )
                return True
            paperless_upload(cfg["paperless_token"], file_bytes, fname, created, tag_id)
            mayan_tag_document(doc_id, moved_tag_id)
            if i % 20 == 0:
                print(f"  ...{i} / {len(ids)}")
        except Exception as exc:
            failure_count += 1
            print(
                f"ERROR: Failed to migrate document {doc_id}: {exc}. Skipping to next."
            )
            if DEBUG_ENABLED:
                traceback.print_exc()
            continue
    if COUNT_ONLY_MODE:
        print(
            f"{type_label}: count-only mode detected {eligible_count} documents "
            "that would be migrated"
        )
        return False
    if failure_count:
        print(
            f"{type_label}: encountered {failure_count} error(s); remaining documents "
            "were processed."
        )
    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate documents from Mayan EDMS to Paperless-ngx."
    )
    parser.add_argument(
        "--config",
        help="Optional path to a JSON file containing configuration overrides.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging of Mayan API calls.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count documents eligible for migration without uploading anything.",
    )
    parser.add_argument(
        "--download-first",
        action="store_true",
        help="Download the first eligible document to disk and exit without migrating.",
    )
    parser.add_argument("--mayan-base", help="Override the Mayan base URL.")
    parser.add_argument("--paperless-base", help="Override the Paperless base URL.")
    parser.add_argument(
        "--export-poll-interval",
        type=float,
        help="Initial poll interval (seconds) when waiting for exports.",
    )
    parser.add_argument(
        "--export-poll-timeout",
        type=float,
        help="Maximum time (seconds) to wait for an export to complete.",
    )
    parser.add_argument(
        "--download-retry-limit",
        type=int,
        help="Maximum number of retries for empty Mayan downloads.",
    )
    parser.add_argument(
        "--download-backoff-min",
        type=float,
        help="Minimum backoff delay (seconds) between download retries.",
    )
    parser.add_argument(
        "--download-backoff-max",
        type=float,
        help="Maximum backoff delay (seconds) between download retries.",
    )
    parser.add_argument(
        "--download-backoff-multiplier",
        type=float,
        help="Multiplier applied to the backoff delay after each retry.",
    )
    args = parser.parse_args()
    if args.count_only and args.download_first:
        parser.error("--download-first cannot be combined with --count-only")
    return args


def build_config_from_args(args) -> MigrationConfig:
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path.resolve()}"
        )
    with open(config_path, "r", encoding="utf-8") as cfg_file:
        config_payload = json.load(cfg_file)

    override_map = {
        "mayan_base": args.mayan_base,
        "paperless_base": args.paperless_base,
        "export_poll_interval": args.export_poll_interval,
        "export_poll_timeout": args.export_poll_timeout,
        "download_retry_limit": args.download_retry_limit,
        "download_backoff_min": args.download_backoff_min,
        "download_backoff_max": args.download_backoff_max,
        "download_backoff_multiplier": args.download_backoff_multiplier,
    }
    for key, value in override_map.items():
        if value is not None:
            config_payload[key] = value

    required_keys = [
        "mayan_base",
        "paperless_base",
        "mayan_user",
        "mayan_pass",
        "mayan_moved_tag",
        "mappings",
        "export_poll_interval",
        "export_poll_timeout",
        "download_retry_limit",
        "download_backoff_min",
        "download_backoff_max",
        "download_backoff_multiplier",
    ]
    missing = [key for key in required_keys if key not in config_payload]
    if missing:
        raise KeyError(
            f"Configuration file {config_path} is missing required keys: {', '.join(missing)}"
        )

    return MigrationConfig(
        mayan_base=str(config_payload["mayan_base"]).rstrip("/"),
        paperless_base=str(config_payload["paperless_base"]).rstrip("/"),
        mayan_user=str(config_payload["mayan_user"]),
        mayan_pass=str(config_payload["mayan_pass"]),
        mayan_moved_tag=str(config_payload["mayan_moved_tag"]),
        mappings=config_payload["mappings"],
        export_poll_interval=float(config_payload["export_poll_interval"]),
        export_poll_timeout=float(config_payload["export_poll_timeout"]),
        download_retry_limit=int(config_payload["download_retry_limit"]),
        download_backoff_min=float(config_payload["download_backoff_min"]),
        download_backoff_max=float(config_payload["download_backoff_max"]),
        download_backoff_multiplier=float(
            config_payload["download_backoff_multiplier"]
        ),
    )


def main():
    global DEBUG_ENABLED, COUNT_ONLY_MODE, DOWNLOAD_FIRST_MODE

    args = parse_args()
    runtime_config = build_config_from_args(args)
    apply_runtime_config(runtime_config)
    reset_mayan_caches()
    DEBUG_ENABLED = args.debug
    COUNT_ONLY_MODE = args.count_only
    DOWNLOAD_FIRST_MODE = args.download_first

    for doc_type in MAPPINGS:
        sample_generated = migrate_one_document_type(doc_type, MAPPINGS[doc_type])
        if DOWNLOAD_FIRST_MODE and sample_generated:
            print("Download-first mode complete; rerun without the flag to migrate.")
            return

    print("Done. Optionally run a verification query in Paperless.")


if __name__ == "__main__":
    main()
def _mayan_get(url: str, **kwargs) -> requests.Response:
    resolved = _resolve_mayan_url(url)
    return sess_mayan.get(resolved, **kwargs)


def _mayan_post(url: str, **kwargs) -> requests.Response:
    resolved = _resolve_mayan_url(url)
    return sess_mayan.post(resolved, **kwargs)
