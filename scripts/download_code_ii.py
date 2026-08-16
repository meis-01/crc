#!/usr/bin/env python
"""Download an official public PhysioNet project without guessing its URL.

The published CODE-II paper currently names PhysioNet but does not expose a
project URL. Pass the URL copied from the official PhysioNet project page once
it is available. This downloader intentionally accepts no mirror hosts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def official_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "physionet.org":
        raise ValueError("Only an HTTPS URL on the official physionet.org host is accepted")
    if not (parsed.path.startswith("/content/") or parsed.path.startswith("/files/")):
        raise ValueError("Expected an official PhysioNet /content/ or /files/ URL")
    return urllib.parse.urlunparse(parsed._replace(fragment="", query=""))


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "CODE-II-reproducible-downloader/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def find_files_root(project_or_files_url: str) -> str:
    url = official_url(project_or_files_url)
    if urllib.parse.urlparse(url).path.startswith("/files/"):
        return url.rstrip("/") + "/"
    parser = LinkParser()
    parser.feed(fetch(url).decode("utf-8", errors="replace"))
    candidates = []
    for href in parser.links:
        resolved = urllib.parse.urljoin(url, href)
        parsed = urllib.parse.urlparse(resolved)
        if parsed.hostname == "physionet.org" and parsed.path.startswith("/files/"):
            candidates.append(official_url(resolved).rstrip("/") + "/")
    roots = sorted(set(candidates), key=len)
    if not roots:
        raise RuntimeError(
            "The project page exposes no public /files/ directory. Confirm that the official release is "
            "published and that any required PhysioNet access steps are complete."
        )
    return roots[0]


def crawl_files(root: str) -> list[str]:
    pending = [root]
    visited: set[str] = set()
    files: set[str] = set()
    root_path = urllib.parse.urlparse(root).path
    while pending:
        directory = pending.pop()
        if directory in visited:
            continue
        visited.add(directory)
        parser = LinkParser()
        parser.feed(fetch(directory).decode("utf-8", errors="replace"))
        for href in parser.links:
            candidate = urllib.parse.urljoin(directory, href)
            parsed = urllib.parse.urlparse(candidate)
            clean = urllib.parse.urlunparse(parsed._replace(query="", fragment=""))
            if parsed.hostname != "physionet.org" or not parsed.path.startswith(root_path):
                continue
            if clean.endswith("/"):
                if clean not in visited:
                    pending.append(clean)
            elif clean != root:
                files.add(clean)
    return sorted(files)


def download_file(url: str, root: str, destination: Path) -> dict[str, object]:
    root_path = urllib.parse.urlparse(root).path
    url_path = urllib.parse.urlparse(url).path
    relative = Path(urllib.parse.unquote(url_path[len(root_path) :]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe relative path from PhysioNet listing: {relative}")
    output = destination / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "CODE-II-reproducible-downloader/1"})
    temporary = output.with_suffix(output.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            handle.write(block)
            digest.update(block)
            size += len(block)
    temporary.replace(output)
    print(f"{relative} ({size} bytes)")
    return {"path": relative.as_posix(), "url": url, "bytes": size, "sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-url",
        required=True,
        help="Exact official PhysioNet project (/content/) or file-root (/files/) URL",
    )
    parser.add_argument("--destination", type=Path, default=Path(r"D:\CODE-II\data\raw"))
    args = parser.parse_args()
    root = find_files_root(args.project_url)
    print(f"Verified official file root: {root}")
    files = crawl_files(root)
    if not files:
        raise RuntimeError(f"No downloadable files found below {root}")
    args.destination.mkdir(parents=True, exist_ok=True)
    entries = [download_file(url, root, args.destination) for url in files]
    manifest = {
        "official_project_url": official_url(args.project_url),
        "official_files_root": root,
        "files": entries,
    }
    (args.destination / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
