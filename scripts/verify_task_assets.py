#!/usr/bin/env python3
"""Validate task-card media, provenance metadata, and website references."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
from html.parser import HTMLParser


class TaskCardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cards: dict[str, dict[str, str]] = {}
        self.current_task: str | None = None

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = dict(attrs_list)
        if tag == "article" and "task-card" in (attrs.get("class") or ""):
            self.current_task = attrs.get("data-task-id")
        elif tag == "video" and self.current_task:
            self.cards[self.current_task] = {
                "video": attrs.get("data-video-src", ""),
                "poster": attrs.get("poster", ""),
            }

    def handle_endtag(self, tag: str) -> None:
        if tag == "article":
            self.current_task = None


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(video: pathlib.Path) -> dict[str, object]:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate",
        "-of", "json", str(video),
    ], text=True)
    return json.loads(raw)["streams"][0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode", action="store_true", help="Decode every MP4 as an additional integrity check.")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "source/tasks/manifest.json").read_text())
    html = TaskCardParser()
    html.feed((root / "index.html").read_text())
    entries = {entry["website_task"]: entry for entry in manifest["entries"]}
    assert manifest["task_count"] == 14 == len(entries), "expected 14 unique manifest entries"
    assert set(html.cards) == set(entries), "task cards and manifest have different task ids"

    expected_probe = {
        "codec_name": "h264", "width": 1536, "height": 1024,
        "pix_fmt": "yuv420p", "r_frame_rate": "15/1",
    }
    for task_id, entry in entries.items():
        card = html.cards[task_id]
        video = root / "source/tasks" / entry["website_video"]
        poster = root / "source/tasks" / entry["poster"]
        assert card["video"] == f"source/tasks/{entry['website_video']}", task_id
        assert card["poster"] == f"source/tasks/{entry['poster']}", task_id
        assert poster.is_file() and poster.stat().st_size > 0, f"missing poster: {poster}"
        assert sha256(video) == entry["sha256"], f"checksum mismatch: {video}"
        assert probe(video) == expected_probe, f"unexpected encoding: {video}"
        if args.decode:
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True)
    print(f"validated {len(entries)} task cards, videos, posters, and manifest entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
