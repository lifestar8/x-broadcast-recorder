#!/usr/bin/env python3
"""Record an X/Twitter (Periscope) broadcast or live stream.

Resolves the broadcast to a playback URL using the same API flow as
offish/twitter-x-broadcast-downloader (broadcasts/show.json ->
media key -> live_video_stream/status), then records the HLS stream
with ffmpeg until the stream ends or the max duration is reached.

Usage:
    python record.py <broadcast_url> [max_minutes]
"""
import subprocess
import sys
import time
from pathlib import Path

import requests

API_URL = "https://api.x.com/1.1"
OUT_DIR = Path("downloads")


def get_broadcast_data(broadcast_url: str) -> dict:
    broadcast_id = broadcast_url.rstrip("/").split("/")[-1]
    r = requests.get(
        f"{API_URL}/broadcasts/show.json",
        params={"ids": broadcast_id, "include_events": "true"},
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"broadcasts/show.json -> HTTP {r.status_code}: {r.text[:300]}"
        )
    return r.json()


def get_media_key(data: dict):
    broadcasts = data.get("broadcasts", {})
    if not broadcasts:
        return None
    return next(iter(broadcasts.values())).get("media_key")


def get_playback_url(media_key: str):
    r = requests.get(
        f"{API_URL}/live_video_stream/status/{media_key}", timeout=30
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"live_video_stream/status -> HTTP {r.status_code}: {r.text[:300]}"
        )
    return r.json().get("source", {}).get("noRedirectPlaybackUrl")


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: record.py <broadcast_url> [max_minutes]")
        return 1

    broadcast_url = sys.argv[1].strip()
    max_minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    deadline = time.time() + max_minutes * 60

    data = get_broadcast_data(broadcast_url)
    info = next(iter(data.get("broadcasts", {}).values()), {})
    print(f"broadcast state={info.get('state')} title={info.get('status')!r}", flush=True)

    media_key = get_media_key(data)
    if not media_key:
        print("ERROR: no media key (broadcast gone or private?)", flush=True)
        return 2
    print(f"Got media key {media_key}", flush=True)

    playback_url = get_playback_url(media_key)
    if not playback_url:
        print("ERROR: no playback URL", flush=True)
        return 2
    kind = "live" if "type=live" in playback_url else "replay"
    print(f"Got playback URL ({kind})", flush=True)

    OUT_DIR.mkdir(exist_ok=True)

    parts = []
    attempt = 0
    while time.time() < deadline:
        remaining = int(deadline - time.time())
        part = OUT_DIR / f"{media_key}_part{attempt:02d}.ts"
        parts.append(part)
        print(
            f"[attempt {attempt}] recording up to {remaining // 60} more minute(s)...",
            flush=True,
        )
        rc = run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning", "-stats",
            "-reconnect", "1", "-reconnect_streamed", "1",
            "-reconnect_delay_max", "60",
            "-i", playback_url,
            "-t", str(remaining),
            "-c", "copy",
            "-f", "mpegts",
            part.as_posix(),
        ])

        recorded = [p for p in parts if p.exists() and p.stat().st_size > 0]
        total_bytes = sum(p.stat().st_size for p in recorded)

        if rc == 0:
            print("ffmpeg finished cleanly (stream end or cap reached).", flush=True)
            break

        print(f"ffmpeg exited rc={rc} (recorded so far: {total_bytes / 1e6:.1f} MB)", flush=True)
        attempt += 1
        if attempt >= 5:
            print("too many failed attempts, giving up", flush=True)
            break

        time.sleep(15)
        try:
            new_url = get_playback_url(media_key)
        except Exception as e:
            print(f"playback URL refresh failed: {e}", flush=True)
            new_url = None

        if new_url and "type=live" not in new_url:
            print("stream has ended (playback is now replay); finalizing.", flush=True)
            break
        if new_url:
            playback_url = new_url
            print("playback URL refreshed, retrying...", flush=True)

    existing = [p for p in parts if p.exists() and p.stat().st_size > 0]
    if not existing:
        print("nothing was recorded", flush=True)
        return 3

    total_bytes = sum(p.stat().st_size for p in existing)
    print(f"recorded {len(existing)} part(s), {total_bytes / 1e6:.1f} MB total", flush=True)

    final = OUT_DIR / f"{media_key}.mp4"
    if len(existing) == 1:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", existing[0].as_posix(),
            "-c", "copy", "-bsf:a", "aac_adtstoasc",
            "-f", "mp4", final.as_posix(),
        ]
    else:
        concat = OUT_DIR / "concat.txt"
        concat.write_text("".join(f"file '{p.as_posix()}'\n" for p in existing))
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", concat.as_posix(),
            "-c", "copy", "-bsf:a", "aac_adtstoasc",
            "-f", "mp4", final.as_posix(),
        ]

    rc = run(cmd)
    if rc != 0:
        print("remux to mp4 failed; keeping raw .ts as fallback", flush=True)
    else:
        for p in existing:
            p.unlink(missing_ok=True)

    if final.exists():
        print(f"DONE -> {final} ({final.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
