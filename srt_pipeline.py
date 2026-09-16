#!/usr/bin/env python3
"""Persian subtitle pipeline for X broadcast recordings.

Timing from Groq Whisper (accurate), translation from Gemini (free tier).

Stages (each resumable, driven by env vars):
  prep       - download replay chunk audio -> out/audio.mp3 (16k mono mp3)
  transcribe - Groq Whisper (verbose_json) -> out/whisper.json
  translate  - Gemini batch translation -> out/segments.json (fa per segment)
  srt        - build Persian SRT -> out/subtitles.srt

Env:
  GROQ_API_KEY   (required for transcribe)
  GEMINI_API_KEY (required for translate)
  BROADCAST_URL, START_MIN (0), DURATION_MIN (15)
  WHISPER_MODEL (whisper-large-v3), TRANSCRIBE_WINDOW_S (600)
  BATCH_SIZE (60), STAGES (all)
"""
import base64
import json
import os
import re
import subprocess
import time
from pathlib import Path

import requests

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GROQ_BASE = "https://api.groq.com/openai/v1"
X_API = "https://api.x.com/1.1"
OUT = Path("out")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3")
TRANSCRIBE_WINDOW_S = float(os.environ.get("TRANSCRIBE_WINDOW_S", "600"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "60"))
TRANSLATE_MODEL = os.environ.get("TRANSLATE_MODEL", "gemini-2.5-flash")

VIDEO_CTX = ('a casual multi-speaker tech livestream ("Grok Bot Galaxy '
             'Livestream") about building an AI company in days')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_state(**kv):
    p = OUT / "state.json"
    st = {}
    if p.exists():
        st = json.loads(p.read_text())
    st.update(kv)
    p.write_text(json.dumps(st, ensure_ascii=False, indent=1))


def run_ffmpeg(args):
    shown = " ".join(str(a) for a in args[:10])
    log(f"+ ffmpeg {shown} ...")
    return subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel",
                           "error", "-y"] + [str(a) for a in args]).returncode


# ---------------------------------------------------------------- stage: prep
def get_playback_url(broadcast_url: str) -> str:
    bid = broadcast_url.rstrip("/").split("/")[-1]
    r = requests.get(f"{X_API}/broadcasts/show.json",
                     params={"ids": bid, "include_events": "true"}, timeout=30)
    r.raise_for_status()
    media_key = next(iter(r.json().get("broadcasts", {}).values()),
                     {}).get("media_key")
    if not media_key:
        raise RuntimeError("no media key")
    r = requests.get(f"{X_API}/live_video_stream/status/{media_key}", timeout=30)
    r.raise_for_status()
    url = r.json().get("source", {}).get("noRedirectPlaybackUrl")
    if not url:
        raise RuntimeError("no playback url")
    log(f"playback URL ok ({'live' if 'type=live' in url else 'replay'})")
    return url


def stage_prep():
    OUT.mkdir(exist_ok=True)
    start_min = float(os.environ.get("START_MIN", "0"))
    dur_min = float(os.environ.get("DURATION_MIN", "15"))
    start_s, dur_s = start_min * 60, dur_min * 60
    url = get_playback_url(os.environ["BROADCAST_URL"])
    audio = OUT / "audio.mp3"
    rc = run_ffmpeg(["-reconnect", "1", "-reconnect_streamed", "1",
                     "-reconnect_delay_max", "60",
                     "-ss", str(start_s), "-t", str(dur_s),
                     "-i", url, "-vn", "-ac", "1", "-ar", "16000",
                     "-b:a", "48k", audio.as_posix()])
    if rc != 0 or not audio.exists() or audio.stat().st_size < 10_000:
        raise RuntimeError(f"prep: audio download failed rc={rc}")
    log(f"prep done: audio={audio.stat().st_size / 1e6:.1f}MB")
    save_state(prep="done", start_s=start_s, dur_s=dur_s)


# ---------------------------------------------------------- stage: transcribe
def groq_whisper(audio: Path, offset: float) -> list:
    data = {
        "model": WHISPER_MODEL,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    with audio.open("rb") as fh:
        r = requests.post(
            f"{GROQ_BASE}/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            files={"file": (audio.name, fh, "audio/mpeg")},
            data=data, timeout=900,
        )
    if r.status_code == 401:
        raise RuntimeError("groq 401: invalid API key")
    if r.status_code == 413:
        raise RuntimeError("groq 413: file too large (reduce TRANSCRIBE_WINDOW_S)")
    if r.status_code in (429, 500, 502, 503):
        raise RuntimeError(f"groq {r.status_code} (retryable): {r.text[:200]}")
    if r.status_code != 200:
        raise RuntimeError(f"groq -> {r.status_code}: {r.text[:300]}")
    doc = r.json()
    segs = []
    for s in doc.get("segments", []):
        try:
            st = float(s.get("start", 0.0)) + offset
            en = float(s.get("end", 0.0)) + offset
        except (TypeError, ValueError):
            continue
        txt = str(s.get("text", "") or "").strip()
        if not txt or en <= st:
            continue
        segs.append({"start": round(st, 3), "end": round(en, 3), "text": txt})
    return segs


def stage_transcribe():
    if not GROQ_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    audio = OUT / "audio.mp3"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", audio.as_posix()], capture_output=True, text=True)
    try:
        total_dur = float(probe.stdout.strip())
    except ValueError:
        raise RuntimeError("transcribe: cannot probe audio duration")
    log(f"transcribe: audio {total_dur:.1f}s, window={TRANSCRIBE_WINDOW_S:.0f}s")

    windows = []
    t0 = 0.0
    while t0 < total_dur - 0.5:
        windows.append((t0, min(TRANSCRIBE_WINDOW_S, total_dur - t0)))
        t0 += TRANSCRIBE_WINDOW_S
    log(f"transcribe: {len(windows)} window(s)")

    all_segs = []
    for wi, (w0, wdur) in enumerate(windows):
        part = OUT / f"_tw_{wi:02d}.mp3"
        rc = run_ffmpeg(["-ss", f"{w0:.2f}", "-t", f"{wdur:.2f}",
                         "-i", audio.as_posix(), "-c", "copy",
                         part.as_posix()])
        if rc != 0 or not part.exists() or part.stat().st_size < 1000:
            raise RuntimeError(f"transcribe: window {wi} cut failed")
        for attempt in range(4):
            try:
                segs = groq_whisper(part, w0)
                break
            except RuntimeError as e:
                if "retryable" in str(e) and attempt < 3:
                    wait = 20 * (attempt + 1)
                    log(f"transcribe window {wi}: {e} -> retry in {wait}s")
                    time.sleep(wait)
                    continue
                part.unlink(missing_ok=True)
                raise
        part.unlink(missing_ok=True)
        log(f"transcribe: window {wi + 1}/{len(windows)} -> {len(segs)} segments")
        all_segs.extend(segs)

    all_segs.sort(key=lambda s: s["start"])
    (OUT / "whisper.json").write_text(
        json.dumps({"segments": all_segs}, ensure_ascii=False, indent=1))
    log(f"transcribe done: {len(all_segs)} segments total")
    save_state(transcribe="done", segments=len(all_segs))


# ----------------------------------------------------------- stage: translate
TRANSLATE_PROMPT = """You are a professional Persian (Farsi) subtitle translator for __CTX__.
Translate every English subtitle segment into natural, conversational Persian. Keep proper names, product names and numbers in Latin script. Preserve tone and energy; keep each translation short and subtitle-friendly. Never invent or merge content.
Return ONLY a JSON object (no markdown fences) shaped exactly:
{"translations": [{"id": 1, "fa": "ترجمهٔ فارسی"}]}
covering EVERY input id, same order.

Segments:
__SEGS__"""


def gemini_generate(model, payload, retries=4):
    for attempt in range(retries):
        r = requests.post(
            f"{API_BASE}/models/{model}:generateContent",
            headers={"x-goog-api-key": GEMINI_KEY,
                     "Content-Type": "application/json"},
            json=payload, timeout=300,
        )
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 503):
            wait = 15 * (attempt + 1)
            log(f"gemini {r.status_code}, retry in {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"gemini {model} -> {r.status_code}: {r.text[:400]}")
    raise RuntimeError("gemini retries exhausted")


def stage_translate():
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    wdoc = json.loads((OUT / "whisper.json").read_text())
    segs = wdoc["segments"]
    tr_dir = OUT / "tr"
    tr_dir.mkdir(exist_ok=True)
    batches = [segs[i:i + BATCH_SIZE]
               for i in range(0, len(segs), BATCH_SIZE)]
    log(f"translate: {len(segs)} segments in {len(batches)} batch(es) "
        f"of <= {BATCH_SIZE}")
    for bi, batch in enumerate(batches):
        dest = tr_dir / f"batch_{bi:03d}.json"
        if dest.exists() and dest.stat().st_size > 10:
            log(f"translate: batch {bi + 1}/{len(batches)} cached")
            continue
        base = bi * BATCH_SIZE
        items = [{"id": base + k + 1, "text": s["text"]}
                 for k, s in enumerate(batch)]
        prompt = (TRANSLATE_PROMPT
                  .replace("__CTX__", VIDEO_CTX)
                  .replace("__SEGS__", json.dumps(items, ensure_ascii=False)))
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "temperature": 0.2,
                                 "maxOutputTokens": 65536},
        }
        resp = gemini_generate(TRANSLATE_MODEL, payload)
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
        text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                raise
            doc = json.loads(m.group(0))
        tmp = Path(str(dest) + ".tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        tmp.rename(dest)
        log(f"translate: batch {bi + 1}/{len(batches)} -> "
            f"{len(doc.get('translations', []))} translations")
        time.sleep(1)
    # merge
    fa_by_id = {}
    for f in sorted(tr_dir.glob("batch_*.json")):
        for t in json.loads(f.read_text()).get("translations", []):
            try:
                fa_by_id[int(t["id"])] = str(t.get("fa", "") or "")
            except (TypeError, ValueError):
                continue
    out_segs = []
    for i, s in enumerate(segs):
        out_segs.append({"id": i + 1, "start": s["start"], "end": s["end"],
                         "text": s["text"],
                         "fa": fa_by_id.get(i + 1, "")})
    (OUT / "segments.json").write_text(
        json.dumps({"segments": out_segs}, ensure_ascii=False, indent=1))
    got = sum(1 for s in out_segs if s["fa"])
    log(f"translate done: {got}/{len(out_segs)} translated")
    if got == 0:
        raise RuntimeError("translate: no translations produced")
    save_state(translate="done", translated=got)


# ------------------------------------------------------------------ stage: srt
def srt_time(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def stage_srt():
    doc = json.loads((OUT / "segments.json").read_text())
    segs = sorted(doc["segments"], key=lambda s: s["start"])
    lines = []
    for i, s in enumerate(segs):
        text = (s.get("fa") or "").strip() or s.get("text", "").strip()
        if not text:
            continue
        lines.append(f"{i + 1}\n{srt_time(s['start'])} --> "
                     f"{srt_time(s['end'])}\n{text}\n")
    srt = OUT / "subtitles.srt"
    srt.write_text("\n".join(lines), encoding="utf-8")
    log(f"srt done: {len(lines)} cues -> {srt}")
    save_state(srt="done", cues=len(lines))


def main():
    stages_env = os.environ.get("STAGES", "all")
    wanted = (["prep", "transcribe", "translate", "srt"]
              if stages_env == "all" else
              [s.strip() for s in stages_env.split(",") if s.strip()])
    for stage in wanted:
        log(f"=== stage: {stage} ===")
        {"prep": stage_prep, "transcribe": stage_transcribe,
         "translate": stage_translate, "srt": stage_srt}[stage]()
    log("pipeline finished: " + ", ".join(wanted))


if __name__ == "__main__":
    main()
