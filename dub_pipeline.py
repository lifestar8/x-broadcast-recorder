#!/usr/bin/env python3
"""Gemini-powered Persian dubbing pipeline for X broadcast recordings.

Stages (each resumable, driven by env vars):
  prep     - download replay chunk [start, start+dur] -> out/video.ts + out/audio.mp3
  analyze  - Gemini: diarization + transcript + Persian translation -> out/segments.json
  tts      - Gemini TTS per segment, one voice per speaker -> out/tts/seg_XXXXX.wav
  assemble - timed dub track + Persian SRT + final MP4 -> out/dubbed.mp4, out/subtitles.srt

Env:
  GEMINI_API_KEY   (required for analyze/tts/assemble)
  BROADCAST_URL    (required for prep)
  START_MIN        (default 0)
  DURATION_MIN     (default 15)
  RPM              (default 10) TTS requests-per-minute budget
  STAGES           (default all) comma list: prep,analyze,tts,assemble
"""
import base64
import json
import os
import re
import subprocess
import time
import wave
from pathlib import Path

import requests

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
X_API = "https://api.x.com/1.1"
OUT = Path("out")
TTS_DIR = OUT / "tts"
ANALYZE_MODEL = os.environ.get("ANALYZE_MODEL", "gemini-2.5-flash")
TTS_MODEL = os.environ.get("TTS_MODEL", "gemini-2.5-flash-preview-tts")
KEY = os.environ.get("GEMINI_API_KEY", "")

MALE_VOICES = ["Charon", "Puck", "Fenrir", "Orus", "Iapetus",
               "Algenib", "Rasalgethi", "Alnilam", "Schedar", "Enceladus"]
FEMALE_VOICES = ["Kore", "Zephyr", "Leda", "Aoede", "Callirrhoe",
                 "Autonoe", "Laomedeia", "Achernar", "Despina", "Vindemiatrix"]

SAMPLE_RATE = 24000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_state(**kv):
    p = OUT / "state.json"
    st = {}
    if p.exists():
        st = json.loads(p.read_text())
    st.update(kv)
    p.write_text(json.dumps(st, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- stage: prep
def get_playback_url(broadcast_url: str) -> str:
    bid = broadcast_url.rstrip("/").split("/")[-1]
    r = requests.get(f"{X_API}/broadcasts/show.json",
                     params={"ids": bid, "include_events": "true"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    media_key = next(iter(data.get("broadcasts", {}).values()), {}).get("media_key")
    if not media_key:
        raise RuntimeError("no media key")
    r = requests.get(f"{X_API}/live_video_stream/status/{media_key}", timeout=30)
    r.raise_for_status()
    url = r.json().get("source", {}).get("noRedirectPlaybackUrl")
    if not url:
        raise RuntimeError("no playback url")
    log(f"playback URL ok ({'live' if 'type=live' in url else 'replay'})")
    return url


def run_ffmpeg(args):
    shown = " ".join(str(a) for a in args[:10])
    log(f"+ ffmpeg {shown} ...")
    return subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel",
                           "error", "-y"] + [str(a) for a in args]).returncode


def stage_prep():
    OUT.mkdir(exist_ok=True)
    start_min = float(os.environ.get("START_MIN", "0"))
    dur_min = float(os.environ.get("DURATION_MIN", "15"))
    start_s, dur_s = start_min * 60, dur_min * 60

    url = get_playback_url(os.environ["BROADCAST_URL"])
    video = OUT / "video.ts"
    rc = run_ffmpeg(["-reconnect", "1", "-reconnect_streamed", "1",
                     "-reconnect_delay_max", "60",
                     "-ss", str(start_s), "-t", str(dur_s),
                     "-i", url, "-c", "copy", "-f", "mpegts", video.as_posix()])
    if rc != 0 or not video.exists() or video.stat().st_size < 100_000:
        raise RuntimeError(f"prep: download failed rc={rc}")
    audio = OUT / "audio.mp3"
    rc = run_ffmpeg(["-i", video.as_posix(), "-vn", "-ac", "1", "-ar", "16000",
                     "-b:a", "48k", audio.as_posix()])
    if rc != 0:
        raise RuntimeError("prep: audio extract failed")
    log(f"prep done: video={video.stat().st_size/1e6:.1f}MB "
        f"audio={audio.stat().st_size/1e6:.1f}MB")
    save_state(prep="done", start_s=start_s, dur_s=dur_s)


# ------------------------------------------------------------- stage: analyze
ANALYZE_PROMPT = """You are a professional subtitle and dubbing engine. Listen to this audio and return ONLY a valid JSON object (no markdown fences) with this exact shape:

{
  "speakers": [{"id": "S1", "gender": "male", "description": "short English description: gender, age group, role/mood"}],
  "segments": [
    {"id": 1, "start": 12.4, "end": 15.9, "speaker": "S1", "text": "original speech verbatim", "fa": "ترجمهٔ فارسیِ روان و محاوره‌ای"}
  ]
}

Rules:
- Timestamps in precise seconds measured from the very beginning of the audio.
- Cluster all speech into speakers (S1, S2, ...) by voice characteristics; add gender (male/female/unknown) and a short description per speaker.
- Segment at sentence boundaries; each segment 2-12 seconds long.
- "fa" = natural, conversational Persian (Farsi) translation preserving the tone and energy; keep proper names in Latin script.
- Skip non-speech regions (music, silence, applause). Never invent content.
- Return ONLY the JSON object."""


def gemini_generate(model, payload, retries=4):
    for attempt in range(retries):
        r = requests.post(
            f"{API_BASE}/models/{model}:generateContent",
            headers={"x-goog-api-key": KEY, "Content-Type": "application/json"},
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


def stage_analyze():
    audio = OUT / "audio.mp3"
    data = base64.b64encode(audio.read_bytes()).decode()
    payload = {
        "contents": [{"parts": [
            {"text": ANALYZE_PROMPT},
            {"inline_data": {"mime_type": "audio/mpeg", "data": data}},
        ]}],
        "generationConfig": {"responseMimeType": "application/json",
                             "temperature": 0.2, "maxOutputTokens": 65536},
    }
    resp = gemini_generate(ANALYZE_MODEL, payload)
    text = resp["candidates"][0]["content"]["parts"][0]["text"]
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    doc = json.loads(text)
    segs = doc.get("segments", [])
    if not segs:
        raise RuntimeError("analyze: no segments returned")
    for i, s in enumerate(segs):
        s.setdefault("speaker", "S1")
        s.setdefault("fa", "")
    (OUT / "segments.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1))
    log(f"analyze done: {len(segs)} segments, "
        f"{len(doc.get('speakers', []))} speakers")
    save_state(analyze="done", segments=len(segs))


# ----------------------------------------------------------------- stage: tts
class RateLimiter:
    def __init__(self, rpm):
        self.interval = 60.0 / max(1, rpm)
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            wait = self.next_t - now
            if wait > 0:
                time.sleep(wait)
                self.next_t = time.time() + self.interval
            else:
                self.next_t = now + self.interval


def tts_one(text, voice, dest: Path, limiter: RateLimiter):
    if dest.exists() and dest.stat().st_size > 1000:
        return
    payload = {
        "contents": [{"parts": [{"text": f"با لحن طبیعی و محاوره‌ای فارسی بخوان:\n{text}"}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {
                "prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    for attempt in range(5):
        limiter.wait()
        r = requests.post(
            f"{API_BASE}/models/{TTS_MODEL}:generateContent",
            headers={"x-goog-api-key": KEY, "Content-Type": "application/json"},
            json=payload, timeout=300,
        )
        if r.status_code == 200:
            parts = r.json()["candidates"][0]["content"]["parts"]
            for p in parts:
                if "inlineData" in p:
                    pcm = base64.b64decode(p["inlineData"]["data"])
                    dest.parent.mkdir(exist_ok=True)
                    tmp = Path(str(dest) + ".tmp.wav")
                    with wave.open(str(tmp), "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(SAMPLE_RATE)
                        w.writeframes(pcm)
                    tmp.rename(dest)
                    return
            raise RuntimeError("tts: no audio in response")
        if r.status_code in (429, 500, 503):
            wait = 20 * (attempt + 1)
            log(f"tts {r.status_code}, retry in {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"tts -> {r.status_code}: {r.text[:300]}")
    raise RuntimeError(f"tts retries exhausted (voice={voice})")


def stage_tts():
    global threading
    import threading
    doc = json.loads((OUT / "segments.json").read_text())
    speakers = {s["id"]: s for s in doc.get("speakers", [])}
    voice_map, mi, fi = {}, 0, 0
    for sid in sorted(speakers):
        g = str(speakers[sid].get("gender", "")).lower()
        if "female" in g or "زن" in g:
            voice_map[sid] = FEMALE_VOICES[fi % len(FEMALE_VOICES)]; fi += 1
        else:
            voice_map[sid] = MALE_VOICES[mi % len(MALE_VOICES)]; mi += 1
    for sid, v in voice_map.items():
        log(f"voice: {sid} ({speakers[sid].get('description','')[:40]}) -> {v}")
    save_state(voice_map=voice_map)

    segs = doc["segments"]
    limiter = RateLimiter(float(os.environ.get("RPM", "10")))
    todo = [s for s in segs
            if not (TTS_DIR / f"seg_{s['id']:05d}.wav").exists()]
    log(f"tts: {len(todo)}/{len(segs)} segments to synthesize")
    for k, s in enumerate(todo):
        fa = (s.get("fa") or "").strip()
        if not fa:
            continue
        dest = TTS_DIR / f"seg_{s['id']:05d}.wav"
        try:
            tts_one(fa, voice_map.get(s["speaker"], "Puck"), dest, limiter)
        except RuntimeError as e:
            log(f"segment {s['id']} FAILED: {e} (continuing)")
        if (k + 1) % 25 == 0:
            log(f"tts progress: {k+1}/{len(todo)}")
    log("tts done")


# ------------------------------------------------------------ stage: assemble
def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
        return w.readframes(w.getnframes())


def fit_duration(src: Path, max_s: float, tmp: Path):
    """Speed up clip with atempo if it exceeds its timing window."""
    with wave.open(str(src), "rb") as w:
        dur = w.getnframes() / w.getframerate()
    if dur <= max_s or dur < 0.1:
        return read_wav(src)
    factor = min(1.35, dur / max(0.1, max_s) * 1.03)
    rc = subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                         "-i", src.as_posix(),
                         "-filter:a", f"atempo={factor:.4f}",
                         "-ar", str(SAMPLE_RATE), "-ac", "1",
                         tmp.as_posix()]).returncode
    if rc != 0:
        return read_wav(src)
    return read_wav(tmp)


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def stage_assemble():
    import array
    doc = json.loads((OUT / "segments.json").read_text())
    segs = sorted(doc["segments"], key=lambda s: s["start"])
    total = float(os.environ.get("DURATION_MIN", "15")) * 60 + 5
    buf = array.array("h", bytes(int(total * SAMPLE_RATE) * 2))
    srt_lines, placed, skipped = [], 0, 0
    tmp = OUT / "_tmp_speed.wav"
    for i, s in enumerate(segs):
        dest = TTS_DIR / f"seg_{s['id']:05d}.wav"
        if not dest.exists() or not s.get("fa"):
            skipped += 1
            continue
        next_start = segs[i + 1]["start"] if i + 1 < len(segs) else s["start"] + 60
        window = max(1.0, next_start - s["start"])
        pcm = fit_duration(dest, window, tmp)
        pos = int(s["start"] * SAMPLE_RATE)
        for j in range(0, len(pcm) - 1, 2):
            k = pos + j // 2
            if 0 <= k < len(buf):
                v = int.from_bytes(pcm[j:j + 2], "little", signed=True)
                buf[k] = max(-32768, min(32767, buf[k] + v))
        placed += 1
        srt_lines.append(f"{i+1}\n{srt_time(s['start'])} --> "
                         f"{srt_time(s['end'])}\n{s['fa']}\n")
    tmp.unlink(missing_ok=True)
    dub = OUT / "dubbed.wav"
    with wave.open(str(dub), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        w.writeframes(buf.tobytes())
    m4a = OUT / "dubbed.m4a"
    rc = run_ffmpeg(["-i", dub.as_posix(), "-c:a", "aac", "-b:a", "160k",
                     m4a.as_posix()])
    if rc != 0:
        raise RuntimeError("aac encode failed")
    srt = OUT / "subtitles.srt"
    srt.write_text("\n".join(srt_lines), encoding="utf-8")
    final = OUT / "dubbed.mp4"
    rc = run_ffmpeg(["-i", (OUT / "video.ts").as_posix(),
                     "-i", m4a.as_posix(),
                     "-map", "0:v:0", "-map", "1:a:0",
                     "-c:v", "copy", "-c:a", "copy", "-shortest",
                     final.as_posix()])
    if rc != 0:
        raise RuntimeError("mux failed")
    log(f"assemble done: placed={placed} skipped={skipped} "
        f"final={final.stat().st_size/1e6:.1f}MB")
    save_state(assemble="done", placed=placed)


def main():
    stages_env = os.environ.get("STAGES", "all")
    wanted = (["prep", "analyze", "tts", "assemble"]
              if stages_env == "all" else
              [s.strip() for s in stages_env.split(",") if s.strip()])
    for stage in wanted:
        log(f"=== stage: {stage} ===")
        if stage != "prep" and not KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")
        {"prep": stage_prep, "analyze": stage_analyze,
         "tts": stage_tts, "assemble": stage_assemble}[stage]()
    log("pipeline finished: " + ", ".join(wanted))


if __name__ == "__main__":
    main()
