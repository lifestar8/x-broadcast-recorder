# x-broadcast-recorder

Record X/Twitter (Periscope) broadcasts or live streams on GitHub Actions.

Based on the API flow of [offish/twitter-x-broadcast-downloader](https://github.com/offish/twitter-x-broadcast-downloader):
`broadcasts/show.json` -> media key -> `live_video_stream/status` -> HLS playback URL,
recorded with ffmpeg (stream copy, no re-encode).

## Usage

1. Go to **Actions -> Record X Broadcast -> Run workflow**
2. Set `broadcast_url` (e.g. `https://x.com/i/broadcasts/1AxRnZbVpjaxl`)
3. Optionally set `max_minutes` (default 300; job hard limit ~350 minutes)
4. Run. When the recording finishes (stream ended or cap reached), the video is
   uploaded as the **broadcast-recording** artifact (retained 7 days).

## Local usage

```bash
pip install requests
python record.py "https://x.com/i/broadcasts/XXXX" 300
```

Notes:
- Free GitHub-hosted runners allow max ~6h per job, hence the default cap.
- If the job is cancelled/timed out, raw `.ts` parts are still uploaded as fallback.
