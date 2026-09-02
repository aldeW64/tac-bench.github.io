# Tac-Bench Project Website

Static project website for **Tac-Bench: Benchmarking Diverse Tactile Manipulation and Multimodal World Models**.

Accepted at the [RSS 2026 Workshop on Tactile Sensing for Robotic Foundation Models](https://tac-for-fm.github.io/rss2026/).

The site is self-contained and can be served from the repository root:

```bash
python3 -m http.server 8765
```

## Refreshing task videos

The 14 task-card videos are generated from the success-only ManiFeel Zarr
stores, not from the website's existing MP4 files.  Use the benchmark's virtual
environment and pass its root explicitly:

```bash
/data/user_data/peilinwu/Projects/tactile_benchmark_manifeel/.venv/bin/python \
  scripts/capture_task_videos.py \
  --benchmark-root /data/user_data/peilinwu/Projects/tactile_benchmark_manifeel \
  --allow-store-fallback
```

The capture tool renders a 1536×1024 H.264 video for every paper task only
after the simulator reports success. If a deterministic replay cannot reproduce
a source trajectory, `--allow-store-fallback` permits rendering that store only
when it is explicitly marked `success_only=true`; the manifest labels this
evidence source. `source/tasks/manifest.json` records the source store,
episode, seed, success step, layout, and checksum for each asset.

Each task card uses a lightweight poster and loads its MP4 only near the
viewport or on interaction. Its expandable **Replay provenance** row reads the
capture record from the manifest without exposing machine-local source paths.

Validate the task-card assets locally (the same check runs in GitHub Actions):

```bash
python3 scripts/verify_task_assets.py --decode
```
