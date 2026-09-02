# Website task-video refresh — 2026-09-02

- Full capture job: `10288510`, completed with exit code `0` in `00:16:47`.
- The published manifest contains 14 entries. Each was decoded successfully and
  verified as H.264, 1536×1024, yuv420p, 15 fps; every manifest SHA-256 matched.
- Simulator replay reached its real success predicate for 13 tasks. Sliding Door
  exhausted 10 deterministic replay candidates, then used episode 0 from the
  source Zarr store explicitly marked `success_only=true`; the manifest records
  `success_evidence: success-only-store`.
- `index.html` references exactly the 14 videos named in the manifest.
- The follow-up website check validated 14 task cards, posters, manifest entries,
  checksums, expected video encoding, and full MP4 decode via
  `scripts/verify_task_assets.py --decode`.
- A later layout revision removed task-card provenance and the redundant three-view
  task example, replaced the two method videos with their final full frames, and
  upgraded reachability media from 256×256 front views to 1024×256 multi-view videos.
