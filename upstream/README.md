# Upstream checkout

The NVIDIA source is fetched, not vendored. Its immutable revision and license are in
`LOCK.json`.

```bash
python scripts/fetch_upstream.py
```

The resulting `upstream/Accelerated_TN_PTSBE/` directory is ignored by Git. This avoids
copying NVIDIA's retained paper data into Q-Tensor while keeping the exact source one
command away. Upstream copyright and Apache-2.0 notices remain in that checkout.
