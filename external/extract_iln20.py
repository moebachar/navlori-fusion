"""Extract Indoor Location Competition 2.0 zip into the layout `select_sites.py` expects.

Source zip layout:
  metadata/<site>/<floor>/{floor_image.png,floor_info.json,geojson_map.json}
  train/<site>/<floor>/<trace>.txt
  test/...                            (unlabeled, skipped)
  sample_submission.csv               (skipped)

Output layout (matches the handoff doc):
  data/<site>/<floor>/{floor_image.png,floor_info.json,geojson_map.json}
  data/<site>/<floor>/path_data_files/<trace>.txt

Streaming: no temp scratch, prints progress to stdout (flush=True for bg jobs).
"""
import os
import sys
import time
import zipfile

ZIP = r"X:\navlori-fusion\data\iln20\indoor-location-navigation.zip"
OUT = r"X:\navlori-fusion\data\iln20\data"


def remap(name: str) -> str | None:
    parts = name.split("/")
    if len(parts) < 3:
        return None
    head = parts[0]
    if head == "metadata" and len(parts) == 4:
        # metadata/<site>/<floor>/<file>  ->  data/<site>/<floor>/<file>
        return os.path.join(OUT, *parts[1:])
    if head == "train" and len(parts) == 4 and parts[3].endswith(".txt"):
        # train/<site>/<floor>/<trace>.txt  ->  data/<site>/<floor>/path_data_files/<trace>.txt
        site, floor, fn = parts[1], parts[2], parts[3]
        return os.path.join(OUT, site, floor, "path_data_files", fn)
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    n_kept = n_skipped = bytes_written = 0
    with zipfile.ZipFile(ZIP) as z:
        infos = z.infolist()
        total = len(infos)
        print(f"[extract] {total:,} entries in zip; remapping metadata+train, skipping test", flush=True)
        for i, zi in enumerate(infos):
            if zi.is_dir():
                continue
            target = remap(zi.filename)
            if target is None:
                n_skipped += 1
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target) and os.path.getsize(target) == zi.file_size:
                # idempotent re-runs: skip already-extracted files of correct size
                n_kept += 1
                continue
            with z.open(zi) as src, open(target, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)  # 1 MiB
                    if not chunk:
                        break
                    dst.write(chunk)
                    bytes_written += len(chunk)
            n_kept += 1
            if (i + 1) % 500 == 0 or i + 1 == total:
                elapsed = time.time() - t0
                pct = 100 * (i + 1) / total
                gb = bytes_written / 1024**3
                rate = gb / max(elapsed, 0.001)
                print(f"[extract] {i+1:,}/{total:,} ({pct:5.1f}%) "
                      f"kept={n_kept:,} skipped={n_skipped:,} "
                      f"written={gb:6.2f} GB @ {rate:4.2f} GB/s, "
                      f"elapsed={elapsed:6.0f}s", flush=True)
    print(f"[extract] DONE in {time.time()-t0:.0f}s "
          f"({n_kept:,} files kept, {n_skipped:,} skipped, "
          f"{bytes_written/1024**3:.2f} GB written)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
