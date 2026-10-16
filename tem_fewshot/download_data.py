"""Download and extract the TEM virus dataset (Mendeley Data 10.17632/x4dwwfwtw3.3).

The "Download All" archive is ~5.0 GB compressed (5,034,669,694 bytes); extracted, the patch set
is ~4.7 GB and the RAW micrographs ~3.3 GB.
By default only the official patch set (context_virus_1nm_256x256) is extracted.

    python download_data.py --out data
    python download_data.py --out data --include-raw          # also extract RAW micrographs
    python download_data.py --zip path/to/x4dwwfwtw3-3.zip --out data   # use a manually downloaded zip

The download resumes automatically if interrupted (re-run the same command).
The archive is kept after extraction unless --delete-zip is given.
"""

from __future__ import annotations

import argparse
import http.client
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temfs.constants import MENDELEY_ZIP_URL, PATCH_ROOT_NAME, RAW_ROOT_NAME  # noqa: E402

CHUNK = 1 << 20  # 1 MiB
UA = {"User-Agent": "Mozilla/5.0 (tem_fewshot dataset downloader)"}


def _total_from_headers(resp, offset):
    cr = resp.headers.get("Content-Range")  # "bytes start-end/total"
    if cr and "/" in cr:
        tail = cr.rsplit("/", 1)[1]
        if tail.isdigit():
            return int(tail)
    cl = resp.headers.get("Content-Length")
    return (int(cl) + offset) if cl and cl.isdigit() else None


def download(url, dest: Path, retries=20):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        print(f"Archive already present: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
        return dest
    attempt = 0
    total = None
    while True:
        offset = part.stat().st_size if part.exists() else 0
        if total is not None and offset >= total:
            break
        headers = dict(UA)
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"
        req = urllib.request.Request(url, headers=headers)  # Mendeley redirects to a signed S3 URL
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                status = resp.status
                if offset > 0 and status == 200:
                    print("Server ignored the resume request; restarting from 0.")
                    offset = 0
                    part.unlink(missing_ok=True)
                total = _total_from_headers(resp, offset) or total
                mode = "ab" if offset > 0 else "wb"
                done = offset
                t0, last = time.time(), 0.0
                with open(part, mode) as fh:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        fh.write(block)
                        done += len(block)
                        now = time.time()
                        if now - last > 5:
                            speed = (done - offset) / max(1e-6, now - t0) / 1e6
                            pct = f"{100 * done / total:5.1f}%" if total else "?"
                            print(f"  {done / 1e9:6.2f} GB {pct}  {speed:5.1f} MB/s", flush=True)
                            last = now
            if total is None or done >= total:
                break
            raise urllib.error.URLError(f"connection closed early at {done} of {total} bytes")
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError, OSError) as e:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"download failed after {retries} retries: {e}") from e
            wait = min(60, 5 * attempt)
            print(f"  download interrupted ({e}); retry {attempt}/{retries} in {wait}s (resumes)")
            time.sleep(wait)
    size = part.stat().st_size
    if total is not None and size != total:
        raise RuntimeError(f"size mismatch: got {size} bytes, expected {total}")
    os.replace(part, dest)
    print(f"Downloaded {dest} ({size / 1e9:.2f} GB)")
    return dest


def _wanted(name: str, include_raw: bool) -> bool:
    parts = name.replace("\\", "/").split("/")
    if PATCH_ROOT_NAME in parts:
        return True
    return include_raw and RAW_ROOT_NAME in parts


def extract(zip_path: Path, out_dir: Path, include_raw=False):
    out_dir = out_dir.resolve()
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"{zip_path} is not a valid zip (incomplete download?). Delete it and re-run.") from e
    with zf:
        members = [m for m in zf.infolist() if not m.is_dir() and _wanted(m.filename, include_raw)]
        if not members:
            names = [m.filename for m in zf.infolist()[:20]]
            raise RuntimeError(f"No '{PATCH_ROOT_NAME}' entries found in the archive. First entries: {names}")
        print(f"Extracting {len(members)} files to {out_dir} ...")
        n_new = 0
        for i, m in enumerate(members):
            target = (out_dir / m.filename).resolve()
            if os.path.commonpath([str(out_dir), str(target)]) != str(out_dir):
                raise RuntimeError(f"unsafe path in archive: {m.filename}")
            if target.exists() and target.stat().st_size == m.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            n_new += 1
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(members)}", flush=True)
        print(f"Done: {n_new} files written, {len(members) - n_new} already present.")
        # Nested archives (e.g. the authors' 'dataset code.zip') are left as-is on purpose.


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="output folder (default: data)")
    ap.add_argument("--zip", default=None, help="use an already downloaded archive instead of downloading")
    ap.add_argument("--url", default=MENDELEY_ZIP_URL)
    ap.add_argument("--include-raw", action="store_true", help="also extract the RAW micrographs (~3.3 GB)")
    ap.add_argument("--delete-zip", action="store_true",
                    help="delete the downloaded archive after a successful extraction (default: keep it)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free / 1e9
    print(f"Free disk space at {out.resolve()}: {free:.1f} GB")
    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.exists():
            sys.exit(f"Archive not found: {zip_path}")
    else:
        if free < 12:
            print("WARNING: less than 12 GB free; the archive (~5 GB) plus the extracted patch set (~4.7 GB) "
                  "needs ~10 GB (+3.3 GB with --include-raw).")
        zip_path = download(args.url, out / "x4dwwfwtw3-3.zip")
    extract(zip_path, out, include_raw=args.include_raw)
    if not args.zip and args.delete_zip:
        print(f"Removing archive {zip_path}")
        zip_path.unlink()
    print("\nNext step:  python f0_verify_data.py --data-root", out)


if __name__ == "__main__":
    main()
