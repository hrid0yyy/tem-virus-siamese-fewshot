"""F0 - data audit (no GPU needed).

Checks that the extracted patch set matches the official dataset exactly:
  * patches and micrographs per class and split == expected counts (Table 1 / Mendeley listing)
  * micrograph ids are contiguous 1..N in every folder
  * every patch is a readable 8-bit 256x256 image
  * no byte-identical patch appears in two different splits (leakage check)
  * prints the per-class table and writes reports/f0_report.json

    python f0_verify_data.py --data-root data
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temfs.constants import ALL_CLASSES, AUGMENTED_SPLIT, BASE_CLASSES, SPLITS  # noqa: E402
from temfs.data import (build_index, count_index, find_cross_split_duplicates, find_patch_root,  # noqa: E402
                        load_expected_counts, load_patch, verify_counts)
from temfs.utils import save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="reports/f0_report.json")
    ap.add_argument("--skip-pixels", action="store_true", help="skip reading every image")
    ap.add_argument("--skip-hash", action="store_true", help="skip the cross-split duplicate check")
    args = ap.parse_args()

    root = find_patch_root(args.data_root)
    print(f"Patch root: {root}")
    records, report = build_index(root, SPLITS, include_augmented=True)
    main_recs = [r for r in records if r.split in SPLITS]
    aug_recs = [r for r in records if r.split == AUGMENTED_SPLIT]
    print(f"Indexed {len(main_recs)} patches in train/validation/test and {len(aug_recs)} in {AUGMENTED_SPLIT}.")
    if report["missing_splits"]:
        print(f"MISSING split folders: {report['missing_splits']}")
    if report["unknown_classes"]:
        print(f"Unknown class folders (ignored): {report['unknown_classes']}")
    if report["unexpected_files"]:
        print(f"{len(report['unexpected_files'])} unexpected files (ignored), first: {report['unexpected_files'][:5]}")

    problems = verify_counts(main_recs)
    exp = load_expected_counts()
    aug_counts = Counter(r.cls for r in aug_recs)
    for c in BASE_CLASSES:
        if aug_counts.get(c, 0) != exp[c]["augmented_train"]["patches"]:
            problems.append(f"{AUGMENTED_SPLIT}/{c}: {aug_counts.get(c, 0)} patches, expected "
                            f"{exp[c]['augmented_train']['patches']}")

    got = count_index(main_recs)
    print(f"\n{'class':14s} {'base':>4s} | {'train p/img':>12s} | {'val p/img':>11s} | {'test p/img':>11s}")
    print("-" * 64)
    for c in ALL_CLASSES:
        cells = []
        for s in SPLITS:
            g = got.get(c, {}).get(s, {"patches": 0, "images": 0})
            cells.append(f"{g['patches']:5d}/{g['images']:3d}")
        print(f"{c:14s} {'yes' if c in BASE_CLASSES else 'no':>4s} | {cells[0]:>12s} | {cells[1]:>11s} | {cells[2]:>11s}")
    tot = {s: sum(got.get(c, {}).get(s, {"patches": 0})["patches"] for c in ALL_CLASSES) for s in SPLITS}
    print("-" * 64)
    print(f"{'total':14s}      | {tot['train']:12d} | {tot['validation']:11d} | {tot['test']:11d}")

    bad_pixels = []
    if not args.skip_pixels:
        print("\nReading every patch (shape/dtype check)...")
        for i, r in enumerate(records):
            try:
                load_patch(r.path)
            except Exception as e:  # noqa: BLE001
                bad_pixels.append(f"{r.path}: {e}")
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(records)}", flush=True)
        problems += bad_pixels[:50]

    dups = []
    if not args.skip_hash:
        print("Hashing patches for cross-split duplicates...")
        dups = find_cross_split_duplicates(main_recs)
        if dups:
            problems.append(f"{len(dups)} byte-identical patches shared between splits (see report)")

    save_json({"patch_root": str(root), "problems": problems, "unexpected_files": report["unexpected_files"],
               "cross_split_duplicates": dups, "bad_images": bad_pixels,
               "counts": {c: {s: {k: v for k, v in d.items() if k != "image_ids"} for s, d in got[c].items()}
                          for c in got}}, args.out)
    print()
    if problems:
        print("F0 RESULT: FAIL")
        for p in problems[:40]:
            print("  -", p)
        sys.exit(1)
    print("F0 RESULT: PASS - all counts match the official dataset, images readable, no cross-split duplicates.")
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
