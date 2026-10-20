"""Download the ALASKA v2 JPEG, TIFF and RAW datasets from https://alaska.utt.fr/.

Each dataset has 80,005 images numbered 00001 to 80005. Files that already exist
are skipped, so an interrupted run can simply be restarted.

Examples:
    python download_alaska.py --list
    python download_alaska.py ALASKA_v2_JPG_256_QF75_COLOR
    python download_alaska.py "ALASKA_v2_JPG_256_*" --out D:/alaska
    python download_alaska.py --all --out E:/alaska
    python download_alaska.py "*TIFF_256*" --first 1 --last 1000
    python download_alaska.py --raw --first 1 --last 500
"""

import argparse
import fnmatch
import io
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

BASE_URL = "https://alaska.utt.fr/DATASETS"
NUM_IMAGES = 80005

# RAW camera files have mixed extensions (.NEF, .CR2, .dng, ...), so their names come from the list
# inside the official scripts zip. --all deliberately excludes them: the full set is about 1.8 TB.
RAW_DATASET = "ALASKA_v2_RAWs"
RAW_GB = 1835.0  # estimated from 41 sampled files, about 23 MB each
RAW_ZIP_URL = "https://alaska.utt.fr/ALASKA_v2_RAWs_scripts.zip"
RAW_ZIP_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ALASKA_v2_RAWs_scripts.zip")
RAW_LIST_IN_ZIP = "Download_script/list_RAWs_ALASKA_v2.txt"

# Approximate full-dataset sizes in GB, estimated from 41 sampled files each.
DATASETS = {
    "ALASKA_v2_JPG_256_QF75_COLOR": 1.2,
    "ALASKA_v2_JPG_256_QF75_GrayScale": 0.8,
    "ALASKA_v2_JPG_256_QF80_COLOR": 1.4,
    "ALASKA_v2_JPG_256_QF80_GrayScale": 0.9,
    "ALASKA_v2_JPG_256_QF85_COLOR": 1.6,
    "ALASKA_v2_JPG_256_QF85_GrayScale": 1.1,
    "ALASKA_v2_JPG_256_QF90_COLOR": 2.1,
    "ALASKA_v2_JPG_256_QF90_GrayScale": 1.3,
    "ALASKA_v2_JPG_256_QF95_COLOR": 3.2,
    "ALASKA_v2_JPG_256_QF95_GrayScale": 1.9,
    "ALASKA_v2_JPG_256_QF100_COLOR": 8.0,
    "ALASKA_v2_JPG_256_QF100_GrayScale": 3.3,
    "ALASKA_v2_JPG_256_QFvarious_COLOR": 3.1,
    "ALASKA_v2_JPG_256_QFvarious_GrayScale": 1.7,
    "ALASKA_v2_JPG_512_QF75_COLOR": 4.4,
    "ALASKA_v2_JPG_512_QF75_GrayScale": 3.0,
    "ALASKA_v2_JPG_512_QF80_COLOR": 5.1,
    "ALASKA_v2_JPG_512_QF80_GrayScale": 3.4,
    "ALASKA_v2_JPG_512_QF85_COLOR": 6.2,
    "ALASKA_v2_JPG_512_QF85_GrayScale": 4.0,
    "ALASKA_v2_JPG_512_QF90_COLOR": 8.0,
    "ALASKA_v2_JPG_512_QF90_GrayScale": 5.0,
    "ALASKA_v2_JPG_512_QF95_COLOR": 12.2,
    "ALASKA_v2_JPG_512_QF95_GrayScale": 7.2,
    "ALASKA_v2_JPG_512_QF100_COLOR": 31.5,
    "ALASKA_v2_JPG_512_QF100_GrayScale": 12.7,
    "ALASKA_v2_JPG_512_QFvarious_COLOR": 12.0,
    "ALASKA_v2_JPG_512_QFvarious_GrayScale": 6.6,
    "ALASKA_v2_TIFF_256_COLOR": 15.7,
    "ALASKA_v2_TIFF_256_GrayScale": 5.3,
    "ALASKA_v2_TIFF_512_COLOR": 62.9,
    "ALASKA_v2_TIFF_512_GrayScale": 21.0,
    "ALASKA_v2_TIFF_VariousSize_COLOR": 131.4,
    "ALASKA_v2_TIFF_VariousSize_GrayScale": 43.8,
}


def raw_file_names():
    """Read the 80,005 RAW file names from the scripts zip, downloading the zip if it isn't present."""
    if os.path.exists(RAW_ZIP_LOCAL):
        source = RAW_ZIP_LOCAL
    else:
        print(f"Fetching RAW file list from {RAW_ZIP_URL}")
        with urllib.request.urlopen(RAW_ZIP_URL, timeout=60) as resp:
            source = io.BytesIO(resp.read())
    with zipfile.ZipFile(source) as z:
        names = z.read(RAW_LIST_IN_ZIP).decode().split()
    if len(names) != NUM_IMAGES:
        sys.exit(f"Unexpected RAW list: {len(names)} names instead of {NUM_IMAGES}")
    return names


def file_names(dataset):
    """All 80,005 file names of a dataset, in image-number order."""
    if dataset == RAW_DATASET:
        return raw_file_names()
    ext = ".jpg" if "_JPG_" in dataset else ".tif"
    return [f"{i:05d}{ext}" for i in range(1, NUM_IMAGES + 1)]


def fetch(url, dest, retries):
    """Download url to dest via a .part file. Returns None on success, else an error string."""
    tmp = dest + ".part"
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
            os.replace(tmp, dest)
            return None
        except (urllib.error.URLError, OSError) as e:
            error = str(e)
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                break
            time.sleep(2 * (attempt + 1))
    if os.path.exists(tmp):
        os.remove(tmp)
    return error


def download_dataset(dataset, out_root, first, last, workers, retries, log):
    out_dir = os.path.join(out_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    todo = []
    for name in file_names(dataset)[first - 1:last]:
        dest = os.path.join(out_dir, name)
        if not os.path.exists(dest):
            todo.append((f"{BASE_URL}/{dataset}/{name}", dest, name))

    total = last - first + 1
    print(f"\n{dataset}: {total - len(todo)} of {total} already present, {len(todo)} to download")
    if not todo:
        return 0

    start = time.time()
    done = failed = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(fetch, url, dest, retries): name for url, dest, name in todo}
        pending = set(futures)
        while pending:
            # Wait with a timeout so Ctrl+C is noticed promptly (an untimed wait can't be interrupted on Windows).
            finished, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for fut in finished:
                error = fut.result()
                if error:
                    failed += 1
                    log.write(f"{dataset}/{futures[fut]}\t{error}\n")
                    log.flush()
                done += 1
                if done % 200 == 0 or done == len(todo):
                    rate = done / max(time.time() - start, 1e-9)
                    eta = (len(todo) - done) / rate if rate else 0
                    print(f"  {done}/{len(todo)}  {rate:.1f} img/s  ETA {eta / 60:.0f} min  failed {failed}",
                          flush=True)
    finally:
        # On normal completion nothing is queued; on Ctrl+C or an error this drops queued downloads
        # so only the few already in progress finish before exit.
        pool.shutdown(wait=False, cancel_futures=True)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", help="dataset names or wildcard patterns (e.g. '*TIFF_256*')")
    parser.add_argument("--all", action="store_true", help="download all 34 JPEG and TIFF datasets (not RAW)")
    parser.add_argument("--raw", action="store_true",
                        help=f"download the original RAW camera files into {RAW_DATASET}/ (~{RAW_GB / 1000:.1f} TB in full)")
    parser.add_argument("--list", action="store_true", help="list available datasets with sizes and exit")
    parser.add_argument("--out", default=".", help="output folder (default: current folder)")
    parser.add_argument("--first", type=int, default=1, help="first image number (default 1)")
    parser.add_argument("--last", type=int, default=NUM_IMAGES, help=f"last image number (default {NUM_IMAGES})")
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads (default 8; keep it modest)")
    parser.add_argument("--retries", type=int, default=5, help="retries per file (default 5)")
    parser.add_argument("--force", action="store_true", help="start even if free disk space looks insufficient")
    args = parser.parse_args()

    if args.list:
        for name, gb in DATASETS.items():
            print(f"{gb:7.1f} GB  {name}")
        print(f"{sum(DATASETS.values()):7.1f} GB  TOTAL (JPEG + TIFF, what --all downloads)")
        print(f"{RAW_GB:7.1f} GB  {RAW_DATASET} (only with --raw)")
        return

    if args.all:
        selected = list(DATASETS)
    else:
        selected = []
        for pattern in args.datasets:
            matches = [d for d in DATASETS if fnmatch.fnmatch(d, pattern)]
            if not matches:
                parser.error(f"no dataset matches '{pattern}' (use --list to see names)")
            selected += [m for m in matches if m not in selected]
    if args.raw:
        selected.append(RAW_DATASET)
    if not selected:
        parser.error("give dataset names/patterns, --all or --raw (use --list to see names)")
    if not 1 <= args.first <= args.last <= NUM_IMAGES:
        parser.error(f"need 1 <= --first <= --last <= {NUM_IMAGES}")

    # Estimate space still needed, discounting files that are already downloaded.
    os.makedirs(args.out, exist_ok=True)
    fraction = (args.last - args.first + 1) / NUM_IMAGES
    needed = 0.0
    for d in selected:
        folder = os.path.join(args.out, d)
        have = len(os.listdir(folder)) if os.path.isdir(folder) else 0
        needed += DATASETS.get(d, RAW_GB) * max(0.0, fraction - have / NUM_IMAGES)
    free = shutil.disk_usage(args.out).free / 1e9
    print(f"{len(selected)} dataset(s) selected; about {needed:.1f} GB still needed, {free:.1f} GB free")
    if needed > free * 0.95 and not args.force:
        sys.exit("Not enough free space. Choose fewer datasets, a smaller --first/--last range, "
                 "another --out drive, or pass --force.")

    failed = 0
    with open(os.path.join(args.out, "alaska_failed_downloads.log"), "a") as log:
        try:
            for d in selected:
                failed += download_dataset(d, args.out, args.first, args.last, args.workers, args.retries, log)
        except KeyboardInterrupt:
            print("\nStopped. Downloads in progress will finish; rerun the same command to resume.", flush=True)
            sys.exit(130)

    print(f"\nFinished. {failed} file(s) failed" +
          (" - listed in alaska_failed_downloads.log; rerun to retry them." if failed else "."))


if __name__ == "__main__":
    main()
