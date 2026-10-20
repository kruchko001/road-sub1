# ALASKA v2 dataset download

Scripts to download the [ALASKA v2](https://alaska.utt.fr/) image steganalysis datasets: the JPEG and TIFF image sets, plus the original RAW-processing scripts.

## Quick start

Requires Python 3.9+ (standard library only, nothing to install).

```bash
python download_alaska.py --list                          # show all datasets and their sizes
python download_alaska.py ALASKA_v2_JPG_256_QF75_COLOR    # download one dataset
```

Images are saved to `<out>/<dataset name>/00001.jpg … 80005.jpg` (or `.tif`).

## Available datasets

There are 34 datasets, each with 80,005 images named `00001` to `80005`. All of them together take about 433 GB. The sizes below are estimates from sampling 41 files per dataset.

**JPEG (`.jpg`)**: 28 sets. They come in two image sizes (256 or 512 px), colour or grayscale, at quality factors 75, 80, 85, 90, 95, 100, or "various" (a mix of quality factors).

| Quality factor | 256 colour | 256 gray | 512 colour | 512 gray |
|---|---|---|---|---|
| 75 | 1.2 GB | 0.8 GB | 4.4 GB | 3.0 GB |
| 80 | 1.4 GB | 0.9 GB | 5.1 GB | 3.4 GB |
| 85 | 1.6 GB | 1.1 GB | 6.2 GB | 4.0 GB |
| 90 | 2.1 GB | 1.3 GB | 8.0 GB | 5.0 GB |
| 95 | 3.2 GB | 1.9 GB | 12.2 GB | 7.2 GB |
| 100 | 8.0 GB | 3.3 GB | 31.5 GB | 12.7 GB |
| various | 3.1 GB | 1.7 GB | 12.0 GB | 6.6 GB |

**TIFF (`.tif`, uncompressed)**: 6 sets.

| Size | Colour | Gray |
|---|---|---|
| 256 px | 15.7 GB | 5.3 GB |
| 512 px | 62.9 GB | 21.0 GB |
| various sizes | 131.4 GB | 43.8 GB |

Subtotals: 256 px JPEG ≈ 32 GB, 512 px JPEG ≈ 121 GB, TIFF ≈ 280 GB.

Dataset names follow the pattern `ALASKA_v2_JPG_<size>_QF<quality>_<COLOR|GrayScale>` and `ALASKA_v2_TIFF_<size|VariousSize>_<COLOR|GrayScale>`. Run `--list` to see the exact names.

## Usage

```bash
# Several datasets by wildcard (quote the pattern)
python download_alaska.py "ALASKA_v2_JPG_256_*"
python download_alaska.py "*TIFF_512*" "*JPG_512_QF95*"

# Everything, to a large drive
python download_alaska.py --all --out E:/alaska

# Only part of each dataset (here, images 00001-05000)
python download_alaska.py "*TIFF_256*" --first 1 --last 5000
```

| Option | Default | Meaning |
|---|---|---|
| `datasets` | none | Dataset names or wildcard patterns |
| `--all` | off | Download all 34 datasets |
| `--list` | off | List datasets with estimated sizes and exit |
| `--out DIR` | `.` | Folder that receives one sub-folder per dataset |
| `--first N` / `--last N` | `1` / `80005` | Range of image numbers to download |
| `--workers N` | `8` | Parallel downloads. Keep this modest so the server isn't overloaded |
| `--retries N` | `5` | Retries per file before it is logged as failed |
| `--force` | off | Start even if the free-space check fails |

## Behaviour

- **Resumable.** Images that already exist are skipped, so rerunning the same command continues where it stopped.
- **Ctrl+C stops promptly.** Queued downloads are cancelled and only the few in progress finish, usually within seconds.
- **No partial files.** Each image downloads to a `.part` file and is renamed only when complete.
- **Disk-space check.** Before starting, the script estimates the space still needed (allowing for files already downloaded) and stops if it won't fit.
- **Failures** are appended to `<out>/alaska_failed_downloads.log`. Rerun the same command to retry them.
- **Speed.** Expect roughly 25 images/s for 256 px JPEG and about 10 images/s for TIFF, so a 256 px JPEG set takes about 1 hour. The full TIFF collection takes several days.

## Other files

| File | Purpose |
|---|---|
| `download_alaska.py` | Main downloader for all JPEG/TIFF datasets |
| `download_alaska.sh` | Older bash/curl script that downloads only `ALASKA_v2_JPG_256_QF100_COLOR` |
| `ALASKA_v2_JPG_256_QF100_COLOR.sh` | Original script from the ALASKA site (needs `wget`) |
| `ALASKA_v2_RAWs_scripts.zip` | Official RAW download and conversion scripts, plus `ALASKA2_documentation.pdf` |

## RAW images (not covered by the downloader)

The original RAW photos (80,005 files in about 35 camera formats such as NEF, CR2 and DNG) total roughly **1.8 TB**, at about 23 MB per file. To get them, unzip `ALASKA_v2_RAWs_scripts.zip` and use `Download_script/ALASKA_v2_RAWs_download_script.sh`, which requires `wget` and a drive with at least 2 TB free. You only need the RAWs to develop your own image variants with the conversion scripts. Most steganalysis work can use the ready-made JPEG/TIFF sets.

## Terms of use

The images are provided by the ALASKA team (UTT / CRIStAL). Check the license and disclaimers on [alaska.utt.fr](https://alaska.utt.fr/) before using or redistributing the data.
