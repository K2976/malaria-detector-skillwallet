"""
download_dataset.py
====================
Automatically downloads and extracts the NIH Malaria Cell Images dataset.

The dataset contains ~27,000 microscopic cell images classified as:
  - Parasitized  (malaria-positive)
  - Uninfected   (malaria-negative)

After running this script the folder structure will be:
    dataset/
        Parasitized/
        Uninfected/
"""

import os
import sys
import zipfile
import shutil
import urllib.request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Primary mirror (NIH / National Library of Medicine)
DATASET_URL = (
    "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
)
# Fallback mirror
FALLBACK_URL = (
    "https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip"
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
ZIP_PATH = os.path.join(PROJECT_ROOT, "cell_images.zip")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Print a simple progress bar while downloading."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        bar_len = 40
        filled = int(bar_len * percent / 100)
        bar = "█" * filled + "─" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {percent:5.1f}%  ({downloaded / 1e6:.1f} MB)")
        sys.stdout.flush()


def download_file(url: str, dest: str) -> bool:
    """Download *url* to *dest*. Returns True on success."""
    try:
        print(f"⬇  Downloading from:\n   {url}")
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        print()  # newline after progress bar
        return True
    except Exception as exc:
        print(f"\n⚠  Download failed: {exc}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extract a ZIP archive to *extract_to*."""
    print(f"📦  Extracting archive to {extract_to} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("   Extraction complete.")


def organise_dataset(extract_root: str, dataset_dir: str) -> None:
    """
    The ZIP often extracts into a nested ``cell_images/cell_images/`` folder.
    This function moves Parasitized/ and Uninfected/ directly under *dataset_dir*.
    """
    # Locate the actual image folders (handle nested directories)
    for root, dirs, _files in os.walk(extract_root):
        if "Parasitized" in dirs and "Uninfected" in dirs:
            src_para = os.path.join(root, "Parasitized")
            src_uninf = os.path.join(root, "Uninfected")
            dst_para = os.path.join(dataset_dir, "Parasitized")
            dst_uninf = os.path.join(dataset_dir, "Uninfected")

            if not os.path.exists(dst_para):
                shutil.move(src_para, dst_para)
            if not os.path.exists(dst_uninf):
                shutil.move(src_uninf, dst_uninf)
            print(f"   ✅ Dataset organised under {dataset_dir}")
            return

    print("⚠  Could not locate Parasitized / Uninfected folders after extraction.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Download, extract and organise the malaria cell-images dataset."""

    # 1. Check if dataset already exists
    para_dir = os.path.join(DATASET_DIR, "Parasitized")
    uninf_dir = os.path.join(DATASET_DIR, "Uninfected")
    if os.path.isdir(para_dir) and os.path.isdir(uninf_dir):
        n_para = len(os.listdir(para_dir))
        n_uninf = len(os.listdir(uninf_dir))
        print(
            f"✅ Dataset already exists at {DATASET_DIR}\n"
            f"   Parasitized : {n_para} files\n"
            f"   Uninfected  : {n_uninf} files"
        )
        return

    os.makedirs(DATASET_DIR, exist_ok=True)

    # 2. Download ZIP
    if not os.path.isfile(ZIP_PATH):
        success = download_file(DATASET_URL, ZIP_PATH)
        if not success:
            print("   Trying fallback URL …")
            success = download_file(FALLBACK_URL, ZIP_PATH)
        if not success:
            print(
                "\n❌ Could not download the dataset automatically.\n"
                "   Please download it manually from:\n"
                "     https://ceb.nlm.nih.gov/repositories/malaria-datasets/\n"
                "   and place cell_images.zip in the project root."
            )
            sys.exit(1)
    else:
        print(f"📁 ZIP already present at {ZIP_PATH}")

    # 3. Extract
    tmp_extract = os.path.join(PROJECT_ROOT, "_tmp_extract")
    os.makedirs(tmp_extract, exist_ok=True)
    extract_zip(ZIP_PATH, tmp_extract)

    # 4. Organise
    organise_dataset(tmp_extract, DATASET_DIR)

    # 5. Clean up
    shutil.rmtree(tmp_extract, ignore_errors=True)
    os.remove(ZIP_PATH) if os.path.isfile(ZIP_PATH) else None
    print("🗑  Temporary files removed.\n🎉 Dataset ready!")


if __name__ == "__main__":
    main()
