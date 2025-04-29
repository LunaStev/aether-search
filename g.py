import os
import tarfile
import urllib.request
import shutil

URL = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.6.0.tar.gz"
TARGET_DIR = "external/tensorflow"
LIB_DIR = os.path.join(TARGET_DIR, "lib")
ARCHIVE_NAME = "libtensorflow.tar.gz"
TMP_DIR = "temp_tensorflow"

def download():
    print(f"Downloading from {URL} ...")
    urllib.request.urlretrieve(URL, ARCHIVE_NAME)
    print(f"Downloaded: {ARCHIVE_NAME}")

def extract_all():
    print("Extracting entire archive temporarily...")
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)

    with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
        tar.extractall(TMP_DIR)
    print("Extraction complete.")

def move_lib():
    print("Moving lib/ folder...")
    src_lib_dir = os.path.join(TMP_DIR, "lib")
    if not os.path.exists(src_lib_dir):
        raise Exception(f"Source lib directory not found: {src_lib_dir}")

    if os.path.exists(LIB_DIR):
        shutil.rmtree(LIB_DIR)
    shutil.copytree(src_lib_dir, LIB_DIR)
    print("Move complete.")

def prepare_dirs():
    os.makedirs(TARGET_DIR, exist_ok=True)

def clean():
    if os.path.exists(ARCHIVE_NAME):
        os.remove(ARCHIVE_NAME)
        print(f"Removed temporary file: {ARCHIVE_NAME}")
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
        print(f"Removed temporary directory: {TMP_DIR}")

def main():
    prepare_dirs()
    download()
    extract_all()
    move_lib()
    clean()
    print("TensorFlow C++ library installation (lib only) complete.")

if __name__ == "__main__":
    main()
