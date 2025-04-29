import os
import tarfile
import urllib.request

URL = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.6.0.tar.gz"
TARGET_DIR = "external/tensorflow"
LIB_DIR = os.path.join(TARGET_DIR, "lib")
ARCHIVE_NAME = "libtensorflow.tar.gz"

def download():
    print(f"Downloading from {URL} ...")
    urllib.request.urlretrieve(URL, ARCHIVE_NAME)
    print(f"Downloaded: {ARCHIVE_NAME}")

def extract():
    print("Extracting lib/ directory from archive...")
    with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
        members = [
            m for m in tar.getmembers()
            if m.name.startswith("libtensorflow/lib/") and m.isfile()
        ]
        for member in members:
            original_file = tar.extractfile(member)
            if original_file is None:
                continue
            relative_path = os.path.relpath(member.name, "libtensorflow/lib")
            target_path = os.path.join(LIB_DIR, relative_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "wb") as f:
                f.write(original_file.read())
    print("Extraction complete.")

def prepare_dirs():
    os.makedirs(LIB_DIR, exist_ok=True)

def clean():
    if os.path.exists(ARCHIVE_NAME):
        os.remove(ARCHIVE_NAME)
        print(f"Removed temporary file: {ARCHIVE_NAME}")

def main():
    prepare_dirs()
    download()
    extract()
    clean()
    print("TensorFlow C++ library installation (lib only) complete.")

if __name__ == "__main__":
    main()
