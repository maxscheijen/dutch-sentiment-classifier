import wget
import tarfile
import os


def download_data(url: str) -> None:
    """Download data from url.

    Parameters
    ----------
    url : str
        url to file
    """
    wget.download(url=url, out="data/dutch-sentiment-data.tgz")


def extract_data(path: str) -> None:
    """Extract GZIP file from path.

    Parameters
    ----------
    path : str
        Path to GZIP file
    """
    tar = tarfile.open(path, mode="r")
    tar.extractall(path="data/")
    tar.close()


def move_raw_data() -> None:
    """Move train and test data and remove old directory.
    """
    os.system("mv data/110kDBRD/train data/train")
    os.system("mv data/110kDBRD/test data/test")
    os.system("rm -rf data/110kDBRD/ data/dutch-sentiment-data.tgz")


if __name__ == "__main__":
    # Path to dataset
    url = "https://github.com/benjaminvdb/110kDBRD/releases/download/v2.0/110kDBRD_v2.tgz"

    # Delete data
    os.system("rm -rf data/dutch-sentiment-data* \
        data/110kDBRD/ data/train/ data/test/")

    # Download data
    print("Downloading data...")
    download_data(url=url)

    # Extract data
    print("Extracting data...")
    extract_data("data/dutch-sentiment-data.tgz")

    # Move data
    print("Moving data...")
    move_raw_data()
