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
