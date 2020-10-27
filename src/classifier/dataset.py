import os

from classifier.utils import get_data, create_data


if __name__ == "__main__":
    # Path to dataset
    url = "https://github.com/benjaminvdb/110kDBRD/releases/download/v2.0/110kDBRD_v2.tgz"

    # Delete data
    os.system("rm -rf data/dutch-sentiment-data* \
        data/110kDBRD/ data/train/ data/test/")

    # Download data
    print("Downloading data...")
    get_data.download_data(url=url)

    # Extract data
    print("Extracting data...")
    get_data.extract_data("data/dutch-sentiment-data.tgz")

    # Move data
    print("Moving data...")
    get_data.move_raw_data()

    # Create Dataset object
    data = create_data.Dataset()

    # Load data
    print("Loading data...")
    data.load_data()

    # Save data
    print("Save data...")
    data.save_data()
