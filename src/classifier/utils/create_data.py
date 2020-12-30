import glob
import pandas as pd

from tqdm import tqdm
from classifier import config


class Dataset:
    """Create dataset class"""

    def __init__(self):
        # Get all txt files
        self.paths = sorted(glob.glob("data/*/*/*.txt"))
        self.dataframe = None

    def load_data(self):
        dfs = []  # initialize list for dataframes

        # Loop over all txt files
        for filepath in tqdm(self.paths):

            # Read text files
            with open(filepath, "r") as f:
                text = f.read()

                # Create label from path
                if "pos" in filepath:
                    sentiment = "positief"
                else:
                    sentiment = "negatief"

                # Append dataframe to list
                dfs.append(pd.DataFrame({"text": [text],
                                         "sentiment": [sentiment]}))

        # Concat DataFrames
        self.dataframe = pd.concat(dfs).reset_index(drop=True)

    def save_data(self):
        # Create train and test split
        train_data = self.dataframe.sample(frac=config.SPLIT_SIZE,
                                           random_state=config.SEED)
        test_data = self.dataframe.iloc[train_data.index]

        # Save data
        train_data.to_csv(config.TRAIN_DATA, index=None)
        test_data.to_csv(config.TEST_DATA, index=None)
