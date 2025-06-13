import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
 
# Set Kaggle environment variable (optional if kaggle.json is in ~/.kaggle)
os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser("~/Desktop")
 
# Initialize and authenticate API
api = KaggleApi()
api.authenticate()
 
# Define the dataset
dataset = 'nadyinky/sephora-products-and-skincare-reviews'
download_path = 'data'  # Folder where files will be stored
 
# Create download directory if it doesn't exist
os.makedirs(download_path, exist_ok=True)
 
# Download the dataset
api.dataset_download_files(dataset, path=download_path, unzip=True)
 
# Load datasets into DataFrames
data_dir = "data"
review_files = sorted([
    f for f in os.listdir(data_dir) if f.startswith("reviews") and f.endswith(".csv")
])
 
dfs = []
for f in review_files:
    df = pd.read_csv(
        os.path.join(data_dir, f),
        delimiter=",",
        quotechar='"',
        escapechar="\\",
        engine="python",
        on_bad_lines="skip"
    )
    dfs.append(df)
 
all_reviews_df = pd.concat(dfs, ignore_index=True)
 
print(f"Loaded {len(all_reviews_df)} reviews.")
print(all_reviews_df.head())