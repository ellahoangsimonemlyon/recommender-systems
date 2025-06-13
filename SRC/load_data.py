import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
 
# Set Kaggle environment variable (optional if kaggle.json is in ~/.kaggle)
os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser("~/.kaggle")
 
# Initialize and authenticate API
api = KaggleApi()
api.authenticate()

# Dataset info
dataset = 'nadyinky/sephora-products-and-skincare-reviews'
download_path = 'data'
os.makedirs(download_path, exist_ok=True)

# Download and unzip
api.dataset_download_files(dataset, path=download_path, unzip=True)

# Load each CSV into a named DataFrame
data_dir = "data"
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

dataframes = {}
for f in csv_files:
    try:
        df = pd.read_csv(
            os.path.join(data_dir, f),
            delimiter=",",
            quotechar='"',
            escapechar="\\",
            engine="python",
            on_bad_lines="skip",
            encoding='utf-8'
        )
        dataframes[f] = df
        print(f"Loaded {f} with shape {df.shape}")
    except Exception as e:
        print(f"Failed to load {f}: {e}")

for name, df in dataframes.items():
    print(f"{name}: {df.shape}")

# Example usage: Accessing a specific DataFrame
print(df['review_text'].head())