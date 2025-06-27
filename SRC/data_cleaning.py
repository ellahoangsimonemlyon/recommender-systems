# Imports
import pandas as pd
from utils import clean_data

pd.set_option("display.max_columns", None)

# Data product_info
product_info = pd.read_csv("../data/product_info.csv")
# Data reviews -- merge together datasets into a single one
reviews1 = pd.read_csv(
    "../data/reviews_0-250.csv", dtype={"author_id": "str"}
)
reviews2 = pd.read_csv("../data//reviews_250-500.csv")
reviews3 = pd.read_csv("../data/reviews_500-750.csv")
reviews4 = pd.read_csv(
    "../data/reviews_750-1250.csv", dtype={"author_id": "str"}
)
reviews5 = pd.read_csv(
    "../data/reviews_1250-end.csv", dtype={"author_id": "str"}
)

reviews_df = pd.concat(
    [reviews1, reviews2, reviews3, reviews4, reviews5], axis=0
)

# Datasets
print(f"Product_info overview:{product_info.head()}")
print(f"\nReviews overview: {reviews_df.head()}")

# Print data insights
print(f"\n\n\nProduct_info info: {product_info.info()} ")
print(f"\nReviews info: {reviews_df.info()} ")

print(f"\nProduct_info describe: {product_info.describe()} ")
print(f"\nReviews describe: {reviews_df.describe()} ")

# Print shape of datasets
print(f"\nProduct_info shape: {product_info.shape}")
print(f"Reviews shape: {reviews_df.shape}")

# Data Cleaning -- product_info
missing_product_info = (
    product_info.isnull()
    .sum()
    .to_frame(name='missing_count')
    .assign(
        missing_percent=lambda x: (
            100 * x['missing_count'] / len(product_info)
        )
    )
    .sort_values(by='missing_count', ascending=False)
)

print("\n\nProduct_info missing values:\n", missing_product_info)

# Clean missing values
cols_to_drop = [
    "sale_price_usd",
    "value_price_usd",
    "variation_desc",
    "child_max_price",
    "child_min_price",
    "size",
    "variation_value",
    "variation_type",
    "child_count",
]
product_info_cleaned = clean_data(
    product_info, columns_to_drop=cols_to_drop, do_imputation=True
)  # 0.17 chosen based on the column type and the percentage of missing values

# Change the data type of reviews column to int.
product_info_cleaned["reviews"] = product_info_cleaned["reviews"].astype(int)
product_info_cleaned["loves_count"] = (
    product_info_cleaned["loves_count"]
    .astype(int)
)

# Change the data type of reviews column to int.
product_info_cleaned["reviews"] = product_info_cleaned["reviews"].astype(int)


# Data Cleaning -- reviews
missing_reviews = (
    reviews_df.isnull()
    .sum()
    .to_frame(name='missing_count')
    .assign(
        missing_percent=lambda x: (
            100 * x['missing_count'] / len(reviews_df)
        )
    )
    .sort_values(by='missing_count', ascending=False)
)

print("\n\nReviews missing values:\n", missing_reviews)

cols_to_drop = ["helpfulness", "review_title", "Unnamed: 0"]
reviews_df_cleaned = clean_data(
    reviews_df, columns_to_drop=cols_to_drop
)  # 0.25 based on the number of null values present in the dataset

# Change data time data types
reviews_df_cleaned["submission_time"] = pd.to_datetime(
    reviews_df_cleaned["submission_time"], errors="coerce"
)

# change column values to lower case
reviews_df_cleaned["skin_tone"] = (
    reviews_df_cleaned["skin_tone"].str.lower().str.strip()
)
reviews_df_cleaned["brand_name"] = (
    reviews_df_cleaned["brand_name"].str.lower().str.strip()
)
reviews_df_cleaned["product_name"] = (
    reviews_df_cleaned["product_name"].str.lower().str.strip()
)

# Remove duplicates -- final check
reviews_df_cleaned.drop_duplicates(inplace=True)


# Save the files as csv
product_info_cleaned.to_csv("../data/product_info_df.csv", index=False)
reviews_df_cleaned.to_csv("../data/reviews_df.csv", index=False)
