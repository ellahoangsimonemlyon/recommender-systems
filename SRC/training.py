"""
Training Pipeline
------------------------------------------------------

This script trains the recommendation models and saves them separately.
"""

import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils import SephoraRecommendationSystem

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_directories():
    """Create necessary directories"""
    directories = [
        "../models",
        "../models/collaborative_filtering",
        "../models/content_based",
        "../models/mappings",
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def load_data():
    """Load training data"""
    try:
        reviews_df = pd.read_csv(
            "../data/reviews_df.csv",
            dtype={"author_id": "str"}
        )
        products_df = pd.read_csv("../data/product_info_df.csv")
        return reviews_df, products_df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def train_and_save_models(reviews_df, products_df):
    """Train the recommendation system and save all models separately"""
    print("Training recommendation system")

    # Initialize system
    print("Initializing recommendation system...")
    rec_system = SephoraRecommendationSystem(reviews_df, products_df)

    # Train models
    print("Training models...")
    training_success = rec_system.train_models(test_size=0.2)

    if not training_success:
        print(
            "Warning: Training had issues, ",
            "continuing with available models"
        )

    # Save models separately -- suggestion given by chatgpt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Saving models...")

    # 1. Save Collaborative Filtering Model (ALS)
    if rec_system.user_item_model is not None:
        cf_model_path = f"../models/collaborative_filtering/als_model_{timestamp}.pkl"
        with open(cf_model_path, "wb") as f:
            pickle.dump(rec_system.user_item_model, f)
        print(f"- Collaborative Filtering model saved: {cf_model_path}")
    else:
        print("- Collaborative Filtering model not available")

    # 2. Save Content-Based Model Components
    content_model = {
        "content_similarity_matrix": rec_system.content_similarity_matrix,
        "tfidf_ingredients": rec_system.tfidf_ingredients,
        "tfidf_highlights": rec_system.tfidf_highlights,
        "content_product_to_idx": rec_system.content_product_to_idx,
        "content_idx_to_product": rec_system.content_idx_to_product,
    }

    cb_model_path = f"../models/content_based/content_model_{timestamp}.pkl"
    with open(cb_model_path, "wb") as f:
        pickle.dump(content_model, f)
    print(f"- Content-based model saved: {cb_model_path}")

    # 3. Save User-Item Matrix and Mappings
    matrices_and_mappings = {
        "user_item_matrix": rec_system.user_item_matrix,
        "user_to_idx": rec_system.user_to_idx,
        "idx_to_user": rec_system.idx_to_user,
        "product_to_idx": rec_system.product_to_idx,
        "idx_to_product": rec_system.idx_to_product,
        "user_profiles": rec_system.user_profiles,
        "user_profile_encoders": rec_system.user_profile_encoders,
    }

    mappings_path = f"../models/mappings/mappings_{timestamp}.pkl"
    with open(mappings_path, "wb") as f:
        pickle.dump(matrices_and_mappings, f)
    print(f"- Mappings and matrices saved: {mappings_path}")

    # 4. Save preprocessed data
    processed_data = {
        "reviews_df": rec_system.reviews_df,
        "products_df": rec_system.products_df,
    }

    data_path = f"../models/processed_data_{timestamp}.pkl"
    with open(data_path, "wb") as f:
        pickle.dump(processed_data, f)
    print(f"- Processed data saved: {data_path}")

    # 5. Save model metadata
    metadata = {
        "timestamp": timestamp,
        "training_success": training_success,
        "stats": rec_system.get_user_stats(),
        "categories": rec_system.get_categories(),
        "model_files": {
            "collaborative_filtering": (
                cf_model_path if rec_system.user_item_model is not None else None
            ),
            "content_based": cb_model_path,
            "mappings": mappings_path,
            "processed_data": data_path,
        },
    }

    metadata_path = f"../models/model_metadata_{timestamp}.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"- Model metadata saved: {metadata_path}")

    # 6. Save latest model paths for easy loading
    latest_paths = {
        "collaborative_filtering": (
            cf_model_path if rec_system.user_item_model is not None else None
        ),
        "content_based": cb_model_path,
        "mappings": mappings_path,
        "processed_data": data_path,
        "metadata": metadata_path,
        "timestamp": timestamp,
    }

    with open("../models/latest_models.pkl", "wb") as f:
        pickle.dump(latest_paths, f)
    print("- Latest model paths saved")

    print("\nSuccess")

    # Display training summary
    print("\n\nTRAINING SUMMARY")

    stats = rec_system.get_user_stats()
    print(f"Total users: {stats['total_users']:,}")
    print(f"Total products: {stats['total_products']:,}")
    print(f"Total interactions: {stats['total_interactions']:,}")
    print(f"Matrix sparsity: {stats['sparsity']:.2f}%")
    print(f"Avg ratings per user: {stats['avg_ratings_per_user']:.1f}")
    print(f"Avg ratings per product: {stats['avg_ratings_per_product']:.1f}")

    categories = rec_system.get_categories()
    print("\nAvailable categories:")
    for cat_type, cat_list in categories.items():
        print(f"  {cat_type}: {len(cat_list)} categories")

    return rec_system, timestamp


def test_saved_models(timestamp):
    """Test that saved models can be loaded and work correctly"""
    print("\n\nTESTING SAVED MODELS")

    try:
        # Load latest model paths
        with open("../models/latest_models.pkl", "rb") as f:
            latest_paths = pickle.load(f)
        print("Latest model paths loaded")

        # Test loading each component
        if latest_paths["collaborative_filtering"]:
            with open(latest_paths["collaborative_filtering"], "rb") as f:
                cf_model = pickle.load(f)
            print("Collaborative filtering model loaded")

        with open(latest_paths["content_based"], "rb") as f:
            cb_model = pickle.load(f)
        print("Content-based model loaded")

        with open(latest_paths["mappings"], "rb") as f:
            mappings = pickle.load(f)
        print("Mappings loaded")

        with open(latest_paths["processed_data"], "rb") as f:
            processed_data = pickle.load(f)
        print("Processed data loaded")

        with open(latest_paths["metadata"], "rb") as f:
            metadata = pickle.load(f)
        print("Metadata loaded")

        print("\nAll models successfully saved and can be loaded!")

        return True

    except Exception as e:
        print(f"Error testing saved models: {e}")
        return False


def main():
    """Main training pipeline"""
    print("Sephora Recommendation System - Training Pipeline")

    # Create directories
    create_directories()

    # Load data
    reviews_df, products_df = load_data()

    # Train and save models
    rec_system, timestamp = train_and_save_models(reviews_df, products_df)

    # Test saved models
    test_success = test_saved_models(timestamp)

    if test_success:
        print("\nTraining pipeline completed successfully")
    else:
        print("\nTraining completed but model testing failed")

    return rec_system, timestamp


if __name__ == "__main__":
    rec_system, timestamp = main()
