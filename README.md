# recommender-systems
Final Project

# Sephora Recommendation System

An AI-powered beauty product recommendation system that provides personalized suggestions using collaborative filtering and content-based approaches.

## Features:

- **Personalized Recommendations**: Get tailored product suggestions based on user history and preferences
- **Cold Start Handling**: Recommendations for new users based on profile characteristics
- **Category Filtering**: Filter recommendations by product categories (makeup, skincare, etc.)
- **Hybrid Approach**: Combines collaborative filtering (ALS) and content-based methods (TF-IDF)
- **Interactive Web App**: User-friendly Streamlit interface
- **Comprehensive Evaluation**: Built-in metrics for coverage, diversity, and bias analysis

## Project Structure

```
sephora-recommendation/
├── app/                    # Streamlit web application
│   └── app.py             # Main Streamlit app
├── data/                   # Data files and EDA
│   ├── reviews_df.csv     # Cleaned user reviews data
│   ├── product_info_df.csv # Cleaned product catalog data
|   ├── product_info.csv    # Product catalog data
│   └── reviews_0_250.csv   # User reviews data
├── models/                # Trained model artifacts (generated)
│   ├── collaborative_filtering/
│   ├── content_based/
│   ├── mappings/
│   └── latest_models.pkl
├── src/                   # Core Python modules
|   ├── EDA.ipynb         # Exploratory Data Analysis notebook
│   ├── utils.py          # SephoraRecommendationSystem class and cleaning functions
│   ├── load_data.py      # Data loading pipeline
|   ├── data_cleaning.py  # Cleaning data pipeline
│   ├── training.py       # Model training pipeline
|   ├── recommendations.ipynb    # Recommendation tests
│   ├── inference.py      # Model inference system
│   └── evaluation.py     # Model evaluation metrics
├── .gitignore 
├── environment.yml        # Conda environment file
└── README.md             # This file
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd recommender-system

# Create conda environment
conda env create -f environment.yml
conda activate recommender-system-env
```

### 2. Data Preparation

```bash
cd src
python load_data.py
```

Run the file `load_data.py` inside the `SRC` folder to download the data. A new folder `data` will be created containing:
- `reviews_0-250.csv`, `reviews_250-500.csv`, `reviews_500-750.csv`, `reviews_750-1250.csv`, `reviews_0125-end.csv`: User review data with columns: `author_id`, `rating`, `is_recommended`, `helpfulness`, `total_feedback_count`, `total_neg_feedback_count`, `total_pos_feedback_count`, `submission_time`, `review_text`, `skin_tone`, `eye_color`, `skin_type`, `hair_color`, `product_id`, `product_name`, `brand_name`, `price_usd`
- `product_info.csv`: Product information with columns: `product_id`, `product_name`, `brand_id`,`brand_name`, `loves_count`, `rating`, `reviews`, `size`, `variation_type`, `variation_value`, `variation_desc`, `ingredients`, `price_usd`,`value_price_usd`, `sale_price_usd`, `limited_edition`, `new`, `online_only`, `out_of_stock`, `sephora_exclusive`, `highlights`,`primary_category`, `secondary_category`, `tertiary_category`, `child_count`, `child_max_price`, `child_min_price`


### 3. Data Preprocessing

```bash
cd src
python data_cleaning.py
```

This script cleans and processes raw data by:

- Product Info: Removes columns with excessive nulls, converts data types, imputes category hierarchies, removes duplicates.
- Reviews: Merges CSVs, drops unnecessary columns, standardizes text (e.g. skin tone, brand), handles author IDs, removes duplicates.
- Smart Imputation: Hierarchical category imputation using most frequent values within groups.
- Data QA: Validates types, removes duplicates, checks consistency, and analyzes missing values.

Output files:
- data/product_info_df.csv – cleaned product data
- data/reviews_df.csv – cleaned and merged reviews data

### 4. Train Models

```bash
cd src
python training.py
```

This will:
- Train collaborative filtering (ALS) and content-based models
- Save models separately in the `models/` directory
- Generate training statistics and metadata

### 5. Evaluate Models

```bash
cd src
python inference.py
python evaluation.py
```

This generates a comprehensive evaluation report including:
- Coverage metrics (catalog and user coverage)
- Diversity analysis (category, brand, price diversity)
- Popularity bias detection
- RMSE, MAP@10, NDCG@10 and Precision@10
- Cold start performance assessment
- HTML report saved to `models/evaluation_report.html`

### 6. Launch Web App

```bash
cd app
streamlit python -m streamlit run app.py
```

The web app will be available at `http://localhost:8501`

## How It Works

### Recommendation Strategies

1. **Existing Users (Collaborative Filtering + Content-Based)**:
   - Uses ALS (Alternating Least Squares) matrix factorization
   - Combines with content-based filtering using TF-IDF on product ingredients and highlights
   - Prioritizes collaborative filtering recommendations, fills gaps with content-based

2. **New Users (Cold Start)**:
   - Matches user profile (skin tone, type, eye/hair color) with similar existing users
   - Recommends popular products from users with similar characteristics
   - Falls back to overall popular products if no profile match

3. **Category Filtering**:
   - Supports filtering by primary, secondary, and tertiary categories
   - Fallback system when no products match specific categories
   - Finds similar categories or relaxes filters progressively

### Model Architecture
Training System (`SephoraRecommendationSystem`):

Processes raw data and trains models
Saves trained models separately by type
Used once during development

Inference System (`SephoraInferenceSystem`): 

Loads pre-trained models for fast predictions
Handles real-time recommendation requests
Used by web app and evaluation

## Key Findings from EDA

Machine Learning Metrics:

- **MAP@10: 0.77** - Excellent ranking quality (research-paper level)
- **Precision@10: 0.81** - Very high precision
- **NDCG@10: 0.81** - Excellent normalized ranking
- **RMSE: 3.52** - Rating prediction accuracy (area for improvement)

System Metrics:

- **User Coverage: 100%** - All users receive recommendations
- **Cold Start Success: 100%** - Perfect new user handling
- **Brand Diversity: 0.69** - Good variety of brands
- **Catalog Coverage: 3.4%**- Conservative recommendation strategy

## Web App Features

### User Interface
- **User Selection**: Choose between existing user ID or new user profile
- **Profile Input**: Easy form for skin tone, type, eye color, hair color
- **Category Filters**: Dropdown menus for product categories
- **Recommendation Count**: Slider to select number of recommendations

### Visualizations
- **Price Distribution**: Histogram of recommended product prices
- **Category Distribution**: Pie chart of recommended categories
- **Rating vs Price**: Scatter plot showing value analysis
- **Summary Metrics**: Average price, rating, brand diversity

### Export Features
- **CSV Download**: Export recommendations for external use
- **Shareable Results**: Easy to share recommendation lists

## Model Configuration

### Collaborative Filtering (ALS)
```python
AlternatingLeastSquares(
    factors=50,           # Latent factors
    regularization=0.1,   # L2 regularization
    iterations=20,        # Training iterations
    alpha=40,            # Confidence scaling
    random_state=42
)
```

### Content-Based Filtering
- **TF-IDF Vectorization** on product ingredients and highlights
- **Cosine Similarity** for product-product recommendations
- **Feature Engineering** with text preprocessing and normalization

### Hybrid Combination
- Prioritizes collaborative filtering for existing users
- Uses content-based as fallback and diversity enhancer
- Intelligent blending based on user history and preferences

## Performance Metrics

### Coverage
- **Catalog Coverage**: Percentage of products that get recommended
- **User Coverage**: Percentage of users who receive recommendations

### Diversity
- **Category Diversity**: Variety of product categories in recommendations
- **Brand Diversity**: Variety of brands recommended per user
- **Price Diversity**: Range of price points in recommendations

### Quality
- **Popularity Bias**: Tendency to recommend only popular items
- **Cold Start Success**: Ability to recommend to new users
- **Rating Prediction**: Accuracy of predicted preferences

## Technical Implementation

### Dependencies
- **pandas/numpy**: Data manipulation and numerical computing
- **scikit-learn**: Machine learning utilities and metrics
- **implicit**: Collaborative filtering library (ALS implementation)
- **streamlit**: Web application framework
- **plotly**: Interactive visualizations

### Model Persistence
- Models saved separately for modularity:
  - `collaborative_filtering/`: ALS model artifacts
  - `content_based/`: TF-IDF vectors and similarity matrices
  - `mappings/`: User/product mappings and interaction matrices


## Acknowledgments

- Sephora for inspiration and beauty product domain knowledge
- Implicit library authors for collaborative filtering implementation
- Streamlit team for the web framework
- Open source community for the amazing tools and libraries


---
