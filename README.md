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

## 📁 Project Structure

```
sephora-recommendation/
├── app/                    # Streamlit web application
│   └── app.py             # Main Streamlit app
├── data/                   # Data files and EDA
│   ├── reviews_df.csv     # Cleaned user reviews data
│   ├── product_info_df.csv # Cleaned product catalog data
|   ├── product_info.csv    # Product catalog data
|   ├── reviews_0_250.csv   # User reviews data
│   └── EDA.ipynb         # Exploratory Data Analysis notebook
├── models/                # Trained model artifacts (generated)
│   ├── collaborative_filtering/
│   ├── content_based/
│   ├── mappings/
│   └── latest_models.pkl
├── src/                   # Core Python modules
│   ├── utils.py          # SephoraRecommendationSystem class and cleaning functions
│   ├── load_data.py      # Data loading pipeline
|   ├── data_cleaning     # Cleaning data pipeline
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
cd sephora-recommendation

# Create conda environment
conda env create -f environment.yml
conda activate sephora-recommendation
```

### 2. Data Preparation

Place your data files in the `data/` directory:
- `reviews_df.csv`: User review data with columns: `author_id`, `product_id`, `rating`, `skin_tone`, `skin_type`, `eye_color`, `hair_color`
- `product_info_df.csv`: Product information with columns: `product_id`, `product_name`, `brand_name`, `price_usd`, `primary_category`, `secondary_category`, `tertiary_category`, `ingredients`, `highlights`

### 3. Data Preprocessing (Optional)

```bash
cd src
python preprocessing.py
```

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
python evaluation.py
```

This generates a comprehensive evaluation report including:
- Coverage metrics (catalog and user coverage)
- Diversity analysis (category, brand, price diversity)
- Popularity bias detection
- Cold start performance assessment
- HTML report saved to `models/evaluation_report.html`

### 6. Launch Web App

```bash
cd app
streamlit run app.py
```

The web app will be available at `http://localhost:8501`

## 💡 How It Works

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
   - Intelligent fallback system when no products match specific categories
   - Finds similar categories or relaxes filters progressively

### Data Schema

**Reviews Data** (`reviews_df.csv`):
```
author_id       : Unique user identifier
product_id      : Unique product identifier  
rating          : Rating score (1-5)
skin_tone       : User's skin tone (light, medium, dark, etc.)
skin_type       : User's skin type (dry, oily, combination, etc.)
eye_color       : User's eye color
hair_color      : User's hair color
```

**Products Data** (`product_info_df.csv`):
```
product_id         : Unique product identifier
product_name       : Product name
brand_name         : Brand name
price_usd          : Price in USD
rating             : Average product rating
primary_category   : Main category (Makeup, Skincare, etc.)
secondary_category : Subcategory
tertiary_category  : Specific product type
ingredients        : Product ingredients text
highlights         : Product highlights/features text
```

## 📊 Key Findings from EDA

- **Dataset Size**: ~X users, ~Y products, ~Z interactions
- **Sparsity**: The user-item matrix is highly sparse (~95%+)
- **Popular Categories**: Makeup and Skincare dominate the catalog
- **Rating Distribution**: Most ratings are positive (4-5 stars)
- **Price Range**: Products range from budget-friendly to luxury
- **User Profiles**: Diverse user characteristics enable effective cold start

## 🎮 Web App Features

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

## 🔧 Model Architecture

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

## 📈 Performance Metrics

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

## 🛠️ Technical Implementation

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
  - `processed_data/`: Cleaned and preprocessed datasets

### Scalability Considerations
- **Sparse Matrix Storage**: Efficient memory usage for large datasets
- **Incremental Training**: Models can be retrained with new data
- **Batch Inference**: Support for bulk recommendation generation
- **Caching**: Model loading cached for web app performance

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Make sure you're in the right directory and environment is activated
   conda activate sephora-recommendation
   cd src  # for running training/evaluation scripts
   cd app  # for running web app
   ```

2. **Missing Models**:
   ```bash
   # Train models first
   cd src
   python training.py
   ```

3. **Data Issues**:
   ```bash
   # Check data format and run preprocessing
   cd src
   python preprocessing.py
   ```

4. **Memory Issues**:
   - Reduce matrix dimensions in `utils.py`
   - Use smaller sample for evaluation
   - Increase system memory or use cloud instance

### Performance Optimization

- **For Large Datasets**: Consider using approximate algorithms
- **For Real-time Serving**: Implement model serving with FastAPI
- **For Scale**: Consider distributed computing with Dask/Spark

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Sephora for inspiration and beauty product domain knowledge
- Implicit library authors for collaborative filtering implementation
- Streamlit team for the excellent web framework
- Open source community for the amazing tools and libraries

## 📞 Contact

For questions, issues, or collaboration opportunities, please open an issue or contact the development team.

---

**Built with ❤️ for beauty enthusiasts and data science practitioners**