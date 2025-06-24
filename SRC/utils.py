# Imports
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
import implicit


##### Cleaning data functions #####

def clean_data(df, columns_to_drop, do_imputation = False):
    # Remove columns with many missing values
    final_df = df.drop(columns = columns_to_drop)

    # If imputation is enabled, apply value-based imputation on category columns
    if do_imputation == True:
        final_df = value_impute(final_df, 'primary_category', 'secondary_category')
        final_df = value_impute(final_df, 'secondary_category', 'tertiary_category')

    # Drop null values and duplicates
    final_df = final_df.dropna(axis=0)
    final_df = final_df.drop_duplicates()

    return final_df


# Function to impute the values based on similar columns: if there's a missing value, the function imputes the most common value for each category, or returns the same value
def value_impute(df, col, target_col):
    df = df.copy()

    mode_map = (
        df
        .dropna(subset=[target_col])
        .groupby(col)[target_col]
        .agg(lambda x: x.mode().iloc[0])
    )
    def impute_column(row):
        if pd.isnull(row[target_col]):
            return mode_map.get(row[col], row[col])
        else:
            return row[target_col]
    df[target_col] = df.apply(impute_column, axis=1)
    return df


class SephoraRecommendationSystem:
    def __init__(self, reviews_df, products_df):
        """
        Initialize the recommendation system with reviews and products data
        
        Args:
            reviews_df: DataFrame with review data
            products_df: DataFrame with product information
        """
        self.reviews_df = reviews_df.copy()
        self.products_df = products_df.copy()
        
        # Initialize models
        self.user_item_model = None
        self.item_similarity_model = None
        self.content_similarity_matrix = None
        
        # Mappings
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.product_to_idx = {}
        self.idx_to_product = {}
        
        # User profiles for cold start
        self.user_profiles = {}
        
        # Initialize matrices
        self.user_item_matrix = None
        self.train_matrix = None
        self.test_matrix = None
        self.product_features_matrix = None
        self.user_features_matrix = None
        
        # Content-based mappings (separate from collaborative filtering mappings)
        self.content_product_to_idx = {}
        self.content_idx_to_product = {}
        
        # Encoders
        self.user_profile_encoders = {}
        self.tfidf_ingredients = TfidfVectorizer(max_features=500, stop_words='english')
        self.tfidf_highlights = TfidfVectorizer(max_features=300, stop_words='english')
        
        # Clean and prepare data
        success = self._prepare_data()
        if not success:
            raise ValueError("Failed to prepare data. Please check your input DataFrames.")
        
    def _prepare_data(self):
        """Clean and prepare the data for modeling"""
        try:
            print("Preparing data...")
            
            # Check if required columns exist
            required_review_cols = ['author_id', 'product_id', 'rating']
            required_product_cols = ['product_id']
            
            missing_review_cols = [col for col in required_review_cols if col not in self.reviews_df.columns]
            missing_product_cols = [col for col in required_product_cols if col not in self.products_df.columns]
            
            if missing_review_cols:
                print(f"ERROR: Missing required columns in reviews_df: {missing_review_cols}")
                return False
            if missing_product_cols:
                print(f"ERROR: Missing required columns in products_df: {missing_product_cols}")
                return False
            
            # Clean data
            self._clean_data()
            
            # Create user-item interaction matrix
            success = self._create_user_item_matrix()
            if not success:
                return False
            
            # Create content-based features
            self._create_content_features()
            
            # Create user profiles for cold start
            self._create_user_profiles()
            
            return True
            
        except Exception as e:
            print(f"ERROR in _prepare_data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _clean_data(self):
        """Clean and preprocess the data"""
        # Handle missing values in reviews
        self.reviews_df['rating'] = self.reviews_df['rating'].fillna(3.0)
        self.reviews_df['skin_tone'] = self.reviews_df['skin_tone'].fillna('unknown')
        self.reviews_df['eye_color'] = self.reviews_df['eye_color'].fillna('unknown')
        self.reviews_df['skin_type'] = self.reviews_df['skin_type'].fillna('unknown')
        self.reviews_df['hair_color'] = self.reviews_df['hair_color'].fillna('unknown')
        
        # Handle missing values in products
        self.products_df['ingredients'] = self.products_df['ingredients'].fillna('')
        self.products_df['highlights'] = self.products_df['highlights'].fillna('')
        self.products_df['rating'] = self.products_df['rating'].fillna(3.0)
        self.products_df['price_usd'] = self.products_df['price_usd'].fillna(self.products_df['price_usd'].median())
        
        # Convert string representations to actual strings for TF-IDF
        self.products_df['ingredients_text'] = self.products_df['ingredients'].astype(str)
        self.products_df['highlights_text'] = self.products_df['highlights'].astype(str)
    
    def _create_user_item_matrix(self):
        """Create sparse user-item interaction matrix"""
        try:
            # Create mappings
            unique_users = self.reviews_df['author_id'].unique()
            unique_products = self.reviews_df['product_id'].unique()
            
            if len(unique_users) == 0 or len(unique_products) == 0:
                print("ERROR: No users or products found")
                return False
            
            self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
            self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
            self.product_to_idx = {product: idx for idx, product in enumerate(unique_products)}
            self.idx_to_product = {idx: product for product, idx in self.product_to_idx.items()}
            
            # Create interaction matrix (using rating as implicit feedback)
            rows = self.reviews_df['author_id'].map(self.user_to_idx)
            cols = self.reviews_df['product_id'].map(self.product_to_idx)
            data = self.reviews_df['rating'].values
            
            # Check for any mapping errors
            if rows.isna().any() or cols.isna().any():
                print("ERROR: Failed to map users or products to indices")
                return False
            
            self.user_item_matrix = csr_matrix(
                (data, (rows, cols)), 
                shape=(len(unique_users), len(unique_products))
            )
            
            print(f"User-Item Matrix Shape: {self.user_item_matrix.shape}")
            print(f"Sparsity: {(1 - self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])) * 100:.2f}%")
            
            if self.user_item_matrix.nnz == 0:
                print("ERROR: User-item matrix has no interactions")
                return False
                
            return True
            
        except Exception as e:
            print(f"ERROR in _create_user_item_matrix: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_content_features(self):
        """Create content-based similarity matrix using TF-IDF"""
        try:
            # Get products that are in the reviews (to match matrix dimensions)
            product_ids_in_reviews = set(self.reviews_df['product_id'].unique())
            products_filtered = self.products_df[
                self.products_df['product_id'].isin(product_ids_in_reviews)
            ].copy()
            
            if products_filtered.empty:
                print("Warning: No products found in reviews")
                self.content_similarity_matrix = None
                self.content_product_to_idx = {}
                self.content_idx_to_product = {}
                return
            
            # Create a separate mapping for content similarity that matches the matrix order
            # Sort by product_id to ensure consistent ordering
            products_filtered = products_filtered.sort_values('product_id').reset_index(drop=True)
            
            # Create content-specific mappings
            self.content_product_to_idx = {
                product_id: idx for idx, product_id in enumerate(products_filtered['product_id'])
            }
            self.content_idx_to_product = {
                idx: product_id for product_id, idx in self.content_product_to_idx.items()
            }
            
            # Create TF-IDF features for ingredients and highlights
            if 'ingredients_text' in products_filtered.columns:
                ingredients_tfidf = self.tfidf_ingredients.fit_transform(
                    products_filtered['ingredients_text']
                )
            else:
                ingredients_tfidf = csr_matrix((len(products_filtered), 0))
            
            if 'highlights_text' in products_filtered.columns:
                highlights_tfidf = self.tfidf_highlights.fit_transform(
                    products_filtered['highlights_text']
                )
            else:
                highlights_tfidf = csr_matrix((len(products_filtered), 0))
            
            # Combine content features
            if ingredients_tfidf.shape[1] > 0 and highlights_tfidf.shape[1] > 0:
                combined_features = hstack([ingredients_tfidf, highlights_tfidf])
            elif ingredients_tfidf.shape[1] > 0:
                combined_features = ingredients_tfidf
            elif highlights_tfidf.shape[1] > 0:
                combined_features = highlights_tfidf
            else:
                print("Warning: No content features available")
                self.content_similarity_matrix = None
                self.content_product_to_idx = {}
                self.content_idx_to_product = {}
                return
            
            # Calculate cosine similarity
            self.content_similarity_matrix = cosine_similarity(combined_features)
            print(f"Content Similarity Matrix Shape: {self.content_similarity_matrix.shape}")
            print(f"Content mapping created for {len(self.content_product_to_idx)} products")
            
        except Exception as e:
            print(f"Warning: Could not create content features: {e}")
            self.content_similarity_matrix = None
            self.content_product_to_idx = {}
            self.content_idx_to_product = {}
    
    def _create_user_profiles(self):
        """Create user profiles based on their characteristics for cold start"""
        try:
            self.user_profiles = {}
            
            # Group users by their characteristics
            profile_features = ['skin_tone', 'eye_color', 'skin_type', 'hair_color']
            available_features = [f for f in profile_features if f in self.reviews_df.columns]
            
            if not available_features:
                print("Warning: No profile features available for user profiling")
                return
            
            # Create encoders for user profile features
            for feature in available_features:
                encoder = LabelEncoder()
                self.reviews_df[f'{feature}_encoded'] = encoder.fit_transform(
                    self.reviews_df[feature].astype(str)
                )
                self.user_profile_encoders[feature] = encoder
            
            # Group by profile features
            for features, group in self.reviews_df.groupby(available_features, dropna=False):
                if len(group) >= 5:  # Only consider profiles with enough data
                    # Get popular products for this profile
                    popular_products = (group.groupby('product_id')['rating']
                                      .agg(['mean', 'count'])
                                      .reset_index())
                    popular_products = popular_products[popular_products['count'] >= 2]
                    popular_products = popular_products.sort_values(['mean', 'count'], ascending=False)
                    
                    profile_key = tuple(features) if isinstance(features, tuple) else (features,)
                    self.user_profiles[profile_key] = popular_products['product_id'].head(20).tolist()
                    
            print(f"Created {len(self.user_profiles)} user profiles")
            
        except Exception as e:
            print(f"Warning: Could not create user profiles: {e}")
            self.user_profiles = {}
    
    def train_models(self, test_size=0.2):
        """Train both collaborative filtering and content-based models"""
        try:
            if self.user_item_matrix is None:
                raise ValueError("User-item matrix is None. Data preparation may have failed.")
            
            if self.user_item_matrix.nnz == 0:
                raise ValueError("User-item matrix is empty. No interactions found.")
            
            print(f"Training with matrix shape: {self.user_item_matrix.shape}")
            print(f"Non-zero entries: {self.user_item_matrix.nnz}")
            
            # Split data for evaluation
            try:
                train_matrix, test_matrix = train_test_split(self.user_item_matrix, train_percentage=1-test_size)
                self.train_matrix = train_matrix
                self.test_matrix = test_matrix
                print(f"Train matrix shape: {train_matrix.shape}, nnz: {train_matrix.nnz}")
                print(f"Test matrix shape: {test_matrix.shape}, nnz: {test_matrix.nnz}")
            except Exception as e:
                print(f"Warning: Error in train_test_split: {e}")
                self.train_matrix = self.user_item_matrix
                self.test_matrix = None
                print("Using full matrix for training (no test split)")
            
            print("Training Collaborative Filtering Models...")
            
            # Train ALS model (Matrix Factorization)
            try:
                self.user_item_model = implicit.als.AlternatingLeastSquares(
                    factors=50,
                    regularization=0.1,
                    iterations=20,
                    alpha=40,
                    random_state=42
                )
                self.user_item_model.fit(self.train_matrix)
                print("ALS model trained successfully!")
            except Exception as e:
                print(f"Warning: Error training ALS model: {e}")
                self.user_item_model = None
            
            print("Models training completed!")
            return True
            
        except Exception as e:
            print(f"ERROR in train_models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_user_recommendations(self, user_id=None, user_profile=None, 
                               primary_category=None, secondary_category=None, 
                               tertiary_category=None, n_recommendations=10):
        """
        Get recommendations for a user
        
        Args:
            user_id: Known user ID
            user_profile: Dict with user characteristics if user_id is None
            primary_category: Filter by primary category
            secondary_category: Filter by secondary category  
            tertiary_category: Filter by tertiary category
            n_recommendations: Number of recommendations to return
        """
        try:
            if user_id is not None and user_id in self.user_to_idx:
                return self._get_existing_user_recommendations(
                    user_id, primary_category, secondary_category, 
                    tertiary_category, n_recommendations
                )
            else:
                return self._get_cold_start_recommendations(
                    user_profile, primary_category, secondary_category,
                    tertiary_category, n_recommendations
                )
        except Exception as e:
            print(f"ERROR in get_user_recommendations: {e}")
            return self._get_popular_products_formatted(n_recommendations)
    
    def _get_existing_user_recommendations(self, user_id, primary_category, 
                                        secondary_category, tertiary_category, n_recommendations):
        """Get recommendations for existing user using collaborative filtering + content-based"""
        user_idx = self.user_to_idx[user_id]
        
        # Get user's rated products for content-based filtering
        user_rated_products = set(self.reviews_df[
            self.reviews_df['author_id'] == user_id
        ]['product_id'].values)
        
        recommendations = []
        
        # Get collaborative filtering recommendations
        if self.user_item_model is not None:
            try:
                cf_recs, cf_scores = self.user_item_model.recommend(
                    user_idx, self.user_item_matrix[user_idx], 
                    N=n_recommendations*2, filter_already_liked_items=True
                )
                cf_recs = cf_recs.tolist() if hasattr(cf_recs, 'tolist') else list(cf_recs)
                cf_scores = cf_scores.tolist() if hasattr(cf_scores, 'tolist') else list(cf_scores)
                
                # Convert indices to product IDs
                cf_product_ids = [self.idx_to_product[idx] for idx in cf_recs if idx in self.idx_to_product]
                
            except Exception as e:
                print(f"Warning: Error getting CF recommendations: {e}")
                cf_product_ids = []
        else:
            cf_product_ids = []
        
        # Get content-based recommendations
        cb_product_ids = self._get_content_based_recommendations(user_rated_products, n_recommendations*2)
        
        # Combine recommendations (prioritize CF, then CB)
        combined_recs = []
        seen_products = set()
        
        # Add CF recommendations first
        for product_id in cf_product_ids:
            if product_id not in seen_products and len(combined_recs) < n_recommendations*2:
                combined_recs.append(product_id)
                seen_products.add(product_id)
        
        # Fill with CB recommendations
        for product_id in cb_product_ids:
            if product_id not in seen_products and len(combined_recs) < n_recommendations*2:
                combined_recs.append(product_id)
                seen_products.add(product_id)
        
        # Apply category filters
        filtered_recs = self._apply_category_filters(
            combined_recs, primary_category, secondary_category, tertiary_category
        )
        
        # If no recommendations after filtering and categories are specified,
        # fall back to popular products in that category
        if not filtered_recs and any([primary_category, secondary_category, tertiary_category]):
            print(f"No recommendations found for user {user_id} in specified categories. Using popular products in category.")
            filtered_recs = self._get_popular_products_by_category(
                primary_category, secondary_category, tertiary_category, n_recommendations
            )
        
        return self._format_recommendations(filtered_recs[:n_recommendations])
    
    def _get_content_based_recommendations(self, user_rated_products, n_recommendations):
        """Get content-based recommendations using product similarity"""
        if self.content_similarity_matrix is None or not user_rated_products or not self.content_product_to_idx:
            return []
        
        try:
            # Get indices of user's rated products using content-specific mapping
            rated_indices = []
            
            for product_id in user_rated_products:
                if product_id in self.content_product_to_idx:
                    content_idx = self.content_product_to_idx[product_id]
                    rated_indices.append(content_idx)
            
            if not rated_indices:
                return []
            
            # Validate indices are within bounds
            max_valid_idx = self.content_similarity_matrix.shape[0] - 1
            valid_rated_indices = [idx for idx in rated_indices if idx <= max_valid_idx]
            
            if not valid_rated_indices:
                print(f"Warning: No valid content indices found for user's rated products")
                return []
            
            # Calculate average similarity scores
            similarity_scores = self.content_similarity_matrix[valid_rated_indices].mean(axis=0)
            
            # Get top similar products (excluding already rated ones)
            top_indices = np.argsort(similarity_scores)[::-1]
            
            recommendations = []
            for idx in top_indices:
                # Use content-specific mapping to get product ID
                if idx in self.content_idx_to_product:
                    product_id = self.content_idx_to_product[idx]
                    if product_id and product_id not in user_rated_products:
                        recommendations.append(product_id)
                        if len(recommendations) >= n_recommendations:
                            break
            
            return recommendations
            
        except Exception as e:
            print(f"Warning: Error in content-based recommendations: {e}")
            # Additional debug info
            if self.content_similarity_matrix is not None:
                print(f"Content matrix shape: {self.content_similarity_matrix.shape}")
                print(f"Content mappings: {len(self.content_product_to_idx)} products")
                if rated_indices:
                    print(f"Rated indices: {rated_indices[:5]}... (showing first 5)")
                    print(f"Max rated index: {max(rated_indices) if rated_indices else 'None'}")
            return []
    
    def _get_cold_start_recommendations(self, user_profile, primary_category,
                                      secondary_category, tertiary_category, n_recommendations):
        """Get recommendations for new users (cold start)"""
        if user_profile and self.user_profiles:
            # Try to find similar user profile
            profile_features = ['skin_tone', 'eye_color', 'skin_type', 'hair_color']
            profile_key = tuple(user_profile.get(k, None) for k in profile_features)
            
            if profile_key in self.user_profiles:
                product_recs = self.user_profiles[profile_key]
            else:
                # Find most similar profile
                best_match = None
                best_score = 0
                for stored_profile, products in self.user_profiles.items():
                    score = sum(1 for i, val in enumerate(profile_key) 
                               if i < len(stored_profile) and val == stored_profile[i])
                    if score > best_score:
                        best_score = score
                        best_match = products
                
                product_recs = best_match if best_match else self._get_popular_products(n_recommendations*2)
        else:
            product_recs = self._get_popular_products(n_recommendations*2)
        
        # Apply category filters
        filtered_recs = self._apply_category_filters(
            product_recs, primary_category, secondary_category, tertiary_category
        )
        
        return self._format_recommendations(filtered_recs[:n_recommendations])
    
    def _get_popular_products(self, n_products):
        """Get popular products based on ratings and review count"""
        try:
            popular = (self.reviews_df.groupby('product_id')
                      .agg({'rating': ['mean', 'count']})
                      .reset_index())
            popular.columns = ['product_id', 'avg_rating', 'review_count']
            popular = popular[popular['review_count'] >= 3]
            popular['popularity_score'] = popular['avg_rating'] * np.log1p(popular['review_count'])
            popular = popular.sort_values('popularity_score', ascending=False)
            
            return popular['product_id'].head(n_products).tolist()
        except Exception as e:
            print(f"Error getting popular products: {e}")
            return list(self.products_df['product_id'].head(n_products))
    
    def _get_popular_products_by_category(self, primary_category, secondary_category, tertiary_category, n_products):
        """Get popular products filtered by category with multiple fallback strategies"""
        try:
            # Strategy 1: Try exact category match with rated products
            result = self._get_category_products_with_ratings(primary_category, secondary_category, tertiary_category, n_products)
            if result:
                print(f"Found {len(result)} rated products in specified categories")
                return result
            
            # Strategy 2: Try relaxed category matching (remove one filter at a time)
            fallback_strategies = [
                # Remove tertiary, keep primary + secondary
                (primary_category, secondary_category, None),
                # Remove secondary, keep primary + tertiary  
                (primary_category, None, tertiary_category),
                # Keep only primary category
                (primary_category, None, None),
                # Keep only secondary category
                (None, secondary_category, None),
                # Keep only tertiary category
                (None, None, tertiary_category)
            ]
            
            for strategy_primary, strategy_secondary, strategy_tertiary in fallback_strategies:
                if any([strategy_primary, strategy_secondary, strategy_tertiary]):  # At least one filter
                    result = self._get_category_products_with_ratings(strategy_primary, strategy_secondary, strategy_tertiary, n_products)
                    if result:
                        print(f"Fallback: Found {len(result)} products using relaxed category filters")
                        return result
            
            # Strategy 3: Use products from category even without ratings (content-based)
            result = self._get_unrated_category_products(primary_category, secondary_category, tertiary_category, n_products)
            if result:
                print(f"Fallback: Found {len(result)} unrated products in category using content similarity")
                return result
            
            # Strategy 4: Find similar categories and use their products
            result = self._get_similar_category_products(primary_category, secondary_category, tertiary_category, n_products)
            if result:
                print(f"Fallback: Found {len(result)} products from similar categories")
                return result
            
            # Strategy 5: Ultimate fallback - overall popular products
            print("Using overall popular products as final fallback")
            return self._get_popular_products(n_products)
            
        except Exception as e:
            print(f"Error in category fallback system: {e}")
            return self._get_popular_products(n_products)
    
    def _get_category_products_with_ratings(self, primary_category, secondary_category, tertiary_category, n_products):
        """Get products from category that have ratings"""
        try:
            # Filter products by category
            filtered_products_df = self.products_df.copy()
            
            if primary_category:
                filtered_products_df = filtered_products_df[
                    filtered_products_df['primary_category'] == primary_category
                ]
            if secondary_category:
                filtered_products_df = filtered_products_df[
                    filtered_products_df['secondary_category'] == secondary_category
                ]
            if tertiary_category:
                filtered_products_df = filtered_products_df[
                    filtered_products_df['tertiary_category'] == tertiary_category
                ]
            
            if filtered_products_df.empty:
                return []
            
            # Get product IDs that exist in both filtered products and our reviews
            category_product_ids = set(filtered_products_df['product_id'].unique())
            reviewed_product_ids = set(self.product_to_idx.keys())
            valid_product_ids = category_product_ids.intersection(reviewed_product_ids)
            
            if not valid_product_ids:
                return []
            
            # Use ALS model if available
            if self.user_item_model is not None:
                try:
                    item_factors = self.user_item_model.item_factors
                    popularity_scores = {}
                    
                    for product_id in valid_product_ids:
                        if product_id in self.product_to_idx:
                            item_idx = self.product_to_idx[product_id]
                            
                            # Get interaction count and average rating
                            item_interactions = self.user_item_matrix[:, item_idx].sum()
                            item_ratings = self.reviews_df[self.reviews_df['product_id'] == product_id]['rating']
                            avg_rating = item_ratings.mean() if len(item_ratings) > 0 else 3.0
                            
                            # Combine ALS factors with interactions and ratings
                            if item_idx < len(item_factors):
                                factor_magnitude = np.linalg.norm(item_factors[item_idx])
                                popularity_score = (factor_magnitude * 0.3 + 
                                                  float(item_interactions) * 0.4 + 
                                                  avg_rating * 0.3)
                            else:
                                popularity_score = float(item_interactions) * 0.7 + avg_rating * 0.3
                            
                            popularity_scores[product_id] = popularity_score
                    
                    sorted_products = sorted(popularity_scores.items(), key=lambda x: x[1], reverse=True)
                    return [product_id for product_id, score in sorted_products[:n_products]]
                    
                except Exception:
                    pass  # Fall through to basic calculation
            
            # Basic popularity calculation
            category_reviews = self.reviews_df[self.reviews_df['product_id'].isin(valid_product_ids)]
            
            if category_reviews.empty:
                return []
            
            popular = (category_reviews.groupby('product_id')
                      .agg({'rating': ['mean', 'count']})
                      .reset_index())
            popular.columns = ['product_id', 'avg_rating', 'review_count']
            popular = popular[popular['review_count'] >= 1]  # Very low threshold
            popular['popularity_score'] = popular['avg_rating'] * np.log1p(popular['review_count'])
            popular = popular.sort_values('popularity_score', ascending=False)
            
            return popular['product_id'].head(n_products).tolist()
            
        except Exception as e:
            return []
    
    def _get_unrated_category_products(self, primary_category, secondary_category, tertiary_category, n_products):
        """Get products from category even if they have no ratings, using content similarity"""
        try:
            # Filter products by category
            filtered_products_df = self.products_df.copy()
            
            if primary_category:
                filtered_products_df = filtered_products_df[
                    filtered_products_df['primary_category'] == primary_category
                ]
            if secondary_category:
                filtered_products_df = filtered_products_df[
                    filtered_products_df['secondary_category'] == secondary_category
                ]
            if tertiary_category:
                filtered_products_df = filtered_products_df[
                    filtered_products_df['tertiary_category'] == tertiary_category
                ]
            
            if filtered_products_df.empty:
                return []
            
            # Sort by product attributes as proxy for quality/popularity
            # Prioritize products with complete information
            filtered_products_df = filtered_products_df.copy()
            
            # Create a completeness score
            filtered_products_df['completeness_score'] = 0
            
            # Add points for having key information
            if 'price_usd' in filtered_products_df.columns:
                filtered_products_df['completeness_score'] += filtered_products_df['price_usd'].notna().astype(int) * 2
            if 'brand_name' in filtered_products_df.columns:
                filtered_products_df['completeness_score'] += filtered_products_df['brand_name'].notna().astype(int) * 1
            if 'ingredients' in filtered_products_df.columns:
                filtered_products_df['completeness_score'] += (filtered_products_df['ingredients'].notna() & 
                                                             (filtered_products_df['ingredients'] != '')).astype(int) * 3
            if 'highlights' in filtered_products_df.columns:
                filtered_products_df['completeness_score'] += (filtered_products_df['highlights'].notna() & 
                                                             (filtered_products_df['highlights'] != '')).astype(int) * 2
            
            # Sort by completeness score and take top products
            filtered_products_df = filtered_products_df.sort_values('completeness_score', ascending=False)
            
            return filtered_products_df['product_id'].head(n_products).tolist()
            
        except Exception as e:
            return []
    
    def _get_similar_category_products(self, primary_category, secondary_category, tertiary_category, n_products):
        """Find products from similar categories that have ratings"""
        try:
            # Define category similarity mappings
            category_similarities = {
                'Makeup': ['Beauty', 'Cosmetics', 'Color', 'Face'],
                'Skincare': ['Beauty', 'Face', 'Treatment', 'Care'],
                'Fragrance': ['Perfume', 'Scent', 'Beauty'],
                'Hair': ['Haircare', 'Beauty', 'Care'],
                'Tools': ['Accessories', 'Beauty', 'Application'],
                'Bath': ['Body', 'Care', 'Beauty'],
                'Wellness': ['Health', 'Beauty', 'Care']
            }
            
            similar_categories = []
            
            # Find similar primary categories
            if primary_category:
                if primary_category in category_similarities:
                    similar_categories.extend(category_similarities[primary_category])
                
                # Also try partial matches
                for cat, similar in category_similarities.items():
                    if primary_category.lower() in cat.lower() or cat.lower() in primary_category.lower():
                        similar_categories.extend(similar)
            
            # Remove duplicates and the original category
            similar_categories = list(set(similar_categories))
            if primary_category in similar_categories:
                similar_categories.remove(primary_category)
            
            # Try each similar category
            for similar_cat in similar_categories:
                result = self._get_category_products_with_ratings(similar_cat, None, None, n_products)
                if result:
                    return result
            
            return []
            
        except Exception as e:
            return []
    
    def _get_popular_products_formatted(self, n_products):
        """Get popular products formatted as recommendations"""
        product_ids = self._get_popular_products(n_products)
        return self._format_recommendations(product_ids)
    
    def _apply_category_filters(self, product_ids, primary_category, 
                              secondary_category, tertiary_category):
        """Apply category filters to recommendations"""
        if not any([primary_category, secondary_category, tertiary_category]):
            return product_ids
        
        filtered_products = []
        for product_id in product_ids:
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            
            if product_info.empty:
                continue
                
            product_info = product_info.iloc[0]
            
            # Check category filters
            if primary_category and product_info.get('primary_category') != primary_category:
                continue
            if secondary_category and product_info.get('secondary_category') != secondary_category:
                continue
            if tertiary_category and product_info.get('tertiary_category') != tertiary_category:
                continue
                
            filtered_products.append(product_id)
        
        return filtered_products
    
    def _format_recommendations(self, product_ids):
        """Format recommendations with product details"""
        recommendations = []
        for product_id in product_ids:
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            if not product_info.empty:
                product_info = product_info.iloc[0]
                recommendations.append({
                    'product_id': product_id,
                    'product_name': product_info.get('product_name', 'Unknown'),
                    'brand_name': product_info.get('brand_name', 'Unknown'),
                    'price_usd': product_info.get('price_usd', 0),
                    'rating': product_info.get('rating', 0),
                    'primary_category': product_info.get('primary_category', 'Unknown'),
                    'secondary_category': product_info.get('secondary_category', 'Unknown'),
                    'tertiary_category': product_info.get('tertiary_category', 'Unknown')
                })
        
        return recommendations
    
    def get_categories(self):
        """Get available categories for filtering"""
        categories = {}
        category_columns = ['primary_category', 'secondary_category', 'tertiary_category']
        
        for col in category_columns:
            if col in self.products_df.columns:
                categories[col] = sorted(self.products_df[col].dropna().unique())
            else:
                categories[col] = []
                
        return categories
    
    def get_user_stats(self):
        """Get statistics about users and interactions"""
        stats = {
            'total_users': len(self.user_to_idx),
            'total_products': len(self.product_to_idx),
            'total_interactions': self.user_item_matrix.nnz if self.user_item_matrix is not None else 0,
            'sparsity': (1 - self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])) * 100 if self.user_item_matrix is not None else 0,
            'avg_ratings_per_user': self.reviews_df.groupby('author_id').size().mean(),
            'avg_ratings_per_product': self.reviews_df.groupby('product_id').size().mean()
        }
        return stats

