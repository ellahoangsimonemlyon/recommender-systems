"""
Inference Pipeline for Sephora Recommendation System
===================================================

This script loads trained models and provides inference capabilities.
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SephoraInferenceSystem:
    """
    Inference system that loads pre-trained models and provides recommendations
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the inference system
        
        Args:
            model_path: Path to specific model timestamp, if None uses latest
        """
        self.cf_model = None
        self.cb_model = None
        self.mappings = None
        self.processed_data = None
        self.metadata = None
        
        self.load_models(model_path)
    
    def load_models(self, model_path=None):
        """Load all trained models"""
        try:
            if model_path is None:
                # Load latest models
                with open('../models/latest_models.pkl', 'rb') as f:
                    latest_paths = pickle.load(f)
                
                print(f"Loading latest models (timestamp: {latest_paths['timestamp']})")
                
                # Load collaborative filtering model
                if latest_paths['collaborative_filtering']:
                    with open(latest_paths['collaborative_filtering'], 'rb') as f:
                        self.cf_model = pickle.load(f)
                    print("✓ Collaborative filtering model loaded")
                else:
                    print("⚠ No collaborative filtering model available")
                
                # Load content-based model
                with open(latest_paths['content_based'], 'rb') as f:
                    self.cb_model = pickle.load(f)
                print("✓ Content-based model loaded")
                
                # Load mappings
                with open(latest_paths['mappings'], 'rb') as f:
                    self.mappings = pickle.load(f)
                print("✓ Mappings loaded")
                
                # Load processed data
                with open(latest_paths['processed_data'], 'rb') as f:
                    self.processed_data = pickle.load(f)
                print("✓ Processed data loaded")
                
                # Load metadata
                with open(latest_paths['metadata'], 'rb') as f:
                    self.metadata = pickle.load(f)
                print("✓ Metadata loaded")
                
            else:
                # Load specific model timestamp
                print(f"Loading models from timestamp: {model_path}")
                # Implementation for specific timestamp loading
                pass
            
            print("🎉 All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
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
            if user_id is not None and user_id in self.mappings['user_to_idx']:
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
            print(f"Error in get_user_recommendations: {e}")
            return self._get_popular_products_formatted(n_recommendations)
    
    def _get_existing_user_recommendations(self, user_id, primary_category, 
                                        secondary_category, tertiary_category, n_recommendations):
        """Get recommendations for existing user using collaborative filtering + content-based"""
        user_idx = self.mappings['user_to_idx'][user_id]
        
        # Get user's rated products for content-based filtering
        user_rated_products = set(self.processed_data['reviews_df'][
            self.processed_data['reviews_df']['author_id'] == user_id
        ]['product_id'].values)
        
        recommendations = []
        
        # Get collaborative filtering recommendations
        if self.cf_model is not None:
            try:
                cf_recs, cf_scores = self.cf_model.recommend(
                    user_idx, self.mappings['user_item_matrix'][user_idx], 
                    N=n_recommendations*2, filter_already_liked_items=True
                )
                cf_recs = cf_recs.tolist() if hasattr(cf_recs, 'tolist') else list(cf_recs)
                
                # Convert indices to product IDs
                cf_product_ids = [self.mappings['idx_to_product'][idx] for idx in cf_recs 
                                if idx in self.mappings['idx_to_product']]
                
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
            filtered_recs = self._get_popular_products_by_category(
                primary_category, secondary_category, tertiary_category, n_recommendations
            )
        
        return self._format_recommendations(filtered_recs[:n_recommendations])
    
    def _get_content_based_recommendations(self, user_rated_products, n_recommendations):
        """Get content-based recommendations using product similarity"""
        if (self.cb_model['content_similarity_matrix'] is None or 
            not user_rated_products or 
            not self.cb_model['content_product_to_idx']):
            return []
        
        try:
            # Get indices of user's rated products using content-specific mapping
            rated_indices = []
            
            for product_id in user_rated_products:
                if product_id in self.cb_model['content_product_to_idx']:
                    content_idx = self.cb_model['content_product_to_idx'][product_id]
                    rated_indices.append(content_idx)
            
            if not rated_indices:
                return []
            
            # Validate indices are within bounds
            max_valid_idx = self.cb_model['content_similarity_matrix'].shape[0] - 1
            valid_rated_indices = [idx for idx in rated_indices if idx <= max_valid_idx]
            
            if not valid_rated_indices:
                return []
            
            # Calculate average similarity scores
            similarity_scores = self.cb_model['content_similarity_matrix'][valid_rated_indices].mean(axis=0)
            
            # Get top similar products (excluding already rated ones)
            top_indices = np.argsort(similarity_scores)[::-1]
            
            recommendations = []
            for idx in top_indices:
                # Use content-specific mapping to get product ID
                if idx in self.cb_model['content_idx_to_product']:
                    product_id = self.cb_model['content_idx_to_product'][idx]
                    if product_id and product_id not in user_rated_products:
                        recommendations.append(product_id)
                        if len(recommendations) >= n_recommendations:
                            break
            
            return recommendations
            
        except Exception as e:
            print(f"Warning: Error in content-based recommendations: {e}")
            return []
    
    def _get_cold_start_recommendations(self, user_profile, primary_category,
                                      secondary_category, tertiary_category, n_recommendations):
        """Get recommendations for new users (cold start)"""
        if user_profile and self.mappings['user_profiles']:
            # Try to find similar user profile
            profile_features = ['skin_tone', 'eye_color', 'skin_type', 'hair_color']
            profile_key = tuple(user_profile.get(k, None) for k in profile_features)
            
            if profile_key in self.mappings['user_profiles']:
                product_recs = self.mappings['user_profiles'][profile_key]
            else:
                # Find most similar profile
                best_match = None
                best_score = 0
                for stored_profile, products in self.mappings['user_profiles'].items():
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
            popular = (self.processed_data['reviews_df'].groupby('product_id')
                      .agg({'rating': ['mean', 'count']})
                      .reset_index())
            popular.columns = ['product_id', 'avg_rating', 'review_count']
            popular = popular[popular['review_count'] >= 3]
            popular['popularity_score'] = popular['avg_rating'] * np.log1p(popular['review_count'])
            popular = popular.sort_values('popularity_score', ascending=False)
            
            return popular['product_id'].head(n_products).tolist()
        except Exception as e:
            print(f"Error getting popular products: {e}")
            return list(self.processed_data['products_df']['product_id'].head(n_products))
    
    def _get_popular_products_by_category(self, primary_category, secondary_category, tertiary_category, n_products):
        """Get popular products filtered by category"""
        try:
            # Filter products by category
            filtered_products_df = self.processed_data['products_df'].copy()
            
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
                return self._get_popular_products(n_products)
            
            # Get product IDs that exist in both filtered products and our reviews
            category_product_ids = set(filtered_products_df['product_id'].unique())
            reviewed_product_ids = set(self.mappings['product_to_idx'].keys())
            valid_product_ids = category_product_ids.intersection(reviewed_product_ids)
            
            if not valid_product_ids:
                return list(filtered_products_df['product_id'].head(n_products))
            
            # Basic popularity calculation
            category_reviews = self.processed_data['reviews_df'][
                self.processed_data['reviews_df']['product_id'].isin(valid_product_ids)
            ]
            
            if category_reviews.empty:
                return list(filtered_products_df['product_id'].head(n_products))
            
            popular = (category_reviews.groupby('product_id')
                      .agg({'rating': ['mean', 'count']})
                      .reset_index())
            popular.columns = ['product_id', 'avg_rating', 'review_count']
            popular = popular[popular['review_count'] >= 1]
            popular['popularity_score'] = popular['avg_rating'] * np.log1p(popular['review_count'])
            popular = popular.sort_values('popularity_score', ascending=False)
            
            return popular['product_id'].head(n_products).tolist()
            
        except Exception as e:
            return self._get_popular_products(n_products)
    
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
            product_info = self.processed_data['products_df'][
                self.processed_data['products_df']['product_id'] == product_id
            ]
            
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
            product_info = self.processed_data['products_df'][
                self.processed_data['products_df']['product_id'] == product_id
            ]
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
                    'tertiary_category': product_info.get('tertiary_category', 'Unknown'),
                    'ingredients': product_info.get('ingredients', ''),
                    'highlights': product_info.get('highlights', '')
                })
        
        return recommendations
    
    def get_categories(self):
        """Get available categories for filtering"""
        if self.metadata and 'categories' in self.metadata:
            return self.metadata['categories']
        
        # Fallback: calculate from data
        categories = {}
        category_columns = ['primary_category', 'secondary_category', 'tertiary_category']
        
        for col in category_columns:
            if col in self.processed_data['products_df'].columns:
                categories[col] = sorted(self.processed_data['products_df'][col].dropna().unique())
            else:
                categories[col] = []
                
        return categories
    
    def get_all_users(self):
        """Get list of all user IDs"""
        return sorted(list(self.mappings['user_to_idx'].keys()))
    
    def get_user_stats(self):
        """Get system statistics"""
        if self.metadata and 'stats' in self.metadata:
            return self.metadata['stats']
        
        # Fallback: calculate from data
        stats = {
            'total_users': len(self.mappings['user_to_idx']),
            'total_products': len(self.mappings['product_to_idx']),
            'total_interactions': self.mappings['user_item_matrix'].nnz,
            'model_timestamp': self.metadata.get('timestamp', 'Unknown') if self.metadata else 'Unknown'
        }
        return stats

def test_inference_system():
    """Test the inference system"""
    print("Testing Inference System...")
    
    # Initialize system
    inference_system = SephoraInferenceSystem()
    
    # Test getting categories
    categories = inference_system.get_categories()
    print(f"Available categories: {len(categories)}")
    
    # Test getting users
    users = inference_system.get_all_users()
    print(f"Available users: {len(users)}")
    
    # Test existing user recommendation
    if users:
        test_user = users[0]
        recs = inference_system.get_user_recommendations(user_id=test_user, n_recommendations=5)
        print(f"Recommendations for existing user {test_user}: {len(recs)}")
    
    # Test cold start recommendation
    test_profile = {
        'skin_tone': 'light',
        'skin_type': 'combination',
        'eye_color': 'brown',
        'hair_color': 'blonde'
    }
    recs = inference_system.get_user_recommendations(user_profile=test_profile, n_recommendations=5)
    print(f"Cold start recommendations: {len(recs)}")
    
    print("✅ Inference system test completed!")
    
    return inference_system

if __name__ == "__main__":
    test_inference_system()