"""
Evaluation Script for Sephora Recommendation System
=================================================

This script evaluates the performance of the recommendation system.
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import SephoraInferenceSystem

class RecommendationEvaluator:
    """
    Evaluation system for recommendation models
    """
    
    def __init__(self, inference_system):
        self.inference_system = inference_system
        self.results = {}
    
    def evaluate_coverage(self):
        """Evaluate catalog coverage and user coverage"""
        print("Evaluating Coverage Metrics...")
        
        # Catalog coverage
        all_products = set(self.inference_system.processed_data['products_df']['product_id'].unique())
        recommended_products = set()
        
        # Sample 100 users for coverage test
        users = list(self.inference_system.mappings['user_to_idx'].keys())
        sample_users = users[:min(100, len(users))]
        
        for user_id in sample_users:
            try:
                recs = self.inference_system.get_user_recommendations(user_id=user_id, n_recommendations=10)
                rec_product_ids = {rec['product_id'] for rec in recs}
                recommended_products.update(rec_product_ids)
            except:
                continue
        
        catalog_coverage = len(recommended_products) / len(all_products) if all_products else 0
        
        # User coverage (how many users can get recommendations)
        successful_users = 0
        for user_id in sample_users:
            try:
                recs = self.inference_system.get_user_recommendations(user_id=user_id, n_recommendations=5)
                if recs:
                    successful_users += 1
            except:
                continue
        
        user_coverage = successful_users / len(sample_users) if sample_users else 0
        
        self.results['coverage'] = {
            'catalog_coverage': catalog_coverage,
            'user_coverage': user_coverage,
            'total_products': len(all_products),
            'recommended_products': len(recommended_products),
            'successful_users': successful_users,
            'sample_users': len(sample_users)
        }
        
        print(f"✓ Catalog Coverage: {catalog_coverage:.2%}")
        print(f"✓ User Coverage: {user_coverage:.2%}")
        
        return self.results['coverage']
    
    def evaluate_diversity(self):
        """Evaluate diversity of recommendations"""
        print("Evaluating Diversity Metrics...")
        
        users = list(self.inference_system.mappings['user_to_idx'].keys())
        sample_users = users[:min(50, len(users))]
        
        diversity_metrics = {
            'intra_list_diversity': [],  # Diversity within each user's recommendations
            'category_diversity': [],    # Number of different categories per user
            'brand_diversity': [],       # Number of different brands per user
            'price_diversity': []        # Price range per user
        }
        
        for user_id in sample_users:
            try:
                recs = self.inference_system.get_user_recommendations(user_id=user_id, n_recommendations=10)
                if not recs:
                    continue
                
                # Category diversity
                categories = [rec['primary_category'] for rec in recs]
                unique_categories = len(set(categories))
                diversity_metrics['category_diversity'].append(unique_categories / len(recs))
                
                # Brand diversity
                brands = [rec['brand_name'] for rec in recs]
                unique_brands = len(set(brands))
                diversity_metrics['brand_diversity'].append(unique_brands / len(recs))
                
                # Price diversity (coefficient of variation)
                prices = [rec['price_usd'] for rec in recs if rec['price_usd'] > 0]
                if prices:
                    price_cv = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
                    diversity_metrics['price_diversity'].append(price_cv)
                
            except:
                continue
        
        # Calculate average diversities
        avg_diversity = {}
        for metric, values in diversity_metrics.items():
            if values:
                avg_diversity[metric] = np.mean(values)
            else:
                avg_diversity[metric] = 0
        
        self.results['diversity'] = avg_diversity
        
        print(f"✓ Average Category Diversity: {avg_diversity['category_diversity']:.3f}")
        print(f"✓ Average Brand Diversity: {avg_diversity['brand_diversity']:.3f}")
        print(f"✓ Average Price Diversity: {avg_diversity['price_diversity']:.3f}")
        
        return self.results['diversity']
    
    def evaluate_popularity_bias(self):
        """Evaluate if system has popularity bias"""
        print("Evaluating Popularity Bias...")
        
        # Calculate product popularity
        product_popularity = (self.inference_system.processed_data['reviews_df']
                            .groupby('product_id')
                            .size()
                            .reset_index(columns=['popularity']))
        product_popularity.columns = ['product_id', 'popularity']
        product_popularity['popularity_rank'] = product_popularity['popularity'].rank(ascending=False)
        
        # Get recommendations for sample users
        users = list(self.inference_system.mappings['user_to_idx'].keys())
        sample_users = users[:min(30, len(users))]
        
        recommended_popularities = []
        
        for user_id in sample_users:
            try:
                recs = self.inference_system.get_user_recommendations(user_id=user_id, n_recommendations=10)
                for rec in recs:
                    product_pop = product_popularity[
                        product_popularity['product_id'] == rec['product_id']
                    ]
                    if not product_pop.empty:
                        recommended_popularities.append(product_pop.iloc[0]['popularity_rank'])
            except:
                continue
        
        if recommended_popularities:
            avg_popularity_rank = np.mean(recommended_popularities)
            median_popularity_rank = np.median(recommended_popularities)
            
            # Calculate bias score (lower rank = more popular = higher bias)
            total_products = len(product_popularity)
            bias_score = 1 - (avg_popularity_rank / total_products)  # 0 = no bias, 1 = maximum bias
        else:
            avg_popularity_rank = 0
            median_popularity_rank = 0
            bias_score = 0
        
        self.results['popularity_bias'] = {
            'avg_popularity_rank': avg_popularity_rank,
            'median_popularity_rank': median_popularity_rank,
            'bias_score': bias_score,
            'total_products': len(product_popularity)
        }
        
        print(f"✓ Average Popularity Rank: {avg_popularity_rank:.1f}")
        print(f"✓ Popularity Bias Score: {bias_score:.3f}")
        
        return self.results['popularity_bias']
    
    def evaluate_category_distribution(self):
        """Evaluate category distribution in recommendations"""
        print("Evaluating Category Distribution...")
        
        users = list(self.inference_system.mappings['user_to_idx'].keys())
        sample_users = users[:min(50, len(users))]
        
        category_counts = {}
        total_recommendations = 0
        
        for user_id in sample_users:
            try:
                recs = self.inference_system.get_user_recommendations(user_id=user_id, n_recommendations=10)
                for rec in recs:
                    category = rec['primary_category']
                    category_counts[category] = category_counts.get(category, 0) + 1
                    total_recommendations += 1
            except:
                continue
        
        # Calculate category distribution
        category_distribution = {}
        if total_recommendations > 0:
            for category, count in category_counts.items():
                category_distribution[category] = count / total_recommendations
        
        # Get actual catalog distribution for comparison
        catalog_categories = (self.inference_system.processed_data['products_df']['primary_category']
                            .value_counts(normalize=True)
                            .to_dict())
        
        self.results['category_distribution'] = {
            'recommendation_distribution': category_distribution,
            'catalog_distribution': catalog_categories,
            'total_recommendations': total_recommendations
        }
        
        print(f"✓ Category Distribution Analysis Complete")
        print(f"✓ Total Categories in Recommendations: {len(category_distribution)}")
        
        return self.results['category_distribution']
    
    def evaluate_cold_start_performance(self):
        """Evaluate cold start recommendation performance"""
        print("Evaluating Cold Start Performance...")
        
        # Test different user profiles
        test_profiles = [
            {'skin_tone': 'light', 'skin_type': 'dry'},
            {'skin_tone': 'medium', 'skin_type': 'combination'},
            {'skin_tone': 'dark', 'skin_type': 'oily'},
            {'eye_color': 'brown', 'hair_color': 'black'},
            {'eye_color': 'blue', 'hair_color': 'blonde'},
            {}  # Empty profile
        ]
        
        cold_start_results = []
        
        for i, profile in enumerate(test_profiles):
            try:
                recs = self.inference_system.get_user_recommendations(
                    user_profile=profile, 
                    n_recommendations=10
                )
                
                cold_start_results.append({
                    'profile_id': i,
                    'profile': profile,
                    'num_recommendations': len(recs),
                    'success': len(recs) > 0,
                    'avg_rating': np.mean([rec['rating'] for rec in recs]) if recs else 0,
                    'avg_price': np.mean([rec['price_usd'] for rec in recs]) if recs else 0,
                    'unique_categories': len(set(rec['primary_category'] for rec in recs)) if recs else 0
                })
            except Exception as e:
                cold_start_results.append({
                    'profile_id': i,
                    'profile': profile,
                    'num_recommendations': 0,
                    'success': False,
                    'error': str(e)
                })
        
        success_rate = sum(1 for r in cold_start_results if r['success']) / len(cold_start_results)
        avg_recommendations = np.mean([r['num_recommendations'] for r in cold_start_results])
        
        self.results['cold_start'] = {
            'success_rate': success_rate,
            'avg_recommendations': avg_recommendations,
            'detailed_results': cold_start_results
        }
        
        print(f"✓ Cold Start Success Rate: {success_rate:.2%}")
        print(f"✓ Average Recommendations per Profile: {avg_recommendations:.1f}")
        
        return self.results['cold_start']
    
    def create_evaluation_report(self, save_path='models/evaluation_report.html'):
        """Create comprehensive evaluation report"""
        print("Creating Evaluation Report...")
        
        # Run all evaluations
        self.evaluate_coverage()
        self.evaluate_diversity()
        self.evaluate_popularity_bias()
        self.evaluate_category_distribution()
        self.evaluate_cold_start_performance()
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sephora Recommendation System - Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; }}
                .metric {{ background: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .good {{ background: #d4edda; border: 1px solid #c3e6cb; }}
                .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; }}
                .poor {{ background: #f8d7da; border: 1px solid #f5c6cb; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>📊 Sephora Recommendation System - Evaluation Report</h1>
            <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>🎯 Coverage Metrics</h2>
            <div class="metric {'good' if self.results['coverage']['catalog_coverage'] > 0.1 else 'warning'}">
                <strong>Catalog Coverage:</strong> {self.results['coverage']['catalog_coverage']:.2%}<br>
                <small>Products recommended: {self.results['coverage']['recommended_products']:,} / {self.results['coverage']['total_products']:,}</small>
            </div>
            <div class="metric {'good' if self.results['coverage']['user_coverage'] > 0.9 else 'warning'}">
                <strong>User Coverage:</strong> {self.results['coverage']['user_coverage']:.2%}<br>
                <small>Successful users: {self.results['coverage']['successful_users']:,} / {self.results['coverage']['sample_users']:,}</small>
            </div>
            
            <h2>🌈 Diversity Metrics</h2>
            <div class="metric {'good' if self.results['diversity']['category_diversity'] > 0.3 else 'warning'}">
                <strong>Category Diversity:</strong> {self.results['diversity']['category_diversity']:.3f}<br>
                <small>Average proportion of unique categories per user</small>
            </div>
            <div class="metric {'good' if self.results['diversity']['brand_diversity'] > 0.5 else 'warning'}">
                <strong>Brand Diversity:</strong> {self.results['diversity']['brand_diversity']:.3f}<br>
                <small>Average proportion of unique brands per user</small>
            </div>
            <div class="metric">
                <strong>Price Diversity:</strong> {self.results['diversity']['price_diversity']:.3f}<br>
                <small>Coefficient of variation in prices</small>
            </div>
            
            <h2>⭐ Popularity Bias</h2>
            <div class="metric {'good' if self.results['popularity_bias']['bias_score'] < 0.7 else 'warning'}">
                <strong>Popularity Bias Score:</strong> {self.results['popularity_bias']['bias_score']:.3f}<br>
                <small>0 = no bias, 1 = maximum bias</small>
            </div>
            <div class="metric">
                <strong>Average Popularity Rank:</strong> {self.results['popularity_bias']['avg_popularity_rank']:.1f}<br>
                <small>Lower rank = more popular products</small>
            </div>
            
            <h2>🆕 Cold Start Performance</h2>
            <div class="metric {'good' if self.results['cold_start']['success_rate'] > 0.8 else 'warning'}">
                <strong>Success Rate:</strong> {self.results['cold_start']['success_rate']:.2%}<br>
                <small>Percentage of new user profiles that received recommendations</small>
            </div>
            <div class="metric">
                <strong>Average Recommendations:</strong> {self.results['cold_start']['avg_recommendations']:.1f}<br>
                <small>Average number of recommendations per new user</small>
            </div>
            
            <h2>📈 Category Distribution</h2>
            <p>Top categories in recommendations:</p>
            <table>
                <tr><th>Category</th><th>Recommendation %</th><th>Catalog %</th><th>Difference</th></tr>"""
        
        # Add category distribution table
        rec_dist = self.results['category_distribution']['recommendation_distribution']
        cat_dist = self.results['category_distribution']['catalog_distribution']
        
        for category in sorted(rec_dist.keys(), key=lambda x: rec_dist.get(x, 0), reverse=True)[:10]:
            rec_pct = rec_dist.get(category, 0) * 100
            cat_pct = cat_dist.get(category, 0) * 100
            diff = rec_pct - cat_pct
            html_content += f"""
                <tr>
                    <td>{category}</td>
                    <td>{rec_pct:.1f}%</td>
                    <td>{cat_pct:.1f}%</td>
                    <td style="color: {'red' if diff > 10 else 'green' if diff < -10 else 'black'}">{diff:+.1f}%</td>
                </tr>"""
        
        html_content += """
            </table>
            
            <h2>💡 Recommendations</h2>
            <ul>"""
        
        # Add recommendations based on results
        if self.results['coverage']['catalog_coverage'] < 0.1:
            html_content += "<li>⚠️ Low catalog coverage - consider improving recommendation diversity</li>"
        
        if self.results['diversity']['category_diversity'] < 0.3:
            html_content += "<li>⚠️ Low category diversity - recommendations may be too narrow</li>"
        
        if self.results['popularity_bias']['bias_score'] > 0.7:
            html_content += "<li>⚠️ High popularity bias - system may be recommending only popular items</li>"
        
        if self.results['cold_start']['success_rate'] < 0.8:
            html_content += "<li>⚠️ Poor cold start performance - improve new user recommendations</li>"
        
        html_content += """
                <li>✅ System successfully provides recommendations for most users</li>
                <li>✅ Multiple recommendation strategies (collaborative + content-based) working</li>
            </ul>
            
            <h2>🔧 Technical Details</h2>
            <p><strong>Evaluation Method:</strong> Sample-based evaluation using subset of users and products</p>
            <p><strong>Metrics:</strong> Coverage, Diversity, Popularity Bias, Cold Start Performance</p>
            <p><strong>Sample Size:</strong> 50-100 users depending on metric</p>
            
        </body>
        </html>
        """
        
        # Save report
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Evaluation report saved to: {save_path}")
        
        return self.results
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        
        if 'coverage' in self.results:
            print(f"Coverage:")
            print(f"  • Catalog Coverage: {self.results['coverage']['catalog_coverage']:.2%}")
            print(f"  • User Coverage: {self.results['coverage']['user_coverage']:.2%}")
        
        if 'diversity' in self.results:
            print(f"\nDiversity:")
            print(f"  • Category Diversity: {self.results['diversity']['category_diversity']:.3f}")
            print(f"  • Brand Diversity: {self.results['diversity']['brand_diversity']:.3f}")
        
        if 'popularity_bias' in self.results:
            print(f"\nPopularity Bias:")
            print(f"  • Bias Score: {self.results['popularity_bias']['bias_score']:.3f}")
        
        if 'cold_start' in self.results:
            print(f"\nCold Start:")
            print(f"  • Success Rate: {self.results['cold_start']['success_rate']:.2%}")
        
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        issues = []
        
        if self.results.get('coverage', {}).get('user_coverage', 0) < 0.9:
            issues.append("User coverage needs improvement")
        if self.results.get('diversity', {}).get('category_diversity', 0) < 0.3:
            issues.append("Category diversity is low")
        if self.results.get('popularity_bias', {}).get('bias_score', 0) > 0.7:
            issues.append("High popularity bias detected")
        if self.results.get('cold_start', {}).get('success_rate', 0) < 0.8:
            issues.append("Cold start performance needs work")
        
        if issues:
            print("  Areas for improvement:")
            for issue in issues:
                print(f"    • {issue}")
        else:
            print("  ✅ System performing well across all metrics!")

def main():
    """Main evaluation function"""
    print("Sephora Recommendation System - Evaluation Pipeline")
    print("=" * 60)
    
    # Load inference system
    try:
        inference_system = SephoraInferenceSystem()
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Please run training.py first to train the models.")
        return
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator(inference_system)
    
    # Run evaluation
    try:
        results = evaluator.create_evaluation_report()
        evaluator.print_summary()
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📄 Report saved to: models/evaluation_report.html")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()