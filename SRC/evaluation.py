"""
Evaluation Script
------------------------------------------------------

This script evaluates the performance of the recommendation system.
"""
import os
import sys
import warnings
from pathlib import Path
import numpy as np
import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import SephoraInferenceSystem

# Import train_test_split for evaluation
try:
    from implicit.evaluation import train_test_split
except ImportError:
    # Fallback if not available
    def train_test_split(matrix, train_percentage=0.8):
        return matrix, None


class RecommendationEvaluator:
    """
    Evaluation system for recommendation models
    """

    def __init__(self, inference_system):
        self.inference_system = inference_system
        self.results = {}

        # Load the original trained model for evaluation
        self.trained_model = None
        self.train_matrix = None
        self.test_matrix = None
        self._load_trained_model()

    def _load_trained_model(self):
        """Load the trained collaborative filtering model for evaluation"""
        try:
            if self.inference_system.cf_model is not None:
                self.trained_model = self.inference_system.cf_model
                print("Loaded trained collaborative filtering model for evaluation")

                # Create train/test split for evaluation
                user_item_matrix = self.inference_system.mappings["user_item_matrix"]
                if user_item_matrix is not None:
                    try:
                        self.train_matrix, self.test_matrix = train_test_split(
                            user_item_matrix, train_percentage=0.8
                        )
                        print(f"Created train/test split for evaluation")
                        print(
                            f"   Train matrix: {self.train_matrix.shape}, nnz: {self.train_matrix.nnz}"
                        )
                        if self.test_matrix is not None:
                            print(
                                f"   Test matrix: {self.test_matrix.shape}, nnz: {self.test_matrix.nnz}"
                            )
                    except Exception as e:
                        print(f"Could not create train/test split: {e}")
                        self.train_matrix = user_item_matrix
                        self.test_matrix = None
            else:
                print(
                    "No collaborative filtering model available for advanced evaluation"
                )
        except Exception as e:
            print(f"Error loading trained model: {e}")

    def evaluate_rmse(self):
        """Evaluate RMSE for rating prediction on existing users"""
        print("Evaluating RMSE...")

        if self.trained_model is None or self.test_matrix is None:
            print("Cannot evaluate RMSE: No trained model or test data available")
            self.results["rmse"] = {
                "rmse": None,
                "mae": None,
                "reason": "No test data available",
            }
            return self.results["rmse"]

        try:
            # Get users with test data
            users_with_test_data = np.where(np.diff(self.test_matrix.indptr) > 0)[0]
            sample_users = users_with_test_data[
                : min(500, len(users_with_test_data))
            ]  # Sample for efficiency

            predicted_ratings = []
            actual_ratings = []

            print(f"Evaluating RMSE on {len(sample_users)} users...")

            for user_idx in tqdm.tqdm(sample_users, desc="RMSE Evaluation"):
                # Get test items for this user
                test_start = self.test_matrix.indptr[user_idx]
                test_end = self.test_matrix.indptr[user_idx + 1]

                if test_start == test_end:
                    continue  # No test items for this user

                test_items = self.test_matrix.indices[test_start:test_end]
                test_ratings = self.test_matrix.data[test_start:test_end]

                # Get predictions for test items
                user_factors = self.trained_model.user_factors[user_idx]
                item_factors = self.trained_model.item_factors[test_items]

                # Predict ratings (dot product of user and item factors)
                predictions = np.dot(item_factors, user_factors)

                predicted_ratings.extend(predictions)
                actual_ratings.extend(test_ratings)

            if len(predicted_ratings) > 0:
                # Calculate RMSE and MAE
                rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
                mae = mean_absolute_error(actual_ratings, predicted_ratings)

                # Calculate correlation
                correlation, p_value = pearsonr(actual_ratings, predicted_ratings)

                self.results["rmse"] = {
                    "rmse": rmse,
                    "mae": mae,
                    "correlation": correlation,
                    "p_value": p_value,
                    "num_predictions": len(predicted_ratings),
                    "num_users": len(sample_users),
                }

                print(f"RMSE: {rmse:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"Correlation: {correlation:.4f}")
                print(
                    f"Evaluated on {len(predicted_ratings):,} predictions from {len(sample_users)} users"
                )

            else:
                print("No predictions could be made")
                self.results["rmse"] = {
                    "rmse": None,
                    "mae": None,
                    "reason": "No predictions possible",
                }

            return self.results["rmse"]

        except Exception as e:
            print(f"Error in RMSE evaluation: {e}")
            self.results["rmse"] = {"rmse": None, "mae": None, "error": str(e)}
            return self.results["rmse"]

    def ranking_metrics_at_k(self, K=10):
        """
        Calculate ranking metrics (Precision@K, MAP@K, NDCG@K) for the trained model

        Parameters:
            K : int - Number of items to evaluate (default 10)

        Returns:
            dict : Dictionary with precision, MAP, NDCG scores
        """
        print(f"Evaluating Ranking Metrics at K={K}...")

        if (
            self.trained_model is None
            or self.train_matrix is None
            or self.test_matrix is None
        ):
            print(
                "Cannot evaluate ranking metrics: No trained model or test data available"
            )
            return {
                "precision": None,
                "map": None,
                "ndcg": None,
                "reason": "No test data available",
            }

        try:
            # Ensure matrices are in CSR format
            train_user_items = self.train_matrix.tocsr()
            test_user_items = self.test_matrix.tocsr()
            num_users, num_items = test_user_items.shape

            # Initialize metrics
            relevant = 0
            total_precision_div = 0
            total_map = 0
            total_ndcg = 0
            total_users = 0

            # Compute cumulative gain for NDCG normalization
            cg = 1.0 / np.log2(np.arange(2, K + 2))  # Discount factor
            cg_sum = np.cumsum(cg)  # Ideal DCG normalization

            # Get users with at least one item in the test set
            users_with_test_data = np.where(np.diff(test_user_items.indptr) > 0)[0]
            sample_users = users_with_test_data[
                : min(200, len(users_with_test_data))
            ]  # Sample for efficiency

            print(f"Evaluating on {len(sample_users)} users with test data...")

            # Progress bar
            progress = tqdm.tqdm(total=len(sample_users), desc="Ranking Metrics")

            # Process in batches
            batch_size = 100
            start_idx = 0

            while start_idx < len(sample_users):
                batch_users = sample_users[start_idx : start_idx + batch_size]

                # Get recommendations for batch of users
                try:
                    recommended_items, scores = self.trained_model.recommend(
                        batch_users,
                        train_user_items[batch_users],
                        N=K,
                        filter_already_liked_items=True,
                    )
                except Exception as e:
                    print(f"Error getting recommendations for batch: {e}")
                    start_idx += batch_size
                    progress.update(len(batch_users))
                    continue

                start_idx += batch_size

                for user_idx, user_id in enumerate(batch_users):
                    # Get test items for this user
                    test_start = test_user_items.indptr[user_id]
                    test_end = test_user_items.indptr[user_id + 1]
                    test_items = set(test_user_items.indices[test_start:test_end])

                    if not test_items:
                        continue  # Skip users without test data

                    num_relevant = len(test_items)
                    total_precision_div += min(K, num_relevant)

                    # Calculate metrics for this user
                    ap = 0  # Average Precision
                    hit_count = 0
                    dcg = 0  # Discounted Cumulative Gain
                    idcg = cg_sum[min(K, num_relevant) - 1]  # Ideal DCG

                    user_recommendations = (
                        recommended_items[user_idx]
                        if len(recommended_items.shape) > 1
                        else recommended_items
                    )

                    for rank, item in enumerate(user_recommendations[:K]):
                        if item in test_items:
                            relevant += 1
                            hit_count += 1
                            ap += hit_count / (rank + 1)  # Precision at this rank
                            dcg += cg[rank]  # Add discounted gain

                    # Accumulate metrics
                    if num_relevant > 0:
                        total_map += ap / min(K, num_relevant)
                        if idcg > 0:
                            total_ndcg += dcg / idcg

                    total_users += 1

                progress.update(len(batch_users))

            progress.close()

            # Compute final metrics
            precision_at_k = (
                relevant / total_precision_div if total_precision_div > 0 else 0
            )
            map_at_k = total_map / total_users if total_users > 0 else 0
            ndcg_at_k = total_ndcg / total_users if total_users > 0 else 0

            results = {
                "precision": precision_at_k,
                "map": map_at_k,
                "ndcg": ndcg_at_k,
                "num_users_evaluated": total_users,
                "k": K,
            }

            print(f"- Precision@{K}: {precision_at_k:.4f}")
            print(f"- MAP@{K}: {map_at_k:.4f}")
            print(f"- NDCG@{K}: {ndcg_at_k:.4f}")
            print(f"- Evaluated on {total_users} users")

            return results

        except Exception as e:
            print(f"Error in ranking metrics evaluation: {e}")
            import traceback

            traceback.print_exc()
            return {"precision": None, "map": None, "ndcg": None, "error": str(e)}

    def evaluate_map_at_k(self, K=10):
        """Evaluate MAP@K specifically"""
        ranking_results = self.ranking_metrics_at_k(K)
        self.results[f"map_at_{K}"] = ranking_results
        return ranking_results

    def evaluate_coverage(self):
        """Evaluate catalog coverage and user coverage"""
        print("Evaluating Coverage Metrics...")

        # Catalog coverage
        all_products = set(
            self.inference_system.processed_data["products_df"]["product_id"].unique()
        )
        recommended_products = set()

        # Sample 100 users for coverage test
        users = list(self.inference_system.mappings["user_to_idx"].keys())
        sample_users = users[: min(100, len(users))]

        for user_id in sample_users:
            try:
                recs = self.inference_system.get_user_recommendations(
                    user_id=user_id, n_recommendations=10
                )
                rec_product_ids = {rec["product_id"] for rec in recs}
                recommended_products.update(rec_product_ids)
            except:
                continue

        catalog_coverage = (
            len(recommended_products) / len(all_products) if all_products else 0
        )

        # User coverage (how many users can get recommendations)
        successful_users = 0
        for user_id in sample_users:
            try:
                recs = self.inference_system.get_user_recommendations(
                    user_id=user_id, n_recommendations=5
                )
                if recs:
                    successful_users += 1
            except:
                continue

        user_coverage = successful_users / len(sample_users) if sample_users else 0

        self.results["coverage"] = {
            "catalog_coverage": catalog_coverage,
            "user_coverage": user_coverage,
            "total_products": len(all_products),
            "recommended_products": len(recommended_products),
            "successful_users": successful_users,
            "sample_users": len(sample_users),
        }

        print(f"Catalog Coverage: {catalog_coverage:.2%}")
        print(f"User Coverage: {user_coverage:.2%}")

        return self.results["coverage"]

    def evaluate_diversity(self):
        """Evaluate diversity of recommendations"""
        print("Evaluating Diversity Metrics...")

        users = list(self.inference_system.mappings["user_to_idx"].keys())
        sample_users = users[: min(50, len(users))]

        diversity_metrics = {
            "intra_list_diversity": [],  # Diversity within each user's recommendations
            "category_diversity": [],  # Number of different categories per user
            "brand_diversity": [],  # Number of different brands per user
            "price_diversity": [],  # Price range per user
        }

        for user_id in sample_users:
            try:
                recs = self.inference_system.get_user_recommendations(
                    user_id=user_id, n_recommendations=10
                )
                if not recs:
                    continue

                # Category diversity
                categories = [rec["primary_category"] for rec in recs]
                unique_categories = len(set(categories))
                diversity_metrics["category_diversity"].append(
                    unique_categories / len(recs)
                )

                # Brand diversity
                brands = [rec["brand_name"] for rec in recs]
                unique_brands = len(set(brands))
                diversity_metrics["brand_diversity"].append(unique_brands / len(recs))

                # Price diversity (coefficient of variation)
                prices = [rec["price_usd"] for rec in recs if rec["price_usd"] > 0]
                if prices:
                    price_cv = (
                        np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
                    )
                    diversity_metrics["price_diversity"].append(price_cv)

            except:
                continue

        # Calculate average diversities
        avg_diversity = {}
        for metric, values in diversity_metrics.items():
            if values:
                avg_diversity[metric] = np.mean(values)
            else:
                avg_diversity[metric] = 0

        self.results["diversity"] = avg_diversity

        print(f"Average Category Diversity: {avg_diversity['category_diversity']:.3f}")
        print(f"Average Brand Diversity: {avg_diversity['brand_diversity']:.3f}")
        print(f"Average Price Diversity: {avg_diversity['price_diversity']:.3f}")

        return self.results["diversity"]

    def evaluate_popularity_bias(self):
        """Evaluate if system has popularity bias"""
        print("Evaluating Popularity Bias...")

        # Calculate product popularity
        product_popularity = (
            self.inference_system.processed_data["reviews_df"]
            .groupby("product_id")
            .size()
            .reset_index(name="popularity")
        )
        product_popularity["popularity_rank"] = product_popularity["popularity"].rank(
            ascending=False
        )

        # Get recommendations for sample users
        users = list(self.inference_system.mappings["user_to_idx"].keys())
        sample_users = users[: min(30, len(users))]

        recommended_popularities = []

        for user_id in sample_users:
            try:
                recs = self.inference_system.get_user_recommendations(
                    user_id=user_id, n_recommendations=10
                )
                for rec in recs:
                    product_pop = product_popularity[
                        product_popularity["product_id"] == rec["product_id"]
                    ]
                    if not product_pop.empty:
                        recommended_popularities.append(
                            product_pop.iloc[0]["popularity_rank"]
                        )
            except:
                continue

        if recommended_popularities:
            avg_popularity_rank = np.mean(recommended_popularities)
            median_popularity_rank = np.median(recommended_popularities)

            # Calculate bias score (lower rank = more popular = higher bias)
            total_products = len(product_popularity)
            bias_score = 1 - (
                avg_popularity_rank / total_products
            )  # 0 = no bias, 1 = maximum bias
        else:
            avg_popularity_rank = 0
            median_popularity_rank = 0
            bias_score = 0

        self.results["popularity_bias"] = {
            "avg_popularity_rank": avg_popularity_rank,
            "median_popularity_rank": median_popularity_rank,
            "bias_score": bias_score,
            "total_products": len(product_popularity),
        }

        print(f"Average Popularity Rank: {avg_popularity_rank:.1f}")
        print(f"Popularity Bias Score: {bias_score:.3f}")

        return self.results["popularity_bias"]

    def evaluate_category_distribution(self):
        """Evaluate category distribution in recommendations"""
        print("Evaluating Category Distribution...")

        users = list(self.inference_system.mappings["user_to_idx"].keys())
        sample_users = users[: min(50, len(users))]

        category_counts = {}
        total_recommendations = 0

        for user_id in sample_users:
            try:
                recs = self.inference_system.get_user_recommendations(
                    user_id=user_id, n_recommendations=10
                )
                for rec in recs:
                    category = rec["primary_category"]
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
        catalog_categories = (
            self.inference_system.processed_data["products_df"]["primary_category"]
            .value_counts(normalize=True)
            .to_dict()
        )

        self.results["category_distribution"] = {
            "recommendation_distribution": category_distribution,
            "catalog_distribution": catalog_categories,
            "total_recommendations": total_recommendations,
        }

        print(f"Category Distribution Analysis Complete")
        print(f"Total Categories in Recommendations: {len(category_distribution)}")

        return self.results["category_distribution"]

    def evaluate_cold_start_performance(self):
        """Evaluate cold start recommendation performance"""
        print("Evaluating Cold Start Performance...")

        # Test different user profiles
        test_profiles = [
            {"skin_tone": "light", "skin_type": "dry"},
            {"skin_tone": "medium", "skin_type": "combination"},
            {"skin_tone": "dark", "skin_type": "oily"},
            {"eye_color": "brown", "hair_color": "black"},
            {"eye_color": "blue", "hair_color": "blonde"},
            {},  # Empty profile
        ]

        cold_start_results = []

        for i, profile in enumerate(test_profiles):
            try:
                recs = self.inference_system.get_user_recommendations(
                    user_profile=profile, n_recommendations=10
                )

                cold_start_results.append(
                    {
                        "profile_id": i,
                        "profile": profile,
                        "num_recommendations": len(recs),
                        "success": len(recs) > 0,
                        "avg_rating": (
                            np.mean([rec["rating"] for rec in recs]) if recs else 0
                        ),
                        "avg_price": (
                            np.mean([rec["price_usd"] for rec in recs]) if recs else 0
                        ),
                        "unique_categories": (
                            len(set(rec["primary_category"] for rec in recs))
                            if recs
                            else 0
                        ),
                    }
                )
            except Exception as e:
                cold_start_results.append(
                    {
                        "profile_id": i,
                        "profile": profile,
                        "num_recommendations": 0,
                        "success": False,
                        "error": str(e),
                    }
                )

        success_rate = sum(1 for r in cold_start_results if r["success"]) / len(
            cold_start_results
        )
        avg_recommendations = np.mean(
            [r["num_recommendations"] for r in cold_start_results]
        )

        self.results["cold_start"] = {
            "success_rate": success_rate,
            "avg_recommendations": avg_recommendations,
            "detailed_results": cold_start_results,
        }

        print(f"Cold Start Success Rate: {success_rate:.2%}")
        print(f"Average Recommendations per Profile: {avg_recommendations:.1f}")

        return self.results["cold_start"]

    def create_evaluation_report(self, save_path="evaluation_report.html"):
        """Run comprehensive evaluation and return results"""
        print("Running Comprehensive Evaluation...")

        # Run all evaluations
        print("\n\nEVALUATION")

        # Standard evaluations
        self.evaluate_coverage()
        self.evaluate_diversity()
        self.evaluate_popularity_bias()
        self.evaluate_category_distribution()
        self.evaluate_cold_start_performance()

        # Advanced ML evaluations
        print("\n\nADVANCED ML METRICS")

        self.evaluate_rmse()
        self.evaluate_map_at_k(K=10)

        print("All evaluations completed.")

        # Create HTML report -- code given by Claude.ai
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
                .ml-metrics {{ background: #e3f2fd; border: 1px solid #2196f3; }}
            </style>
        </head>
        <body>
            <h1>📊 Sephora Recommendation System - Evaluation Report</h1>
            
            <h2>🤖 Machine Learning Performance Metrics</h2>"""

        # Add RMSE metrics
        if "rmse" in self.results and self.results["rmse"].get("rmse") is not None:
            rmse_data = self.results["rmse"]
            rmse_class = (
                "good"
                if rmse_data["rmse"] < 1.0
                else "warning" if rmse_data["rmse"] < 1.5 else "poor"
            )
            html_content += f"""
            <div class="metric ml-metrics {rmse_class}">
                <strong>RMSE (Rating Prediction):</strong> {rmse_data['rmse']:.4f}<br>
                <strong>MAE (Mean Absolute Error):</strong> {rmse_data['mae']:.4f}<br>
                <strong>Correlation:</strong> {rmse_data.get('correlation', 'N/A'):.4f}<br>
                <small>Evaluated on {rmse_data.get('num_predictions', 'N/A'):,} predictions from {rmse_data.get('num_users', 'N/A')} users</small>
            </div>"""
        else:
            html_content += f"""
            <div class="metric warning">
                <strong>RMSE:</strong> Not available<br>
                <small>Reason: {self.results.get('rmse', {}).get('reason', 'Unknown error')}</small>
            </div>"""

        # Add MAP@10 metrics
        if (
            "map_at_10" in self.results
            and self.results["map_at_10"].get("map") is not None
        ):
            map_data = self.results["map_at_10"]
            map_class = (
                "good"
                if map_data["map"] > 0.1
                else "warning" if map_data["map"] > 0.05 else "poor"
            )
            html_content += f"""
            <div class="metric ml-metrics {map_class}">
                <strong>MAP@10 (Mean Average Precision):</strong> {map_data['map']:.4f}<br>
                <strong>Precision@10:</strong> {map_data['precision']:.4f}<br>
                <strong>NDCG@10:</strong> {map_data['ndcg']:.4f}<br>
                <small>Evaluated on {map_data.get('num_users_evaluated', 'N/A')} users with test data</small>
            </div>"""
        else:
            html_content += f"""
            <div class="metric warning">
                <strong>MAP@10:</strong> Not available<br>
                <small>Reason: {self.results.get('map_at_10', {}).get('reason', 'Unknown error')}</small>
            </div>"""

        html_content += f"""
            <h2>Coverage Metrics</h2>
            <div class="metric {'good' if self.results['coverage']['catalog_coverage'] > 0.1 else 'warning'}">
                <strong>Catalog Coverage:</strong> {self.results['coverage']['catalog_coverage']:.2%}<br>
                <small>Products recommended: {self.results['coverage']['recommended_products']:,} / {self.results['coverage']['total_products']:,}</small>
            </div>
            <div class="metric {'good' if self.results['coverage']['user_coverage'] > 0.9 else 'warning'}">
                <strong>User Coverage:</strong> {self.results['coverage']['user_coverage']:.2%}<br>
                <small>Successful users: {self.results['coverage']['successful_users']:,} / {self.results['coverage']['sample_users']:,}</small>
            </div>
            
            <h2>Diversity Metrics</h2>
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
            
            <h2>Popularity Bias</h2>
            <div class="metric {'good' if self.results['popularity_bias']['bias_score'] < 0.7 else 'warning'}">
                <strong>Popularity Bias Score:</strong> {self.results['popularity_bias']['bias_score']:.3f}<br>
                <small>0 = no bias, 1 = maximum bias</small>
            </div>
            <div class="metric">
                <strong>Average Popularity Rank:</strong> {self.results['popularity_bias']['avg_popularity_rank']:.1f}<br>
                <small>Lower rank = more popular products</small>
            </div>
            
            <h2>Cold Start Performance</h2>
            <div class="metric {'good' if self.results['cold_start']['success_rate'] > 0.8 else 'warning'}">
                <strong>Success Rate:</strong> {self.results['cold_start']['success_rate']:.2%}<br>
                <small>Percentage of new user profiles that received recommendations</small>
            </div>
            <div class="metric">
                <strong>Average Recommendations:</strong> {self.results['cold_start']['avg_recommendations']:.1f}<br>
                <small>Average number of recommendations per new user</small>
            </div>
            
            <h2>Category Distribution</h2>
            <p>Top categories in recommendations:</p>
            <table>
                <tr><th>Category</th><th>Recommendation %</th><th>Catalog %</th><th>Difference</th></tr>"""

        # Add category distribution table
        rec_dist = self.results["category_distribution"]["recommendation_distribution"]
        cat_dist = self.results["category_distribution"]["catalog_distribution"]

        for category in sorted(
            rec_dist.keys(), key=lambda x: rec_dist.get(x, 0), reverse=True
        )[:10]:
            rec_pct = rec_dist.get(category, 0) * 100
            cat_pct = cat_dist.get(category, 0) * 100
            diff = rec_pct - cat_pct

            # Determine color based on difference
            if diff > 10:
                color = "red"
            elif diff < -10:
                color = "green"
            else:
                color = "black"

            html_content += f"""
                <tr>
                    <td>{category}</td>
                    <td>{rec_pct:.1f}%</td>
                    <td>{cat_pct:.1f}%</td>
                    <td style="color: {color}">{diff:+.1f}%</td>
                </tr>"""

        html_content += """
            </table>
            
            <h2>Performance Analysis & Recommendations</h2>
            <ul>"""

        # Add recommendations based on results
        if (
            self.results.get("rmse", {}).get("rmse")
            and self.results["rmse"]["rmse"] > 1.5
        ):
            html_content += "<li>High RMSE - consider tuning model hyperparameters or adding more features</li>"
        elif (
            self.results.get("rmse", {}).get("rmse")
            and self.results["rmse"]["rmse"] < 1.0
        ):
            html_content += (
                "<li>Good RMSE performance - model predicts ratings accurately</li>"
            )

        if (
            self.results.get("map_at_10", {}).get("map")
            and self.results["map_at_10"]["map"] > 0.1
        ):
            html_content += (
                "<li>Good MAP@10 performance - model ranks relevant items well</li>"
            )
        elif (
            self.results.get("map_at_10", {}).get("map")
            and self.results["map_at_10"]["map"] < 0.05
        ):
            html_content += "<li>Low MAP@10 - improve ranking quality with better features or algorithms</li>"

        if self.results["coverage"]["catalog_coverage"] < 0.1:
            html_content += "<li>Low catalog coverage - consider improving recommendation diversity</li>"

        if self.results["diversity"]["category_diversity"] < 0.3:
            html_content += (
                "<li>Low category diversity - recommendations may be too narrow</li>"
            )

        if self.results["popularity_bias"]["bias_score"] > 0.7:
            html_content += "<li>High popularity bias - system may be recommending only popular items</li>"

        if self.results["cold_start"]["success_rate"] < 0.8:
            html_content += "<li>Poor cold start performance - improve new user recommendations</li>"

        html_content += """
                <li>System successfully provides recommendations for most users</li>
                <li>Multiple recommendation strategies (collaborative + content-based) working</li>
            </ul>
            
            <h2>Technical Details</h2>
            <p><strong>Evaluation Method:</strong> Train/test split evaluation using collaborative filtering metrics</p>
            <p><strong>RMSE:</strong> Root Mean Square Error on held-out ratings</p>
            <p><strong>MAP@10:</strong> Mean Average Precision at 10 recommendations</p>
            <p><strong>Coverage:</strong> Catalog and user coverage analysis</p>
            <p><strong>Diversity:</strong> Category, brand, and price diversity metrics</p>
            <p><strong>Sample Size:</strong> 50-500 users depending on metric</p> 
        </body>
        </html>
        """

        # Save report
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(html_content)

        print(f"Evaluation report saved to: {save_path}")

        return self.results

    def print_summary(self):
        """Print evaluation summary"""
        print("\n\nEVALUATION SUMMARY")

        # ML Performance Metrics
        print("MACHINE LEARNING METRICS:")
        if "rmse" in self.results and self.results["rmse"].get("rmse") is not None:
            rmse_data = self.results["rmse"]
            print(f"• RMSE: {rmse_data['rmse']:.4f}")
            print(f"• MAE: {rmse_data['mae']:.4f}")
            print(f"• Correlation: {rmse_data.get('correlation', 'N/A'):.4f}")

            # RMSE interpretation
            if rmse_data["rmse"] < 1.0:
                print("Excellent rating prediction accuracy")
            elif rmse_data["rmse"] < 1.5:
                print("Good rating prediction accuracy")
            else:
                print("Poor rating prediction - consider model tuning")
        else:
            print("  • RMSE: Not available (no test data)")

        if (
            "map_at_10" in self.results
            and self.results["map_at_10"].get("map") is not None
        ):
            map_data = self.results["map_at_10"]
            print(f"  • MAP@10: {map_data['map']:.4f}")
            print(f"  • Precision@10: {map_data['precision']:.4f}")
            print(f"  • NDCG@10: {map_data['ndcg']:.4f}")

            # MAP interpretation
            if map_data["map"] > 0.1:
                print("Good ranking performance")
            elif map_data["map"] > 0.05:
                print("Moderate ranking performance")
            else:
                print("Poor ranking performance")
        else:
            print("  • MAP@10: Not available (no test data)")

        # Standard Metrics
        print("\nSYSTEM METRICS:")
        if "coverage" in self.results:
            print(
                f"• Catalog Coverage: "
                f"{self.results['coverage']['catalog_coverage']:.2%}"
            )
            print(
                    f"• User Coverage: "
                    f"{self.results['coverage']['user_coverage']:.2%}"
            )

        if "diversity" in self.results:
            print(
                f"• Category Diversity: "
                f"{self.results['diversity']['category_diversity']:.3f}"
            )
            print(
                f"• Brand Diversity: "
                f"{self.results['diversity']['brand_diversity']:.3f}"
            )

        if "popularity_bias" in self.results:
            print(
                f"• Popularity Bias: "
                f"{self.results['popularity_bias']['bias_score']:.3f}"
            )

        if "cold_start" in self.results:
            print(
                f"• Cold Start Success: "
                f"{self.results['cold_start']['success_rate']:.2%}"
            )

        # Overall assessment
        print("\nOVERALL ASSESSMENT:")

        excellent_metrics = []
        good_metrics = []
        issues = []

        # Check ML metrics
        if self.results.get("rmse", {}).get("rmse"):
            if self.results["rmse"]["rmse"] < 1.0:
                excellent_metrics.append("Rating prediction accuracy")
            elif self.results["rmse"]["rmse"] < 1.5:
                good_metrics.append("Rating prediction accuracy")
            else:
                issues.append("High RMSE - poor rating prediction")

        if self.results.get("map_at_10", {}).get("map"):
            if self.results["map_at_10"]["map"] > 0.1:
                excellent_metrics.append("Ranking quality (MAP@10)")
            elif self.results["map_at_10"]["map"] > 0.05:
                good_metrics.append("Ranking quality (MAP@10)")
            else:
                issues.append("Low MAP@10 - poor ranking quality")

        # Check system metrics
        if self.results.get("coverage", {}).get("user_coverage", 0) > 0.95:
            excellent_metrics.append("User coverage")
        elif self.results.get("coverage", {}).get("user_coverage", 0) > 0.8:
            good_metrics.append("User coverage")
        else:
            issues.append("Low user coverage")

        if (
            self.results.get("diversity", {})
            .get("category_diversity", 0) > 0.3
        ):
            good_metrics.append("Category diversity")
        elif (
            self.results.get("diversity", {})
            .get("category_diversity", 0) < 0.2
        ):
            issues.append("Low category diversity")

        if self.results.get("popularity_bias", {}).get("bias_score", 0) < 0.5:
            good_metrics.append("Low popularity bias")
        elif (
            self.results.get("popularity_bias", {})
            .get("bias_score", 0) > 0.8
        ):
            issues.append("High popularity bias")

        if self.results.get("cold_start", {}).get("success_rate", 0) > 0.9:
            excellent_metrics.append("Cold start handling")
        elif self.results.get("cold_start", {}).get("success_rate", 0) > 0.7:
            good_metrics.append("Cold start handling")
        else:
            issues.append("Poor cold start performance")

        # Print assessment
        if excellent_metrics:
            print("Excellent performance:")
            for metric in excellent_metrics:
                print(f"    • {metric}")

        if good_metrics:
            print("Good performance:")
            for metric in good_metrics:
                print(f"    • {metric}")

        if issues:
            print("Areas for improvement:")
            for issue in issues:
                print(f"    • {issue}")

        if not issues:
            print("System performing excellently across all metrics!")

        # Recommendations
        print("\nRECOMMENDATIONS:")
        if (
            self.results.get("rmse", {}).get("rmse")
            and self.results["rmse"]["rmse"] > 1.5
        ):
            print("• Tune ALS hyperparameters")
        if (
            self.results.get("map_at_10", {}).get("map")
            and self.results["map_at_10"]["map"] < 0.05
        ):
            print(
                "• Improve features or add more user interaction data"
            )
        if self.results.get("coverage", {}).get("catalog_coverage", 0) < 0.1:
            print("• Increase recommendation diversity to cover more products")
        if self.results.get("diversity", {}) \
                .get("category_diversity", 0) < 0.2:
            print("• Balance recommendations across more product categories")

        print("• Monitor performance regularly and retrain with new data")
        print("• Consider A/B testing different recommendation strategies")


def main():
    """Main evaluation function"""
    print("Sephora Recommendation System - Evaluation Pipeline")

    # Load inference system
    try:
        inference_system = SephoraInferenceSystem()
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run training.py first to train the models.")
        return

    # Initialize evaluator
    evaluator = RecommendationEvaluator(inference_system)

    # Run evaluation
    try:
        results = evaluator.create_evaluation_report()
        evaluator.print_summary()

        print("\nEvaluation completed successfully!")

        # Print key ML metrics for quick reference
        print("\nQUICK REFERENCE:")
        if results and "rmse" in results and results["rmse"].get("rmse"):
            print(f"   RMSE: {results['rmse']['rmse']:.4f}")
        if (
            results
            and "map_at_10" in results
            and results["map_at_10"].get("map")
        ):
            print(f"   MAP@10: {results['map_at_10']['map']:.4f}")
            print(f"   Precision@10: {results['map_at_10']['precision']:.4f}")
            print(f"   NDCG@10: {results['map_at_10']['ndcg']:.4f}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
