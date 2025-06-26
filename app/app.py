"""
Sephora Recommendation System - Streamlit Web App
-----------------------------------------------------
Interactive web interface for the Sephora recommendation system.
"""

# Imports
import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SRC'))


# Import inference -- try to import it and raise an error if fails (display on streamlit (sr) a message)
try:
    from inference import SephoraInferenceSystem
except ImportError as e:
    st.error(f"Error importing inference system: {e}")
    st.error("Make sure you have trained the models first by running src/training.py")
    st.info("Also ensure 'implicit' library is installed: pip install implicit")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Sephora Recommendation System",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles for HTML -- code created with the help of Claude.ai
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #000000;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333333;
        margin-bottom: 1rem;
    }
    .recommendation-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem;
    }
    .stSelectbox > div > div > select {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource          # only loads the code once per session
def load_inference_system():
    """Load the inference system (cached)"""
    try:
        # Check if implicit is available
        try:
            import implicit
        except ImportError:
            st.error("Missing required library: 'implicit'")
            st.error("Please install it by running: conda install -c conda-forge implicit")
            st.stop()
        
        return SephoraInferenceSystem()
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please ensure models are trained by running: python src/training.py")
        st.info("Or check that you're running the app from the correct directory")
        return None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please ensure models are trained by running: python src/training.py")
        st.info("Check the console for more detailed error information")
        return None

def display_recommendation_card(rec, index):
    """Display a recommendation in a card format"""
    with st.container():
        st.markdown(f"""
        <div class="recommendation-card">
            <h4>#{index + 1}: {rec['product_name']}</h4>
            <p><strong>Brand:</strong> {rec['brand_name']}</p>
            <p><strong>Price:</strong> ${rec['price_usd']:.2f}</p>
            <p><strong>Rating:</strong> {rec['rating']:.1f}/5.0 ⭐</p>
            <p><strong>Category:</strong> {rec['primary_category']}</p>
            {f"<p><strong>Subcategory:</strong> {rec['secondary_category']}</p>" if rec['secondary_category'] != 'Unknown' else ""}
            {f"<p><strong>Type:</strong> {rec['tertiary_category']}</p>" if rec['tertiary_category'] != 'Unknown' else ""}
        </div>
        """, unsafe_allow_html=True)

def create_recommendations_chart(recommendations):
    """Create visualization of recommendations"""
    if not recommendations:
        return None
    
    df = pd.DataFrame(recommendations)
    
    # Price distribution
    fig_price = px.histogram(
        df, 
        x='price_usd', 
        title='Price Distribution of Recommendations',
        nbins=10,
        color_discrete_sequence=['#FF69B4']
    )
    fig_price.update_layout(
        xaxis_title="Price (USD)",
        yaxis_title="Number of Products",
        showlegend=False
    )
    
    # Category distribution
    category_counts = df['primary_category'].value_counts()
    fig_category = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title='Category Distribution of Recommendations',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Rating vs Price scatter
    fig_scatter = px.scatter(
        df,
        x='price_usd',
        y='rating',
        size='price_usd',
        hover_data=['product_name', 'brand_name'],
        title='Rating vs Price of Recommendations',
        color='primary_category',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_scatter.update_layout(
        xaxis_title="Price (USD)",
        yaxis_title="Rating"
    )
    
    return fig_price, fig_category, fig_scatter

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">💄 Sephora Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Get personalized beauty product recommendations powered by AI</p>', unsafe_allow_html=True)
    
    # Load inference system
    with st.spinner("Loading recommendation models..."):
        inference_system = load_inference_system()
    
    if inference_system is None:
        st.stop()
    
    # Sidebar for system info
    with st.sidebar:
        st.markdown("**How it works:**")
        st.markdown("• **Existing users**: Get personalized recommendations using collaborative filtering and content-based methods")
        st.markdown("• **New users**: Get recommendations based on similar user profiles and popular products")
        st.markdown("• **Category filtering**: Narrow down recommendations by product categories")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<h2 class="sub-header">🔍 Get Recommendations</h2>', unsafe_allow_html=True)
        
        # User selection method
        user_method = st.radio(
            "How would you like to get recommendations?",
            ["I'm an existing user", "I'm a new user", "Popular products"],
            help="Choose based on whether you have an account or not"
        )
        
        # Initialize variables
        user_id = None
        user_profile = None
        
        # User input based on method
        if user_method == "I'm an existing user":
            st.subheader("Select Your User ID")
            users = inference_system.get_all_users()
            user_id = st.selectbox(
                "Your User ID:",
                options=users,
                help=f"Select from {len(users):,} available users"
            )
            
            if user_id:
                st.success(f"Selected user: {user_id}")
        
        elif user_method == "I'm a new user":
            st.subheader("Tell Us About Yourself")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                skin_tone = st.selectbox(
                    "Skin Tone:",
                    ["", "light", "medium", "tan", "deep", "dark", "fair"],
                    help="Select your skin tone"
                )
                
                skin_type = st.selectbox(
                    "Skin Type:",
                    ["", "normal", "dry", "oily", "combination", "sensitive"],
                    help="Select your skin type"
                )
            
            with col_b:
                eye_color = st.selectbox(
                    "Eye Color:",
                    ["", "brown", "blue", "green", "hazel", "gray", "amber"],
                    help="Select your eye color"
                )
                
                hair_color = st.selectbox(
                    "Hair Color:",
                    ["", "black", "brown", "blonde", "red", "gray", "other"],
                    help="Select your hair color"
                )
            
            # Create user profile
            user_profile = {
                'skin_tone': skin_tone if skin_tone else None,
                'skin_type': skin_type if skin_type else None,
                'eye_color': eye_color if eye_color else None,
                'hair_color': hair_color if hair_color else None
            }
            
            # Remove None values
            user_profile = {k: v for k, v in user_profile.items() if v is not None}
            
            if user_profile:
                st.success(f"Profile created: {user_profile}")
        
        # Category filters
        st.subheader("🏷️ Filter by Category (Optional)")
        
        categories = inference_system.get_categories()
        
        primary_category = st.selectbox(
            "Primary Category:",
            [""] + categories.get('primary_category', []),
            help="Filter by main product category"
        )
        
        secondary_category = st.selectbox(
            "Secondary Category:",
            [""] + categories.get('secondary_category', []),
            help="Filter by product subcategory"
        )
        
        tertiary_category = st.selectbox(
            "Tertiary Category:",
            [""] + categories.get('tertiary_category', []),
            help="Filter by specific product type"
        )
        
        # Number of recommendations
        n_recommendations = st.slider(
            "Number of Recommendations:",
            min_value=1,
            max_value=20,
            value=10,
            help="How many products to recommend"
        )
        
        # Get recommendations button
        get_recommendations = st.button(
            "🎯 Get My Recommendations",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        st.markdown('<h2 class="sub-header">💫 Your Recommendations</h2>', unsafe_allow_html=True)
        
        if get_recommendations:
            with st.spinner("Generating personalized recommendations..."):
                try:
                    # Prepare parameters
                    params = {
                        'n_recommendations': n_recommendations,
                        'primary_category': primary_category if primary_category else None,
                        'secondary_category': secondary_category if secondary_category else None,
                        'tertiary_category': tertiary_category if tertiary_category else None
                    }
                    
                    if user_method == "I'm an existing user" and user_id:
                        params['user_id'] = user_id
                    elif user_method == "I'm a new user" and user_profile:
                        params['user_profile'] = user_profile
                    
                    # Get recommendations
                    recommendations = inference_system.get_user_recommendations(**params)
                    
                    if recommendations:
                        st.success(f"Found {len(recommendations)} recommendations for you!")
                        
                        # Display recommendations
                        for i, rec in enumerate(recommendations):
                            display_recommendation_card(rec, i)
                        
                        # Create visualizations
                        st.markdown("---")
                        st.subheader("📈 Recommendation Analytics")
                        
                        fig_price, fig_category, fig_scatter = create_recommendations_chart(recommendations)
                        
                        if fig_price:
                            col_chart1, col_chart2 = st.columns(2)
                            
                            with col_chart1:
                                st.plotly_chart(fig_price, use_container_width=True)
                            
                            with col_chart2:
                                st.plotly_chart(fig_category, use_container_width=True)
                            
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Summary statistics
                        df_recs = pd.DataFrame(recommendations)
                        
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        
                        with col_stat1:
                            avg_price = df_recs['price_usd'].mean()
                            st.metric("Average Price", f"${avg_price:.2f}")
                        
                        with col_stat2:
                            avg_rating = df_recs['rating'].mean()
                            st.metric("Average Rating", f"{avg_rating:.1f}/5.0")
                        
                        with col_stat3:
                            unique_brands = df_recs['brand_name'].nunique()
                            st.metric("Unique Brands", unique_brands)
                        
                        with col_stat4:
                            unique_categories = df_recs['primary_category'].nunique()
                            st.metric("Categories", unique_categories)
                        
                        # Download recommendations
                        csv = df_recs.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Recommendations as CSV",
                            data=csv,
                            file_name=f"sephora_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                    else:
                        st.warning("No recommendations found. Try adjusting your filters or preferences.")
                        
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
                    st.error("Please try again or contact support.")
        
        else:
            st.info("👆 Configure your preferences and click 'Get My Recommendations' to see personalized product suggestions!")
            
            # Show sample popular products
            st.subheader("🔥 Popular Products")
            try:
                popular_recs = inference_system.get_user_recommendations(n_recommendations=5)
                if popular_recs:
                    for i, rec in enumerate(popular_recs):
                        with st.expander(f"#{i+1}: {rec['product_name']} by {rec['brand_name']}"):
                            st.write(f"**Price:** ${rec['price_usd']:.2f}")
                            st.write(f"**Rating:** {rec['rating']:.1f}/5.0 ⭐")
                            st.write(f"**Category:** {rec['primary_category']}")
            except:
                st.write("Popular products will appear here")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>💄 Sephora Recommendation System | Powered by Collaborative Filtering & Content-Based AI</p>
            <p>Built with Streamlit | Data Science & Machine Learning</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()