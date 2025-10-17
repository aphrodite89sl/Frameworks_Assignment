"""
CORD-19 Streamlit Application - Updated with Better Error Handling
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """Create sample data if cleaned data is not available"""
    st.warning("📝 Generating sample data for demonstration...")
    
    # Create sample data
    sample_data = {
        'title': [
            'Clinical features of patients infected with COVID-19',
            'A rapid advice guideline for the diagnosis of COVID-19',
            'The epidemiological characteristics of an outbreak of COVID-19',
            'Remdesivir and chloroquine effectively inhibit COVID-19',
            'COVID-19: current knowledge and best practices for clinicians',
            'SARS-CoV-2 viral load in upper respiratory specimens',
            'Effective treatment of severe COVID-19 patients with tocilizumab',
            'The pathogenicity of SARS-CoV-2 in human cells',
            'Mental health during the COVID-19 pandemic',
            'Vaccine development strategies for COVID-19'
        ] * 20,  # Multiply for more data points
        'journal': [
            'Journal of Medical Virology', 'Lancet', 'New England Journal of Medicine',
            'Nature Communications', 'JAMA', 'Clinical Infectious Diseases',
            'Proceedings of the National Academy of Sciences', 'Cell',
            'Journal of Affective Disorders', 'Vaccine'
        ] * 20,
        'year': [2020] * 100 + [2021] * 60 + [2022] * 40,
        'abstract_word_count': np.random.randint(50, 300, 200),
        'source_x': ['PubMed', 'PubMed Central', 'WHO'] * 67
    }
    
    df = pd.DataFrame(sample_data)
    return df

@st.cache_data
def load_data():
    """Load and cache the cleaned data with fallback to sample data"""
    try:
        if os.path.exists('cleaned_metadata.csv'):
            df = pd.read_csv('cleaned_metadata.csv')
            st.success("✅ Successfully loaded cleaned dataset!")
            return df
        else:
            st.info("📝 No cleaned data found. Using sample dataset for demonstration.")
            return create_sample_data()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("📝 Using sample dataset instead.")
        return create_sample_data()

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🦠 CORD-19 Research Explorer</h1>', unsafe_allow_html=True)
    st.markdown("""
    Explore COVID-19 research papers from the CORD-19 dataset. This interactive dashboard 
    allows you to analyze publication trends, top journals, and research focus areas.
    """)
    
    # Load data
    df = load_data()
    
    # Show dataset info
    st.sidebar.title("🔍 Dataset Info")
    st.sidebar.write(f"**Total papers:** {len(df):,}")
    st.sidebar.write(f"**Date range:** {df['year'].min()}-{df['year'].max()}")
    st.sidebar.write(f"**Unique journals:** {df['journal'].nunique()}")
    
    # Sidebar filters
    st.sidebar.title("🎛️ Filters")
    
    # Year filter
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Journal filter
    top_journals = df['journal'].value_counts().head(15).index.tolist()
    selected_journals = st.sidebar.multiselect(
        "Select Journals (optional)",
        options=top_journals,
        default=top_journals[:3]
    )
    
    # Filter data
    filtered_df = df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1])
    ]
    
    if selected_journals:
        filtered_df = filtered_df[filtered_df['journal'].isin(selected_journals)]
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Filtered Papers", f"{len(filtered_df):,}")
    
    with col2:
        st.metric("Selected Years", f"{year_range[0]}-{year_range[1]}")
    
    with col3:
        avg_words = filtered_df['abstract_word_count'].mean()
        st.metric("Avg Abstract Words", f"{avg_words:.0f}")
    
    # Visualizations
    st.markdown("---")
    st.markdown('<h2 class="section-header">📈 Research Trends</h2>', unsafe_allow_html=True)
    
    # Row 1: Time series and journals
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Publications over time
        st.subheader("📅 Publications Over Time")
        yearly_counts = filtered_df['year'].value_counts().sort_index()
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(
            x=yearly_counts.index, 
            y=yearly_counts.values,
            marker_color='lightblue'
        ))
        fig_time.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Publications",
            height=400
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Top journals
        st.subheader("🏥 Top Journals")
        journal_counts = filtered_df['journal'].value_counts().head(8)
        
        fig_journals = go.Figure(go.Bar(
            x=journal_counts.values,
            y=journal_counts.index,
            orientation='h',
            marker_color='lightcoral'
        ))
        fig_journals.update_layout(
            xaxis_title="Number of Papers",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_journals, use_container_width=True)
    
    # Row 2: Word cloud and source distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("☁️ Common Words in Titles")
        
        # Generate word cloud
        all_titles = ' '.join(filtered_df['title'].dropna().astype(str))
        
        if all_titles.strip():
            wordcloud = WordCloud(
                width=600, 
                height=300, 
                background_color='white',
                max_words=50
            ).generate(all_titles)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No titles available for word cloud.")
    
    with col2:
        st.subheader("📚 Paper Sources")
        if 'source_x' in filtered_df.columns:
            source_counts = filtered_df['source_x'].value_counts().head(6)
            
            fig_sources = go.Figure(go.Pie(
                labels=source_counts.index,
                values=source_counts.values,
                hole=0.4
            ))
            fig_sources.update_layout(height=400)
            st.plotly_chart(fig_sources, use_container_width=True)
        else:
            st.info("Source data not available in this dataset.")
    
    # Data sample
    st.markdown("---")
    st.markdown('<h2 class="section-header">📋 Sample Papers</h2>', unsafe_allow_html=True)
    
    # Show sample data
    sample_columns = ['title', 'journal', 'year']
    available_columns = [col for col in sample_columns if col in filtered_df.columns]
    
    st.dataframe(
        filtered_df[available_columns].head(10),
        use_container_width=True,
        height=300
    )
    
    # Setup instructions
    with st.expander("🔧 Setup Instructions"):
        st.markdown("""
        **To use with real CORD-19 data:**
        
        1. Download `metadata.csv` from [Kaggle CORD-19 dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
        2. Run the analysis script:
        ```bash
        python run_analysis.py
        ```
        3. Refresh this app
        
        **Current status:** Using sample dataset for demonstration
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this app:** 
    - Built with Streamlit
    - Data from CORD-19 dataset (sample data for demo)
    - Analysis includes publication trends, journal rankings, and topic analysis
    """)

if __name__ == "__main__":
    main()