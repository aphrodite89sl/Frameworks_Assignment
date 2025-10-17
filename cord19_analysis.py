"""
CORD-19 Data Analysis
Part 1-3: Data Loading, Cleaning, and Visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class CORD19Analyzer:
    def __init__(self, file_path='metadata.csv'):
        self.file_path = file_path
        self.df = None
        self.cleaned_df = None
        
    def load_and_explore(self):
        """
        Part 1: Data Loading and Basic Exploration
        """
        print("=" * 60)
        print("PART 1: DATA LOADING AND BASIC EXPLORATION")
        print("=" * 60)
        
        try:
            # Load the dataset
            self.df = pd.read_csv(self.file_path, low_memory=False)
            print("✅ Dataset loaded successfully!")
            print(f"📊 Dataset shape: {self.df.shape}")
            
            # Display first few rows
            print("\n📋 First 5 rows of the dataset:")
            print(self.df.head())
            
            # Basic information
            print("\n🔍 Dataset information:")
            print(self.df.info())
            
            # Check for missing values
            print("\n❓ Missing values in key columns:")
            key_columns = ['title', 'abstract', 'publish_time', 'journal', 'authors']
            missing_data = {}
            for col in key_columns:
                if col in self.df.columns:
                    missing_count = self.df[col].isnull().sum()
                    missing_percent = (missing_count / len(self.df)) * 100
                    missing_data[col] = missing_percent
                    print(f"{col}: {missing_count} missing ({missing_percent:.2f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def clean_and_prepare(self):
        """
        Part 2: Data Cleaning and Preparation
        """
        print("\n" + "=" * 60)
        print("PART 2: DATA CLEANING AND PREPARATION")
        print("=" * 60)
        
        # Create a cleaned copy
        self.cleaned_df = self.df.copy()
        
        # Handle missing values
        print("🧹 Handling missing values...")
        
        # Remove rows with completely missing titles
        initial_count = len(self.cleaned_df)
        self.cleaned_df = self.cleaned_df.dropna(subset=['title'])
        print(f"Removed {initial_count - len(self.cleaned_df)} rows with missing titles")
        
        # Fill missing abstracts with empty string
        self.cleaned_df['abstract'] = self.cleaned_df['abstract'].fillna('')
        
        # Convert publish_time to datetime and extract year
        print("📅 Processing dates...")
        self.cleaned_df['publish_time'] = pd.to_datetime(
            self.cleaned_df['publish_time'], errors='coerce'
        )
        self.cleaned_df['year'] = self.cleaned_df['publish_time'].dt.year
        self.cleaned_df['month'] = self.cleaned_df['publish_time'].dt.month
        
        # Create abstract word count
        self.cleaned_df['abstract_word_count'] = self.cleaned_df['abstract'].apply(
            lambda x: len(str(x).split())
        )
        
        # Clean journal names
        self.cleaned_df['journal'] = self.cleaned_df['journal'].fillna('Unknown Journal')
        
        print(f"✅ Cleaning complete. Final dataset shape: {self.cleaned_df.shape}")
        return True
    
    def analyze_and_visualize(self):
        """
        Part 3: Data Analysis and Visualization
        """
        print("\n" + "=" * 60)
        print("PART 3: DATA ANALYSIS AND VISUALIZATION")
        print("=" * 60)
        
        # Create visualizations directory
        import os
        os.makedirs('visualizations', exist_ok=True)
        
        # 1. Publications over time
        print("📈 Creating time series visualization...")
        self._create_time_series_plot()
        
        # 2. Top journals
        print("📊 Creating top journals visualization...")
        self._create_journals_plot()
        
        # 3. Word cloud
        print("☁️ Creating word cloud...")
        self._create_word_cloud()
        
        # 4. Source distribution
        print("📚 Creating source distribution...")
        self._create_source_distribution()
        
        # Basic analysis results
        self._print_analysis_results()
    
    def _create_time_series_plot(self):
        """Plot number of publications over time"""
        # Filter valid years
        valid_years = self.cleaned_df[self.cleaned_df['year'].between(2019, 2023)]
        yearly_counts = valid_years['year'].value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=8)
        plt.title('COVID-19 Research Publications Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Number of Publications')
        plt.grid(True, alpha=0.3)
        plt.xticks(yearly_counts.index)
        plt.tight_layout()
        plt.savefig('visualizations/publications_over_time.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_journals_plot(self):
        """Bar chart of top publishing journals"""
        top_journals = self.cleaned_df['journal'].value_counts().head(10)
        
        plt.figure(figsize=(12, 8))
        top_journals.plot(kind='barh')
        plt.title('Top 10 Journals Publishing COVID-19 Research', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Publications')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('visualizations/top_journals.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_word_cloud(self):
        """Generate word cloud from paper titles"""
        # Combine all titles
        all_titles = ' '.join(self.cleaned_df['title'].dropna().astype(str))
        
        # Remove common words and clean text
        stop_words = ['using', 'based', 'study', 'analysis', 'model', 'method', 'covid', '19', 'sars', 'cov', '2']
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_titles.lower())
        words = [word for word in words if word not in stop_words]
        
        word_freq = Counter(words)
        
        plt.figure(figsize=(12, 8))
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100
        ).generate_from_frequencies(word_freq)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Frequent Words in Paper Titles', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/word_cloud.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_source_distribution(self):
        """Plot distribution of papers by source"""
        source_counts = self.cleaned_df['source_x'].value_counts().head(8)
        
        plt.figure(figsize=(10, 8))
        plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of Papers by Source', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/source_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _print_analysis_results(self):
        """Print key analysis findings"""
        print("\n🔍 KEY FINDINGS:")
        print("-" * 40)
        
        # Total papers
        print(f"Total papers analyzed: {len(self.cleaned_df):,}")
        
        # Time range
        valid_years = self.cleaned_df[self.cleaned_df['year'].between(2019, 2023)]
        yearly_counts = valid_years['year'].value_counts().sort_index()
        print(f"\nPublications by year:")
        for year, count in yearly_counts.items():
            print(f"  {year}: {count:,} papers")
        
        # Top journals
        top_journal = self.cleaned_df['journal'].value_counts().index[0]
        top_journal_count = self.cleaned_df['journal'].value_counts().iloc[0]
        print(f"\nTop journal: '{top_journal}' with {top_journal_count:,} papers")
        
        # Abstract statistics
        avg_words = self.cleaned_df['abstract_word_count'].mean()
        print(f"Average abstract length: {avg_words:.0f} words")
        
        # Missing data summary
        missing_titles = self.cleaned_df['title'].isnull().sum()
        print(f"Papers with missing titles: {missing_titles}")

def main():
    """Main execution function"""
    print("🦠 CORD-19 DATASET ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = CORD19Analyzer('metadata.csv')
    
    # Execute all parts
    if analyzer.load_and_explore():
        analyzer.clean_and_prepare()
        analyzer.analyze_and_visualize()
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📁 Files created:")
        print("   - visualizations/publications_over_time.png")
        print("   - visualizations/top_journals.png")
        print("   - visualizations/word_cloud.png")
        print("   - visualizations/source_distribution.png")
        
        # Save cleaned data for Streamlit app
        analyzer.cleaned_df.to_csv('cleaned_metadata.csv', index=False)
        print("   - cleaned_metadata.csv (for Streamlit app)")
        
    else:
        print("❌ Analysis failed. Please check the error messages above.")

if __name__ == "__main__":
    main()