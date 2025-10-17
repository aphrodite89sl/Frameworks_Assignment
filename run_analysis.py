"""
Run this script first to generate the cleaned data for the Streamlit app
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import os

def create_sample_data():
    """
    Since the full CORD-19 dataset might be large or unavailable,
    let's create a functioning version with sample data structure
    """
    print("🔄 Creating sample dataset structure...")
    
    # Sample COVID-19 related data
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
        ],
        'abstract': [
            'This study describes the clinical features of COVID-19 patients in Wuhan.',
            'We developed rapid diagnostic guidelines for COVID-19 detection.',
            'Epidemiological analysis of COVID-19 outbreak patterns and spread.',
            'Antiviral efficacy of remdesivir and chloroquine against coronavirus.',
            'Comprehensive review of COVID-19 clinical management strategies.',
            'Analysis of viral load dynamics in COVID-19 patient specimens.',
            'Clinical trial results for tocilizumab in severe COVID-19 cases.',
            'Investigation of SARS-CoV-2 pathogenicity mechanisms in vitro.',
            'Assessment of mental health impacts during pandemic lockdowns.',
            'Overview of current vaccine development approaches for coronavirus.'
        ],
        'publish_time': [
            '2020-01-15', '2020-02-01', '2020-02-10', '2020-03-05', 
            '2020-03-15', '2020-04-01', '2020-04-10', '2020-05-01',
            '2020-05-15', '2020-06-01'
        ],
        'journal': [
            'Journal of Medical Virology', 'Lancet', 'New England Journal of Medicine',
            'Nature Communications', 'JAMA', 'Clinical Infectious Diseases',
            'Proceedings of the National Academy of Sciences', 'Cell',
            'Journal of Affective Disorders', 'Vaccine'
        ],
        'authors': [
            'Smith A, Johnson B', 'Brown C, Davis D', 'Wilson E, Lee F',
            'Garcia H, Martinez I', 'Thompson J, White K', 'Anderson L, Clark M',
            'Rodriguez N, Lewis O', 'Patel P, Kumar Q', 'Wang R, Zhang S',
            'Taylor T, Harris U'
        ],
        'source_x': [
            'PubMed', 'PubMed Central', 'WHO', 'PubMed', 'PubMed Central',
            'WHO', 'PubMed', 'PubMed Central', 'PubMed', 'WHO'
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Add more data points for better visualization
    expanded_data = [df.copy() for _ in range(50)]
    df = pd.concat(expanded_data, ignore_index=True)
    
    # Add some variation
    import numpy as np
    np.random.seed(42)
    
    # Vary publication dates
    dates = pd.date_range('2019-12-01', '2022-12-31', freq='D')
    df['publish_time'] = np.random.choice(dates, len(df))
    
    # Vary journals
    journals = [
        'Journal of Medical Virology', 'Lancet', 'New England Journal of Medicine',
        'Nature Communications', 'JAMA', 'Clinical Infectious Diseases',
        'Proceedings of the National Academy of Sciences', 'Cell',
        'Journal of Affective Disorders', 'Vaccine', 'Science', 'Nature',
        'BMJ', 'PLOS One', 'Journal of Virology'
    ]
    df['journal'] = np.random.choice(journals, len(df))
    
    # Add some missing values to simulate real data
    df.loc[df.sample(frac=0.1).index, 'abstract'] = None
    df.loc[df.sample(frac=0.05).index, 'journal'] = None
    
    return df

def clean_data(df):
    """Clean and prepare the dataset"""
    print("🧹 Cleaning data...")
    
    cleaned_df = df.copy()
    
    # Handle missing values
    cleaned_df['abstract'] = cleaned_df['abstract'].fillna('No abstract available')
    cleaned_df['journal'] = cleaned_df['journal'].fillna('Unknown Journal')
    
    # Convert publish_time to datetime
    cleaned_df['publish_time'] = pd.to_datetime(cleaned_df['publish_time'], errors='coerce')
    
    # Extract year and month
    cleaned_df['year'] = cleaned_df['publish_time'].dt.year
    cleaned_df['month'] = cleaned_df['publish_time'].dt.month
    
    # Create abstract word count
    cleaned_df['abstract_word_count'] = cleaned_df['abstract'].apply(
        lambda x: len(str(x).split())
    )
    
    return cleaned_df

def create_visualizations(df):
    """Create basic visualizations"""
    print("📊 Creating visualizations...")
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Publications over time
    plt.figure(figsize=(12, 6))
    yearly_counts = df['year'].value_counts().sort_index()
    plt.bar(yearly_counts.index, yearly_counts.values)
    plt.title('COVID-19 Publications by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Publications')
    plt.tight_layout()
    plt.savefig('visualizations/publications_over_time.png')
    plt.close()
    
    # 2. Top journals
    plt.figure(figsize=(10, 8))
    top_journals = df['journal'].value_counts().head(10)
    top_journals.plot(kind='barh')
    plt.title('Top 10 Journals Publishing COVID-19 Research')
    plt.xlabel('Number of Publications')
    plt.tight_layout()
    plt.savefig('visualizations/top_journals.png')
    plt.close()
    
    # 3. Word cloud
    plt.figure(figsize=(12, 8))
    all_titles = ' '.join(df['title'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Frequent Words in Paper Titles')
    plt.tight_layout()
    plt.savefig('visualizations/word_cloud.png')
    plt.close()
    
    # 4. Source distribution
    plt.figure(figsize=(10, 8))
    source_counts = df['source_x'].value_counts()
    plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Papers by Source')
    plt.tight_layout()
    plt.savefig('visualizations/source_distribution.png')
    plt.close()

def main():
    """Main function to generate the data"""
    print("🚀 Generating CORD-19 sample data...")
    
    # Check if metadata.csv exists, otherwise create sample data
    if os.path.exists('metadata.csv'):
        print("📁 Found metadata.csv, loading...")
        df = pd.read_csv('metadata.csv', low_memory=False)
    else:
        print("📝 Creating sample dataset...")
        df = create_sample_data()
        # Save the sample as metadata.csv for consistency
        df.to_csv('metadata.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_data(df)
    
    # Create visualizations
    create_visualizations(cleaned_df)
    
    # Save cleaned data for Streamlit
    cleaned_df.to_csv('cleaned_metadata.csv', index=False)
    
    print("✅ Data generation complete!")
    print("📁 Files created:")
    print("   - metadata.csv (original/sample data)")
    print("   - cleaned_metadata.csv (cleaned data for Streamlit)")
    print("   - visualizations/ (chart images)")
    print("\n🎯 Now you can run: streamlit run app.py")

if __name__ == "__main__":
    main()