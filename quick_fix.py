# quick_fix.py - Run this to generate the required files
import pandas as pd
import numpy as np

# Create a simple cleaned_metadata.csv file
data = {
    'title': [f'COVID-19 Research Paper {i}' for i in range(100)],
    'journal': ['Journal of Virology', 'Lancet', 'Nature', 'Science'] * 25,
    'year': [2020] * 40 + [2021] * 35 + [2022] * 25,
    'abstract_word_count': np.random.randint(100, 500, 100),
    'source_x': ['PubMed', 'PMC', 'WHO'] * 34
}

df = pd.DataFrame(data)
df.to_csv('cleaned_metadata.csv', index=False)
print("✅ Created cleaned_metadata.csv!")
print("🎯 Now run: streamlit run app.py")