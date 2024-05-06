import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def clean_text(text):
    
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(?i)^\s*(chapter|chap|ch)\.?\s+([0-9]+|[ivxlcdm]+)\s*(:|\.|-)?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text.lower())  # Lowercase transform, we can experiment with this also to see how it affects accuracy.  
    return ' '.join(tokens)

def clean_dataframe(df):
    df['Cleaned_Text'] = df['Text'].apply(clean_text)
    return df

dataframes = {
    "HuckleBerry": pd.read_csv('Adventure Of HuckleBerry.txt', sep='\n\n', header=None, names=['Text']),
    "Alonzo": pd.read_csv('Alonzo Fitz and other Stories.txt', sep='\n\n', header=None, names=['Text']),
    "ChristianScience": pd.read_csv('Christian Science.txt', sep='\n\n', header=None, names=['Text']),
    "TomSawyer": pd.read_csv('Tom Sawyer.txt', sep='\n\n', header=None, names=['Text'])
}

# Clean all DataFrames
cleaned_dataframes = {name: clean_dataframe(df) for name, df in dataframes.items()}

print(cleaned_dataframes['TomSawyer'].head(15))