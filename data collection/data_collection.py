import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
import os
nltk.download('punkt')

def clean_text(text):
    
    text = re.sub(r'\[Illustration:[^\]]+\]', '', text)
    text = re.sub(r'M\^{r\.\}', 'Mr.', text)
    text = re.sub(r'M\^{rs\.\}', 'Mrs.', text)
    text = re.sub(r'\[_Copyright [^\]]+\]', '', text)
    text = re.sub(r'(?i)^\s*(chapter|chap|ch)\.?\s+([0-9]+|[ivxlcdm]+)\s*(:|\.|-)?\s*$', '', text, flags=re.MULTILINE)
    tokens = word_tokenize(text.lower())  # Lowercase transform, we can experiment with this also to see how it affects accuracy.  
    return ' '.join(tokens)
    
def clean_and_save_file(input_file, output_dir):
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
        cleaned_text = clean_text(text)
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        with open(output_file, 'w', encoding='utf-8') as output:
            output.write(cleaned_text)

input_dir = './raw_text'
output_dir = './clean_text'
for file in os.listdir(input_dir):
    if file.endswith('.txt'):
        input_file = os.path.join(input_dir, file)
        clean_and_save_file(input_file, output_dir)
