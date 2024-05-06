# tokenizes all the txt files in the input folder and saves them to the output folder
import re
import os
from transformers import BertTokenizer
import numpy as np
import json

def pad_tokens(token_lists, desired_length, pad_token_id):
    padded_lists = [tokens + [pad_token_id] * (desired_length - len(tokens)) for tokens in token_lists]
    return padded_lists

def tokenize_text_file(text, tokenizer, chunk_size=512, test_split=0.1):
    # tokenize text and split into chunks
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    
    # split into test and train sets
    np.random.shuffle(chunks)
    split_index = int(len(chunks) * (1 - test_split))
    train_chunks = chunks[:split_index]
    test_chunks = chunks[split_index:]
    
    # pad chunks to make sure all are 512 long
    train_chunks = pad_tokens(train_chunks, chunk_size, tokenizer.pad_token_id)
    test_chunks = pad_tokens(test_chunks, chunk_size, tokenizer.pad_token_id)
    
    return train_chunks, test_chunks
    
input_dir = r'.\clean_text'
output_dir = r'.\tokenized_text'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # tokenize and split the cleaned text file
        train_data, test_data = tokenize_text_file(text, tokenizer)

        # save the tokenized data
        train_data_file = os.path.join(output_dir, f'train_{filename}.json')
        test_data_file = os.path.join(output_dir, f'test_{filename}.json')
        with open(train_data_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f)
        with open(test_data_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        print(f"Data saved to {train_data_file} and {test_data_file}")