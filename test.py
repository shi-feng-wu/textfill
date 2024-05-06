def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def mask_tokens(inputs, tokenizer, mask_probability=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mask_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer.vocab), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels

def prepare_data(data, tokenizer):
    input_tokens = torch.tensor([tokenizer.convert_tokens_to_ids(tokens) for tokens in data])
    return mask_tokens(input_tokens, tokenizer)

# calculates loss for model on dataset in loader
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    average_loss = total_loss / len(loader)
    return average_loss

# calculates accuracy for model on dataset in loader
def calculate_accuracy(model, loader):
    model.eval()
    total_correct = 0
    total_masked_tokens = 0
    
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = model(input_ids, labels=labels)
            predictions = outputs.logits.argmax(dim=-1)
            mask = labels != -100  # masked tokens are labeled with -100
            correct_predictions = (predictions == labels) & mask
            total_correct += correct_predictions.sum().item()
            total_masked_tokens += mask.sum().item()
    
    accuracy = total_correct / total_masked_tokens
    return accuracy
    
import torch
import json
import os
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from torch.utils.data import DataLoader, TensorDataset

# load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForMaskedLM.from_pretrained('bert-base-uncased')

#config = BertConfig.from_pretrained('bert-base-uncased')
#model = BertForMaskedLM(config)
#model_state_dict = torch.load('jane_austen_2.pth')
#model.load_state_dict(model_state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# directory containing the data files
data_directory = './data collection/tokenized_text'

# load and prepare data
for filename in os.listdir(data_directory):
    if 'test' in filename:
        filepath = os.path.join(data_directory, filename)
        data = load_data(filepath)
        inputs, labels = prepare_data(data, tokenizer)
        data_loader = DataLoader(TensorDataset(inputs, labels), batch_size=16)

        # calculate accuracy and loss
        average_loss = evaluate(model, data_loader)
        accuracy = calculate_accuracy(model, data_loader)
        print(f"File: {filename}")
        print(f"Accuracy: {accuracy:.4f}, Loss: {average_loss:.4f}")
