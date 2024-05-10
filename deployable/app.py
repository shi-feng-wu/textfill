from flask import Flask, request, render_template
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig


def fill_mask(text):
    tokenized_input = tokenizer(text, return_tensors='pt')
    mask_token_index = torch.where(tokenized_input['input_ids'][0] == tokenizer.mask_token_id)[0]

    input_ids = tokenized_input['input_ids'].to(device)

    with torch.no_grad():
        predictions = model(input_ids)[0]

    for mask_index in mask_token_index:
        predicted_token_id = predictions[0, mask_index].argmax(dim=-1)
        predicted_token = tokenizer.decode([predicted_token_id])
        text = text.replace(tokenizer.mask_token, predicted_token, 1)

    return text


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForMaskedLM.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process():
    input_text = request.form['input_text']
    filled_text = fill_mask(input_text)
    output_text = filled_text
    return render_template('index.html', input_text=input_text, output_text=output_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
