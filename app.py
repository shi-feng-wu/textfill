from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process():
    input_text = request.form['input_text']
    output_text = f"Processed: {input_text}"
    return render_template('index.html', input_text=input_text, output_text=output_text)

if __name__ == '__main__':
    app.run(debug=True)
