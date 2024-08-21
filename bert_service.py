from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    correct_answer = data['correct_answer']
    student_answer = data['student_answer']

    correct_vec = encode_text(correct_answer)
    student_vec = encode_text(student_answer)
    similarity = cosine_similarity(correct_vec, student_vec)[0][0]

    score = 0
    if similarity > 0.8:
        score = 1
    elif similarity > 0.5:
        score = 0.5

    return jsonify({'similarity': similarity, 'score': score})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
