from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

app = Flask(__name__)

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def analysis(input_text):
    encoded_text = tokenizer(input_text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)
    # Convert float32 values to regular Python floats
    scores = [float(score) for score in scores]
    scores_dict = {
        'NEGATIVE_SCORE': scores[0],
        'NEUTRAL_SCORE': scores[1],
        'POSITIVE_SCORE': scores[2]
    }
    return scores_dict

@app.route('/analyze-emotion', methods=['POST'])
def analyze_emotion():
    data = request.get_json()

    if 'text_data' not in data:
        return jsonify({'error': 'Missing text_data in request payload'}), 400

    text_data = data['text_data']
    scores_dict = analysis(text_data)

    return jsonify(scores_dict)

if __name__ == '__main__':
    app.run(debug=True)
