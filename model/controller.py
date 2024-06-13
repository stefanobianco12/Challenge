from model import ReviewModel
import torch
from transformers import BertTokenizer
from flask import Flask, request, jsonify
import os

class Controller():
    def __init__(self):
        self.model = ReviewModel()
        self.model_path = 'trained_models/model.pth'
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print("Model loaded from model.pth")
        else:
            print("No saved model found.")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length=128
        
    def preprocess_text(self,text):
        tokens = self.tokenizer.encode_plus(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return tokens['input_ids'], tokens['attention_mask']

    def predict_review(self,text):
        input_ids, attention_mask = self.preprocess_text(text)
        with torch.no_grad():
            food_rating, delivery_rating, approval_prob = self.model(input_ids, attention_mask)
        food_probs,delivery_probs,approval_prob=self.model.from_logits(food_rating, delivery_rating, approval_prob)
        food_rating = torch.argmax(food_probs, dim=1).item() + 1  # Convert to rating scale 1-5
        delivery_rating = torch.argmax(delivery_probs, dim=1).item() + 1
        approval = 1 if approval_prob.item() > 0.5 else 0
        reliability = abs(approval_prob.item()-0.5)

        return {
            'foodRating':food_rating,
            'deliveryRating':delivery_rating,
            'approval':approval,
            'estimateReliability':reliability
        }
    
app = Flask(__name__)

@app.route('/update-model', methods=['POST'])
def update_model():
    # Perform model update logic here
    print("Received model update trigger")
    controller.model.load_state_dict(torch.load(controller.model_path))
    return 'Model update triggered'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print(data)
    prediction = controller.predict_review(data)
    return jsonify(prediction)

if __name__ == '__main__':
    controller=Controller()
    app.run(host='0.0.0.0', port=5000)