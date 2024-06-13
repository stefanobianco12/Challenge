from model import ReviewModel
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from sklearn.metrics import accuracy_score
import requests

class Trainer():
    def __init__(self,threshold):
        self.model = ReviewModel()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.food_criterion = nn.CrossEntropyLoss()
        self.delivery_criterion = nn.CrossEntropyLoss()
        self.approval_criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.epochs=1
        self.batch_size=1
        self.threshold=threshold
        self.model_file_path="trained_models/model.pth"
        self.flask_url='http://localhost:5000'

    def send_trigger_update(self):
        response = requests.post(self.flask_url + '/update-model')
        if response.status_code == 200:
            print("Model update trigger sent successfully")
        else:
            print("Failed to send model update trigger")


    def generate_tensor_dataset(self,dataset_json):
        input_ids = []
        attention_masks = []
        food_labels=[]
        delivery_labels=[]
        approvals=[]
        for data in dataset_json:
            tokens = self.tokenizer.encode_plus(data['comment'],
                                max_length=128,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt')
            
            input_ids.append(tokens['input_ids'].squeeze())
            attention_masks.append(tokens['attention_mask'].squeeze())
            food_labels.append(data['foodRating'])
            delivery_labels.append(data['deliveryRating'])
            approvals.append(data['approval'])

        return TensorDataset(torch.stack(input_ids),
                                torch.stack(attention_masks),
                                torch.tensor(food_labels),
                                torch.tensor(delivery_labels),
                                torch.tensor(approvals))

    def training(self):
        with open("dataset/train_set.json", 'r') as file:
            train_set_json = json.load(file)

        with open("dataset/val_set.json", 'r') as file:
            val_set_json = json.load(file)
        print("GENERATE TRAINING SET")
        train_dataset=self.generate_tensor_dataset(train_set_json)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        print("GENERATING VAL DATASET")
        val_dataset=self.generate_tensor_dataset(val_set_json)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        print("START TRAINING")
        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            print(f"Epoch {epoch + 1}/{self.epochs}")
            for batch in train_dataloader:
                input_ids, attention_mask, food_labels, delivery_labels, approval_labels = batch

                self.optimizer.zero_grad()

                food_logits, delivery_logits, approval_logit = self.model(input_ids, attention_mask)
                food_loss = self.food_criterion(food_logits, food_labels)
                delivery_loss = self.delivery_criterion(delivery_logits, delivery_labels)
                approval_loss = self.approval_criterion(approval_logit.squeeze(dim=1), approval_labels.float())

                total_loss = food_loss + delivery_loss + approval_loss
                total_loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch + 1} complete. Total loss: {total_loss.item()}")

        print("START VAL")
        self.model.eval()
        food_preds=[]
        food_targets=[]
        delivery_preds=[]
        delivery_targets=[]
        approval_preds=[]
        approval_targets=[]
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, food_labels, delivery_labels, approval_labels = batch
                food_logits, delivery_logits, approval_logit = self.model(input_ids, attention_mask)
                food_probs,delivery_probs,approval_prob=self.model.from_logits(food_logits, delivery_logits, approval_logit)
                food_preds.append(torch.argmax(food_probs, dim=1).item() + 1)
                delivery_preds.append(torch.argmax(delivery_probs, dim=1).item() + 1)
                approval_preds.append(1 if approval_prob.item() > 0.5 else 0)
                food_targets.append(food_labels)
                delivery_targets.append(delivery_labels)
                approval_targets.append(approval_labels)
        food_accuracy = accuracy_score(food_targets, food_preds)
        delivery_accuracy = accuracy_score(delivery_targets, delivery_preds)
        approval_accuracy = accuracy_score(approval_targets, approval_preds)
        print(f'Food Accuracy: {food_accuracy}, Delivery Accuracy: {delivery_accuracy}, Approval Accuracy: {approval_accuracy}')

        if food_accuracy>= self.threshold and delivery_accuracy >= self.threshold and approval_accuracy >= approval_accuracy:
            torch.save(self.model.state_dict(), 'trained_models/model.pth')
            self.send_trigger_update()


        