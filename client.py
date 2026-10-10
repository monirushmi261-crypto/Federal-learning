import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from explainability import XAIAuditor

class FederatedClient:
    def __init__(self, client_id, dataloader, model_cls, device):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.model = model_cls().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def set_weights(self, global_weights):
        self.model.load_state_dict(global_weights)

    def get_weights(self):
        return self.model.state_dict()

    def train(self, epochs=2, lr=0.01):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        total_loss, correct, total = 0.0, 0, 0

        for _ in range(epochs):
            for X, y in self.dataloader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(X)
                loss = self.criterion(out, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * X.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return total_loss / total, correct / total

    def evaluate_explainability(self, reference_data):
        auditor = XAIAuditor(self.model, reference_data)
        eval_batch = next(iter(self.dataloader))[0].cpu().numpy()[:20]
        return auditor.compute_shap_values(eval_batch)
