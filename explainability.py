import shap
import torch
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy

class XAIAuditor:
    def __init__(self, model, background_data):
        self.device = next(model.parameters()).device
        self.background = torch.tensor(background_data, dtype=torch.float32).to(self.device)
        self.model = model

    def compute_shap_values(self, eval_data):
        self.model.eval()
        eval_tensor = torch.tensor(eval_data, dtype=torch.float32).to(self.device)
        explainer = shap.DeepExplainer(self.model, self.background)
        shap_vals = explainer.shap_values(eval_tensor, check_additivity=False)
        
        if isinstance(shap_vals, list):
            # Binary classification -> take target class 1
            shap_vals = shap_vals[1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            if shap_vals.shape[-1] == 2:
                shap_vals = shap_vals[:, :, 1]
            elif shap_vals.shape[0] == 2:
                shap_vals = shap_vals[1]
        return np.mean(np.abs(shap_vals), axis=0) # Global feature importance vector

    @staticmethod
    def compute_cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    @staticmethod
    def compute_kl_divergence(vec1, vec2, epsilon=1e-8):
        # Softmax normalization to build probability distribution
        p = np.exp(vec1) / np.sum(np.exp(vec1)) + epsilon
        q = np.exp(vec2) / np.sum(np.exp(vec2)) + epsilon
        return float(entropy(p, q))
