import torch
import copy
import numpy as np
from model import ClassifierNN, get_device
from explainability import XAIAuditor
from database import log_client_metrics, log_global_metrics

class FederatedServer:
    def __init__(self, num_features=10):
        self.device = get_device()
        self.global_model = ClassifierNN(input_dim=num_features).to(self.device)

    def aggregate(self, client_weights_list):
        avg_weights = copy.deepcopy(client_weights_list[0])
        for key in avg_weights.keys():
            for i in range(1, len(client_weights_list)):
                avg_weights[key] += client_weights_list[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(client_weights_list))
        self.global_model.load_state_dict(avg_weights)

    def audit_and_log(self, round_num, clients, reference_data):
        global_auditor = XAIAuditor(self.global_model, reference_data)
        global_shap = global_auditor.compute_shap_values(reference_data[:20])

        client_losses, client_accs, cos_sims, kl_divs = [], [], [], []

        for client in clients:
            loss, acc = client.train()
            client_shap = client.evaluate_explainability(reference_data)

            cos_sim = XAIAuditor.compute_cosine_similarity(global_shap, client_shap)
            kl_div = XAIAuditor.compute_kl_divergence(global_shap, client_shap)

            log_client_metrics(round_num, client.client_id, loss, acc, cos_sim, kl_div, client_shap)

            client_losses.append(loss)
            client_accs.append(acc)
            cos_sims.append(cos_sim)
            kl_divs.append(kl_div)

        # Aggregate Global
        self.aggregate([c.get_weights() for c in clients])
        log_global_metrics(round_num, float(np.mean(client_losses)), float(np.mean(client_accs)), 
                           float(np.mean(cos_sims)), float(np.mean(kl_divs)))
        
        return np.mean(client_accs), np.mean(cos_sims)
