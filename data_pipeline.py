import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np

def generate_non_iid_data(num_clients=5, num_samples=2000, num_features=10, beta=0.5):
    X, y = make_classification(
        n_samples=num_samples, n_features=num_features, n_informative=8, 
        n_redundant=2, n_classes=2, random_state=42
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Dirichlet distribution for class drift / Non-IID partitioning
    min_size = 0
    K = 2
    N = y.shape[0]
    client_indices = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    client_data = {}
    for i in range(num_clients):
        idxs = idx_batch[i]
        X_c, y_c = torch.tensor(X[idxs], dtype=torch.float32), torch.tensor(y[idxs], dtype=torch.long)
        dataset = TensorDataset(X_c, y_c)
        client_data[i] = DataLoader(dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return client_data, test_loader, X[:100]  # Return baseline reference sample for SHAP
