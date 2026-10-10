from database import init_db
from data_pipeline import generate_non_iid_data
from client import FederatedClient
from server import FederatedServer
from model import ClassifierNN, get_device

def run_federated_pipeline(num_rounds=10, num_clients=5):
    print("[+] Initializing Persistent Audit Database...")
    init_db()

    print("[+] Generating Non-IID Partitioned Data Distributions...")
    client_loaders, _, ref_data = generate_non_iid_data(num_clients=num_clients)

    device = get_device()
    print(f"[+] Launching FL Protocol on Execution Unit: {device}")

    server = FederatedServer()
    clients = [FederatedClient(cid, loader, ClassifierNN, device) for cid, loader in client_loaders.items()]

    for r in range(1, num_rounds + 1):
        # Sync global weights to local clients
        global_weights = server.global_model.state_dict()
        for client in clients:
            client.set_weights(global_weights)

        acc, cos_sim = server.audit_and_log(r, clients, ref_data)
        print(f"| Round {r:02d}/{num_rounds:02d} | Mean Acc: {acc:.4f} | XAI Cosine Consistency: {cos_sim:.4f} |")

    print("[+] FL Execution Sequence Complete. Database populated.")

if __name__ == "__main__":
    run_federated_pipeline(num_rounds=15, num_clients=5)
