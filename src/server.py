import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
import numpy as np
from dotenv import load_dotenv
import os

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy by number of examples
    accuracies = [num_examples * m["val_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    return {"val_loss": sum(accuracies) / sum(examples)}

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 8,
        "local_epochs": 1,  # local epochs for each client
        "current_round": server_round,
    }
    return config

# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,                                  # Sample 100% of available clients for training
    fraction_evaluate=1.0,                             # Sample 100% of available clients for evaluation
    min_fit_clients=2,                                 # Never sample less than 2 clients for training
    min_evaluate_clients=2,                            # Never sample less than 2 clients for evaluation
    min_available_clients=2,                           # Wait until at least 2 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,  # Custom aggregation function
    on_fit_config_fn=fit_config,                       # Configuration function for fit
)
if __name__ == "__main__":

    # Load the .env file
    load_dotenv()

    server_address = os.getenv("SERVER_ADDRESS")
    server_port = os.getenv("SERVER_PORT")


    # print(f"Server address: {server_address}:{server_port}")
    # Start server
    fl.server.start_server(
        server_address=f"{server_address}:{server_port}",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
