import os
import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple
from pathlib import Path
from dotenv import load_dotenv


# Add the parent directory to system path to import from playground
from model import SegformerTrainer

class SegformerClient(fl.client.NumPyClient):
    def __init__(self, trainer: SegformerTrainer):
        self.trainer = trainer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        # Get model parameters as a list of NumPy arrays
        state_dict = self.trainer.model.state_dict()
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy arrays
        params_dict = zip(self.trainer.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.trainer.model.load_state_dict(state_dict, strict=True)
        
    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters
        self.set_parameters(parameters)
        
        # Get training config
        batch_size = config["batch_size"]
        epochs = config["local_epochs"]
        
        # Train model
        self.trainer.config['batch_size'] = batch_size
        self.trainer.config['epochs'] = epochs

        # Train for one epoch
        train_loss = self.trainer.train_epoch()
        
        # Return updated model parameters and training results
        return self.get_parameters(config={}), len(self.trainer.train_loader), {
            "train_loss": float(train_loss),
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate model
        val_loss = self.trainer.validate()  # Assuming this returns the validation loss
        
        # Return evaluation results
        return float(val_loss), len(self.trainer.val_loader), {
            "val_loss": float(val_loss)
        }

def main(image_path: Path, gt_path: Path):
    # Define training configuration using your existing config
    config = {
        'img_path': image_path,
        'gt_path': gt_path,
        'num_classes': 14,
        'class_names': [
            'background', 'avalanche', 'building_undamaged', 'building_damaged',
            'cracks/fissure/subsidence', 'debris/mud//rock flow', 'fire/flare',
            'flood/water/river/sea', 'ice_jam_flow', 'lava_flow', 'person',
            'pyroclastic_flow', 'road/railway/bridge', 'vehicle'
        ],
        'batch_size': 8,
        'epochs': 10,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'min_lr': 1e-6,
        'model_name': 'nvidia/mit-b0',
        'val_split': 0.2,
        'save_dir': './checkpoints',
        'save_every': 5
    }
    
    # Initialize trainer
    trainer = SegformerTrainer(config)
    
    # Start Flower client
    client = SegformerClient(trainer)
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )

if __name__ == "__main__":
    load_dotenv()
    image_path = os.getenv('IMAGE_PATH')
    gt_path = os.getenv('GT_PATH')
    main(image_path, gt_path)
    # print(image_path, gt_path)
