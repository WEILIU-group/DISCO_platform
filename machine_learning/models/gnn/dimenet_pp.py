from .base_gnn import BaseGNNModel
import torch
from torch_geometric.nn import DimeNetPlusPlus as PyGDimeNetPP
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import List, Dict

class DimeNetPPModel(BaseGNNModel):
    def __init__(self, params=None):
        super().__init__(params)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default params
        hidden_channels = self.params.get('hidden_channels', 128)
        out_channels = self.params.get('out_channels', 1)
        num_blocks = self.params.get('num_blocks', 4)
        num_bilinear = self.params.get('num_bilinear', 8)
        num_spherical = self.params.get('num_spherical', 7)
        num_radial = self.params.get('num_radial', 6)
        cutoff = self.params.get('cutoff', 5.0)
        
        self.model = PyGDimeNetPP(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            num_bilinear=num_bilinear,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.get('lr', 0.001))
        self.criterion = torch.nn.MSELoss()
        self.epochs = self.params.get('epochs', 100)
        self.batch_size = self.params.get('batch_size', 32)

    def _convert_to_pyg_data(self, dataset: List[Dict]) -> List[Data]:
        data_list = []
        for item in dataset:
            z = torch.tensor(item['atomic_numbers'], dtype=torch.long)
            pos = torch.tensor(item['positions'], dtype=torch.float)
            y = torch.tensor([item['target']], dtype=torch.float)
            data = Data(z=z, pos=pos, y=y)
            data_list.append(data)
        return data_list

    def train(self, dataset: List[Dict], val_dataset: List[Dict] = None):
        train_data = self._convert_to_pyg_data(dataset)
        loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(batch.z, batch.pos, batch.batch)
                loss = self.criterion(out.view(-1), batch.y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")

    def predict(self, dataset: List[Dict]) -> np.ndarray:
        data_list = self._convert_to_pyg_data(dataset)
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.z, batch.pos, batch.batch)
                preds.extend(out.view(-1).cpu().numpy())
        
        return np.array(preds)
