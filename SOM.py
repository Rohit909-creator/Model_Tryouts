import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import torch.nn.functional as F
from sklearn.datasets import load_iris
from torchmetrics import Accuracy

acc = Accuracy(task='multiclass', num_classes=3)
# Fixed SOM Implementation
class SOM(nn.Module):
    def __init__(self, input_size, som_shape, learning_rate=0.5):
        super().__init__()
        
        self.som_shape = som_shape
        self.input_size = input_size
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        
        # Initialize weights with better initialization
        self.weights = nn.Parameter(torch.randn(som_shape[0], som_shape[1], input_size) * 0.1)
        
        # Initialize neighborhood function parameters
        self.initial_radius = max(som_shape) / 2
        self.radius = self.initial_radius
        
    def get_bmu(self, x):
        """Find Best Matching Unit for each input"""
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, input_size]
        weights_expanded = self.weights.unsqueeze(0)  # [1, height, width, input_size]
        
        # Calculate distances (using Euclidean distance)
        distances = torch.sum((x_expanded - weights_expanded) ** 2, dim=-1)
        
        # Find BMU indices
        distances_flat = distances.view(x.size(0), -1)
        bmu_indices = torch.argmin(distances_flat, dim=1)
        
        # Convert flat indices to 2D coordinates
        bmu_coords = torch.stack([
            bmu_indices // self.som_shape[1],
            bmu_indices % self.som_shape[1]
        ], dim=1)
        
        return bmu_coords, bmu_indices
    
    def get_neighborhood(self, bmu_coords):
        """Calculate neighborhood function"""
        # Create coordinate grids
        height, width = self.som_shape
        y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        coords = torch.stack([y_coords, x_coords], dim=-1).float()
        
        # Calculate distances from BMU
        bmu_expanded = bmu_coords.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, 2]
        coord_expanded = coords.unsqueeze(0)  # [1, height, width, 2]
        
        distances = torch.norm(bmu_expanded - coord_expanded, dim=-1)
        
        # Apply Gaussian neighborhood function
        neighborhood = torch.exp(-(distances ** 2) / (2 * self.radius ** 2))
        
        return neighborhood
    
    def update_weights(self, x, epoch, total_epochs):
        """Update weights using SOM learning rule"""
        # Update learning rate and radius
        self.learning_rate = self.initial_learning_rate * (1 - epoch / total_epochs)
        self.radius = self.initial_radius * (1 - epoch / total_epochs)
        
        # Get BMU
        bmu_coords, _ = self.get_bmu(x)
        
        # Get neighborhood function
        neighborhood = self.get_neighborhood(bmu_coords)
        
        # Update weights
        with torch.no_grad():
            for i in range(x.size(0)):
                # Calculate weight updates
                x_expanded = x[i].unsqueeze(0).unsqueeze(0)  # [1, 1, input_size]
                neighborhood_expanded = neighborhood[i].unsqueeze(-1)  # [height, width, 1]
                
                # Apply updates
                weight_diff = x_expanded - self.weights
                update = self.learning_rate * neighborhood_expanded * weight_diff
                self.weights += update
    
    def predict(self, x):
        """Predict cluster for input"""
        _, bmu_indices = self.get_bmu(x)
        return bmu_indices
    
    def get_weights_2d(self):
        """Get weights as 2D array for visualization"""
        return self.weights.view(-1, self.input_size).detach().numpy()


# Demonstration and comparison
def demonstrate_unsupervised_learning():
    # Create data
    # data_tensor = torch.tensor([
    #     [1, 0, 0],  # Class 1
    #     [0, 1, 0],  # Class 1  
    #     [0, 0, 1],  # Class 1
    #     [1, 1, 0],  # Class 2
    #     [1, 0, 1],  # Class 2
    #     [0, 1, 1]   # Class 2
    # ], dtype=torch.float32)
    
    # # True labels for visualization
    # true_labels = [0, 0, 0, 1, 1, 1]
    
    x, true_labels = load_iris(return_X_y=True)
    data_tensor = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(true_labels)
    print(data_tensor.shape, y.shape)
    
    print("=== SOM Training ===")
    # Train SOM
    som = SOM(input_size=4, som_shape=(3, 1), learning_rate=0.5)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        som.update_weights(data_tensor, epoch, num_epochs)
    
    # Test SOM
    som_predictions = som.predict(data_tensor)
    print(f"SOM predictions: {som_predictions}")
    
    accuracy = acc(som_predictions, y)
    print(f"Accuracy: {accuracy}")
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: SOM weights
    plt.subplot(1, 3, 1)
    som_weights = som.get_weights_2d()
    plt.imshow(som_weights, cmap='viridis', aspect='auto')
    plt.title('SOM Weights')
    plt.xlabel('Input Features')
    plt.ylabel('SOM Neurons')
    plt.colorbar()
    
    return som

# Run the demonstration
if __name__ == "__main__":
    som = demonstrate_unsupervised_learning()