import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

class HebbianSpikeLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, learning_rate=0.01, 
                 competition_strength=0.5, activation=False):
        super().__init__()
        
        # Initialize weights with small random values
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        # LTP/LTD strength - this will be updated during learning
        self.ltp_strength = torch.zeros(out_features, in_features)
        
        # Individual thresholds for each neuron (adaptive)
        self.thresholds = nn.Parameter(torch.full((out_features,), 0.5))
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.competition_strength = competition_strength
        self.activation = activation
        
        # For tracking neuron activity history
        self.activity_history = torch.zeros(out_features)
        self.homeostasis_rate = 0.001
        
    def forward(self, x, learn=True):
        # Compute membrane potentials
        membrane_potential = torch.matmul(x, (self.weights + self.ltp_strength).T)
        
        # Apply competition (lateral inhibition)
        if self.competition_strength > 0:
            # Find the maximum activation for each sample
            max_vals, max_indices = torch.max(membrane_potential, dim=1, keepdim=True)
            # Create inhibition mask
            inhibition = torch.zeros_like(membrane_potential)
            inhibition.scatter_(1, max_indices, max_vals * self.competition_strength)
            membrane_potential = membrane_potential - inhibition + inhibition.gather(1, max_indices)
        
        # Determine which neurons fire
        if self.activation:
            # spikes = membrane_potential >= self.thresholds.unsqueeze(0)
            spikes = nn.functional.relu(membrane_potential)
            print(f"Membrane potentials: {membrane_potential}")
            print(f"Thresholds: {self.thresholds}")
            print(f"Neuron Firings:\n{spikes}\n")
            
            if learn:
                self.hebbian_update(x, spikes, membrane_potential)
                
            return spikes.float()
        else:
            return membrane_potential
    
    def hebbian_update(self, inputs, spikes, membrane_potential):
        """Update weights based on Hebbian learning rule"""
        batch_size = inputs.shape[0]
        
        for i in range(batch_size):
            input_vec = inputs[i].unsqueeze(0)  # Shape: (1, in_features)
            spike_vec = spikes[i].unsqueeze(1)  # Shape: (out_features, 1)
            
            # Hebbian rule: strengthen connections between active input and output
            hebbian_update = torch.matmul(spike_vec, input_vec) * self.learning_rate
            
            # Apply LTP (strengthen) where both pre and post are active
            self.ltp_strength += hebbian_update
            
            # Update thresholds based on activity (homeostasis)
            self.activity_history += spikes[i].float()
            
        # Homeostatic threshold adjustment
        avg_activity = self.activity_history / (batch_size + 1e-8)
        target_activity = 0.3  # Target 30% activation
        threshold_adjustment = (avg_activity - target_activity) * self.homeostasis_rate
        
        with torch.no_grad():
            self.thresholds += threshold_adjustment
            self.thresholds.clamp_(0.1, 2.0)  # Keep thresholds in reasonable range
        
        # Decay activity history
        self.activity_history *= 0.9
    
    def reset_learning(self):
        """Reset learning-related parameters"""
        self.ltp_strength.fill_(0.0)
        self.activity_history.fill_(0.0)
        with torch.no_grad():
            self.thresholds.fill_(0.5)

class SpikingNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = HebbianSpikeLayer(input_size, hidden_size, 
                                       learning_rate=0.05, 
                                       competition_strength=0.3, 
                                       activation=True)
        self.layer2 = HebbianSpikeLayer(hidden_size, output_size, 
                                       learning_rate=0.02, 
                                       competition_strength=0.5, 
                                       activation=False)
    
    def forward(self, x, learn=True):
        hidden = self.layer1(x, learn=learn)
        output = self.layer2(hidden, learn=learn)
        return output
    
    def predict(self, x):
        """Prediction without learning"""
        return self.forward(x, learn=False)

def train_and_test():
    # Load and preprocess data
    x, y = load_iris(return_X_y=True)
    
    # Normalize the data
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x)
    
    # Convert to tensors
    x_tensor = torch.tensor(x_normalized, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Create network
    input_size = x_tensor.shape[1]
    hidden_size = 9  # 3 neurons per class
    output_size = 3
    
    network = SpikingNeuralNetwork(input_size, hidden_size, output_size)
    
    print("=== Training Phase ===")
    # Train on representative samples from each class
    train_indices = [10, 25, 40, 50, 65, 80, 110, 125, 140]  # 3 per class
    
    # Multiple training epochs
    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}")
        for idx in train_indices:
            sample = x_tensor[idx].unsqueeze(0)
            label = y_tensor[idx]
            print(f"Training on sample {idx}, class {label}")
            output = network(sample, learn=True)
            print(f"Output: {output}")
    
    print("\n=== Testing Phase ===")
    # Test on the same samples to see if patterns are learned
    test_indices = [10, 50, 110]
    
    for idx in test_indices:
        sample = x_tensor[idx].unsqueeze(0)
        label = y_tensor[idx]
        print(f"\nTesting sample {idx}, true class: {label}")
        output = network.predict(sample)
        print(f"Network output: {output}")
        predicted_class = torch.argmax(output, dim=1)
        print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    train_and_test()