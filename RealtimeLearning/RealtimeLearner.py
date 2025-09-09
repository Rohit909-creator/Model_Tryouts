import torch
import torch.nn as nn
from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)

# data.target[[10, 25, 50]]
# array([0, 0, 1])

# load_iris()
print(x[[10, 50, 110]], y[[10, 50, 110]])


class SpikeLayer(nn.Module):
    
    def __init__(self, in_features:int, out_features:int, activation=False):
        super().__init__()
        self.weights = torch.randn((out_features, in_features))
        self.ltp = torch.randn((out_features, in_features)).fill_(0.3)

        self.activation = activation

        self.threshold = torch.tensor(0.5, dtype=torch.float32)
    def forward(self, x):
        
        weight = self.weights.T + self.ltp.T
        
        # update_matrix =  weight>= self.threshold
        out = x @ weight
        
        if self.activation:
            out = nn.functional.sigmoid(out) >= self.threshold
            print(f"Neuron Firings:\n{out}\n")
        return out.to(torch.float32)

    def predict(self, x):
        
        out = x @ (self.weights.T + self.ltp.T)
        return out
    

if __name__ == "__main__":
    
    input_size = len(x[0])
    
    layer = SpikeLayer(input_size, 6, activation=True)
    layer2 = SpikeLayer(6, 3)
    data = torch.tensor([x[0]], dtype=torch.float32)
    out = layer2(layer(data))
    # print(f"Output: {out}")
    # print(f"shape: {out.shape}\n")
    
    data = torch.tensor([x[50]], dtype=torch.float32)
    out = layer2(layer(data))
    # print(f"Output: {out}")
    # print(f"shape: {out.shape}\n")

    data = torch.tensor([x[110]], dtype=torch.float32)
    out = layer2(layer(data))
    # print(f"Output: {out}")
    # print(f"shape: {out.shape}\n")
    