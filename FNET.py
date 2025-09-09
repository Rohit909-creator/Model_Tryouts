import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
    def forward(self, x):
        # SwiGLU: Swish(xW1) ⊙ (xW3) W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # More numerically stable RMSNorm
        # Calculate mean of squares
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        # RMS normalization
        rms = torch.sqrt(mean_square + self.eps)
        return self.weight * x / rms

class FnetBlock(nn.Module):
    
    def __init__(self, embed_dim):
        super().__init__()
        self.rmsnorm1 = RMSNorm(embed_dim)
        
        self.rmsnorm2 = RMSNorm(embed_dim)
        
        # self.mlp = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim*2),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim*2, embed_dim*2),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim*2, embed_dim)
        # )
        
        self.mlp = SwiGLU(embed_dim, embed_dim*4)
        
    def forward(self, x):
        
        out = x + torch.fft.fft(self.rmsnorm1(x), dim=1).real
        out = out + self.mlp(self.rmsnorm2((out)))
        return out
    
    
class FNET(nn.Module):
    
    def __init__(self, embed_dim, context_length, vocab_size, num_layers=3, lr=0.0001):
        super().__init__()
        
        self.context_length = context_length
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        self.pos_embeddings = nn.Embedding(context_length, embed_dim)
        
        self.blocks = nn.ModuleList([FnetBlock(embed_dim) for _ in range(num_layers)])
        
        self.norm = RMSNorm(embed_dim)
        
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)
        
        print(f"self.output.weight.shape: {self.output.weight.shape}")
        
        print(f"self.word_embeddings.weight.shape: {self.word_embeddings.weight.shape}")
        
        self.output.weight = self.word_embeddings.weight
        
    def forward(self, input_ids, attention_mask:Optional[torch.tensor]=None):
        
        embs = self.word_embeddings(input_ids) + self.pos_embeddings(torch.arange(0, self.context_length).to(input_ids.device))
        
        if attention_mask:
            attention_mask = torch.tril(torch.ones((self.context_length, self.context_length), device=input_ids.device))
            mask = attention_mask.unsqueeze(-1).expand_as(embs)
            # print(mask)
            embs = embs*mask
            
        for layer in self.blocks:
            embs = layer(embs)
            
        embs = self.norm(embs)
        
        logits = self.output(embs)
        return logits
    
        
        
        
        
if __name__ == "__main__":
    
    x = torch.randn((1, 5, 512))
    
    block = FnetBlock(512)
    
    out = block(x)
    print(out.shape)
    
    input_ids = torch.randint(0, 20002, (1, 10))
    
    model = FNET(512, 10, 20002)
    
    out = model(input_ids)
    print(out.shape)