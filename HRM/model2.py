import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# =============================================================================
# HELPER FUNCTIONS AND BASIC COMPONENTS
# =============================================================================

def trunc_normal_init_(tensor, std=1.0):
    """Initialize tensor with truncated normal distribution"""
    with torch.no_grad():
        tensor.normal_(0, std)
        # Truncate to [-2*std, 2*std]
        tensor.clamp_(-2*std, 2*std)    
    return tensor

def rms_norm(x, eps=1e-6):
    """Root Mean Square Layer Normalization"""
    # Calculate RMS
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return x / rms

class SwiGLU(nn.Module):
    """SwiGLU activation function: x * swish(Wx) * Vx"""
    def __init__(self, hidden_size, expansion=2.0):
        super().__init__()
        expanded_size = int(hidden_size * expansion)
        self.gate_proj = nn.Linear(hidden_size, expanded_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, expanded_size, bias=False)
        self.down_proj = nn.Linear(expanded_size, hidden_size, bias=False)
    
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))  # SiLU/Swish activation
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, seq_len=None):
        if seq_len is None:
            seq_len = self.max_seq_len
            
        # Create position indices
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        
        # Compute frequencies for each position
        freqs = torch.outer(t, self.inv_freq)
        
        # Create cos and sin components
        cos = freqs.cos()
        sin = freqs.sin()
        
        return cos, sin

def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary positional embedding to input tensor"""
    # Split into even and odd dimensions
    x1 = x[..., ::2]   # Even dimensions
    x2 = x[..., 1::2]  # Odd dimensions
    
    # Apply rotation
    rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1).flatten(-2)
    
    return rotated

class SimpleAttention(nn.Module):
    """Simplified Multi-Head Attention with RoPE"""
    def __init__(self, hidden_size, num_heads, causal=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, cos_sin=None):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if provided
        if cos_sin is not None:
            cos, sin = cos_sin
            if cos.dim() == 2:  # Add batch and head dimensions
                cos = cos.unsqueeze(0).unsqueeze(0)
                sin = sin.unsqueeze(0).unsqueeze(0)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed
        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(out)

# =============================================================================
# CORE MODEL COMPONENTS
# =============================================================================

@dataclass
class ModelState:
    """Holds the internal state of the reasoning model"""
    z_high: torch.Tensor    # High-level reasoning state
    z_low: torch.Tensor     # Low-level reasoning state
    steps: torch.Tensor     # Number of reasoning steps taken
    halted: torch.Tensor    # Whether reasoning has halted

class ReasoningBlock(nn.Module):
    """A single transformer block for reasoning"""
    def __init__(self, hidden_size, num_heads, expansion=2.0):
        super().__init__()
        self.attention = SimpleAttention(hidden_size, num_heads, causal=False)
        self.mlp = SwiGLU(hidden_size, expansion)
        self.eps = 1e-6
    
    def forward(self, x, cos_sin=None):
        # Post-norm architecture: Add & Norm
        # Self-attention with residual connection and RMS norm
        attn_out = self.attention(x, cos_sin)
        x = rms_norm(x + attn_out, self.eps)
        
        # MLP with residual connection and RMS norm  
        mlp_out = self.mlp(x)
        x = rms_norm(x + mlp_out, self.eps)
        
        return x

class ReasoningLevel(nn.Module):
    """A reasoning level with multiple transformer blocks"""
    def __init__(self, num_layers, hidden_size, num_heads, expansion=2.0):
        super().__init__()
        self.layers = nn.ModuleList([
            ReasoningBlock(hidden_size, num_heads, expansion) 
            for _ in range(num_layers)
        ])
    
    def forward(self, hidden_states, input_injection, cos_sin=None):
        # Add input injection (this allows information flow between levels)
        x = hidden_states + input_injection
        
        # Pass through all transformer blocks
        for layer in self.layers:
            x = layer(x, cos_sin)
        
        return x

# =============================================================================
# MAIN MODEL
# =============================================================================

class SimpleHierarchicalReasoningModel(nn.Module):
    """
    Simplified Hierarchical Reasoning Model with Adaptive Computation Time (ACT)
    
    Key ideas:
    1. Two-level hierarchy: High-level (strategic) and Low-level (detailed) reasoning
    2. Adaptive computation: Model decides when to stop reasoning
    3. Information flows: Low->High and High->Low
    """
    
    def __init__(self, 
                 vocab_size=1000,
                 seq_len=512, 
                 hidden_size=512,
                 num_heads=8,
                 high_layers=2,
                 low_layers=4,
                 high_cycles=2,
                 low_cycles=3,
                 max_reasoning_steps=5):
        super().__init__()
        
        # Store config
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.max_reasoning_steps = max_reasoning_steps
        self.high_cycles = high_cycles
        self.low_cycles = low_cycles
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_scale = math.sqrt(hidden_size)
        
        # Positional embeddings (RoPE)
        self.rotary_emb = RotaryEmbedding(hidden_size // num_heads, seq_len)
        
        # Two reasoning levels
        self.high_level = ReasoningLevel(high_layers, hidden_size, num_heads)
        self.low_level = ReasoningLevel(low_layers, hidden_size, num_heads)
        
        # Output heads
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)  # Language modeling
        self.halt_head = nn.Linear(hidden_size, 1, bias=True)  # Halting decision
        
        # Initial states for reasoning (learned parameters)
        self.high_init = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.low_init = nn.Parameter(torch.randn(hidden_size) * 0.02)
        
        # Initialize halt head to prefer continuing initially
        with torch.no_grad():
            self.halt_head.weight.zero_()
            self.halt_head.bias.fill_(-2.0)  # Bias toward continuing
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_initial_state(self, batch_size, device):
        """Create initial reasoning state"""
        return ModelState(
            z_high=torch.zeros(batch_size, self.seq_len, self.hidden_size, device=device),
            z_low=torch.zeros(batch_size, self.seq_len, self.hidden_size, device=device),
            steps=torch.zeros(batch_size, dtype=torch.long, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device)  # Start halted
        )
    
    def reset_reasoning_state(self, state: ModelState, reset_mask: torch.Tensor):
        """Reset reasoning state for sequences that have halted"""
        batch_size, seq_len, hidden_size = state.z_high.shape
        
        # Expand initial states to match batch and sequence dimensions
        high_init = self.high_init.expand(batch_size, seq_len, hidden_size)
        low_init = self.low_init.expand(batch_size, seq_len, hidden_size)
        
        # Reset states where mask is True
        reset_mask_expanded = reset_mask.view(-1, 1, 1)
        new_z_high = torch.where(reset_mask_expanded, high_init, state.z_high)
        new_z_low = torch.where(reset_mask_expanded, low_init, state.z_low)
        
        return ModelState(
            z_high=new_z_high,
            z_low=new_z_low,
            steps=torch.where(reset_mask, torch.zeros_like(state.steps), state.steps),
            halted=state.halted
        )
    
    def reasoning_step(self, state: ModelState, input_embeddings: torch.Tensor):
        """Perform one step of hierarchical reasoning"""
        # Get positional embeddings
        cos_sin = self.rotary_emb(self.seq_len)
        
        z_high, z_low = state.z_high.clone(), state.z_low.clone()
        
        # Hierarchical reasoning cycles
        for h_step in range(self.high_cycles):
            for l_step in range(self.low_cycles):
                # Low-level reasoning: gets input from high-level + original input
                if not (h_step == self.high_cycles - 1 and l_step == self.low_cycles - 1):
                    z_low = self.low_level(z_low, z_high + input_embeddings, cos_sin)
            
            # High-level reasoning: gets input from low-level
            if h_step < self.high_cycles - 1:
                z_high = self.high_level(z_high, z_low, cos_sin)
        
        # Final step with gradients enabled for learning
        z_low = self.low_level(z_low, z_high + input_embeddings, cos_sin)
        z_high = self.high_level(z_high, z_low, cos_sin)
        
        return ModelState(
            z_high=z_high,
            z_low=z_low,
            steps=state.steps + 1,
            halted=state.halted
        )
    
    def forward(self, input_ids: torch.Tensor, max_reasoning_steps: Optional[int] = None):
        """
        Forward pass with adaptive reasoning
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_reasoning_steps: Override default max steps
        
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            num_steps: Number of reasoning steps taken [batch_size]
        """
        if max_reasoning_steps is None:
            max_reasoning_steps = self.max_reasoning_steps
            
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create input embeddings
        input_embeddings = self.embed_tokens(input_ids) * self.embed_scale
        
        # Initialize reasoning state
        state = self.create_initial_state(batch_size, device)
        
        # Adaptive reasoning loop
        for step in range(max_reasoning_steps):
            # Reset states for sequences that have halted (new problems)
            state = self.reset_reasoning_state(state, state.halted)
            
            # Perform reasoning step
            state = self.reasoning_step(state, input_embeddings)
            
            # Decide whether to halt (only if not training or if enabled)
            if not self.training:
                # During inference, use halting mechanism
                halt_logits = self.halt_head(state.z_high[:, 0])  # Use first token for decision
                should_halt = torch.sigmoid(halt_logits).squeeze(-1) > 0.5
                
                # Always halt at max steps
                at_max_steps = state.steps >= max_reasoning_steps
                state = ModelState(
                    z_high=state.z_high,
                    z_low=state.z_low, 
                    steps=state.steps,
                    halted=should_halt | at_max_steps
                )
                
                # If all sequences have halted, break early
                if state.halted.all():
                    break
        
        # Generate final output
        logits = self.lm_head(state.z_high)
        
        return logits, state.steps

# =============================================================================
# SIMPLE EXAMPLE USAGE
# =============================================================================

def demo_model():
    """Demonstrate the model with random inputs"""
    print("Creating Hierarchical Reasoning Model...")
    
    # Model configuration
    model = SimpleHierarchicalReasoningModel(
        vocab_size=1000,
        seq_len=64,      # Smaller for demo
        hidden_size=256, # Smaller for demo
        num_heads=4,
        high_layers=2,
        low_layers=3,
        high_cycles=2,
        low_cycles=2,
        max_reasoning_steps=3
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create some random input
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Sample input: {input_ids[0, :10].tolist()}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, num_steps = model(input_ids)
    
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Number of reasoning steps: {num_steps.tolist()}")
    print(f"Predicted tokens (first 10): {logits[0, :10].argmax(-1).tolist()}")
    
    return model, input_ids, logits

if __name__ == "__main__":
    model, inputs, outputs = demo_model()
    
    print("\n" + "="*50)
    print("MODEL READY FOR EXPERIMENTATION!")
    print("="*50)
    print("\nTry these experiments:")
    print("1. model.training = True  # Enable training mode")  
    print("2. Change max_reasoning_steps")
    print("3. Modify input tokens")
    print("4. Check model.high_level.layers[0].attention.scale")
    print("5. Inspect internal states during reasoning")