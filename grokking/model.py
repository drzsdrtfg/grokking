import torch
from torch import nn, Tensor
from einops import rearrange, repeat

class DecoderBlock(torch.nn.Module):
    def __init__(self, dim_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Ensure dim_model is divisible by n_heads for better parallel processing
        assert dim_model % n_heads == 0, "dim_model must be divisible by n_heads"
        
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads)
        self.self_attn_norm = nn.LayerNorm(dim_model)
        
        # Wider FFN with SwiGLU activation for better representation capacity
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 8),  # Increased width
            nn.SiLU(),  # SwiGLU-like activation
            nn.Dropout(dropout),
            nn.Linear(dim_model * 8, dim_model)
        )
        self.ffn_norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        # Causal attention mask
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        
        # Multi-head attention with residual connection
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        attn_out = self.dropout(attn_out)
        x = self.self_attn_norm(x + attn_out)
        
        # FFN with residual connection
        ffn_out = self.dropout(self.ffn(x))
        x = self.ffn_norm(x + ffn_out)
        
        return x

class Transformer(torch.nn.Module):
    def __init__(self, num_layers: int, dim_model: int, num_heads: int, num_tokens: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        
        # Token and position embeddings
        self.token_embeddings = nn.Embedding(num_tokens, dim_model)
        self.position_embeddings = nn.Embedding(seq_len, dim_model)
        
        # Initialize embeddings with small values
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            DecoderBlock(dim_model, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(dim_model)
        self.output = nn.Linear(dim_model, num_tokens)
        
        # Initialize output layer with zeros for better training dynamics
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: Tensor):
        batch_size, context_len = inputs.shape
        
        # Get embeddings
        token_embedding = self.token_embeddings(inputs)
        positions = torch.arange(context_len, device=inputs.device)
        positions = repeat(positions, 'p -> b p', b=batch_size)
        position_embedding = self.position_embeddings(positions)
        
        # Combine embeddings
        x = token_embedding + position_embedding
        x = rearrange(x, 'b s d -> s b d')
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply output head
        x = self.norm(x)
        x = self.output(x)
        
        return x