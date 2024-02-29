import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads  # Dimension of each head

        # Linear layers for query, key, and value projections
        self.query_proj = nn.Linear(d_model, num_heads * self.d_k)
        self.key_proj = nn.Linear(d_model, num_heads * self.d_k)
        self.value_proj = nn.Linear(d_model, num_heads * self.d_k)

        # Output linear layer
        self.output_proj = nn.Linear(num_heads * self.d_k, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Project Q, K, V into multiple heads
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Calculate scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) /  math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute the context vectors for each head
        context = torch.matmul(attention_weights, value)

        # Concatenate the output of all heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Final projection
        output = self.output_proj(context)

        return output, attention_weights
