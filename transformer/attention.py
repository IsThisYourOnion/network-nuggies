import torch
import torch.nn as nn
import math

# basic attention from https://arxiv.org/abs/1706.03762
#
class DotProductAttention(nn.Module):
    def __init__(self, dropout=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout or 0.0)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)  # Get the dimensionality of the query/key vectors

        scores = torch.matmul(query, key.transpose(-2, -1)) /  math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Normalize scores with softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, value)

        return output, attention_weights
