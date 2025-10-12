import torch
import torch.nn as nn
import math

class BERT(nn.Module):
    def __init__(
    		self, vocab_size, embed_size, num_layers, num_heads,
    		ff_hidden_size, max_len=512, dropout=0.1):
        super().__init__()

        self.embeddings = BertEmbeddings(vocab_size, embed_size, max_len, dropout)

        self.encoders = nn.ModuleList(
            [EncoderBlock(embed_size, num_heads, ff_hidden_size, dropout) for _ in range(num_layers)]
        )

    def forward(self, input_ids, segment_ids, mask=None):
        x = self.embeddings(input_ids, segment_ids)

        for encoder in self.encoders:
            x = encoder(x, mask)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_size),
            nn.GELU(), # BERT uses GELU activation
            nn.Linear(ff_hidden_size, embed_size)
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention sub-layer
        attn_output = self.attention(x, x, x, mask)
        # Residual connection and normalization
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        # Residual connection and normalization
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len=512, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size

        # Token embedding layer
        self.token_embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # Position embedding layer
        self.position_embeddings = nn.Embedding(max_len, embed_size)

        # Segment embedding layer (for sentence A/B)
        self.segment_embeddings = nn.Embedding(2, embed_size)

        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids):
        seq_length = input_ids.size(1)

        # Create position IDs (0, 1, 2, ...)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Look up the embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        segment_embeds = self.segment_embeddings(segment_ids)

        # Sum the embeddings
        embeddings = token_embeds + position_embeds + segment_embeds

        # Normalize and apply dropout
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == self.embed_size, "Embedding size must be divisible by num_heads"

        # Linear layers to create Q, K, V
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linearly project and reshape for multi-head
        # Original: (batch_size, seq_len, embed_size)
        # Reshaped: (batch_size, num_heads, seq_len, head_dim)
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 1. Calculate scores (Q * K^T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 2. Apply mask (if provided) to ignore certain positions (e.g., padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # Fill with a very small number

        # 3. Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 4. Multiply weights by V
        context = torch.matmul(attention_weights, v)

        # 5. Concatenate heads and apply final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)
        output = self.out_linear(context)

        return output


