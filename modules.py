# modules.py

import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (batch_size, in_channels, img_size, img_size)
        x = self.proj(x)  # (batch_size, embed_dim, num_patches^(1/2), num_patches^(1/2))
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.size()

        # Linear projection
        qkv = self.qkv(x)  # (batch_size, num_tokens, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv[0], qkv[1], qkv[2]  # Each: (batch_size, num_heads, num_tokens, head_dim)

        # Scaled dot-product attention
        energy = torch.matmul(queries, keys.transpose(-2, -1))  # (batch_size, num_heads, num_tokens, num_tokens)
        scaling = float(self.head_dim) ** -0.5
        energy = energy * scaling
        attention = torch.softmax(energy, dim=-1)  # (batch_size, num_heads, num_tokens, num_tokens)
        attention = self.dropout(attention)

        # Attention output
        out = torch.matmul(attention, values)  # (batch_size, num_heads, num_tokens, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, embed_dim)  # (batch_size, num_tokens, embed_dim)

        # Final linear layer
        out = self.fc_out(out)
        out = self.dropout(out)
        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Multi-Head Self-Attention with residual connection
        x = x + self.mhsa(self.norm1(x))

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=2,
                 embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, dropout=0.1, cls_token=True):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if cls_token else None

        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + (1 if cls_token else 0), embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        self.encoder = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )

        # Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)

        if self.cls_token is not None:
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
            x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)

        x = x + self.pos_embed  # Add positional embedding
        x = self.dropout(x)

        x = self.encoder(x)

        x = self.norm(x)

        if self.cls_token is not None:
            x = x[:, 0]  # Extract the cls token
        else:
            x = x.mean(dim=1)  # Global average pooling

        x = self.head(x)
        return x
