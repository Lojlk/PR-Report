import torch
import torch.nn as nn
from torchvision import transforms

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)  # (batch_size, emb_size, num_patches ** 0.5, num_patches ** 0.5)
        x = x.flatten(2)  # (batch_size, emb_size, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, emb_size)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_size)
        self.msa = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(emb_size)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, emb_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ln = self.ln1(x)
        x_msa, _ = self.msa(x_ln, x_ln, x_ln)
        x = x + self.dropout(x_msa)
        x_ln = self.ln2(x)
        x_ffn = self.ffn(x_ln)
        x = x + self.dropout(x_ffn)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, emb_size=768, num_heads=12, num_layers=12, num_classes=2, ff_dim=3072, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + self.patch_embedding.num_patches, emb_size))
        self.transformer = nn.Sequential(
            *[TransformerBlock(emb_size, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.ln(x)
        return self.head(x[:, 0])
