import numpy as np
import torch
import torch.nn as nn
from transformers import ViTModel


def patchify(images, patch_size):
    n, c, h, w = images.shape
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(n, -1, c * patch_size * patch_size)
    return patches


def get_positional_embeddings(sequence_length, hidden_size, device="cuda"):
    position_ids = torch.arange(sequence_length, dtype=torch.long, device=device)
    position_embeddings = nn.Embedding(sequence_length, hidden_size)
    return position_embeddings(position_ids)


class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.self_attention = nn.MultiheadAttention(d, n_heads)

    def forward(self, x):
        x = x.unsqueeze(0)  # Add a dimension for sequence length
        attn_output, _ = self.self_attention(x, x, x)
        attn_output = attn_output.squeeze(0)  # Remove the added dimension
        return attn_output


class GeoViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(GeoViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.linear1 = nn.Linear(hidden_d, hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )
        self.linear2 = nn.Linear(hidden_d, hidden_d)

    def forward(self, x):
        out = self.linear1(self.norm1(x))
        out = out + self.mhsa(out)
        out = self.linear2(self.norm2(out))
        out = out + self.mlp(out)
        return out


class ExtendedGeoViT(nn.Module):
    def __init__(
        self,
        pretrained_model_name="google/vit-base-patch16-224",
        hidden_d=768,
        n_heads=12,
        n_transformer_blocks=3,
    ):
        super(ExtendedGeoViT, self).__init__()

        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.vit.classifier = nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_d)
        self.dropout = nn.Dropout(0.1)
        self.additional_blocks = nn.ModuleList(
            [
                self._make_block(hidden_d // (2**i), n_heads)
                for i in range(n_transformer_blocks)
            ]
        )
        self.final_classifier = nn.Linear(hidden_d // (2**n_transformer_blocks), 2)

    def _make_block(self, hidden_d, n_heads):
        layers = [
            nn.Linear(hidden_d, hidden_d // 2),
            GeoViTBlock(hidden_d // 2, n_heads),
        ]
        return nn.Sequential(*layers)

    def forward(self, images):
        outputs = self.vit(images)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0]
        normalized_logits = self.layer_norm(cls_output)
        dropped_logits = self.dropout(normalized_logits)
        extended_features = dropped_logits
        for block in self.additional_blocks:
            extended_features = block(extended_features)
        final_logits = self.final_classifier(extended_features)

        return final_logits
