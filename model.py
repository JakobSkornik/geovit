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


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d)))
                if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head : (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head**0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class GeoViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(GeoViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.LayerNorm(mlp_ratio * hidden_d),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class GeoViT(nn.Module):
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224', hidden_d=768, n_heads=12):
        super(GeoViT, self).__init__()

        # Load the pretrained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)

        # Or use timm to load the model
        # self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Replace the classifier with a new one (note that '768' is the dimension of the output tensor from ViT)
        self.vit.classifier = nn.Linear(hidden_d, 2)

    def forward(self, images):
        # Forward pass through the base ViT model
        outputs = self.vit(images)

        # We only need the last hidden state from the outputs
        last_hidden_state = outputs.last_hidden_state

        # Get the output of the classification token
        # The classification token corresponds to the first element in the sequence
        cls_output = last_hidden_state[:, 0]

        # Pass the output through the classifier
        logits = self.vit.classifier(cls_output)

        return logits
