from abc import ABC

import torch
from torch import nn

from bonfire.model.nn_models import MultipleInstanceNN


class AttentionNN(MultipleInstanceNN, ABC):

    name = "AttentionNN"

    def __init__(self, device, n_classes, n_expec_dims, encoder, embedding_classifier,
                 attn_n_heads, attn_d_in, attn_d_hid, attn_dropout):
        super().__init__(device, n_classes, n_expec_dims)
        self.encoder = encoder
        self.aggregator = MultiHeadAttentionBlock(attn_n_heads, attn_d_in, attn_d_hid, attn_dropout)
        self.embedding_classifier = embedding_classifier

    def _internal_forward(self, bags):
        bag_predictions = torch.zeros((len(bags), self.n_classes)).to(self.device)
        all_attention_values = []
        for i, instances in enumerate(bags):
            # Embed instances
            instances = instances.to(self.device)
            instance_embeddings = self.encoder(instances)

            # Classify instances and aggregate
            #  Instance interpretations is the attention value assigned to each instance in the bag
            bag_embedding, instance_attention_values = self.attention_aggregator(instance_embeddings)
            bag_prediction = self.embedding_classifier(bag_embedding)

            # Update outputs
            bag_predictions[i] = bag_prediction
            all_attention_values.append(instance_attention_values)
        return bag_predictions, all_attention_values


class AttentionBlock(nn.Module):

    def __init__(self, d_in, d_attn, dropout):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_in, d_attn, bias=False),
            nn.Tanh(),
            nn.Linear(d_attn, 1, bias=False),
            nn.Softmax(dim=0)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        attn = self.attention(x)
        attn = torch.transpose(attn, 1, 0)
        bag_embedding = torch.mm(attn, x)
        bag_embedding = self.dropout(bag_embedding)
        return bag_embedding, attn


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, n_heads, d_in, d_attn, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([AttentionBlock(d_in, d_attn, dropout=dropout) for _ in range(self.n_heads)])

    def forward(self, x):
        # Pass input through each head
        head_outs = [head(x) for head in self.heads]
        # Concatenate bag representations from each head
        bag_embedding = torch.cat([h[0] for h in head_outs], dim=1)
        # Stack attention outputs from each head
        attn = torch.stack([h[1].squeeze() for h in head_outs])
        return bag_embedding, attn
