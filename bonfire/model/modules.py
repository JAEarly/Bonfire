import torch
from torch import nn


class ConvBlock(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super().__init__()
        conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding)
        relu = nn.ReLU()
        pool = nn.MaxPool2d(kernel_size=2)
        self.block = nn.Sequential(conv, relu, pool)

    def forward(self, x):
        return self.block(x)


class FullyConnectedBlock(nn.Module):

    def __init__(self, d_in, d_out, activation_func=nn.ReLU(), dropout=0):
        super().__init__()
        layers = [nn.Linear(d_in, d_out)]
        if activation_func is not None:
            layers.append(activation_func)
        if dropout != 0:
            layers.append(nn.Dropout(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class FullyConnectedStack(nn.Module):

    def __init__(self, d_in, ds_hid, d_out, activation_func=nn.ReLU(), final_activation_func=nn.ReLU(), dropout=0):
        super().__init__()
        self.d_in = d_in
        self.ds_hid = ds_hid
        self.d_out = d_out
        self.n_blocks = len(ds_hid) + 1
        blocks = []
        for i in range(self.n_blocks):
            in_size = d_in if i == 0 else ds_hid[i - 1]
            out_size = d_out if i == self.n_blocks - 1 else ds_hid[i]
            block_activation = final_activation_func if i == self.n_blocks - 1 else activation_func
            block_dropout = 0 if i == self.n_blocks - 1 else dropout
            blocks.append(FullyConnectedBlock(in_size, out_size,
                                              activation_func=block_activation,
                                              dropout=block_dropout))
        self.stack = nn.Sequential(*blocks)

    def forward(self, x):
        return self.stack(x)


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


class GNNConvBlock(nn.Module):

    def __init__(self, in_size, out_size, conv_clz, activation_func=nn.ReLU(), dropout=0):
        super().__init__()
        # Can't use sequential here as we have to pass two args through conv, but only one through relu and dropout
        self.conv = conv_clz(in_size, out_size)
        self.activation_func = activation_func
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x, edge_index = x
        x = self.conv(x, edge_index)
        if self.activation_func:
            x = self.activation_func(x)
        x = self.dropout(x)
        return x, edge_index


class GNNConvStack(nn.Module):

    def __init__(self, d_in, ds_hid, d_out, conv_clz, activation_func=nn.ReLU(),
                 final_activation_func=nn.ReLU(), dropout=0):
        super().__init__()
        n_blocks = len(ds_hid) + 1
        blocks = []
        for i in range(n_blocks):
            in_size = d_in if i == 0 else ds_hid[i - 1]
            out_size = d_out if i == n_blocks - 1 else ds_hid[i]
            block_activation = final_activation_func if i == self.n_blocks - 1 else activation_func
            blocks.append(GNNConvBlock(in_size, out_size, conv_clz, activation_func=block_activation, dropout=dropout))
        self.stack = nn.Sequential(*blocks)

    def forward(self, x):
        return self.stack(x)


class LstmBlock(nn.Module):

    def __init__(self, d_in, d_hid, n_layers, bidirectional, dropout):
        super().__init__()
        # For the LSTM block, a non-zero dropout expects num_layers greater than 1
        self.lstm = nn.LSTM(input_size=d_in, hidden_size=d_hid, num_layers=n_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=0 if n_layers == 1 else dropout)
        self.bidirectional = bidirectional
        self.d_hid = d_hid
        self.dropout = nn.Dropout(p=dropout)
        self.init_hidden = nn.Parameter(torch.zeros(2 * n_layers if bidirectional else n_layers, 1, d_hid))
        self.init_cell = nn.Parameter(torch.zeros(2 * n_layers if bidirectional else n_layers, 1, d_hid))

    def forward(self, x):
        _, n_instances, _ = x.shape

        # Pass through lstm
        out, (ht, _) = self.lstm(x, (self.init_hidden, self.init_cell))

        # Get lstm output
        if self.bidirectional:
            out_split = out.view(1, n_instances, 2, self.d_hid)
            forward_out = out_split[:, :, 0, :]
            backward_out = out_split[:, :, 1, :]
            bag_repr = torch.cat([forward_out[:, -1, :], backward_out[:, 0, :]], dim=1)
        else:
            bag_repr = out[:, -1, :]

        # bag_repr = self.dropout(bag_repr)
        # out = self.dropout(out)
        return bag_repr, out

    def partial_forward(self, instance_embedding, hidden_state, cell_state):
        # Forward pass but we already have a hidden and cell state, and are only doing it for one instance
        instance_embedding = instance_embedding.unsqueeze(0).unsqueeze(0)  # Need a 3D input, currently 1D
        out, (new_hidden_state, new_cell_state) = self.lstm(instance_embedding, (hidden_state, cell_state))
        bag_repr = out[:, -1, :]
        bag_repr = self.dropout(bag_repr)
        return bag_repr, new_hidden_state, new_cell_state

    def flatten_parameters(self):
        self.lstm.flatten_parameters()
