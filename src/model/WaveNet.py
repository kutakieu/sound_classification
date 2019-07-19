import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveNet(nn.Module):
    def __init__(self, batch_size, n_classes=41, n_layers=2, n_blocks=7, n_hidden=32, embedding_dim=None, dilation_channels=32, residual_channels=32, bias=True, gc_cardinality=None, lc_cardinality=None, skip_channels=32, use_BN=True):
        super(WaveNet, self).__init__()
        self.n_classes = n_classes
        self.n_layers = n_layers
        # print(self.n_layers)
        self.n_blocks = n_blocks
        # print(self.n_blocks)
        self.batch_size = batch_size
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.gc_cardinality = gc_cardinality
        self.lc_cardinality = lc_cardinality
        if use_BN:
            self.BN = nn.BatchNorm1d(skip_channels)
        else:
            self.BN = None

        if embedding_dim is not None:
            self.embedding_sample = nn.Embedding(num_embeddings=256, embedding_dim=32)
        else:
            self.embedding_sample = None
            self.causal_conv_init = nn.Conv1d(in_channels=1,
                                                     out_channels=dilation_channels,
                                                     kernel_size=1,
                                                     bias=bias)


        self.causal_conv_input = nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=residual_channels,
                                                 kernel_size=1,
                                                 bias=bias)


        self.dilations_each_block = [2**i for i in range(n_blocks)]
        dilation_layers = self.create_dilation_layers()
        self.dilation_layers = nn.ModuleList(dilation_layers)

        self.receptive_field = sum(self.dilations_each_block)*n_layers + 1


        self.ReLU = nn.ReLU()
        self.causal_conv_skip = nn.Conv1d(in_channels=skip_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias)

        self.causal_conv_output = nn.Conv1d(in_channels=skip_channels,
                                                 out_channels=n_classes,
                                                 kernel_size=1,
                                                 bias=bias)

        # self.h_class = nn.Conv1d(n_hidden, n_classes, 2)


    def create_dilation_layers(self):
        dilation_layers = []
        for block in range(self.n_layers):
            for layer in range(self.n_blocks):
                dilation_layers.append(Residual_Block(self.batch_size, dilation=self.dilations_each_block[layer], residual_channels=self.residual_channels, gc_cardinality=self.gc_cardinality, lc_cardinality=self.lc_cardinality, skip_channels=self.skip_channels))
        return dilation_layers

    def forward(self, input, gc=None, lc=None, is_training=None):
        # print("in the WaveNet")
        if self.embedding_sample is None:
            x = self.causal_conv_init(input).view(input.shape[0], 32, -1)
        else:
            x = self.embedding_sample(input).view(input.shape[0], 32, -1)
        # x = self.embedding_sample(input)
        # print(x.shape)
        x = self.causal_conv_input(x)

        # skips = []
        # print(x.shape)
        residual_out, skip_out = self.dilation_layers[0](x, gc, lc)
        # skips.append(skip_out)

        for i, dilation_layer in enumerate(self.dilation_layers[1:]):
            # print(i)
            residual_out, current_skip_out = dilation_layer(residual_out, gc, lc)
            skip_out = torch.add(skip_out[:, :, dilation_layer.dilation:], current_skip_out)

            # skips.append(skip_out)

        # x = reduce((lambda a, b : torch.add(a, b)), skips)
        x = self.ReLU(skip_out)
        x = self.ReLU(self.BN(self.causal_conv_skip(x)))
        x = self.causal_conv_output(x)
        return x

class Residual_Block(nn.Module):
    def __init__(self, batch_size, dilation, residual_channels=32, dilation_channels=32, kernel_size=2, bias=False, gc_cardinality=None, gc_channels=32, lc_cardinality=None, lc_channels=32, skip_channels=32, use_BN=True):
        super(Residual_Block, self).__init__()

        self.dilation = dilation
        self.batch_size = batch_size
        if use_BN:
            self.BN = nn.BatchNorm1d(dilation_channels)
        else:
            self.BN = None

        self.dilated_conv_filter = nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   dilation=dilation,
                                                   bias=bias)
        self.dilated_conv_gate = nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   dilation=dilation,
                                                   bias=bias)
        """global conditioning"""
        if gc_cardinality is not None:
            self.embedding_gc = nn.Embedding(num_embeddings=gc_cardinality, embedding_dim=gc_channels)
            self.gc_conv_filter = nn.Conv1d(in_channels=gc_channels,
                                                       out_channels=dilation_channels,
                                                       kernel_size=1,
                                                       dilation=1,
                                                       bias=bias)
            self.gc_conv_gate = nn.Conv1d(in_channels=gc_channels,
                                                       out_channels=dilation_channels,
                                                       kernel_size=1,
                                                       dilation=1,
                                                       bias=bias)

        """local conditioning"""
        if lc_cardinality is not None:
            self.embedding_lc = nn.Embedding(num_embeddings=lc_cardinality, embedding_dim=lc_channels)
            self.lc_conv_filter = nn.Conv1d(in_channels=lc_channels,
                                                       out_channels=dilation_channels,
                                                       kernel_size=1,
                                                       dilation=1,
                                                       bias=bias)
            self.lc_conv_gate = nn.Conv1d(in_channels=lc_channels,
                                                       out_channels=dilation_channels,
                                                       kernel_size=1,
                                                       dilation=1,
                                                       bias=bias)

        self.residual_conv = nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=residual_channels,
                                                 kernel_size=1,
                                                 dilation=1,
                                                 bias=bias)
        self.skip_conv = nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 dilation=1,
                                                 bias=bias)

    def forward(self, input, gc=None, lc=None):
        # print("before " + str(input.shape))
        filter_out = self.dilated_conv_filter(input)
        if gc is not None:
            gc = self.embedding_gc(gc).view(self.batch_size, gc_channels, -1)
            filter_out = filter_out + self.gc_conv_filter(gc)
        if lc is not None:
            lc = self.embedding_lc(lc).view(self.batch_size, lc_channels, -1)
            filter_out = filter_out + self.lc_conv_filter(lc)
        if self.BN is None:
            filter_out = torch.tanh(filter_out)
        else:
            filter_out = torch.tanh(self.BN(filter_out))
        # print("after " + str(input.shape))
        gate_out = self.dilated_conv_gate(input)
        if gc is not None:
            # gc = self.embedding_gc(gc).view(self.batch_size, 32, -1)
            gate_out = gate_out + self.gc_conv_gate(gc)
        if lc is not None:
            # lc = self.embedding_lc(lc).view(self.batch_size, 32, -1)
            gate_out = gate_out + self.lc_conv_gate(lc)
        gate_out = torch.sigmoid(gate_out)

        x = filter_out * gate_out
        if self.BN is None:
            skip_out = self.skip_conv(x)
        else:
            skip_out = self.skip_conv(self.BN(x))
        residual_out = self.residual_conv(x) + input[:, :, self.dilation:]

        return residual_out, skip_out
