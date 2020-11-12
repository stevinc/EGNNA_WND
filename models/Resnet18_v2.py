import torch
import torch.nn as nn
from torchvision import models

from graph.gat_model import EGNNA
from graph.gcn_model import EGNNC


class RESNET18_v2(nn.Module):
    def __init__(self, in_channels_bands=3, colorization=0, in_channels_aux=3, out_cls=2, pretrained=0,
                 device="cuda", use_dropout=0, drop_rate=0.2, use_graph=0, graph_version='egnnc', neighbours_labels=0,
                 layers_graph=2, residual=0):
        super(RESNET18_v2, self).__init__()

        # flag for device
        self.device = device
        # from scratch or pretrained on imagenet
        if pretrained:
            self.model = models.resnet18(pretrained=True)
        else:
            self.model = models.resnet18(pretrained=False)
        # SET THE CORRECT NUMBER OF INPUT CHANNELS
        self.in_channels_bands = in_channels_bands
        self.in_channels_aux = in_channels_aux
        # SET THE CORRECT NUMBER OF INPUT FOR COLORIZATION
        self.colorization = colorization
        if colorization:
            self.conv_1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.conv1 = self.conv_1
        # Feature extractor
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        # Dropout
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(drop_rate)
        # Graph
        self.use_graph = use_graph
        self.layers_graph = layers_graph
        self.residual = residual
        self.classifier = nn.Linear(512*self.in_channels_aux, out_cls) if use_graph else nn.Linear(512, out_cls)
        # choose the graph version
        if self.use_graph and graph_version == 'egnnc':
            self.graph = EGNNC(nin=512,
                               nhid=512,
                               nheads=self.in_channels_aux,
                               dropout=drop_rate,
                               layers=self.layers_graph,
                               residual=residual)
        elif self.use_graph and graph_version == 'egnna':
            self.graph = EGNNA(nin=512,
                               nhid=512,
                               nout=1024,
                               alpha=0.2,
                               nheads=self.in_channels_aux,
                               layers=self.layers_graph,
                               residual=residual)
        # use or not the labels of the neighbourd
        self.neighbours_labels = neighbours_labels

    def forward(self, x, blocks_adj_matrix, neighbours_numbers):
        if self.use_graph:
            # resnet features extraction
            features = self.feature_extractor(x).view(x.shape[0], 512)
            # 2- graph
            features = self.graph(features, blocks_adj_matrix)
            # decide if classify or not the neighbourhood labels
            if not self.neighbours_labels:
                features = features.view(int(features.shape[0] / neighbours_numbers), neighbours_numbers, features.shape[1])
                features = features[:, 0, :]
            if self.use_dropout:
                features = self.dropout(features)
            # final classifier
            out = self.classifier(features)
        else:
            out = self.feature_extractor(x)
            out = out.view(x.shape[0], -1)
            if self.use_dropout:
                out = self.dropout(out)
            out = self.classifier(out)
        return out

    def set_weights_conv1(self):
        """
        Function setting the correct number of channels in the first conv
        :return:
        """
        if (self.colorization and self.in_channels_bands != 4) or (not self.colorization and self.in_channels_bands != 3):
            conv1 = nn.Conv2d(self.in_channels_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.feature_extractor[0] = conv1
        return