import torch
import torch.nn as nn

class CompressorModel(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        self.pretrain = args.pretrain

        # thresholding conv layer
        self.thresholder = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=1)

        # misc layers
        self.pool = nn.AvgPool2d(kernel_size=128)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(512)

        # linear model layers (pv + metadata)
        self.linear0 = nn.Linear(12 + 7 + 12, 256)
        self.linear1 = nn.Linear(256, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, 256)

        # combining layers
        self.linear_out = nn.Linear(256, 48)

        self.main_sequence = nn.Sequential(
            self.linear0,
            self.relu,
            self.linear1,
            self.relu,
            self.linear2,
            self.relu,
            self.linear3,
            self.relu,
            self.linear4,
            self.relu
        )

        self.threshold_sequence = nn.Sequential(
            self.thresholder,   # single kernel threshold value
            self.sigmoid,       # make binary
            self.pool,          # compress
            self.flatten
        )

        self.final = nn.Sequential(
            self.linear_out,
            self.sigmoid
        )


    def forward(self, pv, hrv, weather, metadata):

        hrv_processed = self.threshold_sequence(hrv)
        x = torch.concat((pv, metadata, hrv_processed), dim=-1)
        x = self.main_sequence(x)

        z = self.final(x)

        return z