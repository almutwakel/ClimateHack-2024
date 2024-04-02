import torch
import torch.nn as nn

class MultiModel(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.pretrain = args.pretrain

        # hrv scanner
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3)

        # misc layers
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(512)

        # linear model layers
        self.linear0 = nn.Linear(12 + 7, 256)
        self.linear1 = nn.Linear(256, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, 48)

        # combining layers
        self.linear_out = nn.Linear(96, 48)

        self.tabular_sequence = nn.Sequential(
            self.linear0,
            self.relu,
            self.linear1,
            self.relu,
            self.linear2,
            self.relu,
            self.linear3,
            self.relu,
            self.linear4
        )

        self.conv_sequence = nn.Sequential(
            self.conv1,
            self.pool,
            self.relu,
            self.conv2,
            self.pool,
            self.relu,
            self.conv3,
            self.pool,
            self.relu,
            self.conv4,
            self.pool,
            self.relu,
            self.conv5,
            self.pool,
            self.flatten
        )

        self.combiner = nn.Sequential(
            self.linear_out,
            self.sigmoid
        )


    def forward(self, pv, hrv, weather, metadata):
        # tabular data
        x = torch.concat((pv, metadata), dim=-1)
        x = self.tabular_sequence(x)

        # baseline (prev value)
        # b = torch.ones(48, 1).to(device, dtype=float) * pv[:, -1]
        # b = (pv[:, -1] * b).T

        if self.pretrain:
            y = torch.rand_like(x)
        else:
            y = self.conv_sequence(hrv)

        # combine in last step
        z = torch.concat((x, y), dim=-1)
        z = self.combiner(z)

        return z

    def load_pretrained_section(model, state_dict):
        with torch.no_grad():
            model.linear0a.weight.copy_(state_dict['linear0a.weight'])
            model.linear0a.bias.copy_(state_dict['linear0a.bias'])
            model.linear0b.weight.copy_(state_dict['linear0b.weight'])
            model.linear0b.bias.copy_(state_dict['linear0b.bias'])
            model.linear1.weight.copy_(state_dict['linear1.weight'])
            model.linear1.bias.copy_(state_dict['linear1.bias'])
            model.linear2.weight.copy_(state_dict['linear2.weight'])
            model.linear2.bias.copy_(state_dict['linear2.bias'])
            model.batchnorm.weight.copy_(state_dict['batchnorm.weight'])
            model.batchnorm.bias.copy_(state_dict['batchnorm.bias'])
            model.batchnorma.weight.copy_(state_dict['batchnorma.weight'])
            model.batchnorma.bias.copy_(state_dict['batchnorma.bias'])
            # model.linear3.weight.copy_(state_dict['linear3.weight'])
            # model.linear3.bias.copy_(state_dict['linear3.bias'])
            model.linear4.weight.copy_(state_dict['linear4.weight'])
            model.linear4.bias.copy_(state_dict['linear4.bias'])
        for name, param in model.named_parameters():
            if name in [
                'linear0a.weight', 'linear0a.bias',
                'linear0b.weight', 'linear0b.bias',
                'linear1.weight', 'linear1.bias',
                'linear2.weight', 'linear2.bias',
                'batchnorm.weight', 'batchnorm.bias',
                'batchnorma.weight', 'batchnorma.bias',
                # 'linear3.weight', 'linear3.bias',
                'linear4.weight', 'linear4.bias',
                'follower.0.weight', 'follower.0.bias',
                'follower.2.weight', 'follower.2.bias',
                'follower.5.weight', 'follower.5.bias'
            ]:
                param.requires_grad = False

    def freeze_pretrain(model, freeze=True):

        model.pretrain = not freeze
        for name, param in model.named_parameters():
            if name in [
                'linear0a.weight', 'linear0a.bias',
                'linear0b.weight', 'linear0b.bias',
                'linear1.weight', 'linear1.bias',
                'linear2.weight', 'linear2.bias',
                'batchnorm.weight', 'batchnorm.bias',
                'batchnorma.weight', 'batchnorma.bias',
                # 'linear3.weight', 'linear3.bias',
                'linear4.weight', 'linear4.bias',
                'follower.0.weight', 'follower.0.bias',
                'follower.2.weight', 'follower.2.bias',
                'follower.5.weight', 'follower.5.bias'
            ]:
                param.requires_grad = not freeze
