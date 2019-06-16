from torchvision.models import inception_v3
from torch import nn
import torch.utils.model_zoo as model_zoo


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


class IdModel(nn.Module):
    def __init__(self, cam_num):
        super(IdModel, self).__init__()

        self.inception = inception_v3()
        self.inception.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
        self.fc1 = FCL(in_dim=1000, out_dim=256, bn=True, activation="relu")
        self.fc2 = FCL(in_dim=256, out_dim=cam_num, activation="softmax")

    def forward(self, x):
        x = self.inception(x)[0]
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class FCL(nn.Module):

    def __init__(self, in_dim, out_dim, bn=True, activation="tanh"):
        super(FCL, self).__init__()

        if activation == "tanh":
            self.activation_layer = nn.Tanh()
        elif activation == "sigmoid":
            self.activation_layer = nn.Sigmoid()
        elif activation == "relu":
            self.activation_layer = nn.ReLU(True)
        elif activation == "softmax":
            self.activation_layer = nn.Softmax()

        if not bn:
            self.fcl = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                self.activation_layer
            )
        else:
            self.fcl = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                self.activation_layer
            )

    def forward(self, x):
        x = self.fcl(x)
        return x


# From: https://forums.fast.ai/t/flatten-layer-of-pytorch/4639/5
class Flatten(nn.Module):

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x
