import torch
import torch.nn as nn
import torch.functional as F
from torchvision.models import inception_v3


class CharacterClassifierInceptionV3(nn.Module):
    def __init__(self, num_vowels, num_consonants, param_to_freeze):
        super().__init__()
        self.vowel_output = nn.Linear(in_features=2048, out_features=num_vowels)
        self.consonants_output = nn.Linear(in_features=2048, out_features=num_consonants)
        self.output = nn.LogSoftmax(dim=1)

        self.inception_net = inception_v3(pretrained=True, aux_logits=False)
        del self.inception_net.fc

        for name, param in self.inception_net.named_parameters():
            # We freeze all the parameters uptil the `param_to_freeze`
            if name == param_to_freeze:
                break
            param.requires_grad = False

    def forward(self, input):
        x = input
        # This is a hack to get around the fact that the inception net
        # provided by torchvision does not provide an attribute of max_pool
        # in the inception net constructor
        for name, module in self.inception_net.named_children():
            if name == 'Conv2d_2b_3x3':
                x = self.inception_net.Conv2d_2b_3x3(x)
                x = F.max_pool2d(x, kernel_size=3, stride=2)
                continue

            if name == 'Conv2d_4a_3x3':
                x = self.inception_net.Conv2d_4a_3x3(x)
                x = F.max_pool2d(x, kernel_size=3, stride=2)
                continue
            x = module(x)

        # Adaptive average pooling for inceptionV3
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        v_out = self.output(self.vowel_output(x))
        c_out = self.output(self.consonants_output(x))

        return v_out, c_out
