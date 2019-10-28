import yaml
from torchvision import transforms


def get_transforms(train=False):
    T = []
    if train:
        # Add more transforms here
        T.append(transforms.RandomGrayscale(p=0.3))
    T.append(transforms.Resize((299, 299)))
    T.append(transforms.ToTensor())
    return transforms.Compose(T)


def load_config(config_path):
    
    with open(config_path, 'r') as reader:
        config = yaml.safe_load(reader)
    return config