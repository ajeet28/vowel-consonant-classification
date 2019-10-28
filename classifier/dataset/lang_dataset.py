import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class LangDataset(Dataset):
    def __init__(self, train_dir, transform=None):
        if train_dir.endswith('.zip'):
            extract_path = os.path.join(os.environ['HOME'], 'datasets')
            os.makedirs(extract_path, exist_ok=True)
            filename, _ = os.path.splitext(train_dir)

            # extract the dir
            import zipfile
            with zipfile.ZipFile(train_dir, 'r') as zip:
                zip.extractall(path=extract_path)
            train_dir = os.path.join(extract_path, os.path.basename(filename))

        self.images = []
        self.targets = []  # Stores the target in the form of (vowel, consonant)
        self.transform = transform

        images = os.listdir(train_dir)
        for image in tqdm(images):
            image_path = os.path.join(train_dir, image)
            img = Image.open(image_path).convert('RGB')
            self.images.append(img)
            target, _ = os.path.splitext(image)
            vowel, consonant = target.split('_')[:2]
            vowel_idx = int(vowel[1])
            consonant_idx = int(consonant[1])

            self.targets.append(torch.as_tensor(
                [vowel_idx, consonant_idx],
                dtype=torch.long)
            )

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.images)
