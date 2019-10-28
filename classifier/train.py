import click
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from classifier.models.character_classifier import CharacterClassifierInceptionV3
from classifier.dataset.lang_dataset import LangDataset
from classifier.util import get_transforms, load_config


device = 'cuda' if torch.cuda.is_available() else 'cpu'


@click.group()
def cli():
    pass


@cli.command()
def train():
    """Trains a vowel consonant classifier
    """
    # load the param config
    config = load_config('config.yaml')

    test_split_ratio = config['test_split']
    val_split_ratio = config['val_split']
    batch_size = config['batch_size']
    random_seed = config['random_seed']
    num_vowels = config['num_vowels']
    num_consonants = config['num_consonants']
    train_dir = config['train']['data']
    log_step = config['log_step']
    num_epochs = config['num_epochs']

    # Set the random seed
    torch.manual_seed(random_seed)

    # Load and the dataset and split it
    transform = get_transforms(train=True)
    dataset = LangDataset(train_dir, transform=transform)
    num_items = len(dataset)
    num_train_samples = int((1 - (test_split_ratio + val_split_ratio)) * num_items)
    num_test_samples = int(test_split_ratio * num_items)
    num_val_samples = int(val_split_ratio * num_items)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train_samples, num_val_samples, num_test_samples])

    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, num_workers=4, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, num_workers=4, shuffle=True, batch_size=num_test_samples)

    # Define the loss metric, optimizer and the classifier
    criterion = nn.NLLLoss()
    classifier = CharacterClassifierInceptionV3(num_vowels, num_consonants, param_to_freeze='Mixed_5b.branch3x3dbl_1.conv.weight').to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Main Training Loop
    val_batch, val_targets = next(iter(val_loader))
    val_batch = val_batch.to(device)
    val_targets = val_targets.to(device)
    for epoch_idx in range(num_epochs):
        print(f'Epoch Idx: {epoch_idx}')
        loss = _train_one_epoch(classifier, criterion, optimizer, val_batch, val_targets, log_step=log_step)
        loss_profile.extend(loss)


def _train_one_epoch(model, criterion, optimizer, val_batch, val_targets, log_step=50):
    loss_profile = []
    for idx, (input_batch, target_batch) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        input_batch = input_batch.cuda()
        target_batch = target_batch.cuda()
        v_out, c_out = model(input_batch)

        # compute the vowel and the consonant losses
        vowel_targets = target_batch[:, 0]
        consonant_targets = target_batch[:, 1]
        v_loss = criterion(v_out, vowel_targets)
        c_loss = criterion(c_out, consonant_targets)

        loss = v_loss + c_loss
        # compute the gradients
        loss.backward()

        # optimize the parameters
        optimizer.step()

        if idx % log_step == 0:
            # evaluate the model
            model.eval()
            v_val_out, c_val_out = model(val_batch)
            val_loss = criterion(v_val_out, val_targets[:, 0]) + criterion(c_val_out, val_targets[:, 1])
            print(f'StepIdx: {idx} V_loss: {v_loss.item()} C_loss: {c_loss.item()} Loss: {loss.item()} ValLoss: {val_loss}')
            torch.save(model.state_dict(), 'classifier.pt')
        loss_profile.append(loss.item())
    return loss_profile


@cli.command()
def sanity_check_model():
    """
    Ensures that the forward pass through the classifier works
    """
    # Sanity check the classifier
    dataset = LangDataset(train_dir)
    loader = DataLoader(dataset, num_workers=4, shuffle=True, batch_size=32)
    sample_batch, targets_batch = next(iter(loader))
    sample_batch = sample_batch.to(device)
    targets_batch = targets_batch.to(device)

    classifier = CharacterClassifierInceptionV3(num_vowels, num_consonants).to(device)
    v_out, c_out = classifier(sample_batch)

    assert v_out is not None
    assert c_out is not None


if __name__ == "__main__":
    cli()
