import torch
from tqdm import tqdm
from classifier.models.character_classifier import CharacterClassifierInceptionV3
from classifier.dataset.lang_dataset import LangDataset
from classifier.util import load_config, get_transforms


@click.command()
@click.argument('model_path')
@click.argument('test_dataset_path')
def eval(model_path, test_dataset_path):
    # load the param config
    config = load_config('config.yaml')
    num_vowels = config['num_vowels']
    num_consonants = config['num_consonants']
    model = CharacterClassifierInceptionV3(num_vowels, num_consonants, param_to_freeze='Mixed_5b.branch3x3dbl_1.conv.weight').to(device)

    # Load the model state dict
    model.load_state_dict(torch.load(model_path))
    dataset = LangDataset(test_dataset_path, transform=get_transforms())
    accuracy = _compute_accuracy(dataset, model)
    return accuracy


def _get_predictions(model, test_samples):
    model.eval()
    v_out, c_out = model(test_samples)
    vowel_pred = torch.topk(v_out, 1, dim=1)
    consonant_pred = torch.topk(c_out, 1, dim=1)
    return vowel_pred.indices, consonant_pred.indices


def _compute_accuracy(dataset, model):
    num_correct = 0
    total_samples = len(dataset)
    for sample, target in tqdm(dataset):
        sample = torch.unsqueeze(sample, 0).cuda()
        v_pred, c_pred = _get_predictions(model, sample)
        if v_pred.item() == target[0].item() and c_pred.item() == target[1].item():
            num_correct += 1
    accuracy = num_correct / total_samples
    return accuracy
