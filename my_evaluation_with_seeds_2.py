# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import seaborn as sns
import data
import models
import pandas as pd

# Setup
use_cuda = torch.cuda.is_available()
if not use_cuda:
    raise NotImplementedError("evaluate.py requires a GPU to use.")
device = torch.device('cuda')

# Set random seeds for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_cached_testloaders = {}

def get_testloader(dataset_name, batch_size=128):
    """Returns a cached test DataLoader to speed up evaluation."""
    key = (dataset_name, batch_size)
    if key not in _cached_testloaders:
        testloader = data.get_testloader(dataset_name, batch_size=batch_size)
        _cached_testloaders[key] = testloader
    return _cached_testloaders[key]

_cached_testlabels = {}

def get_testlabels(dataset_name, batch_size=128):
    """Retrieves and caches test labels."""
    key = (dataset_name,)
    if key not in _cached_testlabels:
        testloader = get_testloader(dataset_name, batch_size=batch_size)
        test_labels = [targets.numpy() for _, targets in testloader]
        _cached_testlabels[key] = np.concatenate(test_labels)
    return _cached_testlabels[key]

def modify_classifier(net, model_type, dataset_name):
    """Modifies the classifier dynamically for VGG, ResNet, and ShuffleNetV2."""
    num_classes = 100 if dataset_name == "cifar100" else 10

    if "vgg" in model_type.lower():
        net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, num_classes)
    elif "resnet" in model_type.lower():
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    elif "shufflenet" in model_type.lower():
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return net

def predict(checkpoint_paths, model_type, dataset_name='cifar10', batch_size=128):
    """Returns logits from ensembling the given checkpoint_paths together."""
    for checkpoint_path in checkpoint_paths:
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint path does not exist: {checkpoint_path}")
            return []

    ensemble_preds = []
    for checkpoint_path in checkpoint_paths:
        with torch.no_grad():
            # Ensure the model exists in the module
            net_class = getattr(models, model_type, None)
            if net_class is None:
                raise ValueError(f"Model type '{model_type}' not found in models module.")

            net = net_class()  # Instantiate model
            net = modify_classifier(net, model_type, dataset_name)  # Modify classifier

            checkpoint = torch.load(checkpoint_path, map_location="cuda")
            model_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                               if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict, strict=False)

            net.cuda()
            net.eval()

            model_outputs = []
            testloader = get_testloader(dataset_name, batch_size=batch_size)
            for inputs, targets in testloader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)  # Logits
                model_outputs.append(outputs.detach().cpu().numpy())

            model_outputs = np.concatenate(model_outputs)
            ensemble_preds.append(model_outputs)

    return np.mean(np.stack(ensemble_preds, axis=0), axis=0) if len(ensemble_preds) > 1 else ensemble_preds[0]


def compute_metrics(all_logits, test_labels):
    """Computes accuracy, cross-entropy loss, and per-class statistics."""
    accs = []
    ces = []
    cel = torch.nn.CrossEntropyLoss()

    for logits in all_logits:
        preds = logits.argmax(axis=1)
        accs.append(100 * np.mean(preds == test_labels))
        ces.append(cel(torch.tensor(logits), torch.tensor(test_labels)).item())

    print(f"Test Accuracy: {np.mean(accs):.2f} +/- {np.std(accs):.2f}")
    print(f"Test Cross-Entropy: {np.mean(ces):.4f} +/- {np.std(ces):.4f}")

def run_evaluation(checkpoint_paths, model_type, dataset_name='cifar10', batch_size=128):
    """Runs evaluation on multiple model checkpoints."""
    model_preds = {}
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"Evaluating model {i+1}/{len(checkpoint_paths)}...")
        model_preds[checkpoint_path] = predict([checkpoint_path], model_type, dataset_name, batch_size)

    test_labels = get_testlabels(dataset_name, batch_size=batch_size)
    compute_metrics(list(model_preds.values()), test_labels)

def read_training_loss_data(file_path):
    """Reads training loss data from CSV."""
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        return [float(row[1]) for row in reader]

def plot_combined_training_loss(loss_data, num_seeds, filename='combined_training_loss.png'):
    """Plots training loss across multiple seeds."""
    plt.figure(figsize=(10, 6))
    colors = cm.rainbow(np.linspace(0, 1, num_seeds))
    for seed, color in zip(range(1, num_seeds + 1), colors):
        plt.plot(loss_data[seed], color=color, label=f'Seed {seed}')
    plt.title('Training Loss Over Epochs for Different Seeds')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    num_runs = 5
    dataset_name = 'cifar100'
    model_type = 'shufflenetv2'  # Change to 'vgg16', 'resnet18', or 'shufflenetv2'

    # **Corrected model directories & paths**
    model_dirs = [f'shuff05_c100_norm_{i}' for i in range(1, num_runs + 1)]
    paths = [os.path.join('cifar100_models', d, 'model.ckpt') for d in model_dirs]

    print('Running model evaluation...')
    run_evaluation(paths, model_type=model_type, dataset_name=dataset_name, batch_size=128)

    # Read training loss data
    training_loss_data = {}
    for seed, model_dir in zip(range(1, num_runs + 1), model_dirs):
        file_path = os.path.join('cifar100_models', model_dir, f'training_loss_seed_{seed}.csv')
        training_loss_data[seed] = read_training_loss_data(file_path)

    # Plot combined training loss
    plot_combined_training_loss(training_loss_data, num_runs)


