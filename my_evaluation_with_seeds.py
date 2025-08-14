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
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_cached_testloaders = {}  # Maps (dataset_name, batch_size) to testloader
def get_testloader(dataset_name, batch_size=128):
  key = (dataset_name, batch_size)
  if key in _cached_testloaders:
    return _cached_testloaders[key]
  else:
    testloader = data.get_testloader(dataset_name, batch_size=batch_size)
    _cached_testloaders[key] = testloader
    return testloader

_cached_testlabels = {}  # Maps (dataset_name,) to test labels

def get_testlabels(dataset_name, batch_size=128):
  """Gets test labels for the given dataset."""
  key = (dataset_name,)
  if key not in _cached_testlabels:
    testloader = get_testloader(dataset_name, batch_size=batch_size)
    test_labels = []
    for _, targets in testloader:
      test_labels.append(targets.numpy())
    noaug_testlabels = np.concatenate(test_labels)
    _cached_testlabels[key] = noaug_testlabels
#    noaug_testlabels = _cached_testlabels[key]   # additionally added
    return _cached_testlabels[key]
 #   return noaug_testlabels
'''
def predict(checkpoint_paths, model_type, dataset_name='cifar10', batch_size=128):
  """Returns logits from ensembling the given checkpoint_paths together."""
  for checkpoint_path in checkpoint_paths:
    if not os.path.exists(checkpoint_path):
      print(f"Checkpoint path does not exist: {checkpoint_path}")
      return []
  ensemble_preds = []
  for checkpoint_path in checkpoint_paths:
    with torch.no_grad():
      net = getattr(models, model_type)()
      checkpoint = torch.load(checkpoint_path)
      if 'net' in checkpoint:
        net = checkpoint['net']
      elif 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
      else:
        raise ValueError(f'unable to load model from checkpoint with keys: '
                         f'{list(checkpoint.keys())}')
      net.cuda()
      net.eval()
      model_outputs = []
      testloader = get_testloader(dataset_name, batch_size=batch_size)
      for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)  # Logits
        model_outputs.append(outputs.detach().cpu().numpy())
      model_outputs = np.concatenate(model_outputs)
      ensemble_preds.append(model_outputs)
  if len(ensemble_preds) == 1:
    return ensemble_preds[0]
  else:
    return np.mean(np.stack(ensemble_preds, axis=0), axis=0)

def predict(checkpoint_paths, model_type, dataset_name='cifar10', batch_size=128):
    """Returns logits from ensembling the given checkpoint_paths together."""
    for checkpoint_path in checkpoint_paths:
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint path does not exist: {checkpoint_path}")
            return []

    ensemble_preds = []
    for checkpoint_path in checkpoint_paths:
        with torch.no_grad():
            net = getattr(models, model_type)()

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cuda")

            # Modify classifier layer dynamically
            num_classes = 100 if dataset_name == "cifar100" else 10
            net.classifier = torch.nn.Linear(512, num_classes)  # Adjust output layer

            if 'net' in checkpoint:
                net = checkpoint['net']
            elif 'model_state_dict' in checkpoint:
                # Load only matching layers
                model_dict = net.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                                   if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(pretrained_dict)
                net.load_state_dict(model_dict, strict=False)
            else:
                raise ValueError(f'unable to load model from checkpoint with keys: {list(checkpoint.keys())}')
            
            net.cuda()
            net.eval()

            model_outputs = []
            testloader = get_testloader(dataset_name, batch_size=batch_size)
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)  # Logits
                model_outputs.append(outputs.detach().cpu().numpy())

            model_outputs = np.concatenate(model_outputs)
            ensemble_preds.append(model_outputs)

    if len(ensemble_preds) == 1:
        return ensemble_preds[0]
    else:
        return np.mean(np.stack(ensemble_preds, axis=0), axis=0)
'''
def predict(checkpoint_paths, model_type, dataset_name='cifar10', batch_size=128):
    """Returns logits from ensembling the given checkpoint_paths together."""
    for checkpoint_path in checkpoint_paths:
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint path does not exist: {checkpoint_path}")
            return []

    ensemble_preds = []
    num_classes = 100 if dataset_name == "cifar100" else 10  # Ensure correct num classes

    for checkpoint_path in checkpoint_paths:
        with torch.no_grad():
            # Ensure the model exists
            net_class = getattr(models, model_type, None)
            if net_class is None:
                raise ValueError(f"Model type '{model_type}' not found in models module.")

            net = net_class()  # Instantiate model

            # Modify classifier for CIFAR-100
            # Modify classifier dynamically for different model types
            if hasattr(net, "classifier"):  
                in_features = net.classifier.in_features  
                net.classifier = torch.nn.Linear(in_features, num_classes)  
            elif hasattr(net, "fc"):  
                in_features = net.fc.in_features  
                net.fc = torch.nn.Linear(in_features, num_classes)  
            elif hasattr(net, "linear"):  # ?? Fix for ShuffleNet
                in_features = net.linear.in_features  
                net.linear = torch.nn.Linear(in_features, num_classes)
            elif hasattr(net, "conv5"):  # ?? Some ShuffleNet versions store it here
                in_features = net.conv5.in_channels
                net.conv5 = torch.nn.Conv2d(in_features, num_classes, kernel_size=1, stride=1)
            else:
                raise ValueError(f"Model {model_type} does not have a recognized classification layer.")


            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cuda")
            model_dict = net.state_dict()

            # Load only matching layers
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




def _get_ranks(x):
  tmp = x.argsort()
  ranks = torch.zeros_like(tmp).cuda()
  ranks[tmp] = torch.arange(len(x)).cuda()
  return ranks


def spearman_gpu(x, y):
  """Computes the Spearman correlation between 2 1-D vectors.

  Args:
    x: Shape (N, )
    y: Shape (N, )
  """
  x_rank = _get_ranks(x)
  y_rank = _get_ranks(y)
  
  n = x.size(0)
  upper = 6 * torch.sum((x_rank - y_rank).type(torch.cuda.FloatTensor).pow(2))
  down = n * (n ** 2 - 1.0)
  return 1.0 - (upper / down)


def pearson_gpu(x, y):
  """Computes the Pearson correlation between 2 1-D vectors.

  Args:
    x: Shape (N, )
    y: Shape (N, )
  """
  vx = x - torch.mean(x)
  vy = y - torch.mean(y)
  cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) *
                               torch.sqrt(torch.sum(vy ** 2)))
  return cost
'''
def compute_metrics(all_logits, test_labels,  per_class_accuracies, verbose=False):
  
  # Get baseline accs/CEs.
  accs = []
  ces = []
  cel = torch.nn.CrossEntropyLoss()
  for logits in all_logits:
    preds = logits.argmax(axis=1)
    accs.append(100 * np.mean(preds == test_labels))
    ces.append(cel(torch.tensor(logits), torch.tensor(test_labels)).item())

  # Mean/std stats.
  bootstrap_n = 10000
  np.random.seed(1)
  for (stat_name, stat_list) in [('Test Cross-Entropy', ces),
                                 ('Test Accuracy', accs)]:
    mean = np.mean(stat_list)
    std = np.std(stat_list)
    # Bootstrap the std
    bs_vals = np.random.choice(stat_list, size=(bootstrap_n, len(stat_list)),
                               replace=True)
    bs_std = np.std(bs_vals, axis=1)
    assert len(bs_std) == bootstrap_n
    bs_std = np.std(bs_std)
    print(f'{stat_name}: {mean:g} +- {std:g} (+- {bs_std:g})')

  # Put everything on the GPU once at the beginning if the size is small enough.
  if len(all_logits) > 0:
    bytes_used = (4 * len(all_logits) *
                  all_logits[0].shape[0] * all_logits[0].shape[1])
  else:
    bytes_used = 0
  already_on_gpu = False
  if bytes_used < 3e9:
    for i in range(len(all_logits)):
      all_logits[i] = torch.tensor(all_logits[i]).cuda()
    already_on_gpu = True

  # Compute pairwise metrics.
  acc_deltas = []
  ce_deltas = []
  spearmans = []
  pearsons = []
  disagreements = []
  labels_tensor = torch.tensor(test_labels).cuda()
  for i1 in range(len(all_logits)):
    if already_on_gpu:
      logits1 = all_logits[i1]
    else:
      logits1 = torch.tensor(all_logits[i1]).cuda()
    flat_logits1 = logits1.flatten()
    for i2 in range(i1+1, len(all_logits)):
      if verbose:
        print(f'Pairwise {i1} {i2}')
      if already_on_gpu:
        logits2 = all_logits[i2]
      else:
        logits2 = torch.tensor(all_logits[i2]).cuda()
      ensemble_preds = .5 * (logits1 + logits2)
      acc = 100 * torch.mean(torch.eq(torch.argmax(ensemble_preds, dim=1),
          labels_tensor).type(torch.cuda.FloatTensor)).item()
      ce = cel(ensemble_preds, labels_tensor).item()
      acc_deltas.append(acc - accs[i1])
      acc_deltas.append(acc - accs[i2])
      ce_deltas.append(ce - ces[i1])
      ce_deltas.append(ce - ces[i2])
      preds1 = torch.argmax(logits1, dim=1)
      preds2 = torch.argmax(logits2, dim=1)
      disagreement = 100 * torch.mean(
          (preds1 != preds2).type(torch.cuda.FloatTensor)).item()
      disagreements.append(disagreement)
      flat_logits2 = logits2.flatten()
      rho = spearman_gpu(flat_logits1, flat_logits2).item()
      r = pearson_gpu(flat_logits1, flat_logits2).item()
      spearmans.append(rho)
      pearsons.append(r)
  print('Average pairwise accuracy delta (%%): %g' % np.mean(acc_deltas))
  print('Average pairwise cross-entropy delta: %g' % np.mean(ce_deltas))
  print('Average pairwise correlation (spearman): %g' % np.mean(spearmans))
  print('Average pairwise correlation (pearson): %g' % np.mean(pearsons))
  print('Average pairwise disagreement (%%): %g' % np.mean(disagreements))
'''
def compute_metrics(all_logits, test_labels, per_class_accuracies, verbose=False):
    """Computes and prints all variability metrics, including per-class accuracy statistics."""
    # Calculate baseline accuracies and cross-entropy loss
    accs = []
    ces = []
    cel = torch.nn.CrossEntropyLoss()
    for logits in all_logits:
        preds = logits.argmax(axis=1)
        accs.append(100 * np.mean(preds == test_labels))
        ces.append(cel(torch.tensor(logits), torch.tensor(test_labels)).item())

    # Print mean and standard deviation of test cross-entropy and accuracy
    bootstrap_n = 10000
    np.random.seed(1)
    for (stat_name, stat_list) in [('Test Cross-Entropy', ces), ('Test Accuracy', accs)]:
        mean = np.mean(stat_list)
        std = np.std(stat_list)
        bs_vals = np.random.choice(stat_list, size=(bootstrap_n, len(stat_list)), replace=True)
        bs_std = np.std(bs_vals, axis=1)
        print(f'{stat_name}: {mean:g} +- {std:g} (+- {np.std(bs_std):g})')
        
    if len(all_logits) > 0:
       bytes_used = (4 * len(all_logits) *
                     all_logits[0].shape[0] * all_logits[0].shape[1])
    else:
      bytes_used = 0
    already_on_gpu = False
    if bytes_used < 3e9:
      for i in range(len(all_logits)):
        all_logits[i] = torch.tensor(all_logits[i]).cuda()
      already_on_gpu = True
    # Compute pairwise metrics
    acc_deltas = []
    ce_deltas = []
    spearmans = []
    pearsons = []
    disagreements = []
    labels_tensor = torch.tensor(test_labels).cuda()
    for i1 in range(len(all_logits)):
        logits1 = torch.tensor(all_logits[i1]).cuda() if not already_on_gpu else all_logits[i1]
        flat_logits1 = logits1.flatten()
        for i2 in range(i1+1, len(all_logits)):
            logits2 = torch.tensor(all_logits[i2]).cuda() if not already_on_gpu else all_logits[i2]
            ensemble_preds = .5 * (logits1 + logits2)
            acc = 100 * torch.mean(torch.eq(torch.argmax(ensemble_preds, dim=1),
                 labels_tensor).type(torch.cuda.FloatTensor)).item()
            ce = cel(ensemble_preds, labels_tensor).item()
            acc_deltas.append(acc - accs[i1])
            acc_deltas.append(acc - accs[i2])
            ce_deltas.append(ce - ces[i1])
            ce_deltas.append(ce - ces[i2])
            preds1 = torch.argmax(logits1, dim=1)
            preds2 = torch.argmax(logits2, dim=1)
            disagreement = 100 * torch.mean(
                           (preds1 != preds2).type(torch.cuda.FloatTensor)).item()
            disagreements.append(disagreement)
            flat_logits2 = logits2.flatten()
            rho = spearman_gpu(flat_logits1, flat_logits2).item()
            r = pearson_gpu(flat_logits1, flat_logits2).item()
            spearmans.append(rho)
            pearsons.append(r)

    # Compute per-class accuracy statistics
    accuracies = np.array(per_class_accuracies)
    min_accuracies = np.min(accuracies, axis=0)
    max_accuracies = np.max(accuracies, axis=0)
    mean_accuracies = np.mean(accuracies, axis=0)
    std_accuracies = np.std(accuracies, axis=0)
    deltas = accuracies - mean_accuracies

    # Print per-class accuracy statistics
    print("Per-Class Accuracy Statistics:")
    for i in range(10):  # Assuming first 10 classes
        print(f"Class {i}: Min={min_accuracies[i]:.2f}, Max={max_accuracies[i]:.2f}, Mean={mean_accuracies[i]:.2f}, Std Dev={std_accuracies[i]:.2f}")

    # Print other metrics
    print('Average pairwise accuracy delta (%):', np.mean(acc_deltas))
    print('Average pairwise cross-entropy delta:', np.mean(ce_deltas))
    print('Average pairwise correlation (spearman):', np.mean(spearmans))
    print('Average pairwise correlation (pearson):', np.mean(pearsons))
    print('Average pairwise disagreement (%):', np.mean(disagreements))

def get_per_class_accuracies(checkpoint_path):   # new addition for accuracies per class
    checkpoint = torch.load(checkpoint_path)
    if 'per_class_acc' in checkpoint:
        return checkpoint['per_class_acc']
    else:
        raise ValueError("per_class_acc not found in checkpoint")

def get_overall_accuracy(checkpoint_path):
    """Extracts overall test accuracy from the checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    if 'acc' in checkpoint:
        return checkpoint['acc']
    else:
        raise ValueError("test_accuracy not found in checkpoint")
        
#for reading the csv files generated during training process
def read_training_loss_data(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        losses = [float(row[1]) for row in reader]
    return losses
    
#after reading the files , generating seed vs loss curves 
def plot_combined_training_loss(loss_data, num_seeds, filename='combined_training_loss.png'):
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
'''
def plot_accuracy_density(accuracies, seeds, bins=10, filename='accuracy_density_plot.png'):
    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))
    sns.histplot(accuracies, bins=bins, kde=False, color='blue', alpha=0.3)
    sns.kdeplot(accuracies, color='black', linewidth=2)

    # Scatter plot for seeds
    for acc, seed in zip(accuracies, seeds):
        plt.scatter(acc, 0, label=f'Seed {seed}')  # Plot each seed as a point

    plt.title('Accuracy Density Plot')
    plt.xlabel('Accuracy')
    plt.ylabel('Density')

    # Save the plot in the current working directory
    current_dir = os.getcwd()  # Gets the current working directory
    path = os.path.join(current_dir, filename)  # Joins the directory with the filename
    plt.savefig(path)
    plt.close()  # Close the figure to free memory
'''  
    
def plot_accuracy_density_jitter(accuracies, seeds, bins=10, filename='accuracy_density_plot_jitter.png'):
    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))

    # Jitter the seeds for better visibility
    seed_jitter = np.random.normal(0, 0.1, size=len(seeds))

    # Plot histogram
    sns.histplot(accuracies, bins=bins, kde=False, color='blue', alpha=0.3)

    # Overlay KDE
    sns.kdeplot(accuracies, color='black', linewidth=2)

    # Scatter plot for seeds with jitter
    plt.scatter(accuracies + seed_jitter, [0]*len(seeds), color='red', label='Seeds')

    plt.title('Accuracy Density Plot with Jitter')
    plt.xlabel('Accuracy')
    plt.ylabel('Density')
    plt.legend()
    current_dir = os.getcwd()  # Gets the current working directory
    path = os.path.join(current_dir, filename)  # Joins the directory with the filename
    plt.savefig(path)
    plt.close()  # Close the figure to free memory

def plot_accuracy_boxplot(accuracies, seeds, filename='accuracy_boxplot.png'):
    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))

    # Create a DataFrame for seaborn plotting
    data = pd.DataFrame({'Accuracy': accuracies, 'Seed': seeds})

    # Box plot
    sns.boxplot(x='Seed', y='Accuracy', data=data)

    # Overlay points
    sns.stripplot(x='Seed', y='Accuracy', data=data, color='red', jitter=True, size=4)

    plt.title('Accuracy Box Plot by Seed')
    plt.xlabel('Seed')
    plt.ylabel('Accuracy')
    current_dir = os.getcwd()  # Gets the current working directory
    path = os.path.join(current_dir, filename)  # Joins the directory with the filename
    plt.savefig(path)
    plt.close()  # Close the figure to free memory
   
def plot_accuracy_swarm(accuracies, seeds, filename='accuracy_swarm_plot.png'):
    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))

    # Create a DataFrame for seaborn plotting

    data = pd.DataFrame({'Accuracy': accuracies, 'Seed': seeds})

    # Swarm plot
    sns.swarmplot(x='Seed', y='Accuracy', data=data, size=4)

    plt.title('Accuracy Swarm Plot by Seed')
    plt.xlabel('Seed')
    plt.ylabel('Accuracy')
    current_dir = os.getcwd()  # Gets the current working directory
    path = os.path.join(current_dir, filename)  # Joins the directory with the filename
    plt.savefig(path)
    plt.close()  # Close the figure to free memory

def plot_accuracy_density_bar(accuracies, seeds, bins=10, filename='accuracy_density_plot_bar.png'):
    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 6))

    # Calculate histogram bins and counts
    counts, bin_edges = np.histogram(accuracies, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create a bar plot
    plt.bar(bin_centers, counts, width=np.diff(bin_edges), align='center', alpha=0.3, color='blue')

    # Annotate each bar with seed numbers
    for i in range(len(bin_centers)):
        seeds_in_bin = [seeds[j] for j in range(len(accuracies)) if bin_edges[i] <= accuracies[j] < bin_edges[i+1]]
        if seeds_in_bin:  # Check if there are any seeds in the bin
            seed_text = ', '.join(map(str, seeds_in_bin))
            plt.text(bin_centers[i], counts[i], seed_text, ha='center', va='bottom', fontsize=8, rotation=90)

    # Overlay KDE
    sns.kdeplot(accuracies, color='black', linewidth=2)

    plt.title('Accuracy Density Plot with Seed Annotations')
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    current_dir = os.getcwd()  # Gets the current working directory
    path = os.path.join(current_dir, filename)  # Joins the directory with the filename
    plt.savefig(path)
    plt.close()  # Close the figure to free memory
    
'''
def plot_accuracy_density(accuracies, bins=10, filename='accuracy_density_plot.png'):
    """Plots and saves the accuracy density plot in the current working directory."""
    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))
    sns.kdeplot(accuracies, fill=True, color="r", label="Accuracy KDE")
    sns.histplot(accuracies, bins=bins, kde=True, stat='density', linewidth=0, alpha=0.3, color='blue')
    plt.title('Accuracy Density Plot')
    plt.xlabel('Accuracy')
    plt.ylabel('Density')
    plt.legend()
'''
    

'''
  
def run_evaluation(checkpoint_paths, model_type='',
    dataset_name='cifar10', batch_size=128, verbose=False):
  """Computes variability metrics for the given models."""
  with torch.no_grad():
    model_preds = {}
    per_class_accuracies = []                               #my_addition
    for i, checkpoint_path_or_paths in enumerate(checkpoint_paths):
      if verbose:
        print(f'Predicting for model {i+1}/{len(checkpoint_paths)}')
      if isinstance(checkpoint_path_or_paths, list):
        ensemble_paths = checkpoint_path_or_paths
      else:
        ensemble_paths = [checkpoint_path_or_paths]
      key = '|'.join(ensemble_paths)
      model_preds[key] = predict(
          ensemble_paths, model_type, dataset_name=dataset_name, batch_size=batch_size)
    model_preds = {k:v for k, v in model_preds.items() if len(v) > 0}
    print(f'n={len(model_preds)}')
    checkpoint_paths = list(model_preds.keys())

    if verbose:
      print(f'Loading test labels')
    test_labels = get_testlabels(dataset_name, batch_size=batch_size)
    compute_metrics(list(model_preds.values()), test_labels, verbose=verbose)
'''
def run_evaluation(checkpoint_paths, model_type='', dataset_name='cifar10', batch_size=128, verbose=False):
    with torch.no_grad():
        model_preds = {}
        per_class_accuracies = []
        overall_accuracies = []
        seeds = []  # List to track seeds
        for i, checkpoint_path_or_paths in enumerate(checkpoint_paths):
            if verbose:
                print(f'Predicting for model {i+1}/{len(checkpoint_paths)}')
            if isinstance(checkpoint_path_or_paths, list):
                ensemble_paths = checkpoint_path_or_paths
            else:
                ensemble_paths = [checkpoint_path_or_paths]

            key = '|'.join(ensemble_paths)
            model_preds[key] = predict(ensemble_paths, model_type, dataset_name=dataset_name, batch_size=batch_size)
            per_class_acc = get_per_class_accuracies(ensemble_paths[0])
            per_class_accuracies.append(per_class_acc)
            overall_acc = get_overall_accuracy(ensemble_paths[0])
            overall_accuracies.append(overall_acc)
            seeds.append(i+1)  # Assuming seed is i+1 (or modify as needed)

        model_preds = {k: v for k, v in model_preds.items() if len(v) > 0}
        test_labels = get_testlabels(dataset_name, batch_size=batch_size)
        compute_metrics(list(model_preds.values()), test_labels, per_class_accuracies, verbose=verbose)
 #       plot_accuracy_density(overall_accuracies, seeds)  # Pass seeds here
        plot_accuracy_density_jitter(overall_accuracies,seeds)
        plot_accuracy_boxplot(overall_accuracies,seeds)
        plot_accuracy_swarm(overall_accuracies,seeds)
        plot_accuracy_density_bar(overall_accuracies,seeds)
        
if __name__ == '__main__':
  # Main execution code here (same as in the original script)
  # ...
  num_runs = 5
  model_dirs = [f'Res_14_slr_{i}' for i in range(1, num_runs + 1)]
  paths = [os.path.join('cifar10_models', d, 'model.ckpt') for d in model_dirs]
  dataset_name = 'cifar10'
  
  print('Running model evaluation...')
  run_evaluation(paths, model_type='resnet14', dataset_name=dataset_name, 
                 batch_size=128, verbose=False)
  print()
  
  training_loss_data = {}
  for seed, model_dir in zip(range(1, num_runs + 1), model_dirs):
        # Construct the path to the training loss CSV file
        file_path = os.path.join('cifar10_models', model_dir, f'training_loss_seed_{seed}.csv')
        training_loss_data[seed] = read_training_loss_data(file_path)

    # Plot combined training loss
  plot_combined_training_loss(training_loss_data, num_runs)
