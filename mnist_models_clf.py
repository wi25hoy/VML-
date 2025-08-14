# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import model_utils

# Define the StableLoss class
class StableLoss(nn.Module):
    def __init__(self, loss_fn, weight):
        super(StableLoss, self).__init__()
        self.loss_fn = loss_fn
        self.weight = weight
        self.prev_loss = None

    def forward(self, input, target):
        current_loss = self.loss_fn(input, target)
        if self.prev_loss is not None:
            delta = torch.abs(current_loss - self.prev_loss.detach())
            current_loss = current_loss + self.weight * delta
        self.prev_loss = current_loss.detach().clone()
        return current_loss

# Define the VariancePenalty class
class VariancePenalty(nn.Module):
    def __init__(self, vpl_weight_decay):
        super(VariancePenalty, self).__init__()
        self.vpl_weight_decay = vpl_weight_decay

    def forward(self, input):
        return self.vpl_weight_decay * torch.var(input, dim=0).mean()

class MnistNet(nn.Module):
    def __init__(self, num_classes=10, init_strategy='he', stable_weight=0.1, vpl_weight_decay=0.1, vpl_weight=0.1):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.stable_loss = StableLoss(nn.CrossEntropyLoss(), stable_weight)
        self.variance_penalty = VariancePenalty(vpl_weight_decay)
        self.vpl_weight = vpl_weight
        
        self._initialize_weights(init_strategy)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def _initialize_weights(self, init_strategy):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_strategy == 'he':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_strategy == 'xavier':
                    init.xavier_normal_(m.weight)
                elif init_strategy == 'custom_uniform':
                    init.uniform_(m.weight, -0.0089, 0.0089)
                    print("Custom Uniform Initialization Applied:")
                    print(f"Layer: {m}, Min Weight: {torch.min(m.weight).item()}, Max Weight: {torch.max(m.weight).item()}")
                elif init_strategy == 'custom_xavier':
                    init.xavier_normal_(m.weight)
                    m.weight.data = torch.clamp(m.weight.data, -0.0089, 0.0089)
                elif init_strategy == 'custom_kaiming':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    m.weight.data = torch.clamp(m.weight.data, -0.0089, 0.0089)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

class MnistLinearNet(nn.Module):
    """Linear network"""
    def __init__(self, num_classes=10, init_strategy='he', stable_weight=0.1, vpl_weight_decay=0.1, vpl_weight=0.1):
        super(MnistLinearNet, self).__init__()
        self.fc = nn.Linear(28*28*1, num_classes)
        
        self.stable_loss = StableLoss(nn.CrossEntropyLoss(), stable_weight)
        self.variance_penalty = VariancePenalty(vpl_weight_decay)
        self.vpl_weight = vpl_weight
        
        self._initialize_weights(init_strategy)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self, init_strategy):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_strategy == 'he':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_strategy == 'xavier':
                    init.xavier_normal_(m.weight)
                elif init_strategy == 'custom_uniform':
                    init.uniform_(m.weight, -0.0089, 0.0089)
                    print("Custom Uniform Initialization Applied:")
                    print(f"Layer: {m}, Min Weight: {torch.min(m.weight).item()}, Max Weight: {torch.max(m.weight).item()}")
                elif init_strategy == 'custom_xavier':
                    init.xavier_normal_(m.weight)
                    m.weight.data = torch.clamp(m.weight.data, -0.0089, 0.0089)
                elif init_strategy == 'custom_kaiming':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    m.weight.data = torch.clamp(m.weight.data, -0.0089, 0.0089)
                init.constant_(m.bias, 0)

class MnistHiddenNet1(nn.Module):
    """FC net with one hidden FC layer"""
    def __init__(self, num_classes=10, init_strategy='he', stable_weight=0.1, vpl_weight_decay=0.1, vpl_weight=0.1):
        super(MnistHiddenNet1, self).__init__()
        self.fc1 = nn.Linear(28*28*1, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)
        
        self.stable_loss = StableLoss(nn.CrossEntropyLoss(), stable_weight)
        self.variance_penalty = VariancePenalty(vpl_weight_decay)
        self.vpl_weight = vpl_weight
        
        self._initialize_weights(init_strategy)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def _initialize_weights(self, init_strategy):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_strategy == 'he':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_strategy == 'xavier':
                    init.xavier_normal_(m.weight)
                elif init_strategy == 'custom_uniform':
                    init.uniform_(m.weight, -0.0089, 0.0089)
                    print("Custom Uniform Initialization Applied:")
                    print(f"Layer: {m}, Min Weight: {torch.min(m.weight).item()}, Max Weight: {torch.max(m.weight).item()}")
                elif init_strategy == 'custom_xavier':
                    init.xavier_normal_(m.weight)
                    m.weight.data = torch.clamp(m.weight.data, -0.0089, 0.0089)
                elif init_strategy == 'custom_kaiming':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    m.weight.data = torch.clamp(m.weight.data, -0.0089, 0.0089)
                init.constant_(m.bias, 0)

class MnistHiddenNet2(nn.Module):
    """FC net with one hidden conv layer"""
    def __init__(self, num_classes=10, init_strategy='he', stable_weight=0.1, vpl_weight_decay=0.1, vpl_weight=0.1):
        super(MnistHiddenNet2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5,
                               stride=2, padding=2, bias=True)
        self.fc1 = nn.Linear(14*14*64, num_classes)
        self.relu = nn.ReLU(inplace=True)
        
        self.stable_loss = StableLoss(nn.CrossEntropyLoss(), stable_weight)
        self.variance_penalty = VariancePenalty(vpl_weight_decay)
        self.vpl_weight = vpl_weight
        
        self._initialize_weights(init_strategy)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def _initialize_weights(self, init_strategy):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_strategy == 'he':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_strategy == 'xavier':
                    init.xavier_normal_(m.weight)
                elif init_strategy == 'custom_uniform':
                    init.uniform_(m.weight, -0.0089, 0.0089)
                    print("Custom Uniform Initialization Applied:")
                    print(f"Layer: {m}, Min Weight: {torch.min(m.weight).item()}, Max Weight: {torch.max(m.weight).item()}")
                elif init_strategy == 'custom_xavier':
                    init.xavier_normal_(m.weight)
                    m.weight.data = torch.clamp(m.weight.data, -0.0089, 0.0089)
                elif init_strategy == 'custom_kaiming':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    m.weight.data = torch.clamp(m.weight.data, -0.0089, 0.0089)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

def mnistnet(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = MnistNet(**kwargs)
    model_utils.restore_rng_state(old_state)
    return model

def mnistlinearnet(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = MnistLinearNet(**kwargs)
    model_utils.restore_rng_state(old_state)
    return model

def mnisthiddennet1(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = MnistHiddenNet1(**kwargs)
    model_utils.restore_rng_state(old_state)
    return model

def mnisthiddennet2(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = MnistHiddenNet2(**kwargs)
    model_utils.restore_rng_state(old_state)
    return model
