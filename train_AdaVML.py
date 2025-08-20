# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse
import csv
import math
import os
import struct
import time
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')  # headless plotting just in case
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import data
import models_clf
import utils
import ssl

# If you re-exported in models_clf/__init__.py:
from models_clf import AdaptiveLabelVariancePenalty as AVPLClass
# Otherwise, use:
# from models_clf.resnet_cifar import AdaptiveLabelVariancePenalty as AVPLClass

# ---------------------------
# TLS relax (your existing block)
# ---------------------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# ---------------------------
# CLI
# ---------------------------
parser = argparse.ArgumentParser()

# General training args
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name')
parser.add_argument('--model_type', default='resnet14', type=str, help='Model name, from models_clf.py')
parser.add_argument('--model_dir', default='tmp_model', type=str, help='Directory to save model and logs in')
parser.add_argument('--resume_from', default='', type=str, help='Path to saved model to initialize all state from')

# Optimizer args
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--num_epochs', default=200, type=int, help='Total number of epochs to train for')
parser.add_argument('--max_lr', default=.40, type=float, help='Max learning rate')
parser.add_argument('--lr_schedule', default='cosine', type=str, help='Name of learning rate schedule')
parser.add_argument('--warmup_epochs', default=3., type=float, help='Number of epochs for learning rate warmup')
parser.add_argument('--momentum', default=0.9, type=float)

# Seeds
parser.add_argument('--data_aug_seed', default=1, type=int, help='Seed for data augmentation')
parser.add_argument('--cudnn_seed', default=1, type=int, help='Seed for cuDNN (1=deterministic)')
parser.add_argument('--shuffle_train_seed', default=1, type=int, help='Seed for training data shuffling')
parser.add_argument('--init_seed', default=1, type=int, help='Seed for weight initialization')
parser.add_argument('--use_seeds', type=int, default=1,
                    help='1=use the provided seeds (default). 0=ignore seeds and run fully random.')
# Bit flip experiment
parser.add_argument('--random_bit_change', default=0, type=int, help='Type of random bit change to do (see code), 0=no change.')
parser.add_argument('--random_bit_change_seed', default=0, type=int, help='Seed for random bit changes')

# Model init + legacy start gate for custom losses (kept for compatibility)
parser.add_argument('--init_strategy', default='he', type=str,
                    help='Weight initialization strategy (he, xavier, custom_uniform, custom_xavier, custom_kaiming)')
parser.add_argument('--custom_loss_start_epoch', default=150, type=int,
                    help='[Compat] Epoch to start custom losses if explicit flags not provided')

# ---------------------------
# VPL settings + toggles (Adaptive VPL)
# ---------------------------
parser.add_argument('--use_vpl', action='store_true',
                    help='Enable Variance Penalty Loss (Adaptive VPL)')
parser.add_argument('--vpl_start_epoch', type=int, default=None,
                    help='Epoch to start VPL (defaults to custom_loss_start_epoch if None)')
parser.add_argument('--vpl_weight', default=0.05, type=float, help='Global master multiplier on VPL (in addition to adaptive lambda)')
parser.add_argument('--vpl_weight_decay', default=1.0, type=float, help='Fixed scale inside adaptive VPL (acts like old decay)')
# Adaptive VPL internals
parser.add_argument('--avpl_lambda_base', default=1.0, type=float, help='Base gain for adaptive VPL controller')
parser.add_argument('--avpl_alpha', default=0.10, type=float, help='EMA smoothing for v_ema')
parser.add_argument('--avpl_warmup_steps', default=100, type=int, help='Warmup steps to set v_ref')
parser.add_argument('--avpl_lambda_min', default=0.0, type=float, help='Min clamp for adaptive VPL lambda')
parser.add_argument('--avpl_lambda_max', default=2.0, type=float, help='Max clamp for adaptive VPL lambda')
parser.add_argument('--avpl_use_entropy', default=0, type=int, help='1=modulate lambda by entropy; 0=off')
parser.add_argument('--avpl_entropy_scale', default=0.5, type=float, help='Strength of entropy modulation')

# NEW: VPL statistic selector
parser.add_argument('--vpl_stat', type=str, default='vector',
                    choices=['vector', 'true'],
                    help="Statistic for VPL: 'vector' = whole logit vector variance; 'true' = true-class logit variance.")

# ---------------------------
# Adaptive Stable Loss (SL) toggles + hyperparams
# ---------------------------
parser.add_argument('--use_sl', action='store_true',
                    help='Enable Adaptive Stable Loss (EMA baseline + Huber + volatility EMA)')
parser.add_argument('--sl_start_epoch', type=int, default=None,
                    help='Epoch to start SL (defaults to custom_loss_start_epoch if None)')
parser.add_argument('--stable_weight', default=0.5, type=float,
                    help='Base scale (lambda_base) for SL, passed to model')

# SL internals (match your math)
parser.add_argument('--sl_alpha', default=0.10, type=float, help='EMA smoothing rate for baseline Lbar_t (alpha)')
parser.add_argument('--sl_beta', default=0.05, type=float, help='EMA smoothing rate for volatility sigma_t (beta)')
parser.add_argument('--sl_delta_frac', default=0.10, type=float, help='Huber threshold as fraction of scale (delta = frac * scale)')
parser.add_argument('--sl_lambda_min', default=0.0, type=float, help='Min clamp for lambda_t')
parser.add_argument('--sl_lambda_max', default=1.0, type=float, help='Max clamp for lambda_t')
parser.add_argument('--sl_warmup_steps', default=100, type=int, help='Warmup steps to build sigma_ref/median')
parser.add_argument('--sl_use_running_ref', default=1, type=int, help='1=use running reference for sigma_ref, 0=use first sigma_ema')
parser.add_argument('--sl_eps', default=1e-8, type=float, help='Small epsilon for lambda_t stability')

# ---------------------------
# W&B logging (opt-in)
# ---------------------------
parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default='vml-experiments', help='W&B project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (team/user). Leave None to use default.')
parser.add_argument('--wandb_group', type=str, default=None, help='W&B group, defaults to f\"{dataset}-{model_type}\"')
parser.add_argument('--wandb_job_type', type=str, default=None, help='W&B job_type, defaults to condition (base/sl/vpl/vml)')
parser.add_argument('--wandb_name', type=str, default=None, help='W&B run name, defaults to f\"{condition}-seed{init_seed:02d}\"')
parser.add_argument('--wandb_mode', type=str, default=None, help='W&B mode (online/offline/disabled). None respects env.')
parser.add_argument('--log_interval', type=int, default=50, help='Log train scalars/internals every N batches when using W&B')
parser.add_argument('--auto_wandb_live', type=int, default=0,
                    help='If 1, auto-run wandb_live_viz.py in background when W&B is enabled')

# ---------------------------
# Batch CSV + gradient sanity
# ---------------------------
parser.add_argument('--log_batches_csv', type=int, default=1,
                    help='Write per-batch CSV with step metrics (1=yes)')
parser.add_argument('--check_grad_every', type=int, default=50,
                    help='Check/record gradient stats every N steps')
parser.add_argument('--grad_zero_alert', type=int, default=1,
                    help='1=alert if grad norm is zero or NaN/Inf')

args = parser.parse_args()
args.dataset = args.dataset.lower()

# Backward-compatible defaults for start epochs
if args.sl_start_epoch is None:
    args.sl_start_epoch = args.custom_loss_start_epoch
if args.vpl_start_epoch is None:
    args.vpl_start_epoch = args.custom_loss_start_epoch


# ---------------------------
# Utilities
# ---------------------------
def _git_commit_hash_or_none():
    try:
        h = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL)
        return h.decode('utf-8').strip()
    except Exception:
        return None

def _infer_condition_name(use_sl: bool, use_vpl: bool) -> str:
    if use_sl and use_vpl: return "vml"
    if use_sl and not use_vpl: return "sl"
    if use_vpl and not use_sl: return "vpl"
    return "base"

def _to_float_or_nan(x):
    try:
        return float(x)
    except Exception:
        try:
            return float(x.item())
        except Exception:
            return float('nan')

def grad_stats(model):
    total_sq = 0.0
    zero_norm_params = 0
    none_grad_params = 0
    param_count = 0
    has_nan_inf = False
    with torch.no_grad():
        for p in model.parameters():
            param_count += 1
            if p.grad is None:
                none_grad_params += 1
                continue
            g = p.grad
            if torch.any(torch.isnan(g)) or torch.any(torch.isinf(g)):
                has_nan_inf = True
            n = torch.norm(g).item()
            if n == 0.0:
                zero_norm_params += 1
            total_sq += float((g * g).sum().item())
    total_norm = math.sqrt(total_sq)
    return {
        "grad_norm": total_norm,
        "zero_grad_params": zero_norm_params,
        "none_grad_params": none_grad_params,
        "param_count": param_count,
        "has_nan_inf": has_nan_inf,
    }


# ---------------------------
# Functions
# ---------------------------
def _variance_penalty_safe(net, outputs, targets):
    """Always calls (outputs, targets) for adaptive VPL (and falls back to (outputs) if needed)."""
    vp = torch.zeros((), device=outputs.device)
    if hasattr(net, 'variance_penalty') and callable(getattr(net, 'variance_penalty')):
        try:
            vp = net.variance_penalty(outputs, targets)
        except TypeError:
            vp = net.variance_penalty(outputs)
    return vp


def train(epoch):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0.0
    correct = 0
    total = 0
    epoch_losses = []
    class_correct = list(0. for _ in range(num_classes))
    class_total = list(0. for _ in range(num_classes))

    start_t = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        assert targets.max() < num_classes, f"Target label `{targets.max().item()}` is out of bounds."
        assert targets.min() >= 0, f"Target label `{targets.min().item()}` is out of bounds."

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        base_loss = cel(outputs, targets)
        total_loss = base_loss

        # --- Adaptive Stable Loss (SL) ---
        sl_penalty = torch.zeros((), device=device)
        if args.use_sl and epoch >= args.sl_start_epoch:
            if hasattr(net, 'adaptive_stable_loss') and callable(getattr(net, 'adaptive_stable_loss')):
                sl_penalty = net.adaptive_stable_loss(base_loss)
            elif hasattr(net, 'stable_mod') and callable(getattr(net, 'stable_mod')):
                sl_penalty = net.stable_mod(base_loss)
            elif hasattr(net, 'stable_loss'):
                sl_penalty = net.stable_loss(outputs, targets)  # legacy
            total_loss = total_loss + sl_penalty

        # --- VPL eligibility (sanity) ---
        eligible_classes = 0
        eligible_frac = float('nan')
        if args.use_vpl:
            counts = torch.bincount(targets, minlength=num_classes)
            eligible_classes = int((counts >= 2).sum().item())
            denom = int((counts > 0).sum().item())
            eligible_frac = (eligible_classes / max(1, denom)) if denom > 0 else float('nan')

        # --- Adaptive VPL (always adaptive) ---
        vpl_penalty = torch.zeros((), device=device)
        if args.use_vpl and epoch >= args.vpl_start_epoch:
            vp = _variance_penalty_safe(net, outputs, targets)
            vpl_penalty = net.vpl_weight * vp
            total_loss = total_loss + vpl_penalty

        # record for timing/throughput
        step_t0 = time.time()
        total_loss.backward()

        grad_stat_sample = None
        if (batch_idx % max(1, args.check_grad_every)) == 0:
            grad_stat_sample = grad_stats(net)

        optimizer.step()
        lr_scheduler.step()
        step_time = time.time() - step_t0
        throughput = float(inputs.size(0)) / max(step_time, 1e-8)

        # running aggregates
        train_loss += total_loss.item()
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == targets).squeeze()
        epoch_losses.append(train_loss / (batch_idx + 1))

        for i in range(min(targets.size(0), num_classes)):
            label = targets[i].item()
            if label < num_classes:
                class_correct[label] += c[i].item()
                class_total[label] += 1

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total

        # quick glance at VPL controller
        if args.use_vpl and hasattr(net, 'variance_penalty') and hasattr(net.variance_penalty, 'last_lambda'):
            if (batch_idx % max(1, args.log_interval)) == 0:
                lam_val = float(net.variance_penalty.last_lambda)
                vbatch  = float(net.variance_penalty.last_vbatch)
                print(f"[VPL] lambda_t={lam_val:.4f}  v_batch={vbatch:.6f}  eligible={eligible_classes}")

        # ---- W&B step logging (every N batches) ----
        if use_wandb and ((batch_idx % max(1, args.log_interval)) == 0):
            lr_now = lr_scheduler.get_last_lr()
            lr_now = float(lr_now[0] if isinstance(lr_now, list) else lr_now)
            step_payload = {
                "train/ce": float(base_loss.item()),
                "train/total_loss": float(total_loss.item()),
                "train/acc_running": float(acc),
                "lr": lr_now,
                "time/step_s": step_time,
                "time/throughput_img_s": throughput,
                "vpl/eligible_classes": eligible_classes,
                "vpl/eligible_frac": eligible_frac,
            }
            if args.use_sl and epoch >= args.sl_start_epoch:
                step_payload["train/sl_penalty"] = float(sl_penalty.item())
            if args.use_vpl and epoch >= args.vpl_start_epoch:
                step_payload["train/vpl_penalty"] = float(vpl_penalty.item())

                # VPL internals
                vp_obj = net.variance_penalty
                if hasattr(vp_obj, "last_lambda"):
                    step_payload["vpl/lambda_t"] = float(vp_obj.last_lambda)
                if hasattr(vp_obj, "last_vbatch"):
                    step_payload["vpl/v_batch"] = float(vp_obj.last_vbatch)
                if hasattr(vp_obj, "v_ema"):
                    step_payload["vpl/v_ema"] = float(vp_obj.v_ema)
                if hasattr(vp_obj, "v_ref"):
                    step_payload["vpl/v_ref"] = float(vp_obj.v_ref)

            # SL internals
            sl_obj = getattr(net, 'stable_mod', None)
            if sl_obj is None:
                sl_obj = getattr(net, 'adaptive_stable_loss', None)
            if sl_obj is not None:
                lam_val = getattr(sl_obj, "last_lambda", None)
                if lam_val is None and hasattr(sl_obj, "lambda_t"):
                    lam_val = getattr(sl_obj, "lambda_t")
                if lam_val is not None:
                    step_payload["sl/lambda_t"] = _to_float_or_nan(lam_val)
                if hasattr(sl_obj, "sigma_ema"):
                    step_payload["sl/sigma_ema"] = _to_float_or_nan(sl_obj.sigma_ema)
                if hasattr(sl_obj, "sigma_ref"):
                    step_payload["sl/sigma_ref"] = _to_float_or_nan(sl_obj.sigma_ref)
                if hasattr(sl_obj, "l_ema"):
                    step_payload["sl/l_ema"] = _to_float_or_nan(sl_obj.l_ema)

            # W&B live helper if available; else fallback
            if wb is not None:
                wb.log_step_scalars(step_payload, step=global_step)
                wb.maybe_log_grad_hist(net, step=global_step)
            else:
                import wandb
                wandb.log(step_payload, step=global_step)

        # Write per-batch CSV if enabled (and stream to W&B table)
        if batch_csv:
            vp_obj = getattr(net, 'variance_penalty', None)
            sl_obj = getattr(net, 'stable_mod', None) or getattr(net, 'adaptive_stable_loss', None)
            lr_now = lr_scheduler.get_last_lr()
            lr_now = float(lr_now[0] if isinstance(lr_now, list) else lr_now)

            row = [
                epoch, batch_idx, global_step, lr_now,
                float(base_loss.item()),
                float(sl_penalty.item()) if args.use_sl and epoch >= args.sl_start_epoch else 0.0,
                float(vpl_penalty.item()) if args.use_vpl and epoch >= args.vpl_start_epoch else 0.0,
                float(total_loss.item()), float(acc),
            ]
            # grads
            if grad_stat_sample is not None:
                row += [
                    grad_stat_sample["grad_norm"],
                    grad_stat_sample["zero_grad_params"],
                    grad_stat_sample["none_grad_params"],
                    grad_stat_sample["param_count"],
                    int(grad_stat_sample["has_nan_inf"]),
                ]
            else:
                row += [float('nan'), 0, 0, 0, 0]

            # VPL internals
            row += [
                _to_float_or_nan(getattr(vp_obj, "last_lambda", float('nan'))) if vp_obj else float('nan'),
                _to_float_or_nan(getattr(vp_obj, "v_ema", float('nan'))) if vp_obj else float('nan'),
                _to_float_or_nan(getattr(vp_obj, "v_ref", float('nan'))) if vp_obj else float('nan'),
                _to_float_or_nan(getattr(vp_obj, "last_vbatch", float('nan'))) if vp_obj else float('nan'),
            ]
            # SL internals
            lam_val = None
            if sl_obj is not None:
                lam_val = getattr(sl_obj, "last_lambda", None)
                if lam_val is None and hasattr(sl_obj, "lambda_t"):
                    lam_val = getattr(sl_obj, "lambda_t")
            row += [
                _to_float_or_nan(lam_val) if lam_val is not None else float('nan'),
                _to_float_or_nan(getattr(sl_obj, "sigma_ema", float('nan'))) if sl_obj else float('nan'),
                _to_float_or_nan(getattr(sl_obj, "sigma_ref", float('nan'))) if sl_obj else float('nan'),
                _to_float_or_nan(getattr(sl_obj, "l_ema", float('nan'))) if sl_obj else float('nan'),
            ]
            # eligibility + timing
            row += [eligible_classes, eligible_frac, step_time, throughput]

            with open(batch_csv, 'a') as f:
                csv.writer(f).writerow(row)

            # Stream same row to W&B table
            if use_wandb and wb is not None:
                wb.add_table_row(row)

        utils.progress_bar(batch_idx, len(trainloader),
                           'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                           (train_loss / (batch_idx + 1), acc, correct, total))

        global_step += 1

    per_class_acc = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                     for i in range(min(num_classes, 10))]

    avg_train_loss = train_loss / (batch_idx + 1)
    epoch_time_s = time.time() - start_t
    return avg_train_loss, acc, per_class_acc, epoch_losses, epoch_time_s


def test(epoch):
    """Tests at the given epoch and calculates per-class accuracy."""
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    # infer #classes from loader
    num_classes_test = len(torch.unique(torch.cat([targets for _, targets in testloader], dim=0)))
    class_correct = list(0. for _ in range(num_classes_test))
    class_total = list(0. for _ in range(num_classes_test))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = cel(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == targets).squeeze()
            for i in range(targets.size(0)):
                label = targets[i]
                if label < 10:  # keep your original reporting style
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(testloader),
                               'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                               (test_loss / (batch_idx + 1),
                                100. * float(correct) / float(total), correct, total))

    per_class_acc = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                     for i in range(min(10, num_classes_test))]
    acc = 100. * float(correct) / float(total)
    avg_test_loss = round(test_loss / (batch_idx + 1), 2)
    return avg_test_loss, acc, per_class_acc


def checkpoint(model_dir, acc, epoch, per_class_acc, checkpoint_filename='model.ckpt'):
    """Saves a model checkpoint (with adaptive loss states if present)."""
    print('Saving..')
    state = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'per_class_acc': per_class_acc,
        'rng_state': torch.get_rng_state(),
        'train_shuffle_state': trainloader.sampler.state,
        'stable_mod_state_dict': None,
        'vpl_state_dict': None,
    }
    # Save SL if present
    if hasattr(net, 'stable_mod') and hasattr(net.stable_mod, 'state_dict'):
        state['stable_mod_state_dict'] = net.stable_mod.state_dict()
    elif hasattr(net, 'adaptive_stable_loss') and hasattr(getattr(net, 'adaptive_stable_loss'), 'state_dict'):
        state['stable_mod_state_dict'] = net.adaptive_stable_loss.state_dict()
    # Save adaptive VPL state if present
    if hasattr(net, 'variance_penalty') and hasattr(net.variance_penalty, 'state_dict'):
        state['vpl_state_dict'] = net.variance_penalty.state_dict()

    torch.save(state, os.path.join(model_dir, checkpoint_filename))


def learning_rate_fn(step):
    """Learning rate function, given a global step."""
    steps_per_epoch = int(math.ceil(len(trainloader)))
    epoch = step / float(steps_per_epoch)

    if args.lr_schedule == 'cosine':
        if epoch <= args.warmup_epochs:
            lr = step / (args.warmup_epochs * steps_per_epoch) * args.max_lr
        else:
            lr_min = 0.
            lr = lr_min + .5 * (args.max_lr - lr_min) * (1. + math.cos(min(epoch / args.num_epochs, 1.) * math.pi))
        return lr
    elif args.lr_schedule == 'snapshot5':
        if args.num_epochs % 5:
            raise ValueError('Number of epochs for snapshot5 learning rate schedule must be divisible by 5.')
        cycle_length = args.num_epochs // 5
        start_epochs = range(0, args.num_epochs, cycle_length)
        for start_epoch in start_epochs:
            if epoch < start_epoch + cycle_length:
                lr_min = 0.
                if epoch <= args.warmup_epochs + start_epoch:
                    lr = (epoch - start_epoch) / args.warmup_epochs * args.max_lr
                else:
                    lr = lr_min + .5 * (args.max_lr - lr_min) * (1. + math.cos(min((epoch - start_epoch) / cycle_length, 1.) * math.pi))
                return min(max(lr, 0.), args.max_lr)
        return 0.
    else:
        raise ValueError(f'Unsupported learning rate: {args.lr_schedule}')


def resume(checkpoint_path, model):
    """Restores network, optimizer/scheduler, RNG state, and adaptive loss states (if present)."""
    print(f'Resuming from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    epoch = checkpoint['epoch']

    # Restore rng state & sampler state
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
    trainloader.sampler.state = checkpoint['train_shuffle_state']

    # Restore SL state if present
    if 'stable_mod_state_dict' in checkpoint and checkpoint['stable_mod_state_dict'] is not None:
        if hasattr(model, 'stable_mod'):
            model.stable_mod.load_state_dict(checkpoint['stable_mod_state_dict'])
            print('Restored adaptive stable loss state (stable_mod).')
        elif hasattr(model, 'adaptive_stable_loss'):
            model.adaptive_stable_loss.load_state_dict(checkpoint['stable_mod_state_dict'])
            print('Restored adaptive stable loss state (adaptive_stable_loss).')

    # Restore Adaptive VPL state if present
    if 'vpl_state_dict' in checkpoint and checkpoint['vpl_state_dict'] is not None:
        if hasattr(model, 'variance_penalty'):
            try:
                model.variance_penalty.load_state_dict(checkpoint['vpl_state_dict'])
                print('Restored adaptive VPL state.')
            except Exception as e:
                print('Warning: could not restore adaptive VPL state:', e)

    return epoch


def analyze_initial_weight_distribution(net, seed, model_dir, base_filename='Initial_weights_density_plot.png'):
    conv_weights = []
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            weights = module.weight.data.cpu().numpy()
            conv_weights.extend(weights.flatten())

    if not conv_weights:
        print("No convolutional layer weights to analyze.")
        return None

    conv_weights = np.array(conv_weights)
    min_weight = np.min(conv_weights)
    max_weight = np.max(conv_weights)
    weight_range = max_weight - min_weight

    print(f"Convolutional Layer Weight Distribution:")
    print(f"Min Weight: {min_weight}, Max Weight: {max_weight}, Range: {weight_range}")

    plt.hist(conv_weights, bins=50)
    filename = f"{base_filename}_seed_{seed}.png"
    path = os.path.join(model_dir, filename)
    plt.savefig(path)
    plt.close()
    return min_weight, max_weight, weight_range, path


# ---------------------------
# Main
# ---------------------------
use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')

# Your current cudnn config (kept as-is)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# Data
if int(args.use_seeds) == 1:
    shuffle_seed = args.shuffle_train_seed
else:
    # Fresh random 32-bit int from OS entropy; avoids fixed shuffling across runs
    shuffle_seed = int.from_bytes(os.urandom(4), 'little')
trainloader = data.get_trainloader(args.dataset, args.batch_size, shuffle_seed)
testloader  = data.get_testloader(args.dataset, args.batch_size)

if args.dataset == 'cifar10':
    num_classes = 10
elif args.dataset == 'cifar100':
    num_classes = 100
else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")

# Model
flags_for_model = args if int(args.use_seeds) == 1 else None
net = getattr(models_clf, args.model_type)(
    flags=flags_for_model,
    num_classes=num_classes,
    init_strategy=args.init_strategy,
    stable_weight=args.stable_weight,      # used as lambda_base inside model (SL)
    vpl_weight_decay=args.vpl_weight_decay,
    vpl_weight=args.vpl_weight
)

# Configure adaptive stable loss in the model if you exposed a helper
if hasattr(net, 'configure_adaptive_stable') and callable(getattr(net, 'configure_adaptive_stable')):
    net.configure_adaptive_stable(
        alpha=args.sl_alpha,
        beta=args.sl_beta,
        delta_frac=args.sl_delta_frac,
        lambda_base=args.stable_weight,
        lambda_min=args.sl_lambda_min,
        lambda_max=args.sl_lambda_max,
        warmup_steps=args.sl_warmup_steps,
        use_running_ref=bool(args.sl_use_running_ref),
        eps=args.sl_eps,
    )
elif hasattr(net, 'stable_mod'):
    # direct field update fallback
    for name, value in dict(
        alpha=args.sl_alpha,
        beta=args.sl_beta,
        delta_frac=args.sl_delta_frac,
        lambda_base=args.stable_weight,
        lambda_min=args.sl_lambda_min,
        lambda_max=args.sl_lambda_max,
        eps=args.sl_eps,
    ).items():
        if hasattr(net.stable_mod, name):
            setattr(net.stable_mod, name, value)
    if hasattr(net.stable_mod, 'warmup_steps'):
        net.stable_mod.warmup_steps = args.sl_warmup_steps

# Ensure ALWAYS-ADAPTIVE VPL exists on the model (if your model did not wire it itself)
if args.use_vpl:
    has_callable = hasattr(net, 'variance_penalty') and callable(getattr(net, 'variance_penalty'))
    if not has_callable:
        print("[VPL] Attaching AdaptiveLabelVariancePenalty from models_clf")
        net.variance_penalty = AVPLClass(
            lambda_base=args.avpl_lambda_base,
            scale=args.vpl_weight_decay,
            alpha=args.avpl_alpha,
            warmup_steps=args.avpl_warmup_steps,
            lambda_min=args.avpl_lambda_min,
            lambda_max=args.avpl_lambda_max,
            use_entropy=bool(args.avpl_use_entropy),
            entropy_scale=args.avpl_entropy_scale,
            stat=args.vpl_stat,  # vector|true
        )
    if not hasattr(net, 'vpl_weight'):
        net.vpl_weight = args.vpl_weight

if use_cuda:
    net.cuda()
    print('Using', torch.cuda.device_count(), 'GPU(s).')

# Optionally resume
start_epoch = 0
if args.resume_from:
    epoch = resume(args.resume_from, net)
    start_epoch = epoch + 1

# Model dir
model_dir = os.path.join(args.dataset + '_models', args.model_dir)
if os.path.exists(model_dir):
    print('model_dir already exists: %s' % model_dir)
    print('Exiting...')
    exit()
os.makedirs(model_dir)
print('model_dir: %s' % model_dir)

# --- W&B init ---
use_wandb = bool(args.use_wandb)
wb = None  # live helper (if available)
if use_wandb:
    import wandb
    condition = _infer_condition_name(args.use_sl, args.use_vpl)
    wandb_group = args.wandb_group or f"{args.dataset}-{args.model_type}"
    wandb_job_type = args.wandb_job_type or condition
    seed_tag = f"{args.init_seed:02d}" if int(args.use_seeds) == 1 else "RND"   # <--- NEW
    wandb_name = args.wandb_name or f"{condition}-seed{seed_tag}"
    settings = {}
    if args.wandb_mode is not None:
        settings["mode"] = args.wandb_mode
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=wandb_group,
        job_type=wandb_job_type,
        name=wandb_name,
        config=vars(args),
        **settings
    )
    # Meta
    wandb.config.update({
        "meta/commit_hash": _git_commit_hash_or_none(),
        "meta/num_gpus": torch.cuda.device_count(),
    })
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("test/*", step_metric="epoch")
    wandb.define_metric("time/*", step_metric="epoch")
    wandb.define_metric("vpl/*", step_metric="epoch")
    wandb.define_metric("sl/*", step_metric="epoch")
    wandb.define_metric("grad/*", step_metric="epoch")

    # Try to import live viz helper
    try:
        from wb_live_viz import WBLive
        wb = WBLive(run=wandb.run,
                    use_table=True,          # stream per-batch rows to a W&B Table
                    table_name="batches/stream",
                    grad_hist_every=200)     # set >0 to log gradient histogram every N steps
    except Exception:
        wb = None

    # Save run id for later linking (e.g., wb_push_analysis.py --link_run_id)
    run_id_path = os.path.join(model_dir, "wandb_run_id.txt")
    with open(run_id_path, "w") as f:
        f.write(wandb.run.id + "\n")
    print(f"[W&B] run id: {wandb.run.id} (saved to {run_id_path})")

    # --- Auto-start W&B live viz (background) ---
    if int(args.auto_wandb_live) == 1:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        live_script = os.path.join(this_dir, "wandb_live_viz.py")
        if os.path.isfile(live_script):
            cmd = [sys.executable, live_script, "--project", args.wandb_project]
            if args.wandb_entity:
                cmd += ["--entity", args.wandb_entity]
            if wandb_group:
                cmd += ["--group", wandb_group]
            try:
                with open(os.devnull, 'w') as DEVNULL:
                    proc = subprocess.Popen(cmd, stdout=DEVNULL, stderr=DEVNULL, start_new_session=True)
                pid_path = os.path.join(model_dir, "wandb_live_viz.pid")
                with open(pid_path, "w") as f:
                    f.write(str(proc.pid) + "\n")
                print(f"[W&B] live viz started (pid={proc.pid}) -> {live_script}")
            except Exception as e:
                print(f"[W&B] live viz not started: {e}")
        else:
            print(f"[W&B] live viz skipped: script not found at {live_script}")

# Weight histogram
min_weight, max_weight, weight_range, plot_filepath = analyze_initial_weight_distribution(net, args.init_seed, model_dir)
print(f"Plot saved as {plot_filepath}")
if use_wandb:
    try:
        import wandb
        if wb is not None:
            wb.log_images({"debug/initial_weights_hist": wandb.Image(plot_filepath)}, step=0)
        else:
            wandb.log({"debug/initial_weights_hist": wandb.Image(plot_filepath)}, step=0)
    except Exception:
        pass

# Optimizer / Scheduler
parameters_bias = sorted([p[0] for p in net.named_parameters() if 'bias' in p[0]])
parameters_bnscale = sorted([p[0] for p in net.named_parameters() if 'bn' in p[0] and 'weight' in p[0]])
parameters_others = sorted([p[0] for p in net.named_parameters()
                            if p[0] not in parameters_bias and p[0] not in parameters_bnscale])

def tensor_params(name_list):
    names = set(name_list)
    return [p[1] for p in sorted(net.named_parameters()) if p[0] in name_list]

optimizer = torch.optim.SGD(
    [{'params': tensor_params(parameters_bias), 'lr': .1},
     {'params': tensor_params(parameters_bnscale), 'lr': .1},
     {'params': tensor_params(parameters_others)}],
    lr=1.,  # scheduler will set it
    momentum=args.momentum,
    weight_decay=args.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_fn)
cel = nn.CrossEntropyLoss()

# Logging header
logname = os.path.join(model_dir, 'log.txt')
if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        header = ['epoch', 'lr', 'tr_loss', 'tr_acc']
        header += [f'tr_c {i} acc' for i in range(10)]
        header += ['te_loss', 'te_acc']
        header += [f'te_c {i} acc' for i in range(10)]
        logwriter.writerow(header)

# Internals CSV (for Experiment 0 & analysis without W&B)
internals_csv = os.path.join(model_dir, f"internals_seed_{args.init_seed}.csv")
if not os.path.exists(internals_csv):
    with open(internals_csv, 'w') as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "vpl_lambda", "vpl_v_ema", "vpl_v_ref", "vpl_v_batch",
            "sl_lambda", "sl_sigma_ema", "sl_sigma_ref", "sl_l_ema"
        ])

# Batch CSV (Patch C)
batch_csv = os.path.join(model_dir, f"batches_seed_{args.init_seed}.csv") if args.log_batches_csv else None
if batch_csv and not os.path.exists(batch_csv):
    with open(batch_csv, 'w') as f:
        w = csv.writer(f)
        w.writerow([
            "epoch","batch_idx","global_step","lr",
            "base_loss","sl_pen","vpl_pen","total_loss","acc_running",
            "grad_norm","zero_grad_params","none_grad_params","param_count","has_nan_inf",
            "vpl_lambda","vpl_v_ema","vpl_v_ref","vpl_v_batch",
            "sl_lambda","sl_sigma_ema","sl_sigma_ref","sl_l_ema",
            "eligible_classes","eligible_frac","time_step_s","throughput_img_s"
        ])

# Global step
global_step = 0

# Main training loop
all_epoch_losses = []
for epoch in range(start_epoch, args.num_epochs):
    train_loss, train_acc, train_per_class_acc, epoch_losses, epoch_time_s = train(epoch)
    all_epoch_losses.append(train_loss)

    test_loss, test_acc, test_per_class_acc = test(epoch)

    # W&B epoch-level logging
    if use_wandb:
        payload = {
            "epoch": epoch,
            "train/ce_epoch_avg": float(train_loss),
            "train/acc_epoch": float(train_acc),
            "test/ce": float(test_loss),
            "test/acc": float(test_acc),
            "time/epoch_s": float(epoch_time_s),
        }
        # VPL internals (last values from the epoch)
        if args.use_vpl and hasattr(net, 'variance_penalty'):
            vp = net.variance_penalty
            if hasattr(vp, "last_lambda"): payload["vpl/lambda_t"] = float(vp.last_lambda)
            if hasattr(vp, "v_ema"):       payload["vpl/v_ema"]    = float(vp.v_ema)
            if hasattr(vp, "v_ref"):       payload["vpl/v_ref"]    = float(vp.v_ref)
            if hasattr(vp, "last_vbatch"): payload["vpl/v_batch"]  = float(vp.last_vbatch)
        # SL internals
        sl_obj = getattr(net, 'stable_mod', None)
        if sl_obj is None:
            sl_obj = getattr(net, 'adaptive_stable_loss', None)
        if sl_obj is not None:
            lam_val = getattr(sl_obj, "last_lambda", None)
            if lam_val is None and hasattr(sl_obj, "lambda_t"):
                lam_val = getattr(sl_obj, "lambda_t")
            if lam_val is not None:
                payload["sl/lambda_t"] = _to_float_or_nan(lam_val)
            if hasattr(sl_obj, "sigma_ema"): payload["sl/sigma_ema"] = _to_float_or_nan(sl_obj.sigma_ema)
            if hasattr(sl_obj, "sigma_ref"): payload["sl/sigma_ref"] = _to_float_or_nan(sl_obj.sigma_ref)
            if hasattr(sl_obj, "l_ema"):     payload["sl/l_ema"]     = _to_float_or_nan(sl_obj.l_ema)

        # LR (first param group)
        lr = lr_scheduler.get_last_lr()
        payload["lr"] = float(lr[0] if isinstance(lr, list) else lr)

        if wb is not None:
            wb.log_epoch_scalars(payload, epoch=epoch)
            # Optional: per-class accuracy logs (debug)
            try:
                wb.log_step_scalars({**{f"train/class_acc/{i}": float(a) for i, a in enumerate(train_per_class_acc)}}, step=epoch)
                wb.log_step_scalars({**{f"test/class_acc/{i}": float(a) for i, a in enumerate(test_per_class_acc)}}, step=epoch)
            except Exception:
                pass
        else:
            import wandb
            wandb.log(payload, step=epoch)

        # Optional: confusion matrix snapshot every 10 epochs (safe-guarded)
        LOG_CONFMAT_EVERY = 10
        if (epoch % LOG_CONFMAT_EVERY == 0) or (epoch == args.num_epochs - 1):
            try:
                from sklearn.metrics import confusion_matrix
                net.eval()
                all_preds, all_tgts = [], []
                with torch.no_grad():
                    for x, y in testloader:
                        x = x.to(device); y = y.to(device)
                        logits = net(x)
                        all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                        all_tgts.append(y.cpu().numpy())
                y_pred = np.concatenate(all_preds); y_true = np.concatenate(all_tgts)
                cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
                import wandb
                wandb.log({
                    "eval/confusion_matrix": wandb.plots.HeatMap(
                        x_labels=list(range(num_classes)),
                        y_labels=list(range(num_classes)),
                        matrix_values=cm.tolist(),
                        show_text=False,
                    )
                }, step=epoch)
            except Exception:
                pass

        # Versioned artifacts (CSV + checkpoint) every 25 epochs and at the end
        if (epoch % 25 == 0) or (epoch == args.num_epochs - 1):
            from pathlib import Path
            files_to_save = [
                Path(logname),
                Path(os.path.join(model_dir, f"training_loss_seed_{args.init_seed}.csv")),
                Path(internals_csv),
            ]
            ckpt_path = Path(os.path.join(model_dir, "model.ckpt"))
            if ckpt_path.exists():
                files_to_save.append(ckpt_path)
            # If using live helper, it might expose an artifact API; otherwise use wandb directly
            try:
                import wandb
                art = wandb.Artifact(name=f"{args.model_dir}_epoch{epoch}",
                                     type="training-snapshot",
                                     metadata={"epoch": epoch})
                for fp in files_to_save:
                    if fp.exists():
                        art.add_file(str(fp))
                wandb.log_artifact(art, aliases=["latest", f"epoch-{epoch}"])
            except Exception:
                pass

    # Append internals to CSV (epoch-level snapshot)
    with open(internals_csv, 'a') as f:
        w = csv.writer(f)
        vp = getattr(net, 'variance_penalty', None)
        vpl_lambda = _to_float_or_nan(getattr(vp, "last_lambda", float('nan'))) if vp else float('nan')
        vpl_v_ema  = _to_float_or_nan(getattr(vp, "v_ema", float('nan'))) if vp else float('nan')
        vpl_v_ref  = _to_float_or_nan(getattr(vp, "v_ref", float('nan'))) if vp else float('nan')
        vpl_v_batch= _to_float_or_nan(getattr(vp, "last_vbatch", float('nan'))) if vp else float('nan')

        sl_obj = getattr(net, 'stable_mod', None)
        if sl_obj is None:
            sl_obj = getattr(net, 'adaptive_stable_loss', None)
        lam_val = None
        if sl_obj is not None:
            lam_val = getattr(sl_obj, "last_lambda", None)
            if lam_val is None and hasattr(sl_obj, "lambda_t"):
                lam_val = getattr(sl_obj, "lambda_t")
        sl_lambda   = _to_float_or_nan(lam_val) if lam_val is not None else float('nan')
        sl_sigma_ema= _to_float_or_nan(getattr(sl_obj, "sigma_ema", float('nan'))) if sl_obj else float('nan')
        sl_sigma_ref= _to_float_or_nan(getattr(sl_obj, "sigma_ref", float('nan'))) if sl_obj else float('nan')
        sl_l_ema    = _to_float_or_nan(getattr(sl_obj, "l_ema", float('nan'))) if sl_obj else float('nan')

        w.writerow([epoch, vpl_lambda, vpl_v_ema, vpl_v_ref, vpl_v_batch,
                    sl_lambda, sl_sigma_ema, sl_sigma_ref, sl_l_ema])

    # Format per-class accuracy
    formatted_train_per_class_acc = [f'{acc:.2f}%' for acc in train_per_class_acc]
    formatted_test_per_class_acc  = [f'{acc:.2f}%' for acc in test_per_class_acc]

    # Pad if fewer than 10 classes (for consistent CSV)
    formatted_train_per_class_acc += ['N/A'] * (10 - len(formatted_train_per_class_acc))
    formatted_test_per_class_acc  += ['N/A'] * (10 - len(formatted_test_per_class_acc))

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        lr = lr_scheduler.get_last_lr()
        lr_str = '%g' % lr[0] if isinstance(lr, list) else '%g' % lr
        row = [epoch, lr_str, train_loss, train_acc] + formatted_train_per_class_acc + [test_loss, test_acc] + formatted_test_per_class_acc
        logwriter.writerow(row)

    # Save training loss curve per seed
    seed_tag = f"{args.init_seed}" if int(args.use_seeds) == 1 else "RND"
    loss_filename = f"training_loss_seed_{args.init_seed}.csv"
    loss_filepath = os.path.join(model_dir, loss_filename)
    with open(loss_filepath, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss'])
        for e_i, loss in enumerate(all_epoch_losses, start=1):
            writer.writerow([e_i, loss])
    print(f"CSV file for training loss is saved at {loss_filepath}")

    # Save checkpoint at last epoch
    if epoch >= args.num_epochs - 1:
        checkpoint(model_dir, test_acc, epoch, test_per_class_acc)

    if np.isnan(train_loss):
        print('Detected NaN train loss, terminating early...')
        if use_wandb:
            import wandb; wandb.alert(title="NaN Loss", text=f"NaN train loss at epoch {epoch}")
        exit()

# Finish W&B
if use_wandb:
    import wandb
    wandb.finish()
