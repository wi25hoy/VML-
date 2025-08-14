# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import model_utils

# -----------------------------
# Building blocks
# -----------------------------

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


# -----------------------------
# Adaptive Stable Loss (EMA + Huber + volatility)
# -----------------------------

class AdaptiveStableLoss(nn.Module):
    """
    Notation (batch scalar):
      L_t      = base CE loss for the current batch (scalar tensor)
      Lbar_t   = EMA_alpha(L_t)
      delta_t  = L_t - Lbar_{t-1}
      sigma_t  = EMA_beta(|delta_t|)
      lambda_t = clamp(lambda_base * sigma_t / (sigma_ref or sigma_t), [lambda_min, lambda_max])
      SL_t     = lambda_t * Huber_delta(delta_t)

    Implementation details:
      - No gradient through EMA buffers (Lbar_t, sigma_t, sigma_ref).
      - Grad flows through delta_t and Huber(delta_t).
      - Warmup captures sigma_ref after warmup_steps if zero.
    """
    def __init__(
        self,
        alpha=0.10,
        beta=0.10,
        delta=0.10,
        lambda_base=0.50,
        lambda_min=0.0,
        lambda_max=2.0,
        warmup_steps=200,
        eps=1e-8
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.delta = float(delta)
        self.lambda_base = float(lambda_base)
        self.lambda_min  = float(lambda_min)
        self.lambda_max  = float(lambda_max)
        self.warmup_steps = int(warmup_steps)
        self.eps = float(eps)

        # EMA state (buffers -> saved/restored with state_dict)
        self.register_buffer("l_ema",        torch.tensor(0.0))
        self.register_buffer("sigma_ema",    torch.tensor(0.0))
        self.register_buffer("sigma_ref",    torch.tensor(0.0))
        self.register_buffer("warmup_count", torch.tensor(0))

        # debug/inspection
        self.register_buffer("last_lambda",  torch.tensor(0.0))
        self.register_buffer("last_delta",   torch.tensor(0.0))

        self._initialized = False

    @staticmethod
    def _huber(delta, thresh):
        absd = delta.abs()
        quad = 0.5 * delta * delta
        lin  = thresh * (absd - 0.5 * thresh)
        return torch.where(absd <= thresh, quad, lin)

    @torch.no_grad()
    def _init_if_needed(self, loss_value: torch.Tensor):
        if not self._initialized:
            v = loss_value.detach().to(self.l_ema.device)
            self.l_ema.fill_(v)
            self.sigma_ema.zero_()
            self.sigma_ref.zero_()
            self.warmup_count.zero_()
            self.last_lambda.zero_()
            self.last_delta.zero_()
            self._initialized = True

    def forward(self, base_loss: torch.Tensor) -> torch.Tensor:
        assert base_loss.dim() == 0, "AdaptiveStableLoss expects a scalar batch loss"
        self._init_if_needed(base_loss)

        # Treat EMA baseline as constant target (no grad through EMA)
        l_prev = self.l_ema.detach()

        # 1) Deviation with gradient (do not detach base_loss)
        delta_t = base_loss - l_prev

        # 2) Robust penalty with gradient
        huber_t = self._huber(delta_t, self.delta)

        # 3) Update volatility EMA with detached magnitude (no grad)
        with torch.no_grad():
            self.sigma_ema.mul_(1.0 - self.beta).add_(self.beta * delta_t.detach().abs())
            self.warmup_count.add_(1)
            if int(self.warmup_count.item()) == self.warmup_steps and self.sigma_ref.item() == 0.0:
                self.sigma_ref.copy_(torch.clamp(self.sigma_ema, min=self.eps))

        # 4) Adaptive gain (purely from buffers; no grad)
        denom = self.sigma_ref if self.sigma_ref.item() > 0.0 else self.sigma_ema
        lam_t = self.lambda_base * (self.sigma_ema / (denom + self.eps))
        lam_t = torch.clamp(lam_t, self.lambda_min, self.lambda_max)

        # 5) Final SL term (grad flows through huber_t)
        sl_t = lam_t * huber_t

        # 6) Update EMA of the baseline with detached loss
        with torch.no_grad():
            self.l_ema.mul_(1.0 - self.alpha).add_(self.alpha * base_loss.detach())
            self.last_lambda.copy_(lam_t)
            self.last_delta.copy_(delta_t.detach())

        return sl_t


# -----------------------------
# Variance Penalty (label-aware, adaptive)
class AdaptiveLabelVariancePenalty(nn.Module):
    """
    Adaptive VPL with selectable statistic.

    v_batch (depends on `stat`):
      - 'vector': for each class c with >=2 samples, take the variance of the full
                  logit vector across those samples (var over dim=0), then mean over dims.
                  Finally average across eligible classes.
      - 'true'  : for each class c with >=2 samples, take the variance of the true-class
                  logit (column c) across those samples. Then average across classes.

    Controller:
      v_ema <- EMA_alpha(v_batch)
      after warmup_steps, set v_ref = clamp(v_ema, min=eps) (one-time latch)
      lambda_t = lambda_base * clamp( v_ema / (v_ref + eps), [lambda_min, lambda_max] )

      optional entropy modulation:
        lambda_t <- lambda_t * (1 + entropy_scale * H_norm)
        where H_norm = H(p)/log(C), p = softmax(logits)

    Returns:
      penalty = scale * lambda_t * v_batch
    """

    def __init__(self,
                 lambda_base: float = 1.0,
                 scale: float = 1.0,
                 alpha: float = 0.10,
                 warmup_steps: int = 100,
                 lambda_min: float = 0.0,
                 lambda_max: float = 2.0,
                 use_entropy: bool = False,
                 entropy_scale: float = 0.5,
                 eps: float = 1e-8,
                 stat: str = "vector"):
        super().__init__()
        self.lambda_base  = float(lambda_base)
        self.scale        = float(scale)
        self.alpha        = float(alpha)
        self.warmup_steps = int(warmup_steps)
        self.lambda_min   = float(lambda_min)
        self.lambda_max   = float(lambda_max)
        self.use_entropy  = bool(use_entropy)
        self.entropy_scale= float(entropy_scale)
        self.eps          = float(eps)

        self.stat = str(stat).lower()
        if self.stat not in ("vector", "true"):
            raise ValueError(f"stat must be 'vector' or 'true', got: {stat}")

        # Controller state
        self.register_buffer("v_ema",        torch.tensor(0.0))
        self.register_buffer("v_ref",        torch.tensor(0.0))
        self.register_buffer("warmup_count", torch.tensor(0))
        # Debug/telemetry
        self.register_buffer("last_lambda",  torch.tensor(0.0))
        self.register_buffer("last_vbatch",  torch.tensor(0.0))
        self._initialized = False

    @torch.no_grad()
    def reset_state(self):
        self.v_ema.zero_()
        self.v_ref.zero_()
        self.warmup_count.zero_()
        self.last_lambda.zero_()
        self.last_vbatch.zero_()
        self._initialized = False

    @torch.no_grad()
    def _init_if_needed(self, v0: torch.Tensor):
        if not self._initialized:
            self.v_ema.fill_(v0)
            self.v_ref.zero_()
            self.warmup_count.zero_()
            self.last_lambda.zero_()
            self.last_vbatch.copy_(v0)
            self._initialized = True

    def _v_batch_vector(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # mean variance of full logit vector per class, then average across classes
        B, C = logits.shape
        vals = []
        for c in range(C):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if idx.numel() >= 2:
                class_logits = logits.index_select(0, idx)            # [Nc, C]
                class_var = class_logits.var(dim=0, unbiased=False).mean()  # scalar
                vals.append(class_var)
        if not vals:
            return logits.new_tensor(0.0)
        return torch.stack(vals).mean()

    def _v_batch_true(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # variance of the true-class logit per class, then average across classes
        B, C = logits.shape
        vals = []
        for c in range(C):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if idx.numel() >= 2:
                class_true = logits.index_select(0, idx)[:, c]        # [Nc]
                class_var = class_true.var(unbiased=False)            # scalar
                vals.append(class_var)
        if not vals:
            return logits.new_tensor(0.0)
        return torch.stack(vals).mean()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert logits.dim() == 2, "logits must be [B, C]"
        B, C = logits.shape

        # 1) batch statistic
        if self.stat == "vector":
            v_batch = self._v_batch_vector(logits, labels)
        else:  # "true"
            v_batch = self._v_batch_true(logits, labels)

        self._init_if_needed(v_batch.detach())

        # 2) controller update
        with torch.no_grad():
            self.v_ema.mul_(1.0 - self.alpha).add_(self.alpha * v_batch.detach())
            self.warmup_count.add_(1)
            # use >= to be robust to any off-by-one/skip
            if int(self.warmup_count.item()) >= self.warmup_steps and self.v_ref.item() == 0.0:
                self.v_ref.copy_(torch.clamp(self.v_ema, min=self.eps))
            self.last_vbatch.copy_(v_batch.detach())

        denom = (self.v_ref if self.v_ref.item() > 0.0 else self.v_ema)
        lam = self.lambda_base * (self.v_ema / (denom + self.eps))

        # 3) optional entropy modulation
        if self.use_entropy:
            with torch.no_grad():
                p = torch.softmax(logits, dim=1)
                ent = -(p * (p.clamp_min(self.eps).log())).sum(dim=1).mean()  # mean entropy over batch
                # guard denominator in case C<=1
                denom_h = max(math.log(max(C, 2)), self.eps)
                ent_norm = ent / denom_h
            lam = lam * (1.0 + self.entropy_scale * ent_norm)

        # 4) clamp + expose telemetry
        lam = torch.clamp(lam, self.lambda_min, self.lambda_max)
        self.last_lambda.copy_(lam)

        # 5) final penalty
        return self.scale * lam * v_batch




# -----------------------------
# ResNet backbone with SL/VPL
# -----------------------------

class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,                    # e.g., [2,2,2] or [2,2,2,2]
        num_classes=10,
        init_strategy='he',

        # VPL knobs
        stable_weight=0.0,         # used as default for SL lambda_base if sl_lambda_base is None
        vpl_weight_decay=0.1,
        vpl_weight=0.1,

        # Adaptive SL knobs (can be overridden by configure_adaptive_stable)
        sl_alpha=0.10,
        sl_beta=0.10,
        sl_delta=0.10,
        sl_lambda_base=None,       # if None, we default to stable_weight (or 0.5 if both None/0)
        sl_lambda_min=0.0,
        sl_lambda_max=2.0,
        sl_warmup_steps=200,
        sl_eps=1e-8,
    ):
        super().__init__()
        self.inplanes = 64

        # CIFAR-friendly stem (3x3, stride 1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # Stages
        self.layers = []
        self._create_layers(block, layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(self.inplanes, num_classes)

        # init
        self._initialize_weights(init_strategy)

        # --- Loss modules ---
        # SL: map stable_weight -> lambda_base by default if sl_lambda_base not explicitly given
        if sl_lambda_base is None or (isinstance(sl_lambda_base, float) and sl_lambda_base == 0.0):
            sl_lambda_base = stable_weight if (stable_weight is not None and stable_weight != 0.0) else 0.50

        self.adaptive_stable_loss = AdaptiveStableLoss(
            alpha=sl_alpha, beta=sl_beta, delta=sl_delta,
            lambda_base=sl_lambda_base, lambda_min=sl_lambda_min, lambda_max=sl_lambda_max,
            warmup_steps=sl_warmup_steps, eps=sl_eps
        )
        # compat alias for trainer
        self.stable_mod = self.adaptive_stable_loss

        # Adaptive label-aware VPL
        self.variance_penalty = AdaptiveLabelVariancePenalty(
            lambda_base=1.0,               # base gain for controller (trainer can override by reattaching)
            scale=float(vpl_weight_decay),  # same semantics as prior vpl_weight_decay
            alpha=0.10,
            warmup_steps=100,
            lambda_min=0.0,
            lambda_max=2.0,
            use_entropy=False,
            entropy_scale=0.5,
            eps=1e-8,
        )
        self.vpl_weight = float(vpl_weight)

    # ---- factory for stages ----
    def _create_layers(self, block, layers):
        planes = 64
        for i, num_blocks in enumerate(layers):
            stride = 1 if i == 0 else 2
            layer = self._make_layer(block, planes, num_blocks, stride)
            self.add_module(f'layer{i+1}', layer)
            self.layers.append(layer)
            planes *= 2

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    # ---- init ----
    def _initialize_weights(self, init_strategy):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_strategy == 'he':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_strategy == 'xavier':
                    init.xavier_normal_(m.weight)
                elif init_strategy == 'custom_uniform':
                    init.uniform_(m.weight, -0.0089, 0.0089)
                elif init_strategy == 'custom_xavier':
                    init.xavier_normal_(m.weight)
                    m.weight.data.clamp_(-0.0089, 0.0089)
                elif init_strategy == 'custom_kaiming':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    m.weight.data.clamp_(-0.0089, 0.0089)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

    # ---- runtime config for SL (used by trainer) ----
    def configure_adaptive_stable(
        self,
        alpha=None, beta=None, delta_frac=None,
        lambda_base=None, lambda_min=None, lambda_max=None,
        warmup_steps=None, use_running_ref=True, eps=None
    ):
        """
        delta_frac: if provided, we interpret delta = delta_frac (absolute).
        """
        mod = self.adaptive_stable_loss
        if alpha is not None:       mod.alpha = float(alpha)
        if beta  is not None:       mod.beta  = float(beta)
        if eps   is not None:       mod.eps   = float(eps)
        if lambda_base is not None: mod.lambda_base = float(lambda_base)
        if lambda_min  is not None: mod.lambda_min  = float(lambda_min)
        if lambda_max  is not None: mod.lambda_max  = float(lambda_max)
        if warmup_steps is not None: mod.warmup_steps = int(warmup_steps)
        if delta_frac is not None:  mod.delta = float(delta_frac)
        return self


# -----------------------------
# Simple baselines (kept as-is)
# -----------------------------

class LinearNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(32*32*3, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class HiddenNet1(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class HiddenNet2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2, bias=True)
        self.fc1   = nn.Linear(16*16*64, num_classes)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc1(x)


# -----------------------------
# Factory functions
# -----------------------------

def linearnet(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = LinearNet(num_classes=kwargs.get('num_classes', 10))
    model_utils.restore_rng_state(old_state)
    return model


def hiddennet1(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = HiddenNet1(num_classes=kwargs.get('num_classes', 10))
    model_utils.restore_rng_state(old_state)
    return model


def hiddennet2(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = HiddenNet2(num_classes=kwargs.get('num_classes', 10))
    model_utils.restore_rng_state(old_state)
    return model


def resnet6(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = ResNet(BasicBlock, [2], **kwargs)
    model_utils.restore_rng_state(old_state)
    return model


def resnet10(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = ResNet(BasicBlock, [2, 2], **kwargs)
    model_utils.restore_rng_state(old_state)
    return model


def resnet14(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = ResNet(BasicBlock, [2, 2, 2], **kwargs)
    model_utils.restore_rng_state(old_state)
    return model


def resnet14_0125(flags=None, **kwargs):
    return resnet14(flags=flags, **kwargs)


def resnet14_025(flags=None, **kwargs):
    return resnet14(flags=flags, **kwargs)


def resnet14_050(flags=None, **kwargs):
    return resnet14(flags=flags, **kwargs)


def resnet14_2(flags=None, **kwargs):
    return resnet14(flags=flags, **kwargs)


def resnet14_4(flags=None, **kwargs):
    return resnet14(flags=flags, **kwargs)


def resnet14_8(flags=None, **kwargs):
    return resnet14(flags=flags, **kwargs)


def resnet18(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model_utils.restore_rng_state(old_state)
    return model


def resnet20(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = ResNet(BasicBlock, [3, 3, 3], **kwargs)
    model_utils.restore_rng_state(old_state)
    return model


def resnet32(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    model_utils.restore_rng_state(old_state)
    return model


def resnet44(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = ResNet(BasicBlock, [7, 7, 7], **kwargs)
    model_utils.restore_rng_state(old_state)
    return model


def resnet56(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = ResNet(BasicBlock, [9, 9, 9], **kwargs)
    model_utils.restore_rng_state(old_state)
    return model


def resnet110(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
    model_utils.restore_rng_state(old_state)
    return model


def resnet1202(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = ResNet(BasicBlock, [200, 200, 200], **kwargs)
    model_utils.restore_rng_state(old_state)
    return model

