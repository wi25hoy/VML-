# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import model_utils

# -----------------------------
# Adaptive Stable Loss (EMA + Huber + volatility)
# -----------------------------
class AdaptiveStableLoss(nn.Module):
    """
    L_t: batch scalar CE
    L-_t = EMA_alpha(L_t)
    ?_t = L_t - L-_{t-1}
    s_t = EMA_beta(|?_t|)
    ?_t = clip(?_base * s_t / (s_ref or s_t), [?_min, ?_max])
    SL_t = ?_t * Huber_d(?_t)
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

        self.register_buffer("l_ema",        torch.tensor(0.0))
        self.register_buffer("sigma_ema",    torch.tensor(0.0))
        self.register_buffer("sigma_ref",    torch.tensor(0.0))
        self.register_buffer("warmup_count", torch.tensor(0))
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

    # Treat EMA baseline as a constant target (no grad through EMA buffer)
         l_prev = self.l_ema.detach()

    # 1) Deviation with gradient (DO NOT detach base_loss here)
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
# Label-aware Variance Penalty (adaptive)
# -----------------------------
class AdaptiveLabelVariancePenalty(nn.Module):
    """
    v_batch = mean of per-class logit variance (for classes with >=2 samples in batch).
    Adaptive scaler:
      - v_ema: EMA of v_batch (alpha)
      - v_ref: frozen ref after warmup_steps
      - lambda_vpl = lambda_base * clamp(v_ema / (v_ref+eps), [lambda_min, lambda_max])
      - optional entropy modulation (off by default)
    Returns: scale * lambda_vpl * v_batch
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
                 eps: float = 1e-8):
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

        self.register_buffer("v_ema",        torch.tensor(0.0))
        self.register_buffer("v_ref",        torch.tensor(0.0))
        self.register_buffer("warmup_count", torch.tensor(0))
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

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert logits.dim() == 2, "logits must be [B, C]"
        B, C = logits.shape

        class_vars = []
        for c in range(C):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if idx.numel() >= 2:
                class_preds = logits.index_select(0, idx)
                class_var   = class_preds.var(dim=0, unbiased=False).mean()
                class_vars.append(class_var)

        if not class_vars:
            return logits.new_tensor(0.0)

        v_batch = torch.stack(class_vars).mean()
        self._init_if_needed(v_batch.detach())

        with torch.no_grad():
            self.v_ema.mul_(1.0 - self.alpha).add_(self.alpha * v_batch.detach())
            self.warmup_count.add_(1)
            if int(self.warmup_count.item()) == self.warmup_steps and self.v_ref.item() == 0.0:
                self.v_ref.copy_(torch.clamp(self.v_ema, min=self.eps))
            self.last_vbatch.copy_(v_batch.detach())

        denom = (self.v_ref if self.v_ref.item() > 0.0 else self.v_ema)
        lam = self.lambda_base * (self.v_ema / (denom + self.eps))

        if self.use_entropy:
            with torch.no_grad():
                p = torch.softmax(logits, dim=1)
                ent = -(p * (p.clamp_min(self.eps).log())).sum(dim=1).mean()
                ent_norm = ent / math.log(C)
            lam = lam * (1.0 + self.entropy_scale * ent_norm)

        lam = torch.clamp(lam, self.lambda_min, self.lambda_max)
        self.last_lambda.copy_(lam)

        return self.scale * lam * v_batch


# -----------------------------
# ShuffleNetV2 building blocks
# -----------------------------
class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)

class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]

class BasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5):
        super().__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,
                               padding=1, groups=in_channels, bias=False)
        self.bn2   = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(in_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = F.relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2,
                               padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2,
                               padding=1, groups=mid_channels, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_channels)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channels)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        out1 = self.bn1(self.conv1(x))
        out1 = F.relu(self.bn2(self.conv2(out1)))

        out2 = F.relu(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        out2 = F.relu(self.bn5(self.conv5(out2)))

        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out


# -----------------------------
# ShuffleNetV2 backbone with SL/VPL
# -----------------------------
class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        net_size,
        num_classes=10,
        init_strategy='he',

        # legacy knobs kept for API compatibility
        stable_weight=0.1,
        vpl_weight_decay=0.1,
        vpl_weight=0.1,

        # Adaptive SL defaults (overridable via configure_adaptive_stable)
        sl_alpha=0.10,
        sl_beta=0.10,
        sl_delta=0.10,
        sl_lambda_base=None,     # if None/0, derive from stable_weight (or 0.5)
        sl_lambda_min=0.0,
        sl_lambda_max=2.0,
        sl_warmup_steps=200,
        sl_eps=1e-8,
    ):
        super().__init__()
        out_channels = configs[net_size]['out_channels']
        num_blocks   = configs[net_size]['num_blocks']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
        self.conv2  = nn.Conv2d(out_channels[2], out_channels[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2    = nn.BatchNorm2d(out_channels[3])
        self.linear = nn.Linear(out_channels[3], num_classes)

        # --- Adaptive Stable Loss ---
        if sl_lambda_base is None or (isinstance(sl_lambda_base, float) and sl_lambda_base == 0.0):
            sl_lambda_base = stable_weight if (stable_weight is not None and stable_weight != 0.0) else 0.50

        self.adaptive_stable_loss = AdaptiveStableLoss(
            alpha=sl_alpha, beta=sl_beta, delta=sl_delta,
            lambda_base=sl_lambda_base, lambda_min=sl_lambda_min, lambda_max=sl_lambda_max,
            warmup_steps=sl_warmup_steps, eps=sl_eps
        )
        self.stable_mod = self.adaptive_stable_loss  # alias for trainer

        # --- Label-aware VPL (adaptive) ---
        self.variance_penalty = AdaptiveLabelVariancePenalty(
            lambda_base=1.0,
            scale=float(vpl_weight_decay),
            alpha=0.10,
            warmup_steps=100,
            lambda_min=0.0,
            lambda_max=2.0,
            use_entropy=False,
            entropy_scale=0.5,
            eps=1e-8,
        )
        self.vpl_weight = float(vpl_weight)

        self._initialize_weights(init_strategy)

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels)]
        for _ in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

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

    def configure_adaptive_stable(
        self,
        alpha=None, beta=None, delta_frac=None,
        lambda_base=None, lambda_min=None, lambda_max=None,
        warmup_steps=None, use_running_ref=True, eps=None
    ):
        mod = self.adaptive_stable_loss
        if alpha is not None:        mod.alpha = float(alpha)
        if beta  is not None:        mod.beta  = float(beta)
        if eps   is not None:        mod.eps   = float(eps)
        if lambda_base is not None:  mod.lambda_base = float(lambda_base)
        if lambda_min  is not None:  mod.lambda_min  = float(lambda_min)
        if lambda_max  is not None:  mod.lambda_max  = float(lambda_max)
        if warmup_steps is not None: mod.warmup_steps = int(warmup_steps)
        if delta_frac is not None:   mod.delta = float(delta_frac)
        return self


# -----------------------------
# Preset configs
# -----------------------------
configs = {
    0.5: {'out_channels': (48, 96, 192, 1024), 'num_blocks': (3, 7, 3)},
    1.0: {'out_channels': (116, 232, 464, 1024), 'num_blocks': (3, 7, 3)},
    1.5: {'out_channels': (176, 352, 704, 1024), 'num_blocks': (3, 7, 3)},
    2.0: {'out_channels': (224, 488, 976, 2048), 'num_blocks': (3, 7, 3)},
}

# -----------------------------
# Factory helpers (match trainer)
# -----------------------------
def shufflenet05(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = ShuffleNetV2(net_size=0.5, **kwargs)
    model_utils.restore_rng_state(old_state)
    return model

def shufflenet1(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = ShuffleNetV2(net_size=1.0, **kwargs)
    model_utils.restore_rng_state(old_state)
    return model

def shufflenet15(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = ShuffleNetV2(net_size=1.5, **kwargs)
    model_utils.restore_rng_state(old_state)
    return model

def shufflenet2(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = ShuffleNetV2(net_size=2.0, **kwargs)
    model_utils.restore_rng_state(old_state)
    return model


