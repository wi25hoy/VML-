# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import model_utils

# -----------------------------
# Adaptive Stable Loss (EMA + Huber + volatility)
# -----------------------------
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

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
        alpha: float = 0.10,
        beta: float = 0.10,
        delta: float = 0.10,
        lambda_base: float = 0.50,
        lambda_min: float = 0.0,
        lambda_max: float = 2.0,
        warmup_steps: int = 200,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = float(delta)
        self.lambda_base = float(lambda_base)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.warmup_steps = int(warmup_steps)
        self.eps = float(eps)

        # EMA state (buffers so they save/restore with state_dict)
        self.register_buffer("l_ema", torch.tensor(0.0))
        self.register_buffer("sigma_ema", torch.tensor(0.0))
        self.register_buffer("sigma_ref", torch.tensor(0.0))
        self.register_buffer("warmup_count", torch.tensor(0))
        # debug/inspection
        self.register_buffer("last_lambda", torch.tensor(0.0))
        self.register_buffer("last_delta", torch.tensor(0.0))

        self._initialized = False

    @staticmethod
    def _huber(delta: torch.Tensor, thresh: float) -> torch.Tensor:
        absd = delta.abs()
        quad = 0.5 * delta * delta
        lin = thresh * (absd - 0.5 * thresh)
        return torch.where(absd <= thresh, quad, lin)

    @torch.no_grad()
    def _init_if_needed(self, loss_value: torch.Tensor) -> None:
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

        # 1) Deviation with gradient (do NOT detach base_loss)
        delta_t = base_loss - l_prev

        # 2) Robust penalty with gradient
        huber_t = self._huber(delta_t, self.delta)

        # 3) Update volatility EMA with detached magnitude (no grad)
        with torch.no_grad():
            self.sigma_ema.mul_(1.0 - self.beta).add_(self.beta * delta_t.detach().abs())
            self.warmup_count.add_(1)
            if int(self.warmup_count.item()) == self.warmup_steps and self.sigma_ref.item() == 0.0:
                self.sigma_ref.copy_(torch.clamp(self.sigma_ema, min=self.eps))

        # 4) Adaptive gain (from buffers; no grad)
        denom = self.sigma_ref if self.sigma_ref.item() > 0.0 else self.sigma_ema
        lam_t = self.lambda_base * (self.sigma_ema / (denom + self.eps))
        lam_t = torch.clamp(lam_t, self.lambda_min, self.lambda_max)

        # 5) Final SL term (grad flows through huber_t)
        sl_t = lam_t * huber_t

        # 6) Update EMA baseline with detached loss
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
                 lambda_base: float = 1.0,     # base gain for the adaptive controller
                 scale: float = 1.0,           # fixed multiplier (preserve old 'vpl_weight_decay' semantics)
                 alpha: float = 0.10,          # EMA smoothing for v_ema
                 warmup_steps: int = 100,      # steps to lock v_ref
                 lambda_min: float = 0.0,
                 lambda_max: float = 2.0,
                 use_entropy: bool = False,    # optional modulation by prediction entropy
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

        # state
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
        """
        logits: [B, C], labels: [B]
        returns scalar penalty (always adaptive)
        """
        assert logits.dim() == 2, "logits must be [B, C]"
        B, C = logits.shape

        # compute batch statistic
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

        # update controller state
        with torch.no_grad():
            self.v_ema.mul_(1.0 - self.alpha).add_(self.alpha * v_batch.detach())
            self.warmup_count.add_(1)
            if int(self.warmup_count.item()) == self.warmup_steps and self.v_ref.item() == 0.0:
                self.v_ref.copy_(torch.clamp(self.v_ema, min=self.eps))
            self.last_vbatch.copy_(v_batch.detach())

        denom = (self.v_ref if self.v_ref.item() > 0.0 else self.v_ema)
        lam = self.lambda_base * (self.v_ema / (denom + self.eps))

        # optional entropy modulation
        if self.use_entropy:
            with torch.no_grad():
                p = torch.softmax(logits, dim=1)
                ent = -(p * (p.clamp_min(self.eps).log())).sum(dim=1).mean()  # [0, log C]
                ent_norm = ent / math.log(C)                                  # [0,1]
            lam = lam * (1.0 + self.entropy_scale * ent_norm)

        lam = torch.clamp(lam, self.lambda_min, self.lambda_max)
        self.last_lambda.copy_(lam)

        return self.scale * lam * v_batch


# -----------------------------
# VGG configuration
# -----------------------------
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# -----------------------------
# VGG backbone with SL/VPL
# -----------------------------
class VGG(nn.Module):
    def __init__(
        self,
        vgg_name: str,
        init_strategy='he',
        num_classes=10,

        # VPL knobs
        stable_weight=0.0,          # used as default for SL lambda_base if sl_lambda_base is None
        vpl_weight_decay=0.1,
        vpl_weight=0.1,

        # Adaptive SL defaults (can be overridden by configure_adaptive_stable)
        sl_alpha=0.10,
        sl_beta=0.10,
        sl_delta=0.10,
        sl_lambda_base=None,       # if None, default to stable_weight (or 0.5 if both None/0)
        sl_lambda_min=0.0,
        sl_lambda_max=2.0,
        sl_warmup_steps=200,
        sl_eps=1e-8,
    ):
        super().__init__()
        self.features    = self._make_layers(cfg[vgg_name])
        self.classifier  = nn.Linear(512, num_classes)
        self.init_strategy = init_strategy
        self._initialize_weights()

        # --- Loss modules ---
        if sl_lambda_base is None or (isinstance(sl_lambda_base, float) and sl_lambda_base == 0.0):
            sl_lambda_base = stable_weight if (stable_weight is not None and stable_weight != 0.0) else 0.50

        self.adaptive_stable_loss = AdaptiveStableLoss(
            alpha=sl_alpha, beta=sl_beta, delta=sl_delta,
            lambda_base=sl_lambda_base, lambda_min=sl_lambda_min, lambda_max=sl_lambda_max,
            warmup_steps=sl_warmup_steps, eps=sl_eps
        )
        # Alias for trainer checkpoint logic
        self.stable_mod = self.adaptive_stable_loss

        # always-adaptive VPL
        self.variance_penalty = AdaptiveLabelVariancePenalty(
            lambda_base=1.0,               # base gain for controller
            scale=float(vpl_weight_decay),  # keep old 'vpl_weight_decay' meaning
            alpha=0.10,
            warmup_steps=100,
            lambda_min=0.0,
            lambda_max=2.0,
            use_entropy=False,
            entropy_scale=0.5,
            eps=1e-8,
        )
        self.vpl_weight       = float(vpl_weight)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    # ---- build VGG features ----
    def _make_layers(self, cfg_list):
        layers = []
        in_channels = 3
        for x in cfg_list:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    # ---- init ----
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init_strategy == 'he':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif self.init_strategy == 'xavier':
                    init.xavier_normal_(m.weight)
                elif self.init_strategy == 'custom_uniform':
                    init.uniform_(m.weight, -0.0085, 0.0085)
                elif self.init_strategy == 'custom_xavier':
                    init.xavier_normal_(m.weight)
                    m.weight.data.clamp_(-0.0085, 0.0085)
                elif self.init_strategy == 'custom_kaiming':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    m.weight.data.clamp_(-0.0085, 0.0085)
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
        delta_frac: if provided, we interpret d = delta_frac (absolute).
        """
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
# Factory functions (match your trainer's expectations)
# -----------------------------
def vgg11(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = VGG('VGG11', **kwargs)
    model_utils.restore_rng_state(old_state)
    return model

def vgg13(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = VGG('VGG13', **kwargs)
    model_utils.restore_rng_state(old_state)
    return model

def vgg16(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = VGG('VGG16', **kwargs)
    model_utils.restore_rng_state(old_state)
    return model

def vgg19(flags=None, **kwargs):
    old_state = model_utils.set_rng_state(flags)
    model = VGG('VGG19', **kwargs)
    model_utils.restore_rng_state(old_state)
    return model

