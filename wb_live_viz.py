# wb_live_viz.py
from pathlib import Path
from typing import Optional, Dict, Any, List
import wandb
import torch
import numpy as np

class WBLive:
    """
    Small helper to centralize W&B logging:
    - stream a per-batch Table (optional)
    - log step scalars
    - log epoch scalars + images
    - versioned artifacts (ckpt, CSVs, logs)
    """
    def __init__(self,
                 run: "wandb.sdk.wandb_run.Run",
                 use_table: bool = True,
                 table_name: str = "batches/stream",
                 table_cols: Optional[List[str]] = None,
                 grad_hist_every: int = 0):
        self.run = run
        self.use_table = use_table
        self.grad_hist_every = grad_hist_every
        self.table = None
        if use_table:
            default_cols = [
                "epoch","batch_idx","global_step","lr",
                "base_loss","sl_penalty","vpl_penalty","total_loss","acc_running",
                "grad_norm","zero_grad_params","none_grad_params","param_count","has_nan_inf",
                "vpl/lambda_t","vpl/v_ema","vpl/v_ref","vpl/v_batch",
                "sl/lambda_t","sl/sigma_ema","sl/sigma_ref","sl/l_ema",
                "vpl/eligible_classes","vpl/eligible_frac","time/step_s","time/throughput_img_s"
            ]
            cols = table_cols or default_cols
            self.table = wandb.Table(columns=cols)
            run.log({table_name: self.table}, step=0)
            self._table_name = table_name

    def add_table_row(self, row: List[Any]):
        if self.table is not None:
            self.table.add_data(*row)

    def log_step_scalars(self, payload: Dict[str, Any], step: Optional[int] = None):
        self.run.log(payload, step=step)

    def maybe_log_grad_hist(self, model: torch.nn.Module, step: int):
        if self.grad_hist_every <= 0 or step % self.grad_hist_every != 0:
            return
        grads = []
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.detach().float().flatten().cpu().numpy()
                    if g.size:
                        grads.append(g)
        if not grads:
            return
        arr = np.concatenate(grads)
        self.run.log({"grad/hist": wandb.Histogram(arr)}, step=step)

    def log_epoch_scalars(self, payload: Dict[str, Any], epoch: int):
        payload = dict(payload)
        payload["epoch"] = epoch
        self.run.log(payload, step=epoch)

    def log_images(self, images: Dict[str, Any], step: Optional[int] = None):
        """
        images: dict like {"debug/weights": wandb.Image(path_or_numpy), ...}
        """
        self.run.log(images, step=step)

    def log_artifact(self, name: str, artifact_type: str, files: List[Path], aliases: Optional[List[str]] = None):
        art = wandb.Artifact(name=name, type=artifact_type)
        for f in files:
            if f.exists():
                art.add_file(str(f))
        self.run.log_artifact(art, aliases=aliases or [])

    def link_to_run(self, parent_run_id: str):
        # optional: set a reference so analysis runs/artifacts can find the parent
        self.run.config.update({"parent_run_id": parent_run_id}, allow_val_change=True)
