# wb_push_analysis.py
import argparse
from pathlib import Path
import wandb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", required=True, help="Folder produced by analysis_D.py (contains figs/ and tables/).")
    ap.add_argument("--project", default="vml-experiments")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--group", default=None)
    ap.add_argument("--job_type", default="analysis")
    ap.add_argument("--name", default=None)
    ap.add_argument("--link_run_id", default=None, help="If set, record parent training run id in config.")
    args = ap.parse_args()

    run = wandb.init(project=args.project, entity=args.entity,
                     group=args.group, job_type=args.job_type, name=args.name,
                     config={"linked_training_run": args.link_run_id} if args.link_run_id else None)

    analysis_dir = Path(args.analysis_dir)
    figs = list((analysis_dir / "figs").glob("*.png"))
    tables = list((analysis_dir / "tables").glob("*.csv"))
    report = list(analysis_dir.glob("report.md"))

    # Log images to the run
    for p in figs:
        run.log({f"analysis/{p.stem}": wandb.Image(str(p))})

    # Log tables as artifacts (and as files panel)
    art = wandb.Artifact(name=f"analysis-{analysis_dir.name}", type="analysis")
    for p in figs + tables + report:
        if p.exists():
            art.add_file(str(p))
    run.log_artifact(art, aliases=["latest"])

    run.finish()

if __name__ == "__main__":
    main()
