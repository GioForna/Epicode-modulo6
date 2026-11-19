"""Training e generazione da riga di comando, senza passare dall'API.

Esempio:
    python scripts/train_cli.py data/diabetes.csv --epochs 300 --n-samples 1000 \
        --out data/synthetic_diabetes.csv --report reports/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sandbox.config import TrainConfig
from sandbox.trainer import SyntheticDataModel
from sandbox.validation import (
    comparison_table,
    full_report,
    plot_correlation_heatmaps,
    plot_distributions,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Addestra il VAE e genera un dataset sintetico.")
    p.add_argument("csv", type=Path)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--categorical", nargs="*", default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--report", type=Path, default=None, help="Cartella in cui salvare grafici e metriche")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    real = pd.read_csv(args.csv)
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        hidden_dims=tuple(args.hidden_dims),
        beta=args.beta,
        seed=args.seed,
    )

    def log(epoch: int, total: int, m: dict) -> None:
        if epoch == 1 or epoch % 25 == 0 or epoch == total:
            print(f"[{epoch:4d}/{total}] loss={m['loss']:.4f} recon={m['recon_loss']:.4f} kl={m['kl_loss']:.4f}")

    model = SyntheticDataModel(cfg).fit(real, args.categorical, progress=log)
    if args.model_dir:
        model.save(args.model_dir)

    synth = model.generate(args.n_samples or len(real), seed=args.seed)
    out = args.out or args.csv.with_name(f"synthetic_{args.csv.name}")
    synth.to_csv(out, index=False)
    print(f"\nDataset sintetico salvato in {out}")

    report = full_report(real, synth, model.preprocessor)
    print(f"Fidelity score:          {report['fidelity_score']:.3f}")
    print(f"Δ correlazione media:    {report['correlation']['mean_abs_diff']:.3f}")
    priv = report["privacy"]
    print(f"Record identici ai reali: {priv['exact_matches']}")
    print(f"Privacy check superato:  {priv['passed']}")
    print("\n" + comparison_table(real, synth).to_string())

    if args.report:
        args.report.mkdir(parents=True, exist_ok=True)
        plot_correlation_heatmaps(real, synth).savefig(args.report / "correlation_heatmaps.png", dpi=120)
        plot_distributions(real, synth).savefig(args.report / "distributions.png", dpi=120)
        (args.report / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
        (args.report / "training_history.json").write_text(json.dumps(model.history))
        print(f"Report salvato in {args.report}")


if __name__ == "__main__":
    main()
