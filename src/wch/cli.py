"""Command-line entry point.

    wch search     --config config/kanpur.yaml          # which dates would be used (no pixel download)
    wch run        --config config/kanpur.yaml          # composites + change analysis (10 m)
    wch timeseries --config config/kanpur.yaml          # monthly NDVI per zone (needs `run` first)
    wch sample     --config config/kanpur.yaml          # choose points to label (then use the app)
    wch train      --config config/kanpur.yaml          # train + evaluate the construction classifier
    add --force to `run`/`timeseries` to rebuild cached downloads, or to `sample` to draw new points
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings

from wch.config import load_config
from wch.pipeline import run, run_timeseries, search_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="wch", description="What Changed Here?")
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="show which dates would be used")
    p_search.add_argument("--config", required=True, help="path to YAML config")

    for name, help_text in (("run", "run the change analysis"),
                            ("timeseries", "monthly NDVI per zone with anomaly flags"),
                            ("sample", "choose points to label for the classifier")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True, help="path to YAML config")
        p.add_argument("--force", action="store_true", help="rebuild / redo even if results exist")

    p_train = sub.add_parser("train", help="train and evaluate the construction classifier")
    p_train.add_argument("--config", required=True, help="path to YAML config")

    args = parser.parse_args(argv)

    # Harmless notices: pystac upgrading Earth Search metadata, and numpy noting that
    # a pixel had no clear observation at all (it correctly becomes "no data").
    warnings.filterwarnings("ignore", message="Could not parse bucket")
    warnings.filterwarnings("ignore", message="All-NaN slice")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # rasterio logs "boto3 not available" for every public file it opens; we don't need boto3.
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    cfg = load_config(args.config)

    if args.command == "search":
        print(json.dumps(search_report(cfg), indent=2))
    elif args.command == "run":
        summary = run(cfg, force=args.force)
        print(json.dumps({"ndvi": summary["stats"],
                          "change_types": summary["landcover"]["change_types"]}, indent=2))
    elif args.command == "sample":
        from wch.modeling import run_sample   # imported here: only these commands need scikit-learn
        print(json.dumps(run_sample(cfg, force=args.force), indent=2))
        print("Next: streamlit run app/streamlit_app.py  ->  'Label' tab")
    elif args.command == "train":
        from wch.modeling import run_train
        metrics = run_train(cfg)
        print(f"\n{metrics['n_labels']} labels ({metrics['n_construction']} construction), "
              f"{metrics['n_spatial_blocks']} spatial blocks\n")
        print(f"{'method':<45} {'evaluation':<42} {'prec':>5} {'recall':>6} {'F1':>5}")
        for r in metrics["methods"]:
            print(f"{r['method']:<45} {r['evaluation']:<42} {r['precision']:>5.2f} {r['recall']:>6.2f} {r['f1']:>5.2f}")
        est = metrics["area"]["stratified_estimate_from_labels"]
        if est["estimate_km2"] is not None:
            print(f"\nEstimated new construction (from labels): {est['estimate_km2']} km2 "
                  f"(95% interval {est['ci95_km2'][0]}-{est['ci95_km2'][1]})")
    elif args.command == "timeseries":
        result = run_timeseries(cfg, force=args.force)
        print(json.dumps({k: v for k, v in result.items() if k != "anomalies"}, indent=2))
        for a in result["anomalies"]:
            print(f"  unusual: {a['zone']:<28} {a['year']}-{a['month']:02d}  "
                  f"NDVI {a['ndvi']:.3f} vs typical {a['typical']:.3f} (z={a['robust_z']:+.1f})")


if __name__ == "__main__":
    main()