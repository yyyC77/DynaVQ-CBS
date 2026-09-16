#!/usr/bin/env python3
import argparse

from pathlib import Path

from build_apo_pipeline import build_graph_cache


def main():
    ap = argparse.ArgumentParser(description="Build graph cache from rebuilt entities/chain_npz.")
    ap.add_argument("--out_pipeline_dir", type=Path, default=Path("data/structure_pipeline"))
    ap.add_argument("--dist_cutoff", type=float, default=10.0)
    ap.add_argument("--max_neighbors", type=int, default=40)
    args = ap.parse_args()

    build_graph_cache(args.out_pipeline_dir, args.dist_cutoff, args.max_neighbors)


if __name__ == "__main__":
    main()
