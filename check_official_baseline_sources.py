#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check whether external official/public baseline repositories are available."""

from models.official_baselines.utils import describe_available_sources, official_external_root


def main():
    print(f"OFFICIAL_BASELINES_ROOT = {official_external_root()}")
    print("-" * 120)
    print(f"{'name':24s} {'paper name':18s} {'status':8s} details")
    print("-" * 120)
    for row in describe_available_sources():
        print(f"{row['name']:24s} {row['paper_name']:18s} {row['status']:8s} {row['details']}")
    print("-" * 120)
    print("If a source is missing, run scripts/clone_official_baselines.sh or set the corresponding ECG_BASELINE_*_ROOT variable.")


if __name__ == "__main__":
    main()
