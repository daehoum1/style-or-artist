#!/bin/bash
# Wait until style eval + probes exist for all 12 encoders on both datasets, then rebuild analyses.
cd "$(dirname "$0")/.."
n() { ls artifacts/results/style_eval/$1/*.parquet 2>/dev/null | grep -v per_class | wc -l; }
p() { ls artifacts/results/probes/$1/*.parquet 2>/dev/null | wc -l; }
until [ $(n wikiart) -ge 12 ] && [ $(p wikiart) -ge 12 ] && [ $(n multitask_painting) -ge 12 ] && [ $(p multitask_painting) -ge 12 ]; do sleep 60; done
for d in wikiart multitask_painting; do python scripts/08_analyze.py $d > artifacts/logs/analysis/$d.log 2>&1; done
for v in lenient strict; do for d in wikiart multitask_painting; do python scripts/08_analyze.py $d --split_root ${d}_$v > artifacts/logs/analysis/${d}_$v.log 2>&1; done; done
python scripts/09_summary.py > artifacts/logs/analysis/summary.log 2>&1
python scripts/env_report.py > /dev/null 2>&1
echo FINALIZED $(date +%H:%M) >> artifacts/logs/finalize.log
