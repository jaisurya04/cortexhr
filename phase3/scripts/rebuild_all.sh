#!/bin/sh
# Rebuild every notebook from its builder, then execute all of them in order (about 12 minutes on one CPU).
set -e
cd "$(dirname "$0")/.."
for t in 11 12 13 14 15; do python scripts/build_task$t.py > /dev/null; done
python scripts/build_workflow.py > /dev/null
python scripts/execute_notebooks.py
