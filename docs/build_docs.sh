#!/usr/bin/env bash
set -ex

pip install -r docs/requirements.txt
cd docs
sphinx-build -b html . _build
cd ..

set +ex
#make html
