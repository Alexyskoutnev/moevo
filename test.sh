#!/usr/bin/env bash
# Build the wheel and test the INSTALLED package, not the working tree.
#
# moevo uses a flat layout (the package sits at the repo root), so a bare
# `pytest` from here would import moevo via the current directory and pass even
# if the built wheel were missing a subpackage or a data file. Testing from a
# scratch directory with an installed wheel resolves moevo by import rather than
# by path, so what runs is what a user gets from `pip install`.
#
# Usage: bash test.sh
set -euo pipefail

rm -rf dist _testing

echo "== lint =="
uvx ruff@0.16.3 check .
uvx ruff@0.16.3 format --check .

echo "== build wheel =="
uv build --wheel --out-dir dist

echo "== install the wheel into a scratch environment =="
mkdir -p _testing
uv venv --python 3.11 _testing/venv >/dev/null
VIRTUAL_ENV="$PWD/_testing/venv" uv pip install --quiet dist/moevo-*.whl pytest pytest-asyncio

echo "== run tests against the installed package =="
cp -R tests _testing/tests
cd _testing
./venv/bin/python -m pytest tests -q

echo "== public API import check =="
./venv/bin/python -c "import moevo; print('moevo', moevo.__version__, '->', sorted(moevo.__all__))"
