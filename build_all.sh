#!/bin/sh -e

test -d .venv || python3 -m venv .venv
. .venv/bin/activate

type pip
pip install -r requirements.txt

./fusebuild.sh build,linting,test test/run_test_example1_nosandbox .
