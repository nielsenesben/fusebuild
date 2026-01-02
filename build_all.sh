#!/bin/sh -e

test -d .venv || python3 -m venv .venv
. .venv/bin/activate

type pip
pip install -r requirements.txt

if ./fusebuild.sh shallfail fail; then
    echo "Must fail on a failing action."
    exit 1
else
    echo "Failed explicit calling - ok"
fi

if ./fusebuild.sh shallfail .; then
    echo "Must fail on a failing action, also implicit"
    exit 1
else
    echo "Failed implicit calling - ok"
fi

./fusebuild.sh build,linting,test test/run_test_example1_nosandbox .
