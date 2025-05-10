#!/bin/sh

FUSEBUILD_DIR=$(realpath $(dirname $0))
. $FUSEBUILD_DIR/.venv/bin/activate
export PYTHONPATH=$FUSEBUILD_DIR
python3 -m fusebuild  $*



