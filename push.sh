#!/bin/sh
PYTHONPATH=`dirname $0`:$PYTHONPATH
python3 -m flitter.interface.push "$@"
