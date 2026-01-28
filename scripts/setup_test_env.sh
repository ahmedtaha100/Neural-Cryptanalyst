#!/bin/sh
# Install dependencies for running the unit tests
pip install -r requirements.txt
if [ -f requirements-dev.txt ]; then
    pip install -r requirements-dev.txt
fi
