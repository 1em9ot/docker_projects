#!/bin/sh
set -e
export MPLCONFIGDIR=/tmp/.matplotlib
mkdir -p "$MPLCONFIGDIR"

echo "▶ Running train.py..."
python /myhealth/strategies/train.py

echo "▶ Starting Dash..."
exec python /myhealth/dashboard/app.py
