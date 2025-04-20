#!/bin/sh
set -e
echo "▶ Running train.py..."
python /myhealth/strategies/train.py
echo "▶ Starting Dash on 8060..."
exec python /myhealth/dashboard/app.py
