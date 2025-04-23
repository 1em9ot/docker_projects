#!/bin/sh
set -e
export MPLCONFIGDIR=/tmp/.matplotlib
mkdir -p "$MPLCONFIGDIR"

echo "▶ Generating training features..."
python /myhealth/strategies/auto_generate_features.py || echo "⚠️ Feature generation failed"

echo "▶ Running train.py..."
python /myhealth/strategies/train.py                    # Proceed with model training

echo "▶ Starting Dash..."
exec python /myhealth/dashboard/app.py                  # Launch the Dash web app
