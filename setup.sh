#!/usr/bin/env bash
set -e

VENV_DIR=".venv"

echo "============================================"
echo "  Retail Anomaly Detection POC — Setup"
echo "============================================"

echo ""
echo "[1/4] Creating virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "[2/4] Upgrading pip ..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet

echo "[3/4] Installing dependencies from requirements.txt ..."
"$VENV_DIR/bin/pip" install -r requirements.txt --quiet

echo "[4/4] Registering Jupyter kernel 'retail-anomaly-poc' ..."
"$VENV_DIR/bin/python" -m ipykernel install \
    --user \
    --name retail-anomaly-poc \
    --display-name "Retail Anomaly POC"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Activate:       source $VENV_DIR/bin/activate"
echo "  Run POC:        python main.py"
echo "  Run notebook:   jupyter notebook poc_notebook.ipynb"
echo ""
