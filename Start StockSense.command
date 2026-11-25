#!/bin/zsh

export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/Current/bin:$HOME/Library/Python/3.13/bin:$PATH"
cd "$(dirname "$0")"
echo "Starting StockSense... the website will open in your browser."
python3 -m streamlit run app.py
