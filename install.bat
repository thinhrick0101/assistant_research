@echo off
echo Installing dependencies with binary wheels where possible...
pip install --no-cache-dir --only-binary=:all: pymupdf==1.23.18
pip install -r requirements.txt --no-deps
pip install -r requirements.txt
echo Installation completed!
