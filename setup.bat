@echo off
echo Setting up research assistant environment...

REM Create data and index directories
mkdir data
mkdir index

REM Install PyMuPDF separately first with pre-compiled wheel
pip install --no-cache-dir --only-binary=:all: pymupdf==1.23.18

REM Install Scrapy first to ensure it's properly installed
pip install scrapy==2.8.0

REM Then install other dependencies
pip install -r requirements.txt

echo Setup complete! Run the application with: python run.py
