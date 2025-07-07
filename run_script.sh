#!/bin/bash
set -e

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "📥 Downloading FFmpeg..."
mkdir -p ffmpeg
cd ffmpeg
curl -L -o ffmpeg-release.zip https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
unzip -o ffmpeg-release.zip
cd ..

# Set FFMPEG_PATH for config.ini or use as an env var
export FFMPEG_PATH=$(pwd)/ffmpeg/ffmpeg-*/bin

echo "🎬 Running main script..."
python main.py
