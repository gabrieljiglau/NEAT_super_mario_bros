#!/bin/bash

find /home/gabriel/Documents/PycharmProjects/NEAT_super_mario_bros/runs/winner_run -name "*.mp4" -print0 | while IFS= read -r -d '' file; do
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
    duration=${duration%.*}  # trim decimal part
    if [ "$duration" -gt 30 ]; then
        echo "$file"
    fi
done