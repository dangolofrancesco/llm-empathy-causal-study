#!/bin/bash
# Script to launch incremental scoring from terminal in background
# This allows you to run other notebook cells simultaneously

cd "$(dirname "$0")/.."

echo "Starting incremental scoring..."
echo "Input:  data/filtered/wildchat_full_preprocessed.csv"
echo "Output: data/scores/wildchat_full_scored_incremental.csv"
echo "Chunk size: 1000 conversations"
echo ""
echo "To monitor progress:"
echo "    - Run notebook cell 5 anytime"
echo "    - Or use: tail -f logs/scoring_progress.log"
echo ""
echo "To interrupt:"
echo "    - Press Ctrl+C in this terminal"
echo "    - Progress will be saved automatically"
echo ""
echo "==========================================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Start scoring (output also to log file)
# -u flag for unbuffered output so progress appears in real-time
# --parallel 3: Process 3 conversations in parallel (safer for free tier rate limits)
python3 -u scripts/score_conversations_incremental.py \
    --input data/filtered/wildchat_full_preprocessed.csv \
    --output data/scores/wildchat_full_scored_incremental.csv \
    --chunk-size 1000 \
    --parallel 1 \
    2>&1 | tee logs/scoring_progress.log
