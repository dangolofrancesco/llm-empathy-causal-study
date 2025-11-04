#!/bin/bash

# Script to run the Mistral-based conversation scoring
# Uses Mistral AI API (200 RPM free tier - 6x faster than Groq)
# Expected timeline: 24-40 hours for 144k conversations

# Set working directory
cd "$(dirname "$0")/.."

# Load environment variables
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Please create .env file with your MISTRAL_API_KEY"
    exit 1
fi

# Check if MISTRAL_API_KEY is set
source .env
if [ -z "$MISTRAL_API_KEY" ]; then
    echo "❌ Error: MISTRAL_API_KEY not set in .env file"
    echo "Please add: MISTRAL_API_KEY=your_key_here"
    exit 1
fi

# Configuration
INPUT_FILE="data/filtered/wildchat_full_preprocessed.csv"
OUTPUT_FILE="data/scores/wildchat_full_scored_mistral.csv"
CHUNK_SIZE=1000

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "=================================================="
echo "🚀 Starting Mistral AI Scoring Process"
echo "=================================================="
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Chunk:  $CHUNK_SIZE conversations per checkpoint"
echo ""
echo "📊 Expected Performance:"
echo "   - Rate: 1-1.5 conversations/second"
echo "   - Total: 144,439 conversations"
echo "   - ETA: 24-40 hours"
echo ""
echo "⚡ Mistral API: 200 RPM (6x faster than Groq)"
echo "=================================================="
echo ""

# Run the scoring script with unbuffered output
python3 -u scripts/score_conversations_mistral.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --chunk-size "$CHUNK_SIZE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Scoring completed successfully!"
    echo "=================================================="
    echo "Output file: $OUTPUT_FILE"
    echo ""
    echo "Next steps:"
    echo "1. Run analysis notebook: notebooks/03_analysis.ipynb"
    echo "2. Generate embeddings for propensity score matching"
    echo "3. Estimate causal effects (Empathy → Attachment)"
else
    echo ""
    echo "=================================================="
    echo "❌ Scoring failed. Check the logs above."
    echo "=================================================="
    echo ""
    echo "💡 The script auto-resumes from checkpoints."
    echo "   You can safely re-run this script to continue."
    exit 1
fi
