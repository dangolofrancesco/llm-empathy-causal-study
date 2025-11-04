# Mistral AI Scoring Guide

## 🚀 Quick Start

```bash
./scripts/run_scoring_mistral.sh
```

## 📊 Performance Expectations

### Expected Performance (with optimizations)
- **Rate**: 0.8-1.2 conv/s (with delays to avoid rate limits)
- **Timeline**: 33-50 hours for 144,439 conversations
- **Rate limit**: 200 RPM (Mistral free tier)

### Actual vs Theoretical
- **Theoretical max**: 200 RPM = 3.33 req/s = 1.67 conv/s (2 requests per conversation)
- **With delays**: 0.7s per conversation = ~1.4 conv/s
- **With rate limit backoff**: ~0.8-1.2 conv/s (realistic)

## ⚙️ Optimizations Applied

### 1. Delays Between API Calls
- **0.35s** after empathy scoring
- **0.35s** after attachment scoring
- **Total**: ~0.7s per conversation

This keeps us under the 200 RPM limit while maintaining good throughput.

### 2. Exponential Backoff
- First retry: 5s wait (was 4s)
- Second retry: 8s wait (was 8s)
- Third retry: 14s wait (was 16s)
- Extra 2s for rate limit errors

### 3. Auto-Resume from Checkpoints
- Saves every **1,000 conversations**
- Safe to interrupt (Ctrl+C) and resume
- No data loss

## 🔍 Monitoring Progress

### Check Current Progress
```bash
# Check how many rows in the output file
wc -l data/scores/wildchat_full_scored_mistral.csv

# View last few scored conversations
tail data/scores/wildchat_full_scored_mistral.csv
```

### Calculate Completion Percentage
```bash
# Total conversations
total=144439

# Current progress (subtract 1 for header)
current=$(wc -l < data/scores/wildchat_full_scored_mistral.csv)
current=$((current - 1))

# Calculate percentage
echo "scale=2; ($current / $total) * 100" | bc
```

### Estimated Time Remaining
The script shows real-time ETA in the progress bar:
```
Processing: 1,234/144,439 (0.9%) | Rate: 1.12 conv/s | Elapsed: 0.3h | ETA: 35.2h
```

## 🛑 Handling Interruptions

### If Script Stops or You Need to Interrupt

1. **Press Ctrl+C** to gracefully stop
2. **Check checkpoint**: Look at `data/scores/wildchat_full_scored_mistral.csv`
3. **Resume**: Just run `./scripts/run_scoring_mistral.sh` again

The script automatically:
- Detects where you left off
- Skips already-scored conversations
- Continues from the last checkpoint

### If Rate Limits Are Too Aggressive

Edit `scripts/score_conversations_mistral.py` and increase delays:

```python
# Line ~344 - increase these values
time.sleep(0.5)  # was 0.35, between empathy and attachment
# ...
time.sleep(0.5)  # was 0.35, after each conversation
```

This will slow down to ~0.6 conv/s but reduce rate limit errors.

## 📈 Rate Limit Troubleshooting

### Symptoms
```
⚠️  Rate limit hit. Waiting 5s before retry 1/3...
⚠️  Rate limit hit. Waiting 8s before retry 2/3...
```

### Solutions

#### Option 1: Increase Delays (Recommended)
More conservative approach, fewer errors:
```python
time.sleep(0.5)  # instead of 0.35
```

#### Option 2: Wait for Rate Limit Window Reset
Mistral's free tier: **200 requests per minute**
- If you hit the limit, wait 60 seconds for the window to reset
- Script handles this automatically with exponential backoff

#### Option 3: Upgrade Mistral Plan
- **Free tier**: 200 RPM
- **Experiment plan**: 1,000 RPM ($0.14/1M tokens)
- **Developer plan**: 10,000 RPM ($0.14/1M tokens)

See: https://console.mistral.ai/billing/

## 📁 Output Files

### Main Output
- **Path**: `data/scores/wildchat_full_scored_mistral.csv`
- **Columns**: All original columns + `empathy_score` + `attachment_score`
- **Size**: ~50-60 MB (estimated)

### Checkpoint Strategy
- Saves every 1,000 conversations
- File is overwritten (not appended)
- Contains all scored conversations so far

## 🔧 Advanced Usage

### Score a Subset for Testing
```bash
# Create 10k subset
head -n 10001 data/filtered/wildchat_full_preprocessed.csv > data/filtered/wildchat_10k_test.csv

# Score it
python3 scripts/score_conversations_mistral.py \
    --input data/filtered/wildchat_10k_test.csv \
    --output data/scores/wildchat_10k_scored.csv \
    --chunk-size 1000
```

### Start from Specific Row
```python
# In score_conversations_mistral.py, modify:
score_conversations_mistral(
    input_file="data/filtered/wildchat_full_preprocessed.csv",
    output_file="data/scores/wildchat_full_scored_mistral.csv",
    start_from=5000,  # Start from row 5000
    chunk_size=1000
)
```

## ⏱️ Timeline Summary

| Scenario | Rate | Time for 144k |
|----------|------|---------------|
| Theoretical max | 1.67 conv/s | 24 hours |
| **With delays (current)** | **0.8-1.2 conv/s** | **33-50 hours** |
| Very conservative | 0.5 conv/s | 80 hours |

## 💡 Tips

1. **Run overnight**: Start before bed, let it run 8-10 hours
2. **Check morning**: See progress, adjust if needed
3. **Use `screen` or `tmux`**: Keep running if SSH session closes
4. **Monitor rate**: If < 0.5 conv/s consistently, increase delays
5. **Failed scores**: Some will fail (network, parsing), that's normal (<1%)

## 🎯 Success Criteria

- **Rate**: Stable at 0.8-1.2 conv/s
- **Failed scores**: < 1% of total
- **Rate limit errors**: Occasional, handled by backoff
- **Checkpoints**: Saving successfully every 1k conversations

## 📞 Need Help?

Common issues and solutions are documented above. If you encounter persistent problems:

1. Check the terminal output for specific errors
2. Look at the last few lines of the output CSV
3. Verify API key is correctly set in `.env`
4. Ensure sufficient disk space (~100 MB free)
