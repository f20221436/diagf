# Quick Start Checklist for DiagFusion

## Pre-Execution Checklist

### ✅ Environment Setup
- [ ] Virtual environment activated: `(venvDiagFusion)` visible in prompt
- [ ] CUDA availability verified: Run `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Located in diagf directory: `C:\Users\DEVESH PALO\projects\DiagFusionWorking\diagf`

### ✅ Dataset Verification
- [ ] MicroSS path exists: `C:\Users\DEVESH PALO\projects\GAIA-DataSet-main\GAIA-DataSet-main\GAIA-DataSet-main\MicroSS`
- [ ] Required files present:
  - [ ] `gaia_resplit.csv`
  - [ ] `anomalies/demo_metric.json`
  - [ ] `anomalies/demo_trace.json`
  - [ ] `anomalies/stratification_logs.npy`

### ✅ Configuration Check
- [ ] `config/gaia_config.yaml` paths are correct
- [ ] `demo_path` points to MicroSS directory
- [ ] All anomaly file paths are relative to MicroSS

---

## Execution Command

```powershell
python main.py --config gaia_config.yaml
```

---

## Execution Stages (All Automated)

### Stage 1: Event Parsing (5-15 min)
**Input:** Metric, trace, log JSON/NPY files
**Output:** `parse/stratification_texts.pkl`
**Memory:** Low (sequential processing)

### Stage 2: FastText Embedding (10-30 min)
**Input:** `parse/stratification_texts.pkl`
**Output:** `fasttext/event_embedding.pkl`
**Memory:** Medium (text vectorization)

### Stage 3: Sentence Embedding (5-10 min)
**Input:** `fasttext/event_embedding.pkl`
**Output:** `sentence_embedding.pkl`
**Memory:** Medium (TF-IDF computation)

### Stage 4: GNN Training (30-120 min)
**Input:** `sentence_embedding.pkl`, `gaia_resplit.csv`
**Output:** `dgl/stratification_10/1/` (predictions & evaluations)
**Memory:** High (GPU utilized)
**GPU:** CUDA utilized automatically

---

## Progress Indicators to Watch For

### ✅ Successful Start Messages
```
================================================================================
STEP 1: Parsing Events from Metric, Trace, and Log Data
================================================================================
Processing metric, trace, and log data...
Parsing 2000 cases...
```

### ✅ GPU Confirmation (Step 4)
```
Using device: cuda
GPU: NVIDIA <GPU Name>
Memory Available: XX.XX GB
```

### ✅ Completion Messages
Each step should end with:
```
[<Step Name> Resource Usage]
```
Followed by a resource usage plot

---

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError` | Check paths in `gaia_config.yaml` |
| `CUDA out of memory` | Reduce `batch_size` from 1000 to 500 in config |
| Slow processing | Verify GPU is being used (check "Using device" message) |
| `KeyError` in CSV | Verify `gaia_resplit.csv` has correct columns |
| Memory error during parsing | Already optimized with chunking, should not occur |

---

## Final Results Location

After successful completion, find results at:

```
C:\Users\DEVESH PALO\projects\GAIA-DataSet-main\...\MicroSS\dgl\stratification_10\1\

preds/
  - instance_pred_multi_v0.csv    (service instance predictions)
  - anomaly_pred_multi_v0.csv     (anomaly type predictions)

evaluations/
  - instance/instance_acc_multi_v0.csv    (Top-K accuracy for instances)
  - anomaly/anomaly_acc_multi_v0.csv      (Top-K accuracy for anomaly types)
```

---

## Emergency Stop

If you need to stop execution:
- Press `Ctrl + C` to interrupt
- Resources will be cleaned up automatically
- Intermediate files are saved, so you can resume from checkpoints

---

## Post-Execution Verification

- [ ] Check final accuracy in evaluation CSV files
- [ ] Verify prediction files contain expected number of rows
- [ ] Review resource logs for memory/GPU usage patterns
- [ ] Compare results with baseline (if available)

---

## System Optimizations Applied

✅ **Memory Optimizations:**
- Chunked graph loading (50 graphs at a time)
- Explicit memory cleanup after each processing stage
- Sparse matrix handling for TF-IDF

✅ **GPU Optimizations:**
- Automatic CUDA device detection
- Pin memory for faster GPU transfers
- Model and data automatically moved to GPU

✅ **CPU Threading:**
- DataLoader num_workers=0 (recommended for Windows)
- Can be increased to 2-4 on Linux systems

---

**Total Expected Runtime:** 50-175 minutes (full pipeline)
**Peak RAM Usage:** ~12-14 GB
**GPU Utilization:** ~80-100% during training (Step 4)
