# START HERE - DiagFusion Full Dataset Execution

## 🎯 What I Need to Do

Run DiagFusion on the full GAIA dataset with these commands:

## 📋 Before Starting (5 minutes)

### 1. Check I'm in the right place:
```powershell
# Should show: (venvDiagFusion) PS C:\Users\DEVESH PALO\projects\DiagFusionWorking\diagf>
pwd
```

### 2. Verify CUDA is working:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

**Expected:** `CUDA Available: True` and `CUDA Version: 11.5`

### 3. Check dataset files exist:
```powershell
ls "C:\Users\DEVESH PALO\projects\GAIA-DataSet-main\GAIA-DataSet-main\GAIA-DataSet-main\MicroSS\gaia_resplit.csv"
ls "C:\Users\DEVESH PALO\projects\GAIA-DataSet-main\GAIA-DataSet-main\GAIA-DataSet-main\MicroSS\anomalies\"
```

**Expected:** Files should exist without errors

---

## ▶️ Run the Complete Pipeline (One Command!)

```powershell
python main.py --config gaia_config.yaml
```

**Total Time:** ~50-175 minutes (depending on dataset size)

---

## 📊 What Will Happen (4 Automatic Steps)

### STEP 1: Event Parsing (5-15 min)
- Reads metric, trace, log files
- Creates text sequences
- **Watch for:** `Parsing events: 100%`

### STEP 2: FastText Embedding (10-30 min)
- Trains word embeddings
- Data augmentation
- **Watch for:** `fasttext time used: XXX s`

### STEP 3: Sentence Embedding (5-10 min)
- Creates TF-IDF weighted vectors
- **Watch for:** `sentence_embedding shape: X * Y * Z`

### STEP 4: GNN Training (30-120 min)
- Trains neural network on GPU
- **Watch for:** `Using device: cuda` and `GPU: NVIDIA ...`
- **Watch for:** Training progress bar reaching 100%

---

## ✅ Success Indicators

Each step should show:
```
================================================================================
STEP X: <Step Name>
================================================================================
[Processing messages...]

[<Step Name> Resource Usage]
<Plot displayed>
```

For Step 4 specifically, look for:
```
Using device: cuda
GPU: NVIDIA <Your GPU Name>
Memory Available: XX.XX GB
```

---

## 🎁 Where Are My Results?

After completion, find results at:

```
C:\Users\DEVESH PALO\projects\GAIA-DataSet-main\GAIA-DataSet-main\GAIA-DataSet-main\MicroSS\dgl\stratification_10\1\
```

**Key Files:**
- `preds/instance_pred_multi_v0.csv` - Predicted root cause services
- `preds/anomaly_pred_multi_v0.csv` - Predicted fault types
- `evaluations/instance/instance_acc_multi_v0.csv` - Accuracy scores

---

## 🚨 If Something Goes Wrong

| Error | Quick Fix |
|-------|-----------|
| "No such file" | Check path in `config/gaia_config.yaml` |
| "CUDA out of memory" | Edit `gaia_config.yaml`, change `batch_size: 1000` to `batch_size: 500` |
| "No module named..." | Activate venv: `.\venvDiagFusion\Scripts\Activate.ps1` |
| Process is very slow | Check that "Using device: cuda" appears in Step 4 |

---

## 📚 More Detailed Help?

- **Full guide:** `EXECUTION_GUIDE.md` (detailed troubleshooting)
- **Quick checklist:** `QUICK_START_CHECKLIST.md` (step-by-step verification)
- **Code changes:** `CODE_MODIFICATIONS_SUMMARY.md` (what was modified)

---

## 💡 What Was Optimized?

✅ **Memory:** Processes data in chunks (won't crash with 16GB RAM)
✅ **GPU:** Automatically uses your CUDA 11.5 GPU
✅ **Speed:** GPU training is 3-4x faster than CPU
✅ **Visibility:** Progress bars and status messages for each step

---

## 🎬 Ready to Start?

Just run:
```powershell
python main.py --config gaia_config.yaml
```

Then wait and watch the progress! ☕

**Note:** You can safely leave it running. Each step saves results, so even if interrupted, you won't lose all progress.

---

**Questions?** Check the detailed guides mentioned above or review error messages carefully.
