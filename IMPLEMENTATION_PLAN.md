# LLM Systems Engineer Assessment — Implementation Plan

## Status: COMPLETE

All 4 tasks implemented and verified. Actual results recorded below.

| Module | Status | Notes |
|---|---|---|
| Module 0: Environment | ✅ Done | MPS (Apple Silicon) device, requirements.txt pinned |
| Module 1: Architecture Parser | ✅ Done | GPT-2 + TinyLlama parsed, comparison + visualization |
| Module 2: Fine-Tuning | ✅ Done | 82.2% eval accuracy, adapter saved to `gpt2-lora-imdb/` |
| Module 3: Composition | ✅ Done | merge_and_unload + weight averaging sweep fixed |
| Module 4: Eval + Plots | ✅ Done | Loss curve, confusion matrix, bar chart, interpolation |
| Module 5: Notebook | ✅ Done | 26 cells, sequential, all outputs inline |
| Module 6: README | ✅ Done | All 4 Task 4 sections complete including Honest Reflection |

### Actual Measured Results
| Variant | Accuracy (100 samples) | F1 |
|---|---|---|
| Base GPT-2 (untrained) | 0.51 | 0.04 |
| LoRA Fine-Tuned (2 epochs) | 0.79 | 0.79 |
| Merged (merge_and_unload) | 0.79 | 0.79 |
| Full eval set (500 samples) | **0.822** | — |

### Bug Fixed Post-Implementation
Weight averaging cell originally returned 0.79 for all alpha values. Root cause: PEFT wraps `base_model` in-place — after training, `base_model` and `merged_model` were the same object. Fixed by reloading a fresh untrained GPT-2 as the true baseline endpoint for the interpolation sweep.

---

## Overview

This document provides a module-by-module implementation plan for the Junior LLM Systems Engineer assessment. Models chosen: **GPT-2 (124M)** and **TinyLlama-1.1B-Chat-v1.0 (1.1B)**, with fine-tuning performed on GPT-2.

---

## Chosen Stack

| Component | Choice | Reason |
|---|---|---|
| Models | GPT-2 + TinyLlama | Low VRAM, fast to load, architecturally different (encoder-only vs decoder) |
| Fine-tune dataset | IMDB sentiment (HuggingFace) | Small, well-known, clean labels |
| PEFT method | LoRA | Low memory, easy merge, widely understood |
| Merge strategy | LoRA adapter merge into base + weight averaging | Covers both composition approaches |
| Notebook format | Jupyter (.ipynb) | Required by assignment |

---

## Module 0: Environment Setup

**Goal:** Reproducible install, no version conflicts.

```
requirements.txt contents:
  torch
  transformers
  peft
  datasets
  bitsandbytes      # for 4-bit (optional, used if Phi-4 added)
  accelerate
  evaluate
  scikit-learn
  matplotlib
  seaborn
  tqdm
  jupyter
  ipywidgets
```

**Steps:**
1. Create `requirements.txt`
2. Pin major versions where stability matters (`peft>=0.9`, `transformers>=4.40`)
3. Add a setup cell at top of notebook: imports + device detection

---

## Module 1: Model Architecture Parser

**Goal:** Reusable parser that returns a nested dict representing any HuggingFace model's internal structure.

### Design

```
parse_model_architecture(model, model_name) -> dict
```

Recursive walker that inspects every `nn.Module`:
- Layer name (key in parent module)
- Module class name
- Parameter count (trainable + frozen)
- Shape of weight tensors
- Children (recursive)

Output structure:
```json
{
  "model_name": "GPT2LMHeadModel",
  "total_params": 124439808,
  "trainable_params": 124439808,
  "children": {
    "transformer": {
      "class": "GPT2Model",
      "children": {
        "wte": { "class": "Embedding", "shape": [50257, 768], "params": 38597376 },
        "wpe": { "class": "Embedding", "shape": [1024, 768], "params": 786432 },
        "h": {
          "class": "ModuleList",
          "children": {
            "0": {
              "class": "GPT2Block",
              "children": {
                "ln_1": { "class": "LayerNorm", ... },
                "attn": { "class": "GPT2Attention", ... },
                "ln_2": { "class": "LayerNorm", ... },
                "mlp":  { "class": "GPT2MLP", ... }
              }
            }
            ...
          }
        }
      }
    }
  }
}
```

### Key implementation details
- Use `model.named_children()` (not `named_modules()`) to avoid flattened output — children gives one level at a time for true tree shape
- Count params with `sum(p.numel() for p in m.parameters())`
- Separate `trainable` vs `frozen` counts (relevant post-LoRA)
- Add a `print_tree(parsed_dict, indent=0)` utility for readable console output
- Add a `compare_architectures(parsed_a, parsed_b)` utility that diffs two parsed dicts (shows structural differences between GPT-2 and TinyLlama)

### Why this approach
- Recursive walk gives arbitrary depth, works on any `nn.Module` without model-specific logic
- Nested dict is serializable (can be saved as JSON), queryable, and diffable
- Clean separation between parsing logic and display logic (reusable)

---

## Module 2: Fine-Tuning with LoRA on GPT-2

**Goal:** Fine-tune GPT-2 on IMDB (binary sentiment) using PEFT/LoRA. Keep it short (2 epochs) and logged.

### Dataset prep
- Load `datasets.load_dataset("imdb")`
- Subsample: 2000 train / 500 eval (fast iteration)
- Tokenize with `GPT2Tokenizer`, set `pad_token = eos_token`
- Use causal LM framing: input = review text, label = same tokens shifted (standard CLM)
- Alternatively: add a classification head and train as sequence classifier — **chosen approach** since it directly maps to sentiment

### LoRA config
```python
LoraConfig(
    r=8,                        # rank — small enough to be fast
    lora_alpha=32,              # scaling factor
    target_modules=["c_attn"],  # GPT-2's combined QKV projection
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
```

### Training setup
```python
TrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,           # if CUDA available
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)
```

### Metrics tracked
- Training loss per step
- Eval accuracy and F1 per epoch
- Before-training baseline (zero-shot accuracy on 100 samples)

### Output artifacts
- `gpt2-lora-imdb/` — saved adapter weights (only ~2MB, not full model)
- Loss curve plot
- Confusion matrix

---

## Module 3: Model Composition

**Goal:** Two composition experiments showing before/after.

### 3A: Merge LoRA adapter into GPT-2 base
```python
merged_model = peft_model.merge_and_unload()
```
- Before: base GPT-2, PEFT wrapper with frozen base + trainable LoRA
- After: single standard GPT-2 with LoRA weights absorbed — same inference, no PEFT overhead
- Show: parameter counts are identical, but accuracy changes; no PEFT dependency at inference

### 3B: Weight averaging between GPT-2 base and TinyLlama (same-architecture demonstration)

Since GPT-2 and TinyLlama have different architectures, weight averaging is demonstrated on:
- GPT-2 base vs GPT-2 fine-tuned (merged) — same architecture, direct layer averaging
- This is a clean, honest demonstration of linear interpolation between two checkpoints

```python
def weight_average(model_a, model_b, alpha=0.5):
    """
    Returns a new state_dict that is alpha * A + (1-alpha) * B
    Only averages layers that exist in both models with identical shapes.
    """
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    merged = {}
    for key in sd_a:
        if key in sd_b and sd_a[key].shape == sd_b[key].shape:
            merged[key] = alpha * sd_a[key] + (1 - alpha) * sd_b[key]
        else:
            merged[key] = sd_a[key]  # keep A's weights for mismatched layers
    return merged
```

- Show accuracy at alpha = 0.0, 0.25, 0.5, 0.75, 1.0 (interpolation curve)
- alpha=0.0 → 100% fine-tuned (merged), alpha=1.0 → 100% fresh untrained base
- Curve degrades monotonically from ~0.79 toward ~0.51, confirming task-specific signal erodes under interpolation with random weights
- **Note**: original implementation had alpha endpoints pointing to the same model (PEFT in-place modification bug) — fixed by reloading a truly fresh GPT-2 checkpoint as the untrained endpoint

---

## Module 4: Evaluation & Visualization

**Goal:** Quantify before/after differences and visualize architecture.

### Evaluations
1. **Base GPT-2** accuracy on IMDB test (100 samples) — baseline
2. **LoRA fine-tuned GPT-2** accuracy — after training
3. **Merged GPT-2** accuracy — should match fine-tuned
4. **Weight-averaged (alpha=0.5)** accuracy — interpolated

### Visualizations
1. **Architecture tree diagram** — nested box/tree plot using `matplotlib` showing GPT-2 layer hierarchy
2. **Training loss curve** — from Trainer logs
3. **Before/after accuracy bar chart** — all 4 model variants
4. **Weight averaging interpolation curve** — accuracy vs alpha

---

## Module 5: Notebook Structure

**Notebook cell order (clean, sequential, reproducible):**

```
[0] Title & Overview markdown
[1] Environment setup & imports
[2] Device detection & seed
[3] --- TASK 1: Architecture Parser ---
[4] Define parse_model_architecture()
[5] Define print_tree() and compare_architectures()
[6] Load GPT-2 → parse → display
[7] Load TinyLlama → parse → display
[8] Architecture comparison diff
[9] Architecture visualization
[10] --- TASK 2: Fine-Tuning ---
[11] Load IMDB dataset & subsample
[12] Tokenization
[13] Baseline evaluation (pre-training)
[14] Define LoRA config & wrap model
[15] Training loop (Trainer API)
[16] Post-training evaluation
[17] Loss curve plot
[18] Save adapter
[19] --- TASK 3: Model Composition ---
[20] 3A: Merge adapter → unload → verify
[21] 3B: Weight averaging function
[22] Interpolation curve sweep
[23] Before/after accuracy bar chart
[24] --- SUMMARY ---
[25] Final metrics table (markdown cell)
```

---

## Module 6: README.md Answers (Design Thinking)

The README answers Task 4 directly. Key points to cover:

### Design Decisions
- Recursive child-walk vs flat `named_modules()`: chose recursive for true tree shape
- Separation of parse/display/compare functions: single responsibility, reusable
- Chose IMDB classification over CLM fine-tuning: cleaner before/after metric (accuracy vs perplexity)

### Working Conditions
- Hardware: Apple M-series / Intel CPU (document actual specs)
- Times: parsing ~5s, dataset prep ~2min, training 2 epochs ~15–25min on CPU, ~5min on GPU
- Bottleneck: tokenization of IMDB on CPU; solved by `dataset.map(..., num_proc=2)`

### Extensibility & Scalability
- 10 models: parser already handles any `nn.Module`, add a model registry dict
- 50 models: parallelize loading with `concurrent.futures`, stream architecture dicts to disk
- Different architectures: parser is architecture-agnostic; weight averaging needs shape-matching guard (already implemented); add an architecture family tag to parsed dict

### Creativity & Future Vision
1. **Auto-gap detection**: parse architecture → run eval on diverse benchmarks → identify weak attention heads via attribution → target only those heads for LoRA fine-tuning
2. **Evolutionary merging**: maintain a population of weight-averaged checkpoints, score each, keep top-k, repeat — gradient-free model improvement loop
3. **Architecture-aware composition**: instead of naive weight averaging, align layers by functional role (embed layer ↔ embed layer, last attn ↔ last attn) even across architectures with different depths

### Honest Reflection
- Straightforward: LoRA setup with PEFT, Trainer API, basic parsing
- Challenging: making the parser output truly hierarchical (not flat), handling TinyLlama's different module naming convention
- External help: HuggingFace docs for PEFT LoRA target_modules per architecture, consulted for correct GPT-2 attention module names

---

## File Structure (Final Deliverables)

```
LLm-assessment/
├── notebook.ipynb          # all tasks, outputs, inline explanations
├── requirements.txt
├── README.md               # Task 4 analysis (max 1.5 pages)
├── IMPLEMENTATION_PLAN.md  # this file
└── gpt2-lora-imdb/         # saved adapter (generated at runtime)
    ├── adapter_config.json
    └── adapter_model.safetensors
```

---

## Execution Order

1. Set up environment (`pip install -r requirements.txt`)
2. Run notebook cells top-to-bottom — fully sequential, no hidden state
3. All outputs (plots, metrics) are inline in notebook
4. README is written after notebook confirms results

---

## Time Estimate

| Module | Estimated Time |
|---|---|
| Environment + imports | 15 min |
| Task 1: Parser | 45 min |
| Task 2: Fine-tuning | 60–90 min (mostly waiting) |
| Task 3: Composition | 30 min |
| Task 4: Evaluation + plots | 30 min |
| README.md write-up | 45 min |
| **Total** | **~4 hours** |
