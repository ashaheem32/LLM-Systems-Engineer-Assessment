# Junior LLM Systems Engineer Assessment

## Design Thinking & Key Decisions

### Recursive-Walk vs. Flat Model Iteration
When parsing the model architecture for the visualizer, I opted for a recursive traversal using `model.named_children()` instead of a flat list approach like `model.named_modules()`. 
* **Reasoning**: A flat list creates a 1-dimensional array that fundamentally collapses the hierarchical reality of transformer architectures. By utilizing recursion on `named_children()`, the code captures the true nested depth of the model (e.g., `GPT2LMHeadModel -> transformer -> h -> 0 -> attn`). This inherently enables rich structured visualizations and allows the user direct tracking of sub-component sizes, making parameter distribution comparisons between modules much easier and visually representative.

### IMDB Classification vs. Sequence Causal LM Tuning
For the LoRA fine-tuning task, I utilized the `AutoModelForSequenceClassification` object on the IMDB sentiment dataset rather than continuing with traditional next-token Causal Language Modeling wrapper tasks.
* **Reasoning**: Performing a binary sequence classification creates a crisp, clear baseline metric. Standard Causal LM fine-tuning on IMDB necessitates evaluating via perplexity or unstructured generation, which is difficult to quantify cleanly in an automated script. Sequence classification outputs a strict Accuracy and F1 score, offering a highly defined "before and after" boundary that perfectly illustrates the LoRA injection performance against the baseline.

## Working Conditions & Hardware Context
* **Hardware Profile**: The pipeline assumes and runs standard testing workflows mapping correctly to the active hardware state, natively binding PyTorch configurations directly to Apple Silicon (`mps`), NVIDIA sets (`cuda`), or fallback generic `cpu` structures. 
* **Timings & Bottlenecks**: Execution times reflect the hardware compute ceiling:
    * **Parsing & Base Setup**: Negligible latency (~5 seconds per parsing iteration on raw files).
    * **Tokenization Bottleneck**: String manipulation constraints are handled cleanly using `datasets` parallel multiprocessing (`num_proc=2`).
    * **Training Loop Iteration**: Processing subset arrays utilizing LoRA rank `r=8` over 2,000 IMDB samples averages ~5-10 minutes on a local accelerated pipeline (`cuda` / `mps`) but can heavily jump upwards toward ~25+ minutes utilizing isolated CPU instruction sets.

## Scalability & System Extensibility
* **Scaling to 50 Models**: Sequentially parsing and loading the architecture bounds of 50 models inside a Jupyter notebook would crash the underlying state memory or massively throttle the IDE. Scaling to that volume requires shifting the core design loop into an asynchronous data ingestion python node application utilizing `concurrent.futures`. This application would initialize, parse the graph locally, and stream out generic hierarchical JSON artifact files natively to disk. Then, a front-end UI framework (like Streamlit or FastAPI + React) can dynamically serve comparisons spanning those 50 raw JSON artifacts dynamically, circumventing the need for torch to continually manage tensor block RAM loads in the background.

## Creative Future Visions for Model Composition
1. **Auto-Gap Detection for Targeted Fine-Tuning**
   Rather than applying LoRA layers equivalently across standard projection targets blindly (e.g., covering all `c_attn` layers across the whole depth uniformly), we can pipe the hierarchical architecture parser explicitly alongside benchmarking. The system evaluates the model against targets, isolates structurally weak internal attention heads using activation gradient checks, and only mounts LoRA matrices dynamically into those explicit layers. This generates an aggressively hyper-optimized sub-fine-tuning matrix mapping over only the core logic faults.
2. **Evolutionary Checkpoint Breeding Arrays**
   Expanding the static logic behind linear weight-averaging between two endpoints, we can establish an evolutionary algorithm loop. Instead of averaging purely the A and B checkpoints, the application loads an entire `history` of locally saved adapter steps spanning the pipeline. It would selectively average (`alpha` blend) random subset layers simulating cross-breeding and generate populations of new models. The system evaluates these sub-model offspring natively, prunes the failed models recursively, and outputs an actively compounded model structure operating over the optimal parameter efficiency Pareto-curve automatically.

3. **Architecture-Aware Cross-Model Composition**
   Naive weight averaging requires identical tensor shapes — it breaks entirely between GPT-2 and TinyLlama because every matrix dimension differs. A more principled approach aligns layers by *functional role* rather than name or index: the embedding layer maps to the embedding layer, the last attention block to the last attention block, and so on, regardless of depth differences. Within each aligned pair, we project the smaller model's weights up to the larger model's shape using a learned linear projection (trained as a lightweight adapter), then interpolate in that shared space. This extends composition to architectures that would otherwise be entirely incompatible and opens a path toward heterogeneous model merging at scale.

---

## Honest Reflection

**What felt straightforward:**
- The PEFT/LoRA setup using HuggingFace's `Trainer` API was smooth — the library handles device placement, gradient checkpointing, and metric logging with minimal boilerplate.
- Loading and tokenizing IMDB was fast and clean. The `datasets` library's batched `.map()` with `num_proc=2` resolved the tokenization bottleneck without any custom DataLoader work.
- Basic architecture parsing once the recursive `named_children()` approach was chosen — the tree falls out naturally from the module hierarchy.

**What was genuinely challenging:**
- **The weight averaging flat-line bug**: After `merge_and_unload()`, PEFT had already modified `base_model` in-place, so both averaging endpoints had identical weights — every alpha value produced the same accuracy (0.79). The fix required understanding that PEFT wraps the original object by reference, not by copy, and reloading a truly fresh untrained checkpoint as one endpoint.
- **Making the parser output a true tree, not a flat list**: `model.named_modules()` returns every submodule in a flat iterator. Switching to `named_children()` with a recursive walk was necessary to get genuine nesting depth, but required explicitly handling `ModuleList` nodes (which have numeric string keys) to avoid display issues.
- **MPS compatibility**: Apple Silicon's MPS backend does not support `pin_memory=True` in DataLoader, and `fp16` training triggers silent NaN issues on MPS. Both required conditional guards (`fp16=torch.cuda.is_available()` and suppressing the pin_memory warning).

**Where external help was used:**
- HuggingFace PEFT documentation for the correct `target_modules` names per architecture (`c_attn` for GPT-2, `q_proj/v_proj` for LLaMA variants).
- PyTorch docs to confirm `named_children()` vs `named_modules()` behavior.
- The final implementation decisions — recursive walker design, classification framing over CLM, weight averaging endpoints — were my own architectural choices made after understanding the constraints.
