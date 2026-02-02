# Dynamic Neuron Pruning for Large Language Models

Paper: DART-ing Through the Drift: Dynamic Tracing of Knowledge Neurons for Adaptive Inference-Time Pruning [[arXiv]](https://arxiv.org/abs/2601.22632)

Abhishek Tyagi*, Yunuo Cen*, Shrey Dhorajiya, Bharadwaj Veeravalli, Xuanyao Fong

To cite this paper, use
```
@misc{tyagi2026dartingdriftdynamictracing,
      title={DART-ing Through the Drift: Dynamic Tracing of Knowledge Neurons for Adaptive Inference-Time Pruning}, 
      author={Abhishek Tyagi and Yunuo Cen and Shrey Dhorajiya and Bharadwaj Veeravalli and Xuanyao Fong},
      year={2026},
      eprint={2601.22632},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.22632}, 
}
```

If you have questions or thoughts regarding the tool or this work, please contact atygai@u.nus.edu or cenyunuo@u.nus.edu.




https://github.com/user-attachments/assets/f5dfd7fb-6261-49dc-bb1c-68d19e3f838f




## Getting Started

This project implements dynamic neuron pruning strategies for large language models with adaptive masking and knowledge drift detection.

### Prerequisites

#### Required Packages

```bash
# Core dependencies
torch==2.2.0
transformers==4.57.3
datasets==4.4.2
numpy==1.26.4
pandas==2.2.0

# Evaluation and visualization
matplotlib==3.8.2
evaluate==0.4.6
cmcrameri==1.8

# Optional but recommended
accelerate==1.12.0
```

You can install all dependencies using:

```bash
pip install torch transformers datasets numpy pandas matplotlib evaluate cmcrameri accelerate
```

### Running Experiments

The project provides two main experiment scripts:

#### 1. **Standard Pruning Experiments** (`run_experiment.sh`)

For running pruning experiments with perplexity and MMLU evaluation:

```bash
# Edit configuration in run_experiment.sh
# Key parameters:
# - MODEL: Choose your model (e.g., "meta-llama/Llama-3.1-8B")
# - LAYER_TOPK: Pruning configuration per layer
# - MASKING_STEP: When to start applying masks
# - EVAL_PERPLEXITY: Enable perplexity evaluation
# - EVAL_MMLU: Enable MMLU evaluation

./run_experiment.sh
```

#### 2. **Knowledge Drift Analysis** (`run_knowledge_drift.sh`)

For analyzing knowledge drift during generation:

```bash
# Edit configuration in run_knowledge_drift.sh
# Additional parameters:
# - KNOWLEDGE_DRIFT: Enable drift detection
# - CUSTOM_PROMPT_TEXT: Specify your prompt
# - GENERATION: Number of tokens to generate

./run_knowledge_drift.sh
```

### Configuration Options

Both scripts support the following key parameters:

**Model Configuration:**
- `MODEL`: Model name from HuggingFace (e.g., "meta-llama/Llama-3.1-8B", "gpt2")
- `CACHE_DIR`: Directory for model weights
- `DEVICE`: CUDA device ID

**Pruning Configuration:**
- `LAYER_TOPK`: Per-layer neuron keep ratios (e.g., "all:auto" or "0:0.9,1:0.8,...")
- `MASKING_STEP`: Step to start masking neurons
- `RELEASE_STEP`: Step to release masks (optional)
- `EMA_DECAY`: Exponential moving average decay factor
- `RANKING_METHOD`: Neuron ranking method ("max", "mean", "combined", "product", "magnitude")
- `PRUNE_STRATEGY`: Pruning strategy ("topk", "auto")
- `TOTAL_PRUNE_PERCENT`: Target total pruning percentage

**Evaluation Configuration:**
- `EVAL_PERPLEXITY`: Enable perplexity evaluation
- `EVAL_MMLU`: Enable MMLU benchmark evaluation
- `EVAL_GENERAL_NLP`: Enable general NLP tasks evaluation

**Prompt Configuration:**
- `PROMPT_TYPE`: "custom" or "mmlu"
- `PROMPT_SUBJECT`: Dataset/subject name
- `CUSTOM_PROMPT_TEXT`: Custom prompt text
- `PROMPT_LENGTH`: Maximum prompt length

### Output

Results are saved to `results/` directory with:
- Output logs
- Timing information
- Configuration JSON
- Perplexity scores
- MMLU accuracy
- Neuron masking statistics
- Knowledge drift metrics (if enabled)

### Example Usage

```bash
# Quick test with GPT-2 on a small dataset
DEVICE=0
MODEL="gpt2"
LAYER_TOPK="all:auto"
MASKING_STEP=10
EVAL_PERPLEXITY=true
./run_experiment.sh

# Knowledge drift analysis with custom prompt
KNOWLEDGE_DRIFT=true
CUSTOM_PROMPT_TEXT="Write about artificial intelligence"
GENERATION=500
./run_knowledge_drift.sh
```

### Project Structure

```
.
├── dynamicPrune.py          # Main execution script
├── run_experiment.sh        # Standard experiment wrapper
├── run_knowledge_drift.sh   # Knowledge drift wrapper
├── src/
│   ├── neuronDefuser.py     # Core pruning logic
│   ├── perplexity_utils.py  # Perplexity evaluation
│   ├── mmlu_utils.py        # MMLU benchmark
│   ├── hook_setup.py        # Model hook registration
│   └── resultCompiler.py    # Result analysis and plotting
├── lib/                     # Additional utilities
├── datasets/                # Dataset storage
├── results/                 # Experiment outputs
└── llm_weights/            # Model cache
```

### Tips

1. **Memory Management**: For large models, reduce batch size or use gradient checkpointing
2. **Pruning Ratios**: Start with conservative values (keep 80-90% neurons) and adjust
3. **Adaptive Pruning**: Use `LAYER_TOPK="all:auto"` for automatic layer-wise optimization
4. **Logging**: Check both `output_*.log` and `timing_*.log` for detailed information

## License

See LICENSE file for details.
