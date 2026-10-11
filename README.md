# Reward Model Evaluation for LLM Responses

This repository evaluates LLM-generated responses using automatic reward models and scoring metrics. It supports generating multiple responses from different language models, then scoring them with both scalar and multi-objective evaluators such as ArmoRM and BARTScore.

The project is organized as a multi-stage evaluation pipeline:

1. Generate multiple responses for each prompt.
2. Score responses using automatic evaluation metrics.
3. Analyze rankings, categories, and reward dimensions across models.

## Features

- JSONL-based evaluation pipeline
- Multi-response sampling for each prompt
- Support for OpenAI, Mistral, and Hugging Face models
- ArmoRM reward model evaluation
- BARTScore evaluation
- Resume support for long-running scoring jobs
- Category-based analysis utilities
- Reward dimension and gating analysis for ArmoRM

## Repository Layout

```text
.
|-- examples/
|   `-- queries_example.jsonl
|-- outputs/
|   |-- gpt-4o-mini/
|   |-- mistral-7b/
|   |-- mistral-12b/
|   `-- ...
|-- src/
|   |-- sampling_pipeline.py
|   |-- armorm_scoring_pipeline.py
|   |-- bart_scoring_pipeline.py
|   |-- model_registry.py
|   |-- analysis/
|   `-- providers/
|-- main.py
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.10+
- GPU recommended for ArmoRM scoring
- CUDA-compatible environment for faster inference

Install dependencies:

```bash
pip install -r requirements.txt
```

Virtual environment example:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Only set the credentials you need:

```env
OPENAI_API_KEY=...
MISTRAL_API_KEY=...
HF_TOKEN=...
```

Environment variables are automatically loaded from `.env` using `python-dotenv`.

## Supported Models

Run:

```bash
python main.py list-models
```

Example supported models:

- `gpt-4o-mini`
- `gpt-4o`
- `mistral-small-latest`
- `mistral-large-latest`
- `phi-3-mini`
- `phi-3-medium`

## Input Format

Input datasets use JSONL format.

Each line contains:

- `id`
- `prompt`

Example:

```json
{"id":"q1","prompt":"Explain what a cache is in 2 sentences."}
{"id":"q2","prompt":"Write a Python function for binary search."}
```

## Stage 1: Generate Responses

Generate multiple responses for each prompt:

```bash
python main.py generate \
  --data-path examples/queries_example.jsonl \
  --out-path outputs/gpt-4o-mini/responses/demo.jsonl \
  --model-name gpt-4o-mini \
  --num-samples 20
```

Useful options:

- `--data-path`: input JSONL file
- `--out-path`: output JSONL file
- `--model-name`: model key
- `--num-samples`: number of responses per prompt
- `--temperature`: sampling temperature
- `--top-p`: nucleus sampling
- `--max-new-tokens`: generation length limit
- `--limit`: process only first `K` prompts
- `--resume/--no-resume`: continue interrupted runs

Example output:

```json
{
  "id": "q1",
  "prompt": "Explain what a cache is.",
  "model": "gpt-4o-mini",
  "responses": [
    {
      "response_text": "...",
      "metadata": {
        "sample_index": 0
      },
      "token_logprobs": null
    }
  ]
}
```

## Stage 2: Score Responses with ArmoRM

Score generated responses using the reward model:

```bash
python main.py score \
  --data-path outputs/gpt-4o-mini/responses/demo.jsonl \
  --out-path outputs/gpt-4o-mini/scores/armorm/demo_scored.jsonl \
  --scorer armorm
```

ArmoRM outputs:

- `armorm_score`
- `armorm_rewards`
- `armorm_gating`

Example:

```json
{
  "armorm_score": 0.071,
  "armorm_rewards": [...],
  "armorm_gating": [...]
}
```

## Stage 3: Score Responses with BARTScore

Evaluate responses with BARTScore:

```bash
python main.py score \
  --data-path outputs/gpt-4o-mini/responses/demo.jsonl \
  --out-path outputs/gpt-4o-mini/scores/bartscore/demo_scored.jsonl \
  --scorer bart
```

Example output:

```json
{
  "bart_score": -15.18
}
```

## Analysis

The repository includes scripts for:

- Overall model ranking comparisons
- Category-based evaluation
- Safety vs coding performance analysis
- ArmoRM gating dimension analysis
- Reward dimension visualization
- Metric agreement comparison between ArmoRM and BARTScore

Example categories:

```python
CODING_PREFIXES = (
    "hep-python",
    "hep-go",
    "hep-cpp",
    "hep-js",
    "hep-rust",
    "hep-java",
)

SAFETY_REFUSE_PREFIXES = (
    "refusals-",
    "xstest-should-refuse",
)

SAFETY_RESPOND_PREFIXES = (
    "xstest-should-respond",
)
```

## Example Experiments

The evaluation pipeline can be used for experiments such as:

- Comparing model rankings across reward metrics
- Measuring category-specific performance
- Studying safety-sensitive prompts
- Investigating reward model mixture-of-experts behavior
- Comparing scalar and interpretable reward modeling approaches

## References

Reward Modeling:

- Wang, H., Xiong, W., Xie, T., Zhao, H., & Zhang, T. (2024). *Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts*. Findings of ACL: EMNLP 2024.
- Yuan, W., Neubig, G., & Liu, P. (2021). *BARTScore: Evaluating Generated Text as Text Generation*. NeurIPS 2021.

LLM Evaluation and Routing:

- Ding, D. et al. (2025). *BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute*. ICML 2025.
- Ong, I. et al. (2025). *RouteLLM: Learning to Route LLMs from Preference Data*. ICLR 2025.

Models:

- OpenAI models
- Mistral AI models
- Meta Llama models