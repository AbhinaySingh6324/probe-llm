# 🔬 probe

**Behavioral testing framework for LLMs — test properties, not benchmarks.**

---

Benchmarks tell you a model scores 85% on MMLU.
`probe` tells you whether the model **actually behaves consistently** on *your* tasks.

```bash
pip install -e ".[openai]"
```

## Quick Start

### Python

```python
from probe import quick_test, print_summary

results = quick_test(
    model="openai:gpt-4o-mini",
    inputs=[
        "What is the capital of France?",
        "Is 17 a prime number?",
        "Explain photosynthesis in 2 sentences.",
    ],
    properties=["consistency", "robustness"],
)

print_summary(results)
```

### CLI

```bash
probe run --model openai:gpt-4o-mini --input "What is the capital of France?" -p consistency
probe run --model openai:gpt-4o-mini --inputs test_cases.txt -p consistency,invariance,robustness
probe compare --model-a openai:gpt-4o --model-b ollama:llama3 --inputs test_cases.txt
probe list-properties
```

## Behavioral Properties

| Property | What It Tests |
|----------|-------------|
| **consistency** | Rephrase question N ways → same answer? |
| **invariance** | Change irrelevant details → answer holds? |
| **negation** | Negate the question → answer flips? |
| **robustness** | Add typos → model still works? |

## Providers

```
openai:gpt-4o-mini        # OpenAI
anthropic:claude-sonnet-4-20250514  # Anthropic
ollama:llama3              # Local via Ollama
```

## License

MIT
