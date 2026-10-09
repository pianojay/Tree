# Tree

A miniature for making the abstract mathematics of generative models explicit and visible. See [whitepaper.md](whitepaper.md).

## Setup (Windows, CPU)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

## Tests

```bash
.venv/Scripts/python -m tests.test_engine   # gradient check of the pure-Python reference engine
.venv/Scripts/python -m tests.test_learns   # perceptron vs. MLP on XOR (pure Python, ~2 min)
```

## License

MIT
