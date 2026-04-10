## Contributing

PRs welcome. Please:
- keep evaluation outputs deterministic
- avoid adding heavyweight dependencies unless necessary
- add tests for new parsers / formats

Run tests:
```bash
python -m pip install -e .
python -m pip install pytest
pytest -q
```
