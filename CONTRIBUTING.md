# Contributing

Contributions are welcome through issues and pull requests.

## Development setup

```bash
python -m pip install -e ".[dev,bench]"
PYTHONPATH=src pytest
ruff check .
mypy src/crisp_route
```

## Contribution priorities

The highest-value contributions for the next release are correlated scenario risk, time-dependent edges, incremental graph updates, formal approximation analysis, benchmark datasets, and additional scalar or exact multi-objective baselines.

Please include tests for behavioral changes and avoid presenting synthetic benchmark results as evidence of real-world superiority.
