"""CLI commands for run_megatron.

Usage:
    python -m miles.debug.run_megatron run ...
    python -m miles.debug.run_megatron compare ...
    python -m miles.debug.run_megatron run-and-compare ...
    python -m miles.debug.run_megatron show-model-args ...
"""

import typer

from miles.debug.run_megatron.cli.commands import compare, run, run_and_compare

app: typer.Typer = typer.Typer(pretty_exceptions_enable=False)

run.register(app)
compare.register(app)
run_and_compare.register(app)
