from invoke import task
from pathlib import Path


@task
def test_integration(ctx):
    dir = Path("test/integration")
    ctx.run(f'python3 {dir / "execute.py" } --test-dir {dir.absolute()}')


@task
def test_unit(ctx):
    ctx.run("python3 -m pytest test/unit")
