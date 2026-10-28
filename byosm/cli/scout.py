from __future__ import annotations

from .base import app, log


@app.command()
def scout() -> None:
    """Read recent papers on arXiv and Hugging Face and propose techniques for the catalogue; you accept or reject them."""
    from ..config import find_workspace
    from ..scout.proposals import scan

    scan(find_workspace(), log)
