"""Entry point script for running the Nero voice assistant."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from nero.config_loader import load_config  # noqa: E402
from nero.core.orchestrator import NeroOrchestrator  # noqa: E402


def main() -> None:
    """Load configuration and start the Nero orchestration loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config, persona = load_config()
    orchestrator = NeroOrchestrator(config, persona)
    try:
        orchestrator.run_forever()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, stopping Nero...")
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    main()