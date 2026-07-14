from __future__ import annotations

import argparse
import logging
import sys

from ingest.run_vbpl import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--no-keyword", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for name in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    run(
        recreate=args.recreate,
        keyword_filter=not args.no_keyword,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
