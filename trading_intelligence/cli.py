from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Intelligence research CLI")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()
    if args.version:
        from trading_intelligence import __version__
        print(__version__)
    else:
        print("Trading Intelligence research scaffold is ready. No live execution is enabled.")


if __name__ == "__main__":
    main()
