import argparse
from pathlib import Path

from wire_inspector.gui import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FC wiring color/order inspector")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    run(parser.parse_args().config)
