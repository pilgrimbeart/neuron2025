import sys

from app import SimulatorApp


if __name__ == "__main__":
    SimulatorApp(display_index=int(sys.argv[1]) if len(sys.argv) > 1 else 0).run()
