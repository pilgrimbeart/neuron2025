"""Start the simulator.

    python neuron.py [DISPLAY] [-c COMMAND]...

DISPLAY picks the screen (default 0). Each -c runs a console command at startup, after the last session has been
loaded, exactly as if typed, e.g.  python neuron.py -c "twophase 3" -c "speed 20"
"""

import argparse

from app import SimulatorApp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Excitable-cell simulator")
    parser.add_argument("display", nargs="?", type=int, default=0, help="display index (default 0)")
    parser.add_argument("-c", "--command", action="append", default=[], help="console command to run at startup (repeatable)")
    args = parser.parse_args()
    SimulatorApp(display_index=args.display).run(args.command)
