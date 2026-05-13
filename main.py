from __future__ import annotations

import sys

from app import SimulatorApp


def main() -> None:
    display_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    app = SimulatorApp(display_index=display_index)
    app.run()


if __name__ == "__main__":
    main()
