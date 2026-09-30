"""Watch an evolved cell rule run the robot.

    python app.py [KERNEL] [-c COMMAND]...

Four quarters: the cells (top left: v green when positive, red when negative; m, or w, blue), the world (top right), each
actuator's v over time (bottom left, the firing level dotted), and a console (bottom right). Commands are typed at the
console or written to control_in.txt; output also goes to control_out.log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame

import cell
import evolve
import robot

CONTROL_IN, CONTROL_OUT = Path("control_in.txt"), Path("control_out.log")
HELP = """commands:
  kernel NAME         load kernels/NAME.json and restart
  task NAME           move, approach, taste, choose or forage, and restart
  food red|blue       which colour is food, and restart (in taste, the one block is red: food or poison)
  seed N              the world's seed, and restart
  restart             a new life with the same settings
  speed N             ticks per frame (1..50)
  pause / resume
  view m|w            which variable shows in blue on the cells
  rule                print the rule
  quit"""
TICKS_PER_S = 10            # at speed 1


class App:
    def __init__(self, kernel: str):
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.width, self.height = self.screen.get_size()
        self.font = pygame.font.Font(pygame.font.match_font("couriernew"), 16)
        self.lines: list[str] = []
        self.input = ""
        self.kernel, self.task, self.food, self.seed = kernel, "choose", robot.RED, 0
        self.speed, self.paused, self.running = 1, False, True
        self.blue = "m"
        self.log = open(CONTROL_OUT, "a", buffering=1)
        CONTROL_IN.write_text("")
        self.restart()

    # --- console ---------------------------------------------------------------------------------------------------

    def say(self, text: str) -> None:
        for line in text.splitlines():
            self.lines.append(line)
            self.log.write(line + "\n")
        self.lines = self.lines[-200:]

    def command(self, line: str) -> None:
        self.say("> " + line)
        words = line.split()
        if not words:
            return
        name, args = words[0], words[1:]
        try:
            if name == "kernel":
                self.kernel = args[0]
                self.restart()
            elif name == "task":
                assert args[0] in robot.TASKS
                self.task = args[0]
                self.restart()
            elif name == "food":
                self.food = {"red": robot.RED, "blue": robot.BLUE}[args[0]]
                self.restart()
            elif name == "seed":
                self.seed = int(args[0])
                self.restart()
            elif name == "restart":
                self.restart()
            elif name == "speed":
                self.speed = max(1, min(50, int(args[0])))
            elif name in ("pause", "resume"):
                self.paused = name == "pause"
            elif name == "view":
                assert args[0] in cell.VARIABLES
                self.blue = args[0]
            elif name == "rule":
                self.say(cell.describe(self.robot.rule))
            elif name == "quit":
                self.running = False
            else:
                self.say(HELP)
        except (IndexError, ValueError, KeyError, AssertionError, OSError) as error:
            self.say(f"? {error!r}\n{HELP}")

    def restart(self) -> None:
        path = Path(self.kernel) if Path(self.kernel).exists() else evolve.KERNELS / f"{self.kernel}.json"
        self.robot = robot.Robot(evolve.load(path), self.task, self.seed, self.food, world=evolve.load_world(path))
        self.trace = []
        self.say(f"kernel {path.stem}, task {self.task}, {'red' if self.food == robot.RED else 'blue'} is food, "
                 f"seed {self.seed}")

    # --- drawing ---------------------------------------------------------------------------------------------------

    def draw_cells(self, x0, y0, side) -> None:
        r, g = self.robot, self.robot.geometry
        rgb = np.zeros(g.alive.shape + (3,))
        v, blue = r.state[0], r.state[cell.VARIABLES.index(self.blue)]
        rgb[..., 0] = np.clip(-v, 0, 1)
        rgb[..., 1] = np.clip(v, 0, 1)
        rgb[..., 2] = (blue + 1) / 2 * 0.8
        rgb[~g.alive] = 0
        surface = pygame.surfarray.make_surface((rgb * 255).astype(np.uint8))
        self.screen.blit(pygame.transform.scale(surface, (side, side)), (x0, y0))
        scale = side / g.alive.shape[0]
        for s in range(len(g.angles)):
            for colour, rgb_ in ((0, (255, 80, 80)), (1, (80, 120, 255))):
                x, y = g.inputs[s, colour]
                pygame.draw.rect(self.screen, rgb_, (x0 + x * scale, y0 + y * scale, scale, scale), width=2)
            x, y = g.outputs[s]
            pygame.draw.rect(self.screen, (255, 255, 0), (x0 + x * scale, y0 + y * scale, scale, scale), width=2)
        for x, y in g.taste:
            pygame.draw.rect(self.screen, (255, 255, 255), (x0 + x * scale, y0 + y * scale, scale, scale), width=1)

    def draw_world(self, x0, y0, w, h) -> None:
        r, world = self.robot, self.robot.world
        side = min(w, h) - 70
        scale = side / world.size
        ox, oy = x0 + (w - side) / 2, y0 + 60
        at = lambda p: (ox + p[0] * scale, oy + p[1] * scale)
        pygame.draw.rect(self.screen, (80, 80, 80), (ox, oy, side, side), width=1)
        for bx, by, colour in r.blocks:
            rgb = (220, 40, 40) if colour == robot.RED else (40, 80, 255)
            pygame.draw.circle(self.screen, rgb, at((bx, by)), max(3, scale * 0.6))
            if colour == r.food:
                pygame.draw.circle(self.screen, (255, 255, 255), at((bx, by)), max(3, scale * 0.6) + 3, width=1)
        centre, radius = at(r.pos), world.radius * scale
        pygame.draw.circle(self.screen, (160, 160, 160), centre, radius, width=2)
        for s, angle in enumerate(r.geometry.angles):
            lit = r.ticks - r.fired_at[s] < 3
            c, sn = np.cos(angle), np.sin(angle)
            pygame.draw.circle(self.screen, (255, 255, 0) if lit else (70, 70, 0),
                               (centre[0] + c * radius * 0.75, centre[1] + sn * radius * 0.75), 5 if lit else 2)
        food, poison, firings = r.counts
        text = [f"tick {r.ticks}  task {self.task}  {'red' if r.food == robot.RED else 'blue'} is food (ringed)",
                f"ate {food} food, {poison} poison; {firings} thruster firings"]
        for i, line in enumerate(text):
            self.screen.blit(self.font.render(line, True, (255, 255, 255)), (x0 + 10, y0 + 8 + 20 * i))

    def draw_chart(self, x0, y0, w, h) -> None:
        pygame.draw.rect(self.screen, (0, 0, 32), (x0, y0, w, h))
        n = len(self.robot.geometry.angles)
        band = h / n
        for s in range(n):
            mid = y0 + band * (s + 0.5)
            level = mid - cell.FIRE_LEVEL * band * 0.45
            for x in range(int(x0), int(x0 + w), 8):
                self.screen.set_at((x, int(level)), (90, 90, 90))
            points = [(x0 + i * w / 400, mid - v[s] * band * 0.45) for i, v in enumerate(self.trace[-400:])]
            if len(points) > 1:
                pygame.draw.lines(self.screen, (255, 255, 0), False, points, 1)
            self.screen.blit(self.font.render(f"m{s}", True, (160, 160, 160)), (x0 + 4, mid - 8))

    def draw_console(self, x0, y0, w, h) -> None:
        rows, cols = int(h / 18) - 1, int(w / 10)
        wrapped = [line[i:i + cols] for line in self.lines + ["> " + self.input + "_"] for i in range(0, max(1, len(line)), cols)]
        for i, line in enumerate(wrapped[-rows:]):
            self.screen.blit(self.font.render(line, True, (255, 255, 255)), (x0 + 6, y0 + 4 + 18 * i))

    # --- running ---------------------------------------------------------------------------------------------------

    def run(self, commands) -> None:
        for c in commands:
            self.command(c)
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.command(self.input)
                        self.input = ""
                    elif event.key == pygame.K_BACKSPACE:
                        self.input = self.input[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.unicode and event.unicode.isprintable():
                        self.input += event.unicode
            text = CONTROL_IN.read_text() if CONTROL_IN.exists() else ""
            if text:
                CONTROL_IN.write_text("")
                for line in text.splitlines():
                    self.command(line.strip())
            if not self.paused:
                for _ in range(self.speed):
                    self.robot.step()
                    g = self.robot.geometry
                    self.trace.append(self.robot.state[0, g.outputs[:, 0], g.outputs[:, 1]].copy())
                self.trace = self.trace[-400:]
            half_w, half_h = self.width // 2, self.height // 2
            self.screen.fill((0, 0, 0))
            self.draw_cells(0, 0, min(half_w, half_h))
            self.draw_world(half_w, 0, self.width - half_w, half_h)
            self.draw_chart(0, half_h, half_w, self.height - half_h)
            self.draw_console(half_w, half_h, self.width - half_w, self.height - half_h)
            pygame.display.flip()
            clock.tick(TICKS_PER_S)
        self.log.close()
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", nargs="?", default="choose")
    parser.add_argument("-c", dest="commands", action="append", default=[])
    args = parser.parse_args()
    App(args.kernel).run(args.commands)
