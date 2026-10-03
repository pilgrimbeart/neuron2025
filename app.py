"""Watch an evolved cell rule run the robot.

    python app.py [KERNEL] [-c COMMAND]...

Four quarters: the cells (top left: v green when positive, red when negative; m, or w, blue; special cells lettered,
and the cell under the mouse described top right), the world (top right), each
actuator's v over time (bottom left, the firing level dotted), and a console (bottom right). Commands are typed at the
console or written to control_in.txt; output also goes to control_out.log.
"""

from __future__ import annotations

import argparse
import dataclasses
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
  dt X                simulated time per tick (1 = as evolved), and restart
  restart             a new life with the same settings
  speed N             ticks per frame (1..50)
  pause / resume
  view m|w|t          which variable shows in blue on the cells
  rule                print the rule
  quit"""
TICKS_PER_S = 10            # at speed 1


DIRECTIONS = ("E", "SE", "SW", "W", "NW", "NE")     # the sensor groups' directions, as on screen (y downwards)


def food_name(food) -> str:
    return {robot.RED: "red is food", robot.BLUE: "blue is food", robot.BOTH: "both colours are food"}[int(food)]


def roles(g) -> dict:
    """What each special cell of the sheet is: {cell: (letter, lines describing it)}."""
    out = {}
    for s in range(len(g.angles)):
        side = f"sensor group {s}, facing {DIRECTIONS[s]}" if len(g.angles) == 6 else f"sensor group {s}"
        out[g.inputs[s, 0]] = ("R", ["red input", side, "kicked as it sees red"])
        out[g.inputs[s, 1]] = ("B", ["blue input", side, "kicked as it sees blue"])
        out[g.outputs[s]] = ("M", ["thruster", side, "firing pushes the robot away"])
        out[g.taste[s]] = ("t", ["taste cell", side, "t kicked by a bite on this side"])
    for i in g.centre:
        out[i] = ("c", ["centre taste cell", "t kicked by every bite"])
    return out


class App:
    def __init__(self, kernel: str):
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.width, self.height = self.screen.get_size()
        self.font = pygame.font.Font(pygame.font.match_font("couriernew"), 16)
        self.lines: list[str] = []
        self.input = ""
        self.kernel, self.task, self.food, self.seed, self.dt = kernel, "choose", robot.RED, 0, 1.0
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
            elif name == "dt":
                self.dt = float(args[0])
                assert self.dt > 0
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
        world = dataclasses.replace(evolve.load_world(path), dt=self.dt)
        food = robot.BOTH if self.task == "graze" else self.food
        self.robot = robot.Robot(evolve.load(path), self.task, self.seed, food, world=world)
        self.trace = []
        self.roles = roles(self.robot.geometry)
        self.say(f"kernel {path.stem}, task {self.task}, {food_name(food)}, "
                 f"seed {self.seed}, dt {self.dt}")

    # --- drawing ---------------------------------------------------------------------------------------------------

    def draw_cells(self, x0, y0, side) -> None:
        r, g = self.robot, self.robot.geometry
        v, blue = g.grid(r.state[0]), g.grid(r.state[cell.VARIABLES.index(self.blue)])
        alive = g.grid(np.ones(g.sheet.n)) > 0
        rgb = np.zeros(v.shape + (3,))
        rgb[..., 0] = np.clip(-v, 0, 1)
        rgb[..., 1] = np.clip(v, 0, 1)
        rgb[..., 2] = (blue + 1) / 2 * 0.8
        rgb[~alive] = 0
        surface = pygame.surfarray.make_surface((rgb * 255).astype(np.uint8))
        self.screen.blit(pygame.transform.scale(surface, (side, side)), (x0, y0))
        scale = side / g.size
        font = pygame.font.Font(pygame.font.match_font("couriernew", bold=True), max(10, int(scale * 0.6)))
        for i, (letter, _) in self.roles.items():            # a letter in each special cell (not only a colour)
            x, y = x0 + g.xy[i, 0] * scale, y0 + g.xy[i, 1] * scale
            pygame.draw.rect(self.screen, (255, 255, 255), (x, y, scale, scale), width=1)
            text = font.render(letter, True, (255, 255, 255), (0, 0, 0))
            self.screen.blit(text, (x + (scale - text.get_width()) / 2, y + (scale - text.get_height()) / 2))
        mx, my = pygame.mouse.get_pos()                     # the cell under the mouse, described top right
        gx, gy = int((mx - x0) // scale), int((my - y0) // scale)
        under = [i for i, (cx, cy) in enumerate(g.xy) if (cx, cy) == (gx, gy)]
        if under:
            i = under[0]
            values = "  ".join(f"{name} {r.state[a, i]:+.2f}" for a, name in enumerate(cell.VARIABLES))
            lines = [f"cell ({gx}, {gy})"] + self.roles.get(i, ("", ["ordinary cell"]))[1] + [values]
            for k, line in enumerate(lines):
                text = self.font.render(line, True, (255, 255, 255), (0, 0, 0))
                self.screen.blit(text, (x0 + side - text.get_width() - 4, y0 + 4 + 18 * k))

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
            if colour == r.food or r.food == robot.BOTH:
                pygame.draw.circle(self.screen, (255, 255, 255), at((bx, by)), max(3, scale * 0.6) + 3, width=1)
        centre, radius = at(r.pos), world.radius * scale
        pygame.draw.circle(self.screen, (160, 160, 160), centre, radius, width=2)
        for s, angle in enumerate(r.geometry.angles):
            lit = r.ticks - r.fired_at[s] < 3
            c, sn = np.cos(angle), np.sin(angle)
            pygame.draw.circle(self.screen, (255, 255, 0) if lit else (70, 70, 0),
                               (centre[0] + c * radius * 0.75, centre[1] + sn * radius * 0.75), 5 if lit else 2)
        food, poison, firings, spikes = r.counts
        rate = 100 * spikes / max(1, r.geometry.sheet.n * r.ticks * r.world.dt)
        text = [f"tick {r.ticks}  task {self.task}  {food_name(r.food)} (ringed)",
                f"ate {food} food, {poison} poison; {firings} thruster firings; {rate:.1f} spikes per cell per 100"]
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
            points = [(x0 + i * w / 400, mid - float(v[s]) * band * 0.45) for i, v in enumerate(self.trace[-400:])]
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
                    self.trace.append(self.robot.state[0, g.outputs].copy())
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
