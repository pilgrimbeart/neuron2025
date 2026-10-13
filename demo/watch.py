"""Watch a rule run, live, in greys: python watch.py RULE [--size N] [--seed N] [--show VAR ...]

Each shown number is drawn as a grey level (its smallest value black, its largest white; FAR as black), side by side.
Named cells (a number called "name", if the rule has one) are lettered R, B, A, V. Keys: r new seed, space pause,
up/down speed, Escape quit.
"""

from __future__ import annotations

import argparse
import importlib
import time

import numpy as np

import sheet

LETTERS = {1: "R", 2: "B", 3: "A", 4: "V"}


def watch(rule_name: str, n: int, seed: int, show: list) -> None:
    import pygame
    rule = importlib.import_module(f"rules.{rule_name}")
    pygame.init()
    scale = max(1, 440 // n)
    side = n * scale
    screen = pygame.display.set_mode((len(show) * (side + 10), side + 50))
    pygame.display.set_caption(f"{rule_name}: {', '.join(show)}")
    font = pygame.font.SysFont(None, 20)
    s, per_frame, paused, clock = sheet.Sheet(rule, n, seed), 2, False, pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    per_frame *= 2
                elif event.key == pygame.K_DOWN:
                    per_frame = max(1, per_frame // 2)
                elif event.key == pygame.K_r:
                    seed += 1
                    s = sheet.Sheet(rule, n, seed)
        if not paused:
            started = time.time()
            for _ in range(per_frame):
                s.step(1)
                if time.time() - started > 0.2:
                    break
        screen.fill((0, 0, 0))
        for i, var in enumerate(show):
            v = s.view(var)
            known = v < 1e8
            lo, hi = (v[known].min(), v[known].max()) if known.any() else (0.0, 1.0)
            grey = np.where(known, (v - lo) / max(hi - lo, 1e-9), 0.0)
            img = (255 * grey).astype(np.uint8).T.repeat(scale, 0).repeat(scale, 1)
            left = i * (side + 10)
            screen.blit(pygame.surfarray.make_surface(np.stack([img] * 3, axis=-1)), (left, 0))
            screen.blit(font.render(f"{var} ({lo:.0f} to {hi:.0f})", True, (255, 255, 255)), (left + 4, side + 4))
            if "name" in s.index:
                for r, c in np.argwhere(s.view("name") > 0.5):
                    letter = LETTERS.get(int(round(s.view("name")[r, c])), "?")
                    label = font.render(letter, True, (0, 0, 0), (255, 255, 255))
                    screen.blit(label, (left + min(max(0, c * scale - 4), side - 12), min(max(0, r * scale - 4), side - 14)))
        screen.blit(font.render(f"tick {s.t}   {per_frame} ticks per frame   r new seed   space   up/down   Esc",
                                True, (255, 255, 255)), (4, side + 26))
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rule")
    parser.add_argument("--size", type=int, default=49)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", nargs="+", default=["least", "a", "b"])
    args = parser.parse_args()
    watch(args.rule, args.size, args.seed, args.show)
