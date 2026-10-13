"""Hand-designed rules, written by term name, and tools to watch a bare sheet (no robot, no world).

    rule({"v": {"v^2": 2.0, "v": -1.0, ...}, "m": {...}})   ->  a coefficient vector for cell.step
"""
import numpy as np

import cell
import robot


def rule(terms: dict) -> np.ndarray:
    r = np.zeros(cell.N_PARAMS)
    for var, coefficients in terms.items():
        a = cell.VARIABLES.index(var)
        for term, c in coefficients.items():
            r[a * cell.N_TERMS + cell.TERMS.index(term)] = c
    return r


class Sheet:
    """A bare sheet of the robot's body, one life, stepped by hand."""

    def __init__(self, r, seed=0, dt=1.0, start=None):
        self.g = robot.Geometry()
        self.n = self.g.sheet.n
        self.rule = np.asarray(r, dtype=cell.DTYPE)
        self.a, self.b, self.x = robot.working(self.n, 1)
        rng = np.array([seed], dtype=np.uint64)
        robot.start(self.a, self.n, rng)
        if start:
            for var, value in start.items():
                self.a[cell.VARIABLES.index(var), :self.n] = value
        self.dt = dt

    def var(self, name):
        return self.a[cell.VARIABLES.index(name), :self.n]

    def kick(self, i, amount=1.0, var="v"):
        a = cell.VARIABLES.index(var)
        self.a[a, i] = min(1.0, self.a[a, i] + amount)

    def step(self, ticks=1):
        for _ in range(ticks):
            cell.step(self.a, self.b, self.x, self.g.sheet.neighbours, self.g.sheet.weight, self.rule, self.dt)
            self.a, self.b = self.b, self.a

    def grid(self, name):
        return self.g.grid(self.var(name))


def picture(sheets: dict, path: str, scale: int = 6):
    """Save side-by-side images of named grids (values -1..1: red negative, green positive)."""
    import pygame
    pygame.init()
    grids = list(sheets.items())
    size = grids[0][1].shape[0]
    surface = pygame.Surface((len(grids) * (size + 2) * scale, size * scale + 16))
    font = pygame.font.Font(None, 16)
    for k, (label, g) in enumerate(grids):
        rgb = np.zeros(g.shape + (3,))
        rgb[..., 0] = np.clip(-g, 0, 1)
        rgb[..., 1] = np.clip(g, 0, 1)
        img = pygame.surfarray.make_surface((rgb * 255).astype(np.uint8))
        surface.blit(pygame.transform.scale(img, (size * scale, size * scale)), (k * (size + 2) * scale, 16))
        surface.blit(font.render(label, True, (255, 255, 255)), (k * (size + 2) * scale, 0))
    pygame.image.save(surface, path)
