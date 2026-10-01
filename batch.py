"""The robot and its world (robot.py) for many lives at once, in PyTorch: on a GPU when there is one, else the CPU.

The same body, world and cell rule as robot.py and cell.py, laid out as (genomes x lives x cells) and advanced tick by
tick together:
  - the cell rule is, for each variable, constant + linear + a quadratic form in the 36 inputs, so for every cell of
    every life of a genome it is one batched matrix product (what GPUs are fastest at)
  - each genome's sheet is grown once (no inputs reach it while it grows) and copied into all its lives
  - sensing, thrusters, eating, taste and respawning blocks are vectorised over lives

Random numbers come from PyTorch, not Numba, so lives differ from robot.py's tick by tick; growth (no randomness) is
the same, and outcomes agree statistically.

    lives(rules, bodies, task, worlds, dt, duration)   -> per-half counts and positions for every genome and life
"""

from __future__ import annotations

import math

import numpy as np
import torch

import cell
import robot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

_pairs = [(p, q) for p in range(cell.N_INPUTS) for q in range(p, cell.N_INPUTS)]
PAIR_P = torch.tensor([p for p, _ in _pairs])
PAIR_Q = torch.tensor([q for _, q in _pairs])


class Rules:
    """A batch of G rules as tensors: c0 (G, K), linear (G, N_INPUTS, K), quadratic (G, N_INPUTS, K * N_INPUTS)
    with quadratic[g, p, a * N_INPUTS + q] the coefficient of x_p x_q (p <= q) in variable a's change."""

    def __init__(self, rules: np.ndarray, device=DEVICE, dtype=DTYPE):
        r = torch.as_tensor(np.asarray(rules), dtype=dtype, device=device).view(-1, cell.K, cell.N_TERMS)
        g, k, n = r.shape[0], cell.K, cell.N_INPUTS
        self.g = g
        self.c0 = r[:, :, 0]
        self.linear = r[:, :, 1:1 + n].permute(0, 2, 1).contiguous()
        q = torch.zeros(g, k, n, n, dtype=dtype, device=device)
        q[:, :, PAIR_P, PAIR_Q] = r[:, :, 1 + n:]
        self.quadratic = q.permute(0, 2, 1, 3).reshape(g, n, k * n).contiguous()


class Body:
    """robot.Geometry as tensors."""

    def __init__(self, geometry: robot.Geometry, device=DEVICE, dtype=DTYPE):
        g = geometry
        self.n = g.sheet.n
        t = lambda a: torch.as_tensor(np.asarray(a), device=device)
        self.neighbours = t(g.sheet.neighbours).long()
        self.order = t(g.order).long()
        self.inputs = t(g.inputs).long()                                  # (S, colour)
        self.outputs = t(g.outputs).long()                                # (S,)
        self.taste = t(g.taste).long()                                    # (S,)
        self.centre = t(g.centre).long()
        self.facing = torch.stack([torch.cos(t(g.angles)), torch.sin(t(g.angles))], dim=1).to(dtype)   # (S, 2)
        self.rates = t(cell.RATES).to(dtype)


def step(state, rules: Rules, body: Body, dt: float, alive=None):
    """One tick for every cell of every life: state (G, L, K, n + 1) -> the next state. Cells not alive stay 0."""
    g, l, k, _ = state.shape
    n, ni = body.n, cell.N_INPUTS
    own = state[..., :n]                                                  # (G, L, K, n)
    near = state[..., body.neighbours]                                    # (G, L, K, n, 8)
    x = torch.cat([own.permute(0, 1, 3, 2), near.permute(0, 1, 3, 4, 2).reshape(g, l, n, 8 * k)], dim=-1)
    x = x.reshape(g, l * n, ni)                                           # the inputs, in cell.INPUTS order
    change = rules.c0[:, None, :] + torch.bmm(x, rules.linear)            # (G, L n, K)
    change = change + (torch.bmm(x, rules.quadratic).view(g, l * n, k, ni) * x[:, :, None, :]).sum(-1)
    change = change.view(g, l, n, k).permute(0, 1, 3, 2)                  # (G, L, K, n)
    new = torch.clamp(own + body.rates[:, None] * dt * change, -1.0, 1.0)
    if alive is not None:
        new = new * alive
    return torch.cat([new, torch.zeros_like(state[..., n:])], dim=-1)


def grow(rules: Rules, body: Body, world: robot.World):
    """Each genome's grown sheet, (G, K, n + 1): as robot.develop, for all genomes at once."""
    state = torch.zeros(rules.g, 1, cell.K, body.n + 1, dtype=rules.c0.dtype, device=rules.c0.device)
    alive = torch.zeros(body.n, dtype=state.dtype, device=state.device)
    for k in range(body.n):
        alive[body.order[k]] = 1.0
        for _ in range(world.ticks(world.grow)):
            state = step(state, rules, body, world.dt, alive)
    return state[:, 0]


def lives(rules: np.ndarray, bodies: np.ndarray, task: str, worlds: list[tuple[int, int]], dt: float, duration: float,
          seed: int = 0, device=DEVICE, dtype=DTYPE) -> dict:
    """Every genome lives every world: rules (G, N_PARAMS); bodies (G, 2): each genome's taste strengths (contact,
    centre); worlds: L (seed, food) pairs, set up as robot.setup does. Returns numpy arrays: counts (G, L, 2, 4)
    (food, poison, thruster firings, cell spikes in each half of the life), start and end (G, L, 2) positions, and
    first and last (G, L, blocks, 3) blocks."""
    world = robot.World(dt=dt)
    geometry = robot.Geometry()
    body = Body(geometry, device, dtype)
    batch = Rules(rules, device, dtype)
    g, l, n, k = batch.g, len(worlds), body.n, cell.K
    b = g * l
    t = lambda a, d=dtype: torch.as_tensor(np.asarray(a), dtype=d, device=device)

    state = grow(batch, body, world)[:, None].repeat(1, l, 1, 1)          # (G, L, K, n + 1)
    setups = [robot.setup(task, s, food, world) for s, food in worlds]
    pos = t(np.stack([p for p, _ in setups]))[None].repeat(g, 1, 1).view(b, 2)
    blocks = t(np.stack([bl for _, bl in setups]))[None].repeat(g, 1, 1, 1).view(b, -1, 3)
    start, first = pos.clone(), blocks.clone()
    food = t([f for _, f in worlds])[None].repeat(g, 1).view(b)
    contact = t(np.asarray(bodies)[:, 0])[:, None].repeat(1, l).view(b)
    middle = t(np.asarray(bodies)[:, 1])[:, None].repeat(1, l).view(b)
    counts = torch.zeros(b, 2, 4, dtype=torch.long, device=device)
    high = torch.zeros(b, len(body.outputs), dtype=torch.bool, device=device)
    rng = torch.Generator(device=device).manual_seed(seed)
    ticks = world.ticks(duration)
    size, radius, reach = world.size, world.radius, world.reach
    colours = torch.stack([blocks[..., 2] == c for c in (robot.RED, robot.BLUE)], dim=-1).to(dtype)  # (B, blocks, 2)
    inputs = body.inputs.reshape(-1)                                      # (S * 2,), sensor-major then colour

    for tick in range(ticks):
        half = 0 if tick < ticks // 2 else 1
        flat = state.view(b, k, n + 1)
        if blocks.shape[1]:                                               # senses: Poisson pulses into v
            offset = blocks[..., :2] - pos[:, None, :]
            distance = offset.norm(dim=-1) + 1e-9
            square = torch.clamp((offset / distance[..., None]) @ body.facing.T, min=0.0)    # (B, blocks, S)
            seen = torch.einsum("bjs,bjc->bsc", square / (1 + (distance / reach) ** 2)[..., None], colours)
            chance = -torch.expm1(-world.rate * torch.clamp(seen, max=1.0) * dt)
            pulse = torch.rand(chance.shape, generator=rng, device=device, dtype=dtype) < chance
            flat[:, 0, inputs] = torch.clamp(flat[:, 0, inputs] + world.kick * pulse.view(b, -1), max=1.0)
        before = flat[:, 0, :n] > cell.FIRE_LEVEL
        state = step(state, batch, body, dt)
        flat = state.view(b, k, n + 1)
        counts[:, half, 3] += ((flat[:, 0, :n] > cell.FIRE_LEVEL) & ~before).sum(1)       # spikes
        now = flat[:, 0, body.outputs] > cell.FIRE_LEVEL                  # thrusters
        fired = now & ~high
        high = now
        pos = torch.clamp(pos - world.push * fired.to(dtype) @ body.facing, radius, size - radius)
        counts[:, half, 2] += fired.sum(1)
        if blocks.shape[1]:                                               # eating
            offset = blocks[..., :2] - pos[:, None, :]
            eaten = (offset ** 2).sum(-1) < (radius + 0.5) ** 2           # (B, blocks)
            if eaten.any():
                good = blocks[..., 2] == food[:, None]
                counts[:, half, 0] += (eaten & good).sum(1)
                counts[:, half, 1] += (eaten & ~good).sum(1)
                sign = torch.where(good, 1.0, -1.0).to(dtype) * eaten.to(dtype)
                facing = (offset @ body.facing.T).argmax(-1)              # (B, blocks): the sensor group facing it
                taste = flat[:, robot.TASTE]
                taste.scatter_add_(1, body.taste[facing], sign * contact[:, None])
                taste[:, body.centre] += (sign.sum(1) * middle)[:, None]
                taste.clamp_(-1.0, 1.0)
                while eaten.any():                                        # respawn, away from the robot
                    xy = radius + torch.rand(blocks.shape[:2] + (2,), generator=rng, device=device, dtype=dtype) * (
                        size - 2 * radius)
                    ok = ((xy - pos[:, None, :]) ** 2).sum(-1) > (4 * radius) ** 2
                    place = eaten & ok
                    blocks[..., :2] = torch.where(place[..., None], xy, blocks[..., :2])
                    eaten = eaten & ~ok

    shape = lambda a: a.view(g, l, *a.shape[1:]).cpu().numpy()
    return {"counts": shape(counts), "start": shape(start), "end": shape(pos), "first": shape(first),
            "last": shape(blocks)}
