"""Hide & Seek AI — Game Viewer
Loads checkpoint.pth and plays 5-second hide & seek rounds.
Catch = seeker within 1.5 of visible hider → Seeker Wins!
Survive 5 seconds hidden → Hider Wins!
Controls: SPACE=pause  R=reset  S=toggle sensors  ESC=quit
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# ── Hardware ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(0)
    _gn = torch.cuda.get_device_name(0)
    print(f"\u2714 CUDA: {_gn}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("\u2714 Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print("\u26A0 CPU fallback")

try:
    import pygame
except ImportError:
    print("Install pygame:  pip install pygame")
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────
ROUND_FPS = 60
ROUND_SECONDS = 8
ROUND_STEPS = ROUND_FPS * ROUND_SECONDS        # 300 steps = 5 sec
RESULT_PAUSE_FRAMES = ROUND_FPS * 2             # 2 sec pause after result


@dataclass
class Config:
    arena_width: int = 30
    arena_height: int = 30
    num_hiders: int = 2
    num_seekers: int = 2
    num_boxes: int = 5
    max_episode_steps: int = ROUND_STEPS
    box_width: float = 1.5
    box_height: float = 1.5
    agent_radius: float = 0.4
    grab_range: float = 1.5
    num_lidar_rays: int = 9
    lidar_range: float = 10.0
    local_obs_radius: float = 6.0
    max_nearby_entities: int = 6
    catch_radius: float = 2.5
    checkpoint_path: str = "checkpoint.pth"


# ── PPO Network (inference only) ─────────────────────────────────────
class PPONetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),     nn.LayerNorm(256), nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, act_dim)
        self.value_head = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(obs)
        return self.policy_head(f), self.value_head(f).squeeze(-1)

    def get_action(self, obs: torch.Tensor,
                   deterministic: bool = False) -> torch.Tensor:
        logits, _ = self.forward(obs)
        if deterministic:
            return logits.argmax(dim=-1)
        return torch.distributions.Categorical(logits=logits).sample()


# ── Arena (single-env with walls) ────────────────────────────────────
class Arena:
    NUM_ACTIONS: int = 6
    MOVE_SPEED: float = 0.5

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.NH = cfg.num_hiders
        self.NS = cfg.num_seekers
        self.NA = cfg.num_hiders + cfg.num_seekers
        self.NB = cfg.num_boxes
        self.W = float(cfg.arena_width)
        self.H = float(cfg.arena_height)
        self.lidar_angles = np.linspace(
            -np.pi * 0.75, np.pi * 0.75, cfg.num_lidar_rays, dtype=np.float32)
        self.obs_dim = (cfg.num_lidar_rays + 2 + 2 + 1
                        + cfg.max_nearby_entities * 4 + 1 + 1)
        self.agent_pos = np.zeros((self.NA, 2), np.float32)
        self.agent_vel = np.zeros((self.NA, 2), np.float32)
        self.agent_heading = np.zeros(self.NA, np.float32)
        self.agent_grabbed = np.full(self.NA, -1, np.int32)
        self.agent_idle = np.zeros(self.NA, np.int32)
        self.box_pos = np.zeros((self.NB, 2), np.float32)
        self.box_size = np.tile(
            np.array([cfg.box_width, cfg.box_height], np.float32), (self.NB, 1))
        self.box_grabbed_by = np.full(self.NB, -1, np.int32)
        self.step_count = 0
        self.hider_seen = np.zeros(self.NH, dtype=bool)
        self.episode_hider_ever_seen = np.zeros(self.NH, dtype=bool)
        self.lidar_dists = np.zeros((self.NA, cfg.num_lidar_rays), np.float32)
        self.caught = False
        # ── Static walls (16 walls — lots of hiding spots) ──
        T = 0.6
        W, H = self.W, self.H
        wall_defs = [
            # ── Center cross (shorter for wider passages) ──
            (W*0.50, H*0.50, W*0.28, T),      # center H
            (W*0.50, H*0.50, T,      H*0.28), # center V
            # ── Corner alcoves (L-shapes, trimmed) ──
            (W*0.18, H*0.84, W*0.18, T),      # TL H
            (W*0.08, H*0.74, T,      H*0.16), # TL V
            (W*0.82, H*0.84, W*0.18, T),      # TR H
            (W*0.92, H*0.74, T,      H*0.16), # TR V
            (W*0.18, H*0.16, W*0.18, T),      # BL H
            (W*0.08, H*0.26, T,      H*0.16), # BL V
            (W*0.82, H*0.16, W*0.18, T),      # BR H
            (W*0.92, H*0.26, T,      H*0.16), # BR V
            # ── Mid-field barriers (narrower) ──
            (W*0.30, H*0.70, W*0.10, T),      # upper-left H
            (W*0.70, H*0.30, W*0.10, T),      # lower-right H
            # ── Extra hiding spots ──
            (W*0.35, H*0.88, T,      H*0.10), # top-center-left V
            (W*0.65, H*0.12, T,      H*0.10), # bottom-center-right V
            (W*0.13, H*0.50, W*0.08, T),      # mid-left H
            (W*0.87, H*0.50, W*0.08, T),      # mid-right H
            # ── Additional cover (moved further from center) ──
            (W*0.33, H*0.35, T,      H*0.10), # inner-left V
            (W*0.67, H*0.65, T,      H*0.10), # inner-right V
            (W*0.22, H*0.56, W*0.08, T),      # left-mid H
            (W*0.78, H*0.44, W*0.08, T),      # right-mid H
        ]
        self.NW = len(wall_defs)
        self.wall_pos = np.array([[d[0], d[1]] for d in wall_defs], np.float32)
        self.wall_size = np.array([[d[2], d[3]] for d in wall_defs], np.float32)
        self.wall_min = self.wall_pos - self.wall_size * 0.5
        self.wall_max = self.wall_pos + self.wall_size * 0.5
        self.reset()

    def reset(self) -> np.ndarray:
        m = 2.0
        self.agent_pos = np.random.uniform(
            [m, m], [self.W - m, self.H - m], (self.NA, 2)).astype(np.float32)
        self.agent_vel[:] = 0
        self.agent_heading = np.random.uniform(
            -np.pi, np.pi, self.NA).astype(np.float32)
        self.agent_grabbed[:] = -1
        self.agent_idle[:] = 0
        self.box_pos = np.random.uniform(
            [m, m], [self.W - m, self.H - m], (self.NB, 2)).astype(np.float32)
        self.box_grabbed_by[:] = -1
        self.step_count = 0
        self.hider_seen[:] = False
        self.episode_hider_ever_seen[:] = False
        self.caught = False
        # Push agents out of walls
        r = self.cfg.agent_radius
        for w in range(self.NW):
            half = self.wall_size[w] * 0.5 + r
            rel = self.agent_pos - self.wall_pos[w]
            ox = half[0] - np.abs(rel[:, 0])
            oy = half[1] - np.abs(rel[:, 1])
            coll = (ox > 0) & (oy > 0)
            push_x = ox < oy
            self.agent_pos[coll & push_x, 0] += np.sign(rel[coll & push_x, 0]) * ox[coll & push_x]
            self.agent_pos[coll & ~push_x, 1] += np.sign(rel[coll & ~push_x, 1]) * oy[coll & ~push_x]
        # Push boxes out of walls
        for w in range(self.NW):
            wh = self.wall_size[w] * 0.5
            for bi in range(self.NB):
                bh = self.box_size[bi] * 0.5
                th = wh + bh
                rel = self.box_pos[bi] - self.wall_pos[w]
                ox = th[0] - abs(rel[0])
                oy = th[1] - abs(rel[1])
                if ox > 0 and oy > 0:
                    if ox < oy:
                        self.box_pos[bi, 0] += np.sign(rel[0]) * ox
                    else:
                        self.box_pos[bi, 1] += np.sign(rel[1]) * oy
        return self._observe()

    def _observe(self) -> np.ndarray:
        obs = np.zeros((self.NA, self.obs_dim), np.float32)
        lidar = self._cast_lidar()
        self.lidar_dists = lidar * self.cfg.lidar_range
        team = np.zeros((self.NA, 1), np.float32)
        team[self.NH:, 0] = 1.0
        nearby = self._nearby()
        grabbed = (self.agent_grabbed >= 0).astype(np.float32)[:, np.newaxis]
        norm_p = self.agent_pos / np.array([self.W, self.H], np.float32)
        frac = np.full((self.NA, 1), self.step_count / self.cfg.max_episode_steps,
                       np.float32)
        i = 0
        NR = self.cfg.num_lidar_rays
        obs[:, i:i+NR] = lidar;      i += NR
        obs[:, i:i+2] = norm_p;      i += 2
        obs[:, i:i+2] = self.agent_vel; i += 2
        obs[:, i:i+1] = team;        i += 1
        K4 = self.cfg.max_nearby_entities * 4
        obs[:, i:i+K4] = nearby;     i += K4
        obs[:, i:i+1] = grabbed;     i += 1
        obs[:, i:i+1] = frac;        i += 1
        return obs

    def _cast_lidar(self) -> np.ndarray:
        NR = self.cfg.num_lidar_rays
        rng = self.cfg.lidar_range
        angles = self.agent_heading[:, np.newaxis] + self.lidar_angles[np.newaxis, :]
        dx, dy = np.cos(angles), np.sin(angles)
        dist = np.full_like(dx, rng)
        px, py = self.agent_pos[:, 0:1], self.agent_pos[:, 1:2]
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = np.minimum(dist, np.clip(np.where(dx > 1e-6, (self.W-px)/dx, rng), 0, rng))
            dist = np.minimum(dist, np.clip(np.where(dx < -1e-6, -px/dx, rng), 0, rng))
            dist = np.minimum(dist, np.clip(np.where(dy > 1e-6, (self.H-py)/dy, rng), 0, rng))
            dist = np.minimum(dist, np.clip(np.where(dy < -1e-6, -py/dy, rng), 0, rng))
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_dx = np.where(np.abs(dx) > 1e-8, 1.0/dx, 1e8)
            inv_dy = np.where(np.abs(dy) > 1e-8, 1.0/dy, 1e8)
        # Boxes
        bmin = self.box_pos - self.box_size * 0.5
        bmax = self.box_pos + self.box_size * 0.5
        for b in range(self.NB):
            t1x = (bmin[b, 0] - px) * inv_dx
            t2x = (bmax[b, 0] - px) * inv_dx
            t1y = (bmin[b, 1] - py) * inv_dy
            t2y = (bmax[b, 1] - py) * inv_dy
            tmin_xy = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
            tmax_xy = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
            hit = (tmax_xy >= np.maximum(tmin_xy, 0.0)) & (tmin_xy < dist)
            dist = np.where(hit, np.minimum(dist, np.maximum(tmin_xy, 0.0)), dist)
        # Walls
        for w in range(self.NW):
            t1x = (self.wall_min[w, 0] - px) * inv_dx
            t2x = (self.wall_max[w, 0] - px) * inv_dx
            t1y = (self.wall_min[w, 1] - py) * inv_dy
            t2y = (self.wall_max[w, 1] - py) * inv_dy
            tmin_xy = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
            tmax_xy = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
            hit = (tmax_xy >= np.maximum(tmin_xy, 0.0)) & (tmin_xy < dist)
            dist = np.where(hit, np.minimum(dist, np.maximum(tmin_xy, 0.0)), dist)
        return dist / rng

    def _nearby(self) -> np.ndarray:
        K = self.cfg.max_nearby_entities
        rad = self.cfg.local_obs_radius
        out = np.zeros((self.NA, K * 4), np.float32)
        rel = self.agent_pos[np.newaxis, :, :] - self.agent_pos[:, np.newaxis, :]
        d = np.linalg.norm(rel, axis=-1)
        np.fill_diagonal(d, rad + 1)
        te = np.zeros(self.NA, np.float32)
        te[self.NH:] = 1.0
        for k in range(min(K, self.NA - 1)):
            ci = np.argmin(d, axis=-1)
            cd = d[np.arange(self.NA), ci]
            ok = cd < rad
            rp = rel[np.arange(self.NA), ci]
            b = k * 4
            out[:, b] = np.where(ok, rp[:, 0] / rad, 0)
            out[:, b+1] = np.where(ok, rp[:, 1] / rad, 0)
            out[:, b+2] = np.where(ok, 1.0, 0)
            out[:, b+3] = np.where(ok, te[ci], 0)
            d[np.arange(self.NA), ci] = rad + 1
        return out

    def _check_los(self) -> np.ndarray:
        visible = np.zeros(self.NH, dtype=bool)
        hpos = self.agent_pos[:self.NH]
        spos = self.agent_pos[self.NH:]
        bmin = self.box_pos - self.box_size * 0.5
        bmax = self.box_pos + self.box_size * 0.5
        for s in range(self.NS):
            for h in range(self.NH):
                d = hpos[h] - spos[s]
                ln = np.linalg.norm(d)
                if ln < 1e-6:
                    visible[h] = True
                    continue
                dn = d / ln
                blocked = False
                with np.errstate(divide="ignore", invalid="ignore"):
                    inv = np.where(np.abs(dn) > 1e-8, 1.0 / dn, 1e8)
                for b in range(self.NB):
                    t1 = (bmin[b] - spos[s]) * inv
                    t2 = (bmax[b] - spos[s]) * inv
                    tmin = max(float(np.minimum(t1, t2).max()), 0.0)
                    tmax = min(float(np.maximum(t1, t2).min()), ln)
                    if tmax > tmin:
                        blocked = True
                        break
                if not blocked:
                    for w in range(self.NW):
                        t1 = (self.wall_min[w] - spos[s]) * inv
                        t2 = (self.wall_max[w] - spos[s]) * inv
                        tmin = max(float(np.minimum(t1, t2).max()), 0.0)
                        tmax = min(float(np.maximum(t1, t2).min()), ln)
                        if tmax > tmin:
                            blocked = True
                            break
                if not blocked:
                    visible[h] = True
        return visible

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, bool, Dict]:
        prev = self.agent_pos.copy()
        dx = np.zeros(self.NA, np.float32)
        dy = np.zeros(self.NA, np.float32)
        dx[actions == 3] = -self.MOVE_SPEED
        dx[actions == 4] = self.MOVE_SPEED
        dy[actions == 1] = self.MOVE_SPEED
        dy[actions == 2] = -self.MOVE_SPEED
        delta = np.stack([dx, dy], axis=-1)
        moving = np.linalg.norm(delta, axis=-1) > 0.01
        self.agent_heading = np.where(moving, np.arctan2(dy, dx), self.agent_heading)
        self.agent_pos += delta
        r = self.cfg.agent_radius
        self.agent_pos[:, 0] = np.clip(self.agent_pos[:, 0], r, self.W - r)
        self.agent_pos[:, 1] = np.clip(self.agent_pos[:, 1], r, self.H - r)
        # Box collision
        for b in range(self.NB):
            half = self.box_size[b] / 2 + r
            rel = self.agent_pos - self.box_pos[b]
            ox = half[0] - np.abs(rel[:, 0])
            oy = half[1] - np.abs(rel[:, 1])
            coll = (ox > 0) & (oy > 0)
            push_x = ox < oy
            self.agent_pos[coll & push_x, 0] += np.sign(rel[coll & push_x, 0]) * ox[coll & push_x]
            self.agent_pos[coll & ~push_x, 1] += np.sign(rel[coll & ~push_x, 1]) * oy[coll & ~push_x]
        # Wall collision
        for w in range(self.NW):
            hw = self.wall_size[w] * 0.5 + r
            rel = self.agent_pos - self.wall_pos[w]
            ox = hw[0] - np.abs(rel[:, 0])
            oy = hw[1] - np.abs(rel[:, 1])
            coll = (ox > 0) & (oy > 0)
            push_x = ox < oy
            self.agent_pos[coll & push_x, 0] += np.sign(rel[coll & push_x, 0]) * ox[coll & push_x]
            self.agent_pos[coll & ~push_x, 1] += np.sign(rel[coll & ~push_x, 1]) * oy[coll & ~push_x]
        self.agent_vel = self.agent_pos - prev
        # Grab / release
        grab_mask = actions == 5
        for b in range(self.NB):
            d2b = np.linalg.norm(self.agent_pos - self.box_pos[b], axis=-1)
            for a in range(self.NA):
                if grab_mask[a]:
                    if self.agent_grabbed[a] == b:
                        self.agent_grabbed[a] = -1
                        self.box_grabbed_by[b] = -1
                    elif (self.agent_grabbed[a] == -1
                          and self.box_grabbed_by[b] == -1
                          and d2b[a] < self.cfg.grab_range):
                        self.agent_grabbed[a] = b
                        self.box_grabbed_by[b] = a
        # Move grabbed boxes
        for b in range(self.NB):
            g = self.box_grabbed_by[b]
            if g >= 0:
                self.box_pos[b] += self.agent_vel[g]
                hs = self.box_size[b] / 2
                self.box_pos[b, 0] = np.clip(self.box_pos[b, 0], hs[0], self.W - hs[0])
                self.box_pos[b, 1] = np.clip(self.box_pos[b, 1], hs[1], self.H - hs[1])
                for w in range(self.NW):
                    wh = self.wall_size[w] * 0.5
                    th = wh + hs
                    rel = self.box_pos[b] - self.wall_pos[w]
                    ox = th[0] - abs(rel[0])
                    oy = th[1] - abs(rel[1])
                    if ox > 0 and oy > 0:
                        if ox < oy:
                            self.box_pos[b, 0] += np.sign(rel[0]) * ox
                        else:
                            self.box_pos[b, 1] += np.sign(rel[1]) * oy
        self.step_count += 1
        vis = self._check_los()
        self.hider_seen = vis
        self.episode_hider_ever_seen |= vis
        # Catch check
        self.caught = False
        for h_i in range(self.NH):
            if vis[h_i]:
                for s_i in range(self.NS):
                    d = np.linalg.norm(
                        self.agent_pos[self.NH + s_i] - self.agent_pos[h_i])
                    if d < self.cfg.catch_radius:
                        self.caught = True
        time_up = self.step_count >= self.cfg.max_episode_steps
        done = time_up or self.caught
        obs = self._observe()
        return obs, done, {"caught": self.caught, "time_up": time_up}


# ── Renderer ─────────────────────────────────────────────────────────
class Renderer:
    BG          = (13, 17, 23)
    ARENA_BG    = (22, 27, 34)
    WALL_C      = (63, 72, 86)
    WALL_EDGE   = (82, 92, 108)
    GRID_C      = (30, 36, 46)
    HIDER       = (56, 185, 80)
    HIDER_GLOW  = (46, 160, 67)
    SEEKER      = (248, 81, 73)
    SEEKER_GLOW = (218, 54, 51)
    BOX_C       = (110, 118, 129)
    BOX_GRAB    = (240, 136, 62)
    BOX_EDGE    = (72, 79, 88)
    TEXT_C      = (201, 209, 217)
    HUD_BG      = (13, 17, 23)
    SEEN_C      = (255, 200, 50)
    MUTED       = (110, 118, 129)
    BORDER_C    = (48, 54, 61)
    ACCENT      = (88, 166, 255)
    FLOOR_TILE1 = (22, 27, 34)
    FLOOR_TILE2 = (25, 31, 39)
    WIN_GREEN   = (56, 185, 80)
    WIN_RED     = (248, 81, 73)

    def __init__(self, arena: Arena, scale: float = 28.0) -> None:
        self.arena = arena
        self.scale = scale
        self.hud_w = 260
        self.win_w = int(arena.W * scale) + self.hud_w
        self.win_h = int(arena.H * scale)
        pygame.init()
        pygame.display.set_caption("Hide & Seek AI")
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo", 13)
        self.font_b = pygame.font.SysFont("Menlo", 16, bold=True)
        self.font_t = pygame.font.SysFont("Menlo", 22, bold=True)
        self.font_s = pygame.font.SysFont("Menlo", 11)
        self.font_big = pygame.font.SysFont("Menlo", 28, bold=True)
        self.font_timer = pygame.font.SysFont("Menlo", 40, bold=True)
        self.show_sensors = True
        self.paused = False
        # Game state
        self.round_num = 1
        self.hider_wins = 0
        self.seeker_wins = 0
        self.result_text = ""
        self.result_color = self.TEXT_C
        self.result_countdown = 0
        # Floor tiles
        aw = int(arena.W * scale)
        ah = int(arena.H * scale)
        self.floor_surf = pygame.Surface((aw, ah))
        self.floor_surf.fill(self.ARENA_BG)
        tile = int(2 * scale)
        for tx in range(0, aw, tile):
            for ty in range(0, ah, tile):
                if ((tx // tile) + (ty // tile)) % 2 == 0:
                    pygame.draw.rect(self.floor_surf, self.FLOOR_TILE2,
                                     (tx, ty, tile, tile))

    def w2s(self, x: float, y: float) -> Tuple[int, int]:
        return int(x * self.scale), int((self.arena.H - y) * self.scale)

    def draw(self) -> None:
        aw = int(self.arena.W * self.scale)
        self.screen.fill(self.BG)
        self.screen.blit(self.floor_surf, (0, 0))
        # Grid
        for i in range(0, int(self.arena.W) + 1, 5):
            x = int(i * self.scale)
            pygame.draw.line(self.screen, self.GRID_C, (x, 0), (x, self.win_h), 1)
        for i in range(0, int(self.arena.H) + 1, 5):
            y = int(i * self.scale)
            pygame.draw.line(self.screen, self.GRID_C, (0, y), (aw, y), 1)
        # Walls
        for w in range(self.arena.NW):
            wx, wy = self.arena.wall_pos[w]
            ws = self.arena.wall_size[w]
            sx, sy = self.w2s(wx - ws[0]/2, wy + ws[1]/2)
            pw, ph = int(ws[0] * self.scale), int(ws[1] * self.scale)
            rect = pygame.Rect(sx, sy, pw, ph)
            pygame.draw.rect(self.screen, (8, 10, 16),
                             pygame.Rect(sx + 3, sy + 3, pw, ph))
            pygame.draw.rect(self.screen, self.WALL_C, rect)
            pygame.draw.line(self.screen, self.WALL_EDGE,
                             (sx, sy), (sx + pw, sy), 2)
            pygame.draw.line(self.screen, self.WALL_EDGE,
                             (sx, sy), (sx, sy + ph), 1)
            pygame.draw.rect(self.screen, self.BORDER_C, rect, 1)
        pygame.draw.rect(self.screen, self.BORDER_C, (0, 0, aw, self.win_h), 3)
        # Boxes
        for b in range(self.arena.NB):
            bx, by = self.arena.box_pos[b]
            bw, bh = self.arena.box_size[b]
            sx, sy = self.w2s(bx - bw/2, by + bh/2)
            pw, ph = int(bw * self.scale), int(bh * self.scale)
            grabbed = self.arena.box_grabbed_by[b] >= 0
            color = self.BOX_GRAB if grabbed else self.BOX_C
            rect = pygame.Rect(sx, sy, pw, ph)
            pygame.draw.rect(self.screen, (8, 10, 16, 80),
                             pygame.Rect(sx + 2, sy + 2, pw, ph))
            pygame.draw.rect(self.screen, color, rect)
            darker = tuple(max(c - 40, 0) for c in color)
            pygame.draw.line(self.screen, darker, (sx+3, sy+3), (sx+pw-3, sy+ph-3), 1)
            pygame.draw.line(self.screen, darker, (sx+pw-3, sy+3), (sx+3, sy+ph-3), 1)
            edge_c = (255, 160, 70) if grabbed else self.BOX_EDGE
            pygame.draw.rect(self.screen, edge_c, rect, 2)
        # Sensor rays
        if self.show_sensors:
            ray_surf = pygame.Surface((aw, self.win_h), pygame.SRCALPHA)
            for a in range(self.arena.NA):
                ax, ay = self.arena.agent_pos[a]
                h = self.arena.agent_heading[a]
                is_h = a < self.arena.NH
                rc = (*self.HIDER[:3], 30) if is_h else (*self.SEEKER[:3], 30)
                dot_c = (*self.HIDER[:3], 80) if is_h else (*self.SEEKER[:3], 80)
                for ri in range(self.arena.cfg.num_lidar_rays):
                    ang = h + self.arena.lidar_angles[ri]
                    d = self.arena.lidar_dists[a, ri]
                    ex = ax + np.cos(ang) * d
                    ey = ay + np.sin(ang) * d
                    start = self.w2s(ax, ay)
                    end = self.w2s(ex, ey)
                    pygame.draw.line(ray_surf, rc, start, end, 1)
                    pygame.draw.circle(ray_surf, dot_c, end, 2)
            self.screen.blit(ray_surf, (0, 0))
        # Agents
        for a in range(self.arena.NA):
            ax, ay = self.arena.agent_pos[a]
            sx, sy = self.w2s(ax, ay)
            rad = max(int(self.arena.cfg.agent_radius * self.scale), 6)
            is_h = a < self.arena.NH
            color = self.HIDER if is_h else self.SEEKER
            glow = self.HIDER_GLOW if is_h else self.SEEKER_GLOW
            if is_h and self.arena.hider_seen[a]:
                pygame.draw.circle(self.screen, self.SEEN_C, (sx, sy), rad + 8, 2)
                pygame.draw.circle(self.screen, (*self.SEEN_C, 40), (sx, sy), rad + 12, 1)
            if self.arena.agent_grabbed[a] >= 0:
                pygame.draw.circle(self.screen, self.BOX_GRAB, (sx, sy), rad + 6, 2)
            pygame.draw.circle(self.screen, (8, 10, 16), (sx + 2, sy + 2), rad)
            pygame.draw.circle(self.screen, color, (sx, sy), rad)
            pygame.draw.circle(self.screen, glow, (sx - 2, sy - 2), max(rad - 4, 2))
            lighter = tuple(min(c + 50, 255) for c in color)
            pygame.draw.circle(self.screen, lighter, (sx, sy), rad, 2)
            heading = self.arena.agent_heading[a]
            arrow_len = rad * 1.8
            ex = int(math.cos(heading) * arrow_len)
            ey = int(-math.sin(heading) * arrow_len)
            pygame.draw.line(self.screen, (255, 255, 255), (sx, sy),
                             (sx + ex, sy + ey), 2)
            label = f"H{a}" if is_h else f"S{a - self.arena.NH}"
            txt = self.font_s.render(label, True, self.TEXT_C)
            self.screen.blit(txt, (sx - txt.get_width()//2, sy - rad - 15))
        # HUD
        self._draw_hud(aw)

    def _draw_hud(self, aw: int) -> None:
        hud_rect = pygame.Rect(aw, 0, self.hud_w, self.win_h)
        pygame.draw.rect(self.screen, self.HUD_BG, hud_rect)
        pygame.draw.line(self.screen, self.BORDER_C, (aw, 0), (aw, self.win_h), 2)
        x, y = aw + 15, 15
        hw = self.hud_w - 30
        # Title
        title = self.font_t.render("Hide & Seek AI", True, self.ACCENT)
        self.screen.blit(title, (x, y))
        y += 28
        sub = self.font_s.render("Multi-Agent RL", True, self.MUTED)
        self.screen.blit(sub, (x, y))
        y += 22
        pygame.draw.line(self.screen, self.BORDER_C, (x, y), (x + hw, y))
        y += 15
        # Timer
        remaining = max(0, ROUND_STEPS - self.arena.step_count) / ROUND_FPS
        timer_color = self.SEEN_C if remaining < 1.5 else self.TEXT_C
        timer_txt = self.font_timer.render(f"{remaining:.1f}s", True, timer_color)
        self.screen.blit(timer_txt, (x + hw//2 - timer_txt.get_width()//2, y))
        y += 50
        # Timer bar
        frac = self.arena.step_count / ROUND_STEPS
        bar_w = hw
        bar_h = 8
        pygame.draw.rect(self.screen, self.BORDER_C, (x, y, bar_w, bar_h))
        fill_w = int(bar_w * min(frac, 1.0))
        bar_color = self.SEEN_C if remaining < 1.5 else self.ACCENT
        if fill_w > 0:
            pygame.draw.rect(self.screen, bar_color, (x, y, fill_w, bar_h))
        y += 20
        pygame.draw.line(self.screen, self.BORDER_C, (x, y), (x + hw, y))
        y += 12
        # Round + Score
        sec = self.font_b.render(f"Round {self.round_num}", True, self.TEXT_C)
        self.screen.blit(sec, (x, y))
        y += 25
        # Score display
        h_txt = self.font_b.render(f"Hiders  {self.hider_wins}", True, self.HIDER)
        s_txt = self.font_b.render(f"Seekers {self.seeker_wins}", True, self.SEEKER)
        self.screen.blit(h_txt, (x, y))
        y += 22
        self.screen.blit(s_txt, (x, y))
        y += 30
        pygame.draw.line(self.screen, self.BORDER_C, (x, y), (x + hw, y))
        y += 12
        # Visibility
        vt = self.font_b.render("Visibility", True, self.TEXT_C)
        self.screen.blit(vt, (x, y))
        y += 22
        for h in range(self.arena.NH):
            seen = self.arena.hider_seen[h]
            dot_c = self.SEEN_C if seen else self.HIDER
            status = "SEEN" if seen else "HIDDEN"
            pygame.draw.circle(self.screen, dot_c, (x + 8, y + 7), 5)
            if seen:
                pygame.draw.circle(self.screen, self.SEEN_C, (x + 8, y + 7), 8, 1)
            t = self.font.render(f"Hider {h}: {status}", True, dot_c)
            self.screen.blit(t, (x + 20, y))
            y += 20
        y += 8
        pygame.draw.line(self.screen, self.BORDER_C, (x, y), (x + hw, y))
        y += 12
        # Status / Result
        if self.result_countdown > 0:
            rt = self.font_big.render(self.result_text, True, self.result_color)
            self.screen.blit(rt, (x + hw//2 - rt.get_width()//2, y))
            y += 40
        else:
            status = "PAUSED" if self.paused else "PLAYING..."
            st = self.font_b.render(status, True, self.MUTED)
            self.screen.blit(st, (x, y))
            y += 25
        # Arena info
        y += 5
        pygame.draw.line(self.screen, self.BORDER_C, (x, y), (x + hw, y))
        y += 12
        info = self.font_s.render(
            f"Arena {int(self.arena.W)}x{int(self.arena.H)}  "
            f"Walls {self.arena.NW}  Boxes {self.arena.NB}",
            True, self.MUTED)
        self.screen.blit(info, (x, y))
        y += 18
        fps_txt = self.font_s.render(
            f"FPS {self.clock.get_fps():.0f}  "
            f"Sensors {'ON' if self.show_sensors else 'OFF'}",
            True, self.MUTED)
        self.screen.blit(fps_txt, (x, y))
        # Controls at bottom
        y = self.win_h - 90
        pygame.draw.line(self.screen, self.BORDER_C, (x, y), (x + hw, y))
        y += 10
        ct = self.font_b.render("Controls", True, self.TEXT_C)
        self.screen.blit(ct, (x, y))
        y += 20
        for line in ["SPACE  Pause/Resume", "R      New round",
                      "S      Toggle sensors", "ESC    Quit"]:
            t = self.font_s.render(line, True, self.MUTED)
            self.screen.blit(t, (x, y))
            y += 16
        # Footer
        ft = self.font_s.render("Made with \u2764 by uzbtrust", True, self.MUTED)
        self.screen.blit(ft, (x + hw//2 - ft.get_width()//2, self.win_h - 18))


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    config = Config()
    checkpoint = None
    if os.path.exists(config.checkpoint_path):
        print(f"Loading {config.checkpoint_path} ...")
        checkpoint = torch.load(config.checkpoint_path, map_location=DEVICE,
                                weights_only=False)
        if "config" in checkpoint:
            skip = ("checkpoint_path", "max_episode_steps", "max_train_minutes",
                    "num_envs", "batch_size", "num_epochs")
            for k, v in checkpoint["config"].items():
                if hasattr(config, k) and k not in skip:
                    setattr(config, k, v)
            print("  Config restored from checkpoint")
    else:
        print("No checkpoint found \u2014 random policies")

    arena = Arena(config)
    hider_net = PPONetwork(arena.obs_dim, Arena.NUM_ACTIONS).to(DEVICE)
    seeker_net = PPONetwork(arena.obs_dim, Arena.NUM_ACTIONS).to(DEVICE)

    if checkpoint:
        hider_net.load_state_dict(checkpoint["hider_net"])
        seeker_net.load_state_dict(checkpoint["seeker_net"])
        ep = checkpoint.get("episode_count", "?")
        print(f"  Loaded networks (trained {ep} episodes)")
    hider_net.eval()
    seeker_net.eval()

    renderer = Renderer(arena)
    obs = arena.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    renderer.paused = not renderer.paused
                elif event.key == pygame.K_r:
                    obs = arena.reset()
                    renderer.round_num += 1
                    renderer.result_countdown = 0
                elif event.key == pygame.K_s:
                    renderer.show_sensors = not renderer.show_sensors

        # Result pause countdown
        if renderer.result_countdown > 0:
            renderer.result_countdown -= 1
            if renderer.result_countdown == 0:
                obs = arena.reset()
                renderer.round_num += 1
        elif not renderer.paused:
            with torch.no_grad():
                ot = torch.tensor(obs, device=DEVICE)
                h_act = hider_net.get_action(
                    ot[:config.num_hiders], deterministic=False)
                s_act = seeker_net.get_action(
                    ot[config.num_hiders:], deterministic=False)
                actions = torch.cat([h_act, s_act]).cpu().numpy()
            obs, done, info = arena.step(actions)
            if done:
                if info["caught"]:
                    renderer.seeker_wins += 1
                    renderer.result_text = "SEEKER WINS!"
                    renderer.result_color = renderer.WIN_RED
                else:
                    renderer.hider_wins += 1
                    renderer.result_text = "HIDER WINS!"
                    renderer.result_color = renderer.WIN_GREEN
                renderer.result_countdown = RESULT_PAUSE_FRAMES

        renderer.draw()
        pygame.display.flip()
        renderer.clock.tick(ROUND_FPS)

    pygame.quit()
    print(f"\nFinal Score: Hiders {renderer.hider_wins} - "
          f"{renderer.seeker_wins} Seekers")


if __name__ == "__main__":
    main()
