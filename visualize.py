"""Hide & Seek AI — Pygame Visualization
Loads checkpoint.pth and renders multi-agent inference in real-time.
Controls: SPACE=pause  R=reset  UP/DOWN=speed 1x-10x  S=toggle sensors  ESC=quit
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(0)
    _gn = torch.cuda.get_device_name(0)
    _gc = torch.cuda.device_count()
    print(f"\u2714 CUDA: {_gn}" + (f"  ({_gc} GPUs)" if _gc > 1 else ""))
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
@dataclass
class Config:
    arena_width: int = 20
    arena_height: int = 20
    num_hiders: int = 2
    num_seekers: int = 2
    num_boxes: int = 3
    max_episode_steps: int = 7200      # 2 minutes at 60 FPS (visual only)
    box_width: float = 1.5
    box_height: float = 1.5
    agent_radius: float = 0.4
    grab_range: float = 1.5
    num_lidar_rays: int = 9
    lidar_range: float = 8.0
    local_obs_radius: float = 5.0
    max_nearby_entities: int = 6
    checkpoint_path: str = "checkpoint.pth"
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
class Arena:
    NUM_ACTIONS: int = 6
    MOVE_SPEED: float = 0.3
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.NH = cfg.num_hiders
        self.NS = cfg.num_seekers
        self.NA = cfg.num_hiders + cfg.num_seekers
        self.NB = cfg.num_boxes
        self.W = float(cfg.arena_width)
        self.H = float(cfg.arena_height)
        self.lidar_angles = np.linspace(
            -np.pi * 0.75, np.pi * 0.75, cfg.num_lidar_rays, dtype=np.float32
        )
        self.obs_dim = (cfg.num_lidar_rays + 2 + 2 + 1
                        + cfg.max_nearby_entities * 4 + 1 + 1)
        self.agent_pos = np.zeros((self.NA, 2), np.float32)
        self.agent_vel = np.zeros((self.NA, 2), np.float32)
        self.agent_heading = np.zeros(self.NA, np.float32)
        self.agent_grabbed = np.full(self.NA, -1, np.int32)
        self.agent_idle = np.zeros(self.NA, np.int32)
        self.box_pos = np.zeros((self.NB, 2), np.float32)
        self.box_size = np.tile(
            np.array([cfg.box_width, cfg.box_height], np.float32), (self.NB, 1)
        )
        self.box_grabbed_by = np.full(self.NB, -1, np.int32)
        self.step_count = 0
        self.hider_seen = np.zeros(self.NH, dtype=bool)
        self.episode_hider_ever_seen = np.zeros(self.NH, dtype=bool)
        self.lidar_dists = np.zeros((self.NA, cfg.num_lidar_rays), np.float32)
        self.reset()
    def reset(self) -> np.ndarray:
        m = 2.0
        self.agent_pos = np.random.uniform(
            [m, m], [self.W - m, self.H - m], (self.NA, 2)
        ).astype(np.float32)
        self.agent_vel[:] = 0
        self.agent_heading = np.random.uniform(
            -np.pi, np.pi, self.NA
        ).astype(np.float32)
        self.agent_grabbed[:] = -1
        self.agent_idle[:] = 0
        self.box_pos = np.random.uniform(
            [m, m], [self.W - m, self.H - m], (self.NB, 2)
        ).astype(np.float32)
        self.box_grabbed_by[:] = -1
        self.step_count = 0
        self.hider_seen[:] = False
        self.episode_hider_ever_seen[:] = False
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
        bmin = self.box_pos - self.box_size * 0.5
        bmax = self.box_pos + self.box_size * 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_dx = np.where(np.abs(dx) > 1e-8, 1.0/dx, 1e8)
            inv_dy = np.where(np.abs(dy) > 1e-8, 1.0/dy, 1e8)
        for b in range(self.NB):
            t1x = (bmin[b, 0] - px) * inv_dx
            t2x = (bmax[b, 0] - px) * inv_dx
            t1y = (bmin[b, 1] - py) * inv_dy
            t2y = (bmax[b, 1] - py) * inv_dy
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
                for b in range(self.NB):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        inv = np.where(np.abs(dn) > 1e-8, 1.0 / dn, 1e8)
                    t1 = (bmin[b] - spos[s]) * inv
                    t2 = (bmax[b] - spos[s]) * inv
                    tmin = max(float(np.minimum(t1, t2).max()), 0.0)
                    tmax = min(float(np.maximum(t1, t2).min()), ln)
                    if tmax > tmin:
                        blocked = True
                        break
                if not blocked:
                    visible[h] = True
        return visible
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
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
        for b in range(self.NB):
            half = self.box_size[b] / 2 + r
            rel = self.agent_pos - self.box_pos[b]
            ox = half[0] - np.abs(rel[:, 0])
            oy = half[1] - np.abs(rel[:, 1])
            coll = (ox > 0) & (oy > 0)
            push_x = ox < oy
            self.agent_pos[coll & push_x, 0] += np.sign(rel[coll & push_x, 0]) * ox[coll & push_x]
            self.agent_pos[coll & ~push_x, 1] += np.sign(rel[coll & ~push_x, 1]) * oy[coll & ~push_x]
        self.agent_vel = self.agent_pos - prev
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
        for b in range(self.NB):
            g = self.box_grabbed_by[b]
            if g >= 0:
                self.box_pos[b] += self.agent_vel[g]
                hs = self.box_size[b] / 2
                self.box_pos[b, 0] = np.clip(self.box_pos[b, 0], hs[0], self.W - hs[0])
                self.box_pos[b, 1] = np.clip(self.box_pos[b, 1], hs[1], self.H - hs[1])
        self.step_count += 1
        vis = self._check_los()
        self.hider_seen = vis
        self.episode_hider_ever_seen |= vis
        done = self.step_count >= self.cfg.max_episode_steps
        rewards = np.zeros(self.NA, np.float32)
        any_seen = vis.any()
        for s_i in range(self.NS):
            rewards[self.NH + s_i] = 1.0 if any_seen else -1.0
        for h_i in range(self.NH):
            rewards[h_i] = -1.0 if vis[h_i] else 1.0
        obs = self._observe()
        return obs, rewards, done, {"visible": vis.copy()}
class Renderer:
    BG         = (13, 17, 23)
    ARENA_BG   = (15, 20, 28)
    WALL       = (48, 54, 61)
    GRID_C     = (21, 26, 35)
    HIDER      = (56, 185, 80)
    SEEKER     = (248, 81, 73)
    BOX_C      = (139, 148, 158)
    BOX_GRAB   = (240, 136, 62)
    TEXT_C     = (201, 209, 217)
    HUD_BG     = (22, 27, 34)
    SEEN_C     = (255, 200, 50)
    MUTED      = (110, 118, 129)
    def __init__(self, arena: Arena, scale: float = 40.0) -> None:
        self.arena = arena
        self.scale = scale
        self.hud_w = 280
        self.win_w = int(arena.W * scale) + self.hud_w
        self.win_h = int(arena.H * scale)
        pygame.init()
        pygame.display.set_caption("Hide & Seek AI \u2014 Multi-Agent RL")
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo", 14)
        self.font_b = pygame.font.SysFont("Menlo", 18, bold=True)
        self.font_t = pygame.font.SysFont("Menlo", 22, bold=True)
        self.show_sensors = True
        self.paused = False
        self.sim_speed = 1
        self.episode = 1
        self.total_steps = 0
        self.hider_wins = 0
        self.seeker_wins = 0
        self.total_eps = 0
    def w2s(self, x: float, y: float) -> Tuple[int, int]:
        return int(x * self.scale), int((self.arena.H - y) * self.scale)
    def draw(self) -> None:
        aw = int(self.arena.W * self.scale)
        self.screen.fill(self.BG)
        pygame.draw.rect(self.screen, self.ARENA_BG, (0, 0, aw, self.win_h))
        for i in range(int(self.arena.W) + 1):
            x = int(i * self.scale)
            pygame.draw.line(self.screen, self.GRID_C, (x, 0), (x, self.win_h))
        for i in range(int(self.arena.H) + 1):
            y = int(i * self.scale)
            pygame.draw.line(self.screen, self.GRID_C, (0, y), (aw, y))
        pygame.draw.rect(self.screen, self.WALL, (0, 0, aw, self.win_h), 3)
        for b in range(self.arena.NB):
            bx, by = self.arena.box_pos[b]
            bw, bh = self.arena.box_size[b]
            sx, sy = self.w2s(bx - bw/2, by + bh/2)
            pw, ph = int(bw * self.scale), int(bh * self.scale)
            color = self.BOX_GRAB if self.arena.box_grabbed_by[b] >= 0 else self.BOX_C
            rect = pygame.Rect(sx, sy, pw, ph)
            pygame.draw.rect(self.screen, color, rect)
            darker = tuple(max(c - 30, 0) for c in color)
            pygame.draw.rect(self.screen, darker, rect, 2)
        if self.show_sensors:
            ray_surf = pygame.Surface((aw, self.win_h), pygame.SRCALPHA)
            for a in range(self.arena.NA):
                ax, ay = self.arena.agent_pos[a]
                h = self.arena.agent_heading[a]
                rc = (*self.HIDER[:3], 40) if a < self.arena.NH else (*self.SEEKER[:3], 40)
                for r in range(self.arena.cfg.num_lidar_rays):
                    ang = h + self.arena.lidar_angles[r]
                    d = self.arena.lidar_dists[a, r]
                    ex, ey = ax + np.cos(ang) * d, ay + np.sin(ang) * d
                    start = self.w2s(ax, ay)
                    end = self.w2s(ex, ey)
                    pygame.draw.line(ray_surf, rc, start, end, 1)
                    pygame.draw.circle(ray_surf, (*rc[:3], 100), end, 2)
            self.screen.blit(ray_surf, (0, 0))
        for a in range(self.arena.NA):
            ax, ay = self.arena.agent_pos[a]
            sx, sy = self.w2s(ax, ay)
            rad = int(self.arena.cfg.agent_radius * self.scale)
            is_h = a < self.arena.NH
            color = self.HIDER if is_h else self.SEEKER
            if is_h and self.arena.hider_seen[a]:
                pygame.draw.circle(self.screen, self.SEEN_C, (sx, sy), rad + 6, 2)
            pygame.draw.circle(self.screen, color, (sx, sy), rad)
            lighter = tuple(min(c + 40, 255) for c in color)
            pygame.draw.circle(self.screen, lighter, (sx, sy), rad, 2)
            heading = self.arena.agent_heading[a]
            ex = int(np.cos(heading) * rad * 1.5)
            ey = int(-np.sin(heading) * rad * 1.5)
            pygame.draw.line(self.screen, (255, 255, 255), (sx, sy),
                             (sx + ex, sy + ey), 2)
            label = f"H{a}" if is_h else f"S{a - self.arena.NH}"
            txt = self.font.render(label, True, self.TEXT_C)
            self.screen.blit(txt, (sx - txt.get_width()//2, sy - rad - 16))
            if self.arena.agent_grabbed[a] >= 0:
                pygame.draw.circle(self.screen, self.BOX_GRAB, (sx, sy), rad + 4, 2)
        self._draw_hud(aw)
    def _draw_hud(self, aw: int) -> None:
        pygame.draw.rect(self.screen, self.HUD_BG, (aw, 0, self.hud_w, self.win_h))
        pygame.draw.line(self.screen, self.WALL, (aw, 0), (aw, self.win_h), 2)
        x, y = aw + 15, 15
        title = self.font_t.render("Hide & Seek AI", True, (88, 166, 255))
        self.screen.blit(title, (x, y)); y += 35
        sub = self.font.render("Multi-Agent RL", True, self.MUTED)
        self.screen.blit(sub, (x, y)); y += 30
        pygame.draw.line(self.screen, self.WALL, (x, y), (x + self.hud_w - 30, y))
        y += 15
        h_wr = self.hider_wins / max(self.total_eps, 1) * 100
        s_wr = self.seeker_wins / max(self.total_eps, 1) * 100
        stats: List[Tuple[str, str]] = [
            ("Episode", str(self.episode)),
            ("Step", f"{self.arena.step_count}/{self.arena.cfg.max_episode_steps}"),
            ("Total Steps", f"{self.total_steps:,}"),
            ("", ""),
            ("Hider Wins", str(self.hider_wins)),
            ("Seeker Wins", str(self.seeker_wins)),
            ("Hider Win%", f"{h_wr:.1f}%"),
            ("Seeker Win%", f"{s_wr:.1f}%"),
            ("", ""),
            ("Speed", f"{self.sim_speed}x"),
            ("FPS", f"{self.clock.get_fps():.0f}"),
            ("Sensors", "ON" if self.show_sensors else "OFF"),
            ("State", "PAUSED" if self.paused else "RUNNING"),
        ]
        for label, val in stats:
            if not label:
                y += 8
                continue
            ls = self.font.render(f"{label}:", True, self.MUTED)
            vs = self.font.render(val, True, self.TEXT_C)
            self.screen.blit(ls, (x, y))
            self.screen.blit(vs, (x + self.hud_w - 40 - vs.get_width(), y))
            y += 22
        y += 10
        pygame.draw.line(self.screen, self.WALL, (x, y), (x + self.hud_w - 30, y))
        y += 15
        vt = self.font_b.render("Visibility", True, self.TEXT_C)
        self.screen.blit(vt, (x, y)); y += 25
        for h in range(self.arena.NH):
            seen = self.arena.hider_seen[h]
            dot_c = self.SEEN_C if seen else self.HIDER
            status = "SEEN" if seen else "HIDDEN"
            pygame.draw.circle(self.screen, dot_c, (x + 8, y + 8), 5)
            t = self.font.render(f"Hider {h}: {status}", True, dot_c)
            self.screen.blit(t, (x + 20, y))
            y += 22
        y = self.win_h - 130
        pygame.draw.line(self.screen, self.WALL, (x, y), (x + self.hud_w - 30, y))
        y += 10
        ct = self.font_b.render("Controls", True, self.TEXT_C)
        self.screen.blit(ct, (x, y)); y += 25
        for line in ["SPACE  Pause/Resume", "R      Reset",
                      "UP/DN  Speed 1x-10x", "S      Toggle Sensors",
                      "ESC    Quit"]:
            t = self.font.render(line, True, self.MUTED)
            self.screen.blit(t, (x, y))
            y += 18
def main() -> None:
    config = Config()
    checkpoint = None
    if os.path.exists(config.checkpoint_path):
        print(f"Loading {config.checkpoint_path} ...")
        checkpoint = torch.load(config.checkpoint_path, map_location=DEVICE,
                                weights_only=False)
        if "config" in checkpoint:
            skip = ("checkpoint_path", "max_episode_steps",
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
                    renderer.episode += 1
                elif event.key == pygame.K_UP:
                    renderer.sim_speed = min(renderer.sim_speed + 1, 10)
                elif event.key == pygame.K_DOWN:
                    renderer.sim_speed = max(renderer.sim_speed - 1, 1)
                elif event.key == pygame.K_s:
                    renderer.show_sensors = not renderer.show_sensors
        if not renderer.paused:
            for _ in range(renderer.sim_speed):
                with torch.no_grad():
                    ot = torch.tensor(obs, device=DEVICE)
                    h_act = hider_net.get_action(
                        ot[:config.num_hiders], deterministic=True)
                    s_act = seeker_net.get_action(
                        ot[config.num_hiders:], deterministic=True)
                    actions = torch.cat([h_act, s_act]).cpu().numpy()
                obs, _, done, _ = arena.step(actions)
                renderer.total_steps += 1
                if done:
                    renderer.total_eps += 1
                    if not arena.episode_hider_ever_seen.any():
                        renderer.hider_wins += 1
                    else:
                        renderer.seeker_wins += 1
                    renderer.episode += 1
                    obs = arena.reset()
        renderer.draw()
        pygame.display.flip()
        renderer.clock.tick(60)
    pygame.quit()
    print("Visualization ended.")
if __name__ == "__main__":
    main()
