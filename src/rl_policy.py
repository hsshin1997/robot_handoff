"""Tiny numpy policy + REINFORCE trainer for the one-step handoff env.

Policy: MLP (obs -> 64 -> 64) with two heads:
  - gaussian over the 5 continuous handoff parameters (tanh-squashed mean,
    state-independent learned log-std),
  - categorical over the grasp set.
Training: REINFORCE with an EMA baseline and Adam. Pure numpy — no torch,
keeping the project's dependency list intact. One-step episodes make this
plain policy-gradient perfectly adequate.
"""
from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, shapes, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            out.append(p + self.lr * mh / (np.sqrt(vh) + self.eps))  # ascent
        return out


class HandoffPolicy:
    def __init__(self, obs_dim: int, n_cont: int, n_grasps: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        h = 64
        def init(a, b):
            return rng.normal(0, np.sqrt(2.0 / a), (a, b))
        self.params = [
            init(obs_dim, h), np.zeros(h),        # W1 b1
            init(h, h), np.zeros(h),              # W2 b2
            init(h, n_cont) * 0.1, np.zeros(n_cont),      # mean head
            init(h, n_grasps) * 0.1, np.zeros(n_grasps),  # grasp logits head
            np.full(n_cont, -0.7),                # log-std (state-independent)
        ]
        self.n_cont, self.n_grasps = n_cont, n_grasps
        self.opt = Adam([p.shape for p in self.params])
        self.baseline = 0.0

    # -- forward --

    def _trunk(self, obs):
        W1, b1, W2, b2 = self.params[:4]
        h1 = np.tanh(obs @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        return h1, h2

    def forward(self, obs):
        _, h2 = self._trunk(obs)
        Wm, bm, Wg, bg, log_std = self.params[4:]
        mean = np.tanh(h2 @ Wm + bm)
        logits = h2 @ Wg + bg
        return mean, np.exp(log_std), logits

    def act(self, obs, rng: np.random.Generator, deterministic=False):
        mean, std, logits = self.forward(obs)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        if deterministic:
            return mean, int(np.argmax(probs))
        a = np.clip(mean + std * rng.standard_normal(self.n_cont), -1, 1)
        return a, int(rng.choice(self.n_grasps, p=probs))

    # -- REINFORCE update over a batch of (obs, a_cont, grasp, reward) --

    def update(self, batch) -> dict:
        obs = np.array([b[0] for b in batch])
        acts = np.array([b[1] for b in batch])
        gidx = np.array([b[2] for b in batch])
        rew = np.array([b[3] for b in batch])
        adv = rew - self.baseline
        self.baseline = 0.95 * self.baseline + 0.05 * rew.mean()

        W1, b1, W2, b2, Wm, bm, Wg, bg, log_std = self.params
        h1 = np.tanh(obs @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        mean = np.tanh(h2 @ Wm + bm)
        std = np.exp(log_std)
        logits = h2 @ Wg + bg
        pmax = logits.max(axis=1, keepdims=True)
        probs = np.exp(logits - pmax)
        probs /= probs.sum(axis=1, keepdims=True)

        n = len(batch)
        A = adv[:, None] / n
        # gaussian head: d logpi / d mean = (a - mean) / std^2
        z = (acts - mean) / std
        d_mean = (z / std) * A
        d_logstd = ((z ** 2 - 1.0) * A).sum(axis=0)
        d_pre_m = d_mean * (1 - mean ** 2)              # tanh backprop
        # categorical head: d logpi / d logits = onehot - probs
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), gidx] = 1.0
        d_logits = (onehot - probs) * A
        # shared trunk
        dh2 = d_pre_m @ Wm.T + d_logits @ Wg.T
        dpre2 = dh2 * (1 - h2 ** 2)
        dh1 = dpre2 @ W2.T
        dpre1 = dh1 * (1 - h1 ** 2)

        grads = [obs.T @ dpre1, dpre1.sum(0),
                 h1.T @ dpre2, dpre2.sum(0),
                 h2.T @ d_pre_m, d_pre_m.sum(0),
                 h2.T @ d_logits, d_logits.sum(0),
                 d_logstd]
        self.params = self.opt.step(self.params, grads)
        # keep exploration from collapsing too early
        self.params[8] = np.clip(self.params[8], -2.0, 0.5)
        return {"reward_mean": float(rew.mean()),
                "success_rate": float((rew >= 1.0).mean()),
                "baseline": float(self.baseline)}

    # -- persistence --

    def save(self, path: str) -> None:
        np.savez(path, *self.params, baseline=self.baseline)

    @classmethod
    def load(cls, path: str, obs_dim: int, n_cont: int, n_grasps: int):
        pol = cls(obs_dim, n_cont, n_grasps)
        data = np.load(path)
        pol.params = [data[f"arr_{i}"] for i in range(9)]
        pol.baseline = float(data["baseline"])
        return pol
