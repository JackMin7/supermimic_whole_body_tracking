"""Equations 1--3 of SuperMimic; no simulator gradients are required.

The dynamics are goal-independent. Learned observation/reward/termination heads
are allowed by Eq. 1. This implementation is not the authors' released code.
"""
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F


def mlp(n, widths, out):
    layers = []
    for width in widths:
        layers += [nn.Linear(n, width), nn.ELU()]
        n = width
    layers.append(nn.Linear(n, out))
    return nn.Sequential(*layers)


@contextmanager
def frozen(*modules):
    params = [p for m in modules for p in m.parameters()]
    flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
        p.grad = None
    try:
        yield
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)


class Standardizer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer('mean', torch.zeros(n))
        self.register_buffer('scale', torch.ones(n))

    @torch.no_grad()
    def fit(self, x):
        self.mean.copy_(x.mean(0))
        self.scale.copy_(x.std(0, unbiased=False).clamp_min(.1))

    def forward(self, x):
        return (x - self.mean) / self.scale


class WorldModel(nn.Module):
    def __init__(self, ns, na, ng, nc, no, nf=0, widths=(2048, 2048, 2048), analytic_proprio=False):
        super().__init__()
        self.ns, self.na, self.no, self.nf = ns, na, no, nf
        self.analytic_proprio = analytic_proprio
        self.register_buffer("default_pose", torch.zeros(na))
        self.sn, self.gn, self.cn = Standardizer(ns), Standardizer(ng), Standardizer(nc)
        self.on, self.fn = Standardizer(no), Standardizer(nf)
        self.rn = Standardizer(1)
        self.dynamics = mlp(ns + na + nc, widths, ns)
        self.observation = mlp(ns + ng + nc, (512, 256), no + nf)
        self.reward_done = mlp(ns * 2 + na + ng + nc, (512, 256), 2)
        nn.init.zeros_(self.dynamics[-1].weight)
        nn.init.zeros_(self.dynamics[-1].bias)

    def step(self, s, a, c):
        ds = self.dynamics(torch.cat((self.sn(s), a, self.cn(c)), -1))
        nxt = s + .1 * self.sn.scale * ds
        # State layout: qpos (root xyz, wxyz, joints), qvel, executed action.
        quat = F.normalize(nxt[..., 3:7], dim=-1, eps=1e-6)
        return torch.cat((nxt[..., :3], quat, nxt[..., 7:-self.na], a), -1)

    def observe(self, s, g, c):
        if self.analytic_proprio and not self.nf:
            return proprio(s, self.default_pose, c[..., -self.na:]), s[..., :0]
        z = self.observation(torch.cat((self.sn(s), self.gn(g), self.cn(c)), -1))
        obs = z[..., :self.no] * self.on.scale + self.on.mean
        feat = z[..., self.no:] * self.fn.scale + self.fn.mean
        if self.analytic_proprio:
            obs = proprio(s, self.default_pose, c[..., -self.na:])
        return obs, feat

    def outcome(self, s, a, nxt, g, c):
        z = self.reward_done(torch.cat((self.sn(s), a, self.sn(nxt), self.gn(g), self.cn(c)), -1))
        return z[..., 0] * self.rn.scale[0] + self.rn.mean[0], z[..., 1]


def proprio(s, default_pose, encoder_bias):
    """G1: qpos=36, qvel=35, prev_action=29; MuJoCo free angular qvel is local."""
    if s.shape[-1] != 100:
        raise ValueError('Expected G1 29-DoF state layout')
    w, x, y, z = F.normalize(s[..., 3:7], dim=-1).unbind(-1)
    gravity = torch.stack((2*(w*y-x*z), -2*(y*z+w*x), 2*(x*x+y*y)-1), -1)
    return torch.cat((s[..., 7:36]-default_pose+encoder_bias,
                      s[..., 42:71], s[..., 39:42], gravity, s[..., -29:]), -1)


class Actor(nn.Module):
    def __init__(self, no, na, history=10, ng=0, goal_frames=1, size='small'):
        super().__init__()
        self.normalizer = Standardizer(no)
        self.goal_normalizer = Standardizer(ng)
        self.ng, self.goal_frames = ng, goal_frames
        widths = (2048,1024,512,256) if size == 'paper' else (512,256,128)
        self.encoder = mlp(ng*goal_frames, widths, 32) if ng else None
        self.net = mlp(no * history + (32 if ng else 0), widths, na)
        nn.init.normal_(self.net[-1].weight, std=.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, goal=None):
        x = self.normalizer(h).flatten(1)
        if self.encoder is not None:
            if goal is None: raise ValueError('Explicit goal sequence required')
            z = self.encoder(self.goal_normalizer(goal).flatten(1)).tanh()
            x = torch.cat((x,z), -1)
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, wm):
        super().__init__()
        self.net = mlp(wm.ns + wm.gn.mean.numel() + wm.cn.mean.numel(), (256, 128), 1)

    def forward(self, s, g, c, wm):
        return self.net(torch.cat((wm.sn(s), wm.gn(g), wm.cn(c)), -1)).squeeze(-1)


def lambda_returns(rewards, dones, values, gamma=.99, lam=.95):
    """Time-major; values has T+1 entries. Targets never retain a graph."""
    with torch.no_grad():
        out, nxt = [], values[-1]
        for t in reversed(range(len(rewards))):
            nxt = rewards[t] + gamma * (1 - dones[t]) * ((1-lam)*values[t+1] + lam*nxt)
            out.append(nxt)
        return torch.stack(out[::-1])


def world_loss(wm, batch):
    """Autoregressive k-step loss; masks stop at first reset, including terminal transition."""
    s = batch['s'][0]
    alive = torch.ones(s.shape[0], device=s.device)
    losses, metrics = [], []
    for t in range(len(batch['a'])):
        a, g, c = batch['a'][t], batch['g'][t], batch['c'][t]
        nxt = wm.step(s, a, c)
        obs, feat = wm.observe(nxt, batch["gnext"][t], c)
        reward, logit = wm.outcome(s, a, nxt, g, c)
        state_err = (wm.sn(nxt) - wm.sn(batch['nxt'][t])).square().mean(-1)
        obs_err = (wm.on(obs) - wm.on(batch['onext'][t])).square().mean(-1)
        loss = state_err + obs_err + ((reward-batch['r'][t])/wm.rn.scale[0]).square()
        loss += F.binary_cross_entropy_with_logits(logit, batch['term'][t], reduction='none')
        if wm.nf:
            loss += (wm.fn(feat)-wm.fn(batch['fn'][t])).square().mean(-1)
        weight = alive * batch.get('valid', torch.ones_like(batch['reset']))[t]
        losses.append((loss*weight).sum())
        metrics.append((state_err.detach()*weight).sum())
        if t == 0:
            denom = weight.sum()
        else:
            denom = denom + weight.sum()
        alive = weight * (1-batch['reset'][t])
        s = nxt
    return sum(losses)/denom.clamp_min(1), sum(metrics)/denom.clamp_min(1)


class ModelRolloutError(RuntimeError):
    """Reject a numerically invalid/out-of-support model rollout, not its diagnostics."""


def imagine(wm, actor, critic, s, h, goals, context, noise, gamma=.99, max_state_z=float('inf'), sensor_samples=None):
    """Caller freezes WM/critic weights, but never input Jacobians."""
    survival = torch.ones(s.shape[0], device=s.device)
    objective = torch.zeros_like(survival)
    states, rewards, dones, values = [], [], [], []
    horizon = len(goals)-actor.goal_frames
    for t in range(horizon):
        g = goals[t]
        states.append(s)
        values.append(critic(s, g, context, wm))
        a = actor(h, goals[t:t+actor.goal_frames].transpose(0,1))
        nxt = wm.step(s, a, context)
        if not torch.isfinite(nxt).all() or wm.sn(nxt).detach().abs().max() > max_state_z:
            raise ModelRolloutError('Imagined state became nonfinite or exceeded configured support guard')
        reward, logit = wm.outcome(s, a, nxt, g, context)
        done = logit.sigmoid()
        # Eq.2 weights r_(t+1) by c_t, not c_(t+1).
        objective = objective + gamma**t * survival * reward
        survival = survival * (1-done)
        rewards.append(reward)
        dones.append(done)
        obs, _ = wm.observe(nxt, goals[t+1], context)
        obs = obs + ((torch.rand_like(obs)*2-1)*noise if sensor_samples is None else sensor_samples[t])
        h = torch.cat((h[:, 1:], obs[:, None]), 1)
        s = nxt
    values.append(critic(s, goals[horizon], context, wm))
    objective = objective + gamma**horizon*survival*values[-1]
    values = torch.stack(values)
    targets = lambda_returns(torch.stack(rewards), torch.stack(dones), values)
    return -objective.mean(), (torch.stack(states).detach(), goals[:horizon].detach(), context.detach(), targets)


@torch.no_grad()
def rollout_support(wm, actor, s, h, goals, context, noise, max_state_z):
    """Optional adaptation: prefilter individual rollouts, never reject healthy peers.
    The finite-mask decisions have no policy gradient. Accepted rollouts are recomputed
    with autograd using the same fixed sensor noise, avoiding NaN backward through
    discarded branches. With infinity, only numerical nonfinite states are rejected.
    """
    valid = torch.ones(len(s), dtype=torch.bool, device=s.device)
    horizon = len(goals)-actor.goal_frames
    for t in range(horizon):
        a = actor(h, goals[t:t+actor.goal_frames].transpose(0,1))
        nxt = wm.step(s,a,context)
        good = torch.isfinite(nxt).all(-1) & (wm.sn(nxt).abs().amax(-1)<=max_state_z)
        valid &= good
        s = torch.where(valid[:,None], nxt, torch.zeros_like(nxt))
        obs,_ = wm.observe(s,goals[t+1],context)
        valid &= torch.isfinite(obs).all(-1)
        obs = torch.nan_to_num(obs)
        h = torch.cat((h[:,1:], (obs+noise[t])[:,None]),1)
    return valid
