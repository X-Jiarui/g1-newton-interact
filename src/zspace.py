"""zspace: a latent action space for the grasp policy -- encoder, decoder and a state-conditioned prior.

Ported from Omnigrasp's PULSE-X distillation (ZhengyiLuo/Omnigrasp @ f7740cd):

  phc/learning/amp_network_z_builder.py   z_type="vae", use_vae_prior=True, use_vae_clamped_prior=True
  phc/learning/amp_agent.py:_optimize_kin  the supervised "kin" loss (only_kin_loss=True -> no PPO)
  phc/env/tasks/humanoid_z.py:120          downstream use:  z = prior_mu(s) + a_policy ; a = D(s, z)

The three networks, with PULSE's shapes as the defaults:

  encoder  q(z | s, g)  trunk MLP(task_units=[1536,1024,512], SiLU) -> Linear(5E) -> mu(5E->E), logvar(5E->E)
  prior    p(z | s)     trunk MLP(task_units,                  SiLU) -> mu(->E), logvar(->E)
  decoder  a = D(s, z)  MLP(units=[3096,2048,1024], SiLU) -> action_dim

  logvar of both q and p is clamped to [-5, vae_var_clamp_max=2].
  s = proprioception ONLY.  The decoder and the prior never see the goal, so everything the
  action needs to know about the task has to travel through z.  g = goal / task observations,
  seen by the encoder alone; the encoder exists only for distillation and is dropped downstream.

Loss (amp_agent.py:773-826), per minibatch:

  kin_action_loss = mean_b ||a_pred - a_teacher||_2                    (RMSE, not MSE)
  KLD             = mean_b KL( q(z|s,g) || p(z|s) )                     (kl_multi)
  ar1             = mean ||mu_t - phi * mu_{t-1}||_2, phi=0.99, over consecutive samples of one env
  regu            = 0.001*(mu_p^2 + mu_q^2).mean() + 0.001*(logvar_p^2 + logvar_q^2).mean()  [optional]
  loss = kin_action_loss + kld_w * KLD + ar1_w * ar1 + 0.005 * regu
  kld_w anneals linearly 0.01 -> 0.001 (kld_anneal=True).

Reparameterisation noise sampled when the student ACTED is stored and reused in the update
(form_embedding: "bypass reparametrization and use the noise sampled during training"), so the
gradient is taken through the exact z that produced the executed action.

Input normalisation is a running mean/std (rl_games RunningMeanStd, normalize_input=True),
updated from each collected batch and frozen inside the update epochs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------------------------
class RunningMeanStd(nn.Module):
  """Welford running statistics, rl_games-style: normalised inputs are clamped to +-5."""

  def __init__(self, dim: int, epsilon: float = 1e-4, clamp: float = 5.0) -> None:
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(float(epsilon)))
    self.clamp = float(clamp)
    self.frozen = False

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    if self.frozen:
      return
    x = x.reshape(-1, x.shape[-1]).to(torch.float32)
    batch_mean = x.mean(dim=0)
    batch_var = x.var(dim=0, unbiased=False)
    batch_count = float(x.shape[0])
    delta = batch_mean - self.mean
    tot = self.count + batch_count
    new_mean = self.mean + delta * batch_count / tot
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot
    self.mean.copy_(new_mean)
    self.var.copy_(m2 / tot)
    self.count.copy_(tot)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = (x - self.mean) / torch.sqrt(self.var + 1e-5)
    return y.clamp(-self.clamp, self.clamp)


def _activation(name: str) -> nn.Module:
  name = name.lower()
  if name == "silu":
    return nn.SiLU()
  if name == "elu":
    return nn.ELU()
  if name == "relu":
    return nn.ReLU()
  if name == "tanh":
    return nn.Tanh()
  raise ValueError(f"unknown activation {name!r}")


def mlp(in_dim: int, units: tuple[int, ...], activation: str) -> nn.Sequential:
  layers: list[nn.Module] = []
  d = in_dim
  for u in units:
    layers.append(nn.Linear(d, int(u)))
    layers.append(_activation(activation))
    d = int(u)
  return nn.Sequential(*layers)


# ---------------------------------------------------------------------------------------------
# the VAE
# ---------------------------------------------------------------------------------------------
@dataclass
class ZSpaceConfig:
  self_dim: int
  task_dim: int
  action_dim: int
  latent_dim: int = 48
  enc_units: tuple[int, ...] = (1536, 1024, 512)  # PULSE task_mlp
  dec_units: tuple[int, ...] = (3096, 2048, 1024)  # PULSE mlp
  activation: str = "silu"  # z_activation
  logvar_min: float = -5.0
  logvar_max: float = 2.0  # vae_var_clamp_max
  # names of the observation groups that make up s and g, kept so a downstream user can rebuild
  # the exact inputs from a TensorDict without reading the training script
  self_groups: tuple[str, ...] = field(default_factory=tuple)
  task_groups: tuple[str, ...] = field(default_factory=tuple)


class ZSpaceVAE(nn.Module):
  def __init__(self, cfg: ZSpaceConfig) -> None:
    super().__init__()
    self.cfg = cfg
    E = int(cfg.latent_dim)
    self.self_norm = RunningMeanStd(cfg.self_dim)
    self.task_norm = RunningMeanStd(cfg.task_dim)

    # encoder: [s, g] -> 5E -> (mu, logvar)      (z_mlp + z_mu / z_logvar)
    self.z_mlp = mlp(cfg.self_dim + cfg.task_dim, cfg.enc_units, cfg.activation)
    self.z_out = nn.Linear(cfg.enc_units[-1], 5 * E)
    self.z_mu = nn.Linear(5 * E, E)
    self.z_logvar = nn.Linear(5 * E, E)

    # prior: s -> (mu, logvar)                    (z_prior + z_prior_mu / z_prior_logvar)
    self.z_prior = mlp(cfg.self_dim, cfg.enc_units, cfg.activation)
    self.z_prior_mu = nn.Linear(cfg.enc_units[-1], E)
    self.z_prior_logvar = nn.Linear(cfg.enc_units[-1], E)

    # decoder: [s, z] -> action                    (the actor MLP of the z network)
    self.decoder = mlp(cfg.self_dim + E, cfg.dec_units, cfg.activation)
    self.dec_out = nn.Linear(cfg.dec_units[-1], cfg.action_dim)

  # -- pieces -----------------------------------------------------------------------------------
  def _clamp(self, logvar: torch.Tensor) -> torch.Tensor:
    return logvar.clamp(self.cfg.logvar_min, self.cfg.logvar_max)

  def encode(self, s_n: torch.Tensor, g_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = self.z_out(self.z_mlp(torch.cat([s_n, g_n], dim=-1)))
    return self.z_mu(h), self._clamp(self.z_logvar(h))

  def prior(self, s_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = self.z_prior(s_n)
    return self.z_prior_mu(h), self._clamp(self.z_prior_logvar(h))

  def decode(self, s_n: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return self.dec_out(self.decoder(torch.cat([s_n, z], dim=-1)))

  @staticmethod
  def reparameterize(
    mu: torch.Tensor, logvar: torch.Tensor, noise: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
      noise = torch.randn_like(mu)
    return mu + torch.exp(0.5 * logvar) * noise, noise

  # -- whole forward passes ----------------------------------------------------------------------
  def normalize(self, s: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return self.self_norm(s), self.task_norm(g)

  def forward(
    self,
    s: torch.Tensor,
    g: torch.Tensor,
    noise: torch.Tensor | None = None,
    deterministic: bool = False,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Student action from raw (un-normalised) s, g.  deterministic -> z = mu_q (flags.test)."""
    s_n, g_n = self.normalize(s, g)
    mu, logvar = self.encode(s_n, g_n)
    if deterministic:
      z, noise = mu, torch.zeros_like(mu)
    else:
      z, noise = self.reparameterize(mu, logvar, noise)
    a = self.decode(s_n, z)
    p_mu, p_logvar = self.prior(s_n)
    return a, {
      "mu": mu, "logvar": logvar, "z": z, "noise": noise,
      "prior_mu": p_mu, "prior_logvar": p_logvar,
    }

  @torch.no_grad()
  def act_from_latent(self, s: torch.Tensor, a_latent: torch.Tensor) -> torch.Tensor:
    """Downstream interface (humanoid_z.py:120): z = prior_mu(s) + a_policy ; a = D(s, z).

    This is the action channel a latent-space RL policy gets: E-dimensional, residual on the
    prior mean, decoded through the frozen decoder.  a_policy = 0 reproduces the prior's mean
    behaviour for the current proprioceptive state.
    """
    s_n = self.self_norm(s)
    p_mu, _ = self.prior(s_n)
    return self.decode(s_n, p_mu + a_latent)

  @torch.no_grad()
  def act_from_prior(self, s: torch.Tensor, sample: bool = False) -> torch.Tensor:
    """What the prior alone does: z = mu_p (or a sample of p) -> action.  Diagnostic."""
    s_n = self.self_norm(s)
    p_mu, p_logvar = self.prior(s_n)
    z = self.reparameterize(p_mu, p_logvar)[0] if sample else p_mu
    return self.decode(s_n, z)


# ---------------------------------------------------------------------------------------------
# losses (amp_agent.py:_optimize_kin)
# ---------------------------------------------------------------------------------------------
def kl_multi(
  mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor
) -> torch.Tensor:
  """KL( N(mu_q, e^logvar_q) || N(mu_p, e^logvar_p) ), summed over latent dims -> [B]."""
  return 0.5 * (
    logvar_p - logvar_q
    + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2)) / torch.exp(logvar_p)
    - 1.0
  ).sum(dim=-1)


def ar1_prior(mu: torch.Tensor, consecutive: torch.Tensor, phi: float = 0.99) -> torch.Tensor:
  """PULSE's AR(1) smoothness prior on the posterior mean along time.

  mu:          [B, H, E]  posterior means of B envs over H consecutive rollout steps
  consecutive: [B, H-1]   True where step t and t+1 are consecutive steps of the same episode
                          AND neither is in the reset transient (PULSE zeroes idx <= 2; here the
                          caller also zeroes the 36-step startup override)
  """
  err = mu[:, 1:] - mu[:, :-1] * float(phi)
  err = err * consecutive.unsqueeze(-1).to(err.dtype)
  return torch.norm(err, dim=-1).mean()


def kld_weight(
  iteration: int, start: float = 0.01, end: float = 0.001, anneal_iters: int = 1
) -> float:
  """amp_agent.py:842 -- linear from `start` to `end` over anneal_iters, then held at `end`."""
  if anneal_iters <= 0:
    return float(end)
  frac = max((anneal_iters - iteration) / float(anneal_iters), 0.0)
  return (start - end) * frac + end


@dataclass
class KinLossWeights:
  kld: float = 0.01
  ar1: float = 0.005
  regu: float = 0.0  # PULSE default use_vae_prior_regu=False; 0.005 turns it on at their weight
  w_body: float = 1.0  # `use_part` analogue: 1.5 body / 0.5 per hand in PULSE-X, off by default
  w_hand: float = 1.0
  ar1_phi: float = 0.99


def kin_loss(
  model: ZSpaceVAE,
  s: torch.Tensor,  # [B, H, Ds] raw
  g: torch.Tensor,  # [B, H, Dg] raw
  a_teacher: torch.Tensor,  # [B, H, A]
  noise: torch.Tensor,  # [B, H, E]  the noise used when the student acted
  valid: torch.Tensor,  # [B, H] bool  label carries information (not in the startup override)
  consecutive: torch.Tensor,  # [B, H-1] bool
  weights: KinLossWeights,
  num_body: int,
) -> tuple[torch.Tensor, dict[str, float]]:
  B, H, _ = s.shape
  s_n, g_n = model.normalize(s.reshape(B * H, -1), g.reshape(B * H, -1))
  mu, logvar = model.encode(s_n, g_n)
  z, _ = model.reparameterize(mu, logvar, noise.reshape(B * H, -1))
  pred = model.decode(s_n, z)
  p_mu, p_logvar = model.prior(s_n)

  target = a_teacher.reshape(B * H, -1)
  v = valid.reshape(B * H).to(pred.dtype)
  n_valid = v.sum().clamp_min(1.0)

  err = pred - target
  if weights.w_body == 1.0 and weights.w_hand == 1.0:
    per_sample = torch.norm(err, dim=-1)  # RMSE, PULSE's `torch.norm(pred - gt, dim=-1).mean()`
  else:
    per_sample = (
      torch.norm(err[:, :num_body], dim=-1) * weights.w_body
      + torch.norm(err[:, num_body:], dim=-1) * weights.w_hand
    )
  action_loss = (per_sample * v).sum() / n_valid
  kld = (kl_multi(mu, logvar, p_mu, p_logvar) * v).sum() / n_valid
  ar1 = ar1_prior(mu.reshape(B, H, -1), consecutive, weights.ar1_phi)

  regu = torch.zeros((), device=pred.device)
  if weights.regu > 0.0:
    regu = (p_mu.pow(2).mean() + mu.pow(2).mean()) * 0.001 + (
      p_logvar.pow(2).mean() + logvar.pow(2).mean()
    ) * 0.001

  total = action_loss + weights.kld * kld + weights.ar1 * ar1 + weights.regu * regu
  with torch.no_grad():
    body_rmse = ((torch.norm(err[:, :num_body], dim=-1) * v).sum() / n_valid).item()
    hand_rmse = ((torch.norm(err[:, num_body:], dim=-1) * v).sum() / n_valid).item()
    z_std = torch.exp(0.5 * logvar).mean().item()
    p_std = torch.exp(0.5 * p_logvar).mean().item()
  info = {
    "loss": float(total.item()),
    "kin_action_loss": float(action_loss.item()),
    "kin_body_rmse": body_rmse,
    "kin_hand_rmse": hand_rmse,
    "kin_KLD": float(kld.item()),
    "kin_ar1": float(ar1.item()),
    "kin_regu": float(regu.item()),
    "z_post_std": z_std,
    "z_prior_std": p_std,
  }
  return total, info


# ---------------------------------------------------------------------------------------------
# checkpoint I/O
# ---------------------------------------------------------------------------------------------
def save_student(path: str, model: ZSpaceVAE, extra: dict | None = None) -> None:
  payload = {
    "format": "zspace_student.v1",
    "cfg": asdict(model.cfg),
    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
  }
  if extra:
    payload.update(extra)
  torch.save(payload, path)


def load_student(path: str, device: str | torch.device = "cpu") -> tuple[ZSpaceVAE, dict]:
  payload = torch.load(path, map_location="cpu", weights_only=False)
  cfg_d = dict(payload["cfg"])
  for k in ("enc_units", "dec_units", "self_groups", "task_groups"):
    if k in cfg_d:
      cfg_d[k] = tuple(cfg_d[k])
  model = ZSpaceVAE(ZSpaceConfig(**cfg_d))
  model.load_state_dict(payload["state_dict"])
  model.self_norm.frozen = True
  model.task_norm.frozen = True
  return model.to(device).eval(), payload
