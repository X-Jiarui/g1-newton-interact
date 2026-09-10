"""Residual actor that composes a frozen SONIC tracker with a trainable head."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

from mjlab.tasks.residual_interact import mdp
from mjlab.tasks.residual_interact.hand_bc import HandBCPolicy, parse_hidden_dims


def _as_tuple(values: Sequence[str] | str) -> tuple[str, ...]:
  if isinstance(values, str):
    return tuple(v.strip() for v in values.split(",") if v.strip())
  return tuple(values)


def _as_float_tuple(values: Sequence[float] | str) -> tuple[float, ...]:
  if isinstance(values, str):
    text = values.strip().strip("()[]")
    if not text:
      return ()
    return tuple(float(value.strip()) for value in text.split(",") if value.strip())
  return tuple(float(value) for value in values)


def _make_residual_mask(mask: str, device: torch.device | str) -> torch.Tensor:
  name = str(mask).strip().lower().replace("-", "_")
  out = torch.zeros(mdp.ACTION_DIM, device=device)
  if name in {"all", "full", "body_hand", "body_hands"}:
    out[:] = 1.0
  elif name in {"body", "body_only"}:
    out[: mdp.NUM_BODY] = 1.0
  elif name in {"hand", "hands", "hand_only", "hands_only"}:
    out[mdp.NUM_BODY :] = 1.0
  elif name in {"arms_hands", "upper_body_hands"}:
    out[15 : mdp.NUM_BODY] = 0.5
    out[mdp.NUM_BODY :] = 1.0
  elif name in {"arms_hands_full", "upper_body_hands_full", "isaac_arms_hands"}:
    out[15 : mdp.NUM_BODY] = 1.0
    out[mdp.NUM_BODY :] = 1.0
  elif name in {"none", "tracker_only", "zero"}:
    pass
  elif name in {"hands_small_root", "hands+small_root"}:
    out[0:3] = 0.25
    out[mdp.NUM_BODY :] = 1.0
  else:
    raise ValueError(
      f"Unknown residual_mask '{mask}'. Expected all, body, hand, arms_hands, "
      "arms_hands_full, hands_small_root, or none."
    )
  return out


def _make_residual_gain(
  residual_gain: float,
  body_residual_gain: float | None,
  hand_residual_gain: float | None,
  device: torch.device | str,
) -> torch.Tensor:
  out = torch.full((mdp.ACTION_DIM,), float(residual_gain), device=device)
  if body_residual_gain is not None:
    out[: mdp.NUM_BODY] = float(body_residual_gain)
  if hand_residual_gain is not None:
    out[mdp.NUM_BODY :] = float(hand_residual_gain)
  return out


def _mask_decoded_body_delta(
  base_body: torch.Tensor,
  candidate_body: torch.Tensor,
  residual_mask: torch.Tensor,
) -> torch.Tensor:
  """Keep decoded body changes only on body dimensions enabled by the mask."""
  active = residual_mask[: mdp.NUM_BODY].to(
    device=candidate_body.device, dtype=torch.bool
  )
  return torch.where(active.unsqueeze(0), candidate_body, base_body)


def _first_obs_tensor(obs: TensorDict) -> torch.Tensor:
  preferred = (
    "reference_phase",
    "astra_obs",
    "sonic_obs_or_latent",
    "sonic_encoder_obs",
  )
  for key in preferred:
    if key in obs:
      value = obs[key]
      if isinstance(value, torch.Tensor):
        return value
  for key in obs.keys():
    value = obs[key]
    if isinstance(value, torch.Tensor):
      return value
  raise ValueError("ResidualInteractActorModel received no tensor observation groups.")


def _hand_primitive_proxy_ids(close_action: torch.Tensor) -> torch.Tensor:
  """Pick one high-motion joint per hand as the stochastic primitive proxy."""
  local_ids = []
  for start, stop in ((0, 12), (12, mdp.NUM_HAND)):
    side = close_action[start:stop].abs()
    if side.numel() == 0:
      local_ids.append(start)
      continue
    local_ids.append(start + int(side.argmax().detach().cpu().item()))
  return torch.tensor(local_ids, dtype=torch.long, device=close_action.device)


def _reference_hand_action(ref: dict, frame: int) -> torch.Tensor:
  """Convert an absolute reference hand pose to the action adapter's delta space."""
  return ref["dof_pos"][frame, mdp.NUM_BODY : mdp.ACTION_DIM] - ref["default_hand"]


_HAND_PRIMITIVE_MODES = {
  "none",
  "open_close",
  "grail_close_2d",
  "grail_close_2d_hard",
}


def _groot_interaction_dir() -> Path:
  groot_root = Path(
    os.environ.get("GROOT_ROOT", "/home/jiarui/projects/GR00T-WholeBodyControl")
  )
  return groot_root / "interaction"


class OfficialSonicONNX53Actor(nn.Module):
  """Frozen official SONIC ONNX body tracker with an explicit hand base."""

  is_recurrent: bool = False
  token_dim: int = 64

  def __init__(
    self,
    obs: TensorDict,
    output_dim: int,
    sonic_encoder_onnx: str,
    sonic_decoder_onnx: str,
    hand_base_mode: str = "zero",
  ) -> None:
    super().__init__()
    if output_dim != mdp.ACTION_DIM:
      raise ValueError(f"Expected output_dim={mdp.ACTION_DIM}, got {output_dim}.")
    for group in ("sonic_encoder_obs", "sonic_obs_or_latent"):
      if group not in obs:
        raise ValueError(f"Official SONIC base requires observation group {group!r}.")
    hand_base_mode = str(hand_base_mode).strip().lower()
    if hand_base_mode != "zero":
      raise ValueError(
        "Official SONIC only outputs 29 body actions; currently supported "
        f"hand_base_mode is 'zero', got {hand_base_mode!r}."
      )

    interaction_dir = _groot_interaction_dir()
    if str(interaction_dir) not in sys.path:
      sys.path.insert(0, str(interaction_dir))
    sonic_encoder = cast(Any, importlib.import_module("sonic_encoder"))
    sonic_decoder = cast(Any, importlib.import_module("sonic_decoder"))

    self.encoder = sonic_encoder.SONICEncoder()
    sonic_encoder.load_weights_from_onnx(self.encoder, sonic_encoder_onnx)
    self.decoder = sonic_decoder.SONICDecoder()
    sonic_decoder.load_weights_from_onnx(self.decoder, sonic_decoder_onnx)
    self.hand_base_mode = hand_base_mode
    self._output_dim = output_dim
    self.obs_groups = ["sonic_encoder_obs", "sonic_obs_or_latent"]
    for param in self.parameters():
      param.requires_grad_(False)
    self.eval()

    n_total = sum(p.numel() for p in self.parameters())
    print(
      "[OfficialSonicONNX53Actor] "
      f"encoder={sonic_encoder_onnx} decoder={sonic_decoder_onnx} "
      f"hand_base={self.hand_base_mode} params={n_total:,}"
    )

  def _sanitize_obs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    enc_obs = cast(torch.Tensor, obs["sonic_encoder_obs"])
    sonic_obs = cast(torch.Tensor, obs["sonic_obs_or_latent"])
    enc_obs = torch.nan_to_num(enc_obs, nan=0.0, posinf=1e6, neginf=-1e6).clamp(
      -1e6, 1e6
    )
    sonic_obs = torch.nan_to_num(sonic_obs, nan=0.0, posinf=1e6, neginf=-1e6).clamp(
      -1e6, 1e6
    )
    return enc_obs, sonic_obs

  def token_and_body_history(
    self,
    obs: TensorDict,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    enc_obs, sonic_obs = self._sanitize_obs(obs)
    body_hist = sonic_obs[
      :, mdp.ENC_INPUT_DIM : mdp.ENC_INPUT_DIM + mdp.BODY_DEC_HIST_DIM
    ]
    with torch.no_grad():
      tokens = self.encoder(enc_obs)
    return tokens.detach(), body_hist.detach()

  def decode_body(self, tokens: torch.Tensor, body_hist: torch.Tensor) -> torch.Tensor:
    return self.decoder(torch.cat([tokens, body_hist], dim=-1))

  def hand_base(self, batch: int, device, dtype) -> torch.Tensor:
    return torch.zeros(batch, mdp.NUM_HAND, device=device, dtype=dtype)

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    del hidden_state, stochastic_output
    obs_td = cast(
      TensorDict,
      unpad_trajectories(obs, masks)
      if masks is not None and not self.is_recurrent
      else obs,
    )
    tokens, body_hist = self.token_and_body_history(obs_td)
    with torch.no_grad():
      body_action = self.decode_body(tokens, body_hist)
    hand_action = self.hand_base(
      body_action.shape[0], body_action.device, body_action.dtype
    )
    return torch.cat([body_action, hand_action], dim=-1)


class ASTRAONNX29BodyActor(nn.Module):
  """Frozen ASTRA/Humanoid-GPT 29D body tracker with zero hand base."""

  is_recurrent: bool = False
  hidden_dim: int = 512

  def __init__(
    self,
    obs: TensorDict,
    output_dim: int,
    astra_onnx_path: str,
    hand_base_mode: str = "zero",
  ) -> None:
    super().__init__()
    if output_dim != mdp.ACTION_DIM:
      raise ValueError(f"Expected output_dim={mdp.ACTION_DIM}, got {output_dim}.")
    if "astra_obs" not in obs:
      raise ValueError("ASTRA base requires observation group 'astra_obs'.")
    hand_base_mode = str(hand_base_mode).strip().lower()
    if hand_base_mode != "zero":
      raise ValueError(
        "ASTRA release checkpoint only outputs 29 body actions; currently "
        f"supported hand_base_mode is 'zero', got {hand_base_mode!r}."
      )

    path = Path(astra_onnx_path).expanduser()
    if not path.exists():
      raise FileNotFoundError(f"ASTRA ONNX checkpoint not found: {path}")
    self.session = None
    self.input_name = ""
    self.output_name = ""
    self.torch_trunk, self.torch_head = self._load_torch_policy(path)
    self.torch_policy = nn.Sequential(self.torch_trunk, self.torch_head)
    self.backend = "torch"
    requested_backend = os.environ.get("ASTRA_BASE_BACKEND", "torch").strip().lower()
    if requested_backend in {"onnx", "onnxruntime", "onnx_cuda", "cuda"}:
      self.torch_trunk = None
      self.torch_head = None
      self.torch_policy = None
      self.session, self.input_name, self.output_name, self.backend = (
        self._make_onnx_session(path, prefer_cuda=True)
      )
    elif requested_backend in {"onnx_cpu", "cpu"}:
      self.torch_trunk = None
      self.torch_head = None
      self.torch_policy = None
      self.session, self.input_name, self.output_name, self.backend = (
        self._make_onnx_session(path, prefer_cuda=False)
      )
    elif requested_backend not in {"torch", "torch_native", "pytorch"}:
      raise ValueError(
        "ASTRA_BASE_BACKEND must be one of torch, onnx_cuda, or onnx_cpu; "
        f"got {requested_backend!r}."
      )
    self.hand_base_mode = hand_base_mode
    self._output_dim = output_dim
    self.obs_groups = ["astra_obs"]
    self.register_buffer(
      "astra_default_pkl",
      torch.tensor(mdp.ASTRA_DEFAULT_BODY_PKL, dtype=torch.float32),
      persistent=False,
    )
    self.register_buffer(
      "astra_scale_pkl",
      torch.tensor(mdp.ASTRA_ACTION_SCALE_PKL, dtype=torch.float32),
      persistent=False,
    )
    self.register_buffer(
      "mjlab_default_pkl",
      torch.tensor(mdp.apple_mdp.SONIC_DEFAULT_ANGLES_PKL, dtype=torch.float32),
      persistent=False,
    )
    self.register_buffer(
      "mjlab_scale_pkl",
      torch.tensor(mdp.apple_mdp.SONIC_ACTION_SCALE_PKL, dtype=torch.float32),
      persistent=False,
    )
    self.register_buffer(
      "pkl_for_il",
      torch.tensor(mdp.apple_mdp.PKL_FOR_IL, dtype=torch.long),
      persistent=False,
    )
    self.last_astra_action_pkl = torch.zeros(1, mdp.NUM_BODY)
    self.eval()
    print(
      "[ASTRAONNX29BodyActor] "
      f"onnx={path} obs_dim={mdp.ASTRA_OBS_DIM} output=29 "
      f"hand_base={hand_base_mode} backend={self.backend}"
    )
    print("[ASTRAONNX29BodyActor] output action maps ASTRA PKL -> mjlab IL.")

  def hand_base(self, batch: int, device, dtype) -> torch.Tensor:
    return torch.zeros(batch, mdp.NUM_HAND, device=device, dtype=dtype)

  @staticmethod
  def _make_onnx_session(path: Path, prefer_cuda: bool):
    try:
      import onnxruntime as ort
    except Exception as exc:
      raise RuntimeError(
        "onnxruntime is required for ASTRA ONNXRuntime backends."
      ) from exc

    available = set(ort.get_available_providers())
    providers: list[str] = []
    if prefer_cuda and "CUDAExecutionProvider" in available:
      providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(path), providers=providers)
    actual = tuple(session.get_providers())
    backend = "onnx_cuda" if "CUDAExecutionProvider" in actual else "onnx_cpu"
    if prefer_cuda and backend != "onnx_cuda":
      print(
        "[ASTRAONNX29BodyActor] CUDAExecutionProvider unavailable; "
        f"falling back to providers={actual}"
      )
    return session, session.get_inputs()[0].name, session.get_outputs()[0].name, backend

  @staticmethod
  def _load_torch_policy(path: Path) -> tuple[nn.Module, nn.Module]:
    try:
      import onnx
      from onnx import numpy_helper
    except Exception as exc:
      raise RuntimeError("onnx is required for ASTRA torch-native backend.") from exc

    graph = onnx.load(str(path)).graph
    weights = {
      init.name: torch.from_numpy(numpy_helper.to_array(init).copy()).float()
      for init in graph.initializer
    }
    required = (
      "mlp.0.weight",
      "mlp.0.bias",
      "mlp.2.weight",
      "mlp.2.bias",
      "mlp.4.weight",
      "mlp.4.bias",
      "mean_head.weight",
      "mean_head.bias",
    )
    missing = [name for name in required if name not in weights]
    if missing:
      raise RuntimeError(f"ASTRA ONNX is missing expected initializers: {missing}")

    trunk = nn.Sequential(
      nn.Linear(mdp.ASTRA_OBS_DIM, 2048),
      nn.SiLU(),
      nn.Linear(2048, 1024),
      nn.SiLU(),
      nn.Linear(1024, 512),
      nn.SiLU(),
    )
    head = nn.Linear(512, mdp.NUM_BODY)
    linear_names = (
      ("mlp.0.weight", "mlp.0.bias"),
      ("mlp.2.weight", "mlp.2.bias"),
      ("mlp.4.weight", "mlp.4.bias"),
      ("mean_head.weight", "mean_head.bias"),
    )
    linears = [module for module in trunk if isinstance(module, nn.Linear)]
    linears.append(head)
    for module, (weight_name, bias_name) in zip(linears, linear_names, strict=True):
      module.weight.data.copy_(weights[weight_name])
      module.bias.data.copy_(weights[bias_name])
      module.weight.requires_grad_(False)
      module.bias.requires_grad_(False)
    trunk.eval()
    head.eval()
    for param in (*trunk.parameters(), *head.parameters()):
      param.requires_grad_(False)
    return trunk, head

  @staticmethod
  def _sanitize_astra_obs(astra_obs: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(astra_obs.detach(), nan=0.0, posinf=1e6, neginf=-1e6).clamp(
      -1e6, 1e6
    )

  def astra_hidden(self, obs: TensorDict) -> torch.Tensor:
    if self.torch_trunk is None:
      raise RuntimeError(
        "ASTRA hidden residual requires ASTRA_BASE_BACKEND=torch so the "
        "frozen trunk/head are available."
      )
    astra_obs = cast(torch.Tensor, obs["astra_obs"])
    if astra_obs.shape[-1] != mdp.ASTRA_OBS_DIM:
      raise ValueError(
        f"ASTRA obs must have dim {mdp.ASTRA_OBS_DIM}, got {astra_obs.shape}."
      )
    obs_t = self._sanitize_astra_obs(astra_obs).to(dtype=torch.float32)
    trunk = self.torch_trunk.to(device=obs_t.device, dtype=torch.float32)
    with torch.no_grad():
      hidden = trunk(obs_t).detach()
    return hidden.to(device=astra_obs.device, dtype=astra_obs.dtype)

  def decode_hidden_to_mjlab_body_action(self, hidden: torch.Tensor) -> torch.Tensor:
    if self.torch_head is None:
      raise RuntimeError(
        "ASTRA hidden residual requires ASTRA_BASE_BACKEND=torch so the "
        "frozen output head is available."
      )
    head = self.torch_head.to(device=hidden.device, dtype=torch.float32)
    astra_action_pkl = head(hidden.to(dtype=torch.float32)).to(dtype=hidden.dtype)
    self.last_astra_action_pkl = astra_action_pkl.detach()
    return self._astra_to_mjlab_body_action(astra_action_pkl)

  def body_action_from_edited_obs(
    self,
    obs: TensorDict,
    ref_edit_full: torch.Tensor,
  ) -> torch.Tensor:
    """Run the frozen ASTRA tracker on a differentiably edited observation."""
    if self.torch_trunk is None:
      raise RuntimeError("ASTRA reference editing requires ASTRA_BASE_BACKEND=torch.")
    astra_obs = cast(torch.Tensor, obs["astra_obs"])
    obs_t = self._sanitize_astra_obs(astra_obs).to(dtype=torch.float32)
    edit_t = ref_edit_full.to(device=obs_t.device, dtype=torch.float32)
    trunk = self.torch_trunk.to(device=obs_t.device, dtype=torch.float32)
    hidden = trunk(obs_t + edit_t)
    return self.decode_hidden_to_mjlab_body_action(hidden).to(dtype=astra_obs.dtype)

  def hidden_and_body_action(
    self, obs: TensorDict
  ) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = self.astra_hidden(obs)
    with torch.no_grad():
      base_body = self.decode_hidden_to_mjlab_body_action(hidden)
    return hidden.detach(), base_body.detach()

  def _run_onnx(self, astra_obs: torch.Tensor) -> torch.Tensor:
    if self.torch_policy is not None:
      obs_t = self._sanitize_astra_obs(astra_obs)
      policy = self.torch_policy.to(device=obs_t.device, dtype=torch.float32)
      out = policy(obs_t.to(dtype=torch.float32))
      return out.to(device=astra_obs.device, dtype=astra_obs.dtype)

    if self.session is None:
      raise RuntimeError("ASTRA ONNXRuntime session was not initialized.")
    obs_np = (
      torch.nan_to_num(astra_obs.detach(), nan=0.0, posinf=1e6, neginf=-1e6)
      .clamp(-1e6, 1e6)
      .to(device="cpu", dtype=torch.float32)
      .numpy()
    )
    out = self.session.run([self.output_name], {self.input_name: obs_np})[0]
    return torch.as_tensor(out, device=astra_obs.device, dtype=astra_obs.dtype)

  def _astra_to_mjlab_body_action(self, astra_action_pkl: torch.Tensor) -> torch.Tensor:
    dtype = astra_action_pkl.dtype
    device = astra_action_pkl.device
    astra_default = self.astra_default_pkl.to(device=device, dtype=dtype)
    astra_scale = self.astra_scale_pkl.to(device=device, dtype=dtype) * float(
      mdp.ASTRA_POLICY_ACTION_SCALE
    )
    mjlab_default = self.mjlab_default_pkl.to(device=device, dtype=dtype)
    mjlab_scale = self.mjlab_scale_pkl.to(device=device, dtype=dtype)
    target_pkl = astra_default.unsqueeze(0) + astra_action_pkl * astra_scale.unsqueeze(
      0
    )
    body_action_pkl = (target_pkl - mjlab_default.unsqueeze(0)) / mjlab_scale.unsqueeze(
      0
    )
    return body_action_pkl.index_select(1, self.pkl_for_il.to(device=device))

  def sync_last_astra_action_from_mjlab(self, body_action_il: torch.Tensor) -> None:
    """Store the ASTRA-native action corresponding to the executed MJLab action."""
    dtype = body_action_il.dtype
    device = body_action_il.device
    pkl_for_il = cast(torch.Tensor, self.pkl_for_il).to(device=device)
    body_action_pkl = torch.empty_like(body_action_il)
    body_action_pkl.scatter_(
      1,
      pkl_for_il.unsqueeze(0).expand(body_action_il.shape[0], -1),
      body_action_il,
    )
    mjlab_default = cast(torch.Tensor, self.mjlab_default_pkl).to(
      device=device, dtype=dtype
    )
    mjlab_scale = cast(torch.Tensor, self.mjlab_scale_pkl).to(
      device=device, dtype=dtype
    )
    astra_default = cast(torch.Tensor, self.astra_default_pkl).to(
      device=device, dtype=dtype
    )
    astra_scale = cast(torch.Tensor, self.astra_scale_pkl).to(
      device=device, dtype=dtype
    )
    target_pkl = mjlab_default.unsqueeze(0) + body_action_pkl * mjlab_scale.unsqueeze(0)
    denom = astra_scale * float(mdp.ASTRA_POLICY_ACTION_SCALE)
    astra_action_pkl = (target_pkl - astra_default.unsqueeze(0)) / denom.unsqueeze(0)
    self.last_astra_action_pkl = astra_action_pkl.detach()

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    del hidden_state, stochastic_output
    obs_td = cast(
      TensorDict,
      unpad_trajectories(obs, masks)
      if masks is not None and not self.is_recurrent
      else obs,
    )
    astra_obs = cast(torch.Tensor, obs_td["astra_obs"])
    if astra_obs.shape[-1] != mdp.ASTRA_OBS_DIM:
      raise ValueError(
        f"ASTRA obs must have dim {mdp.ASTRA_OBS_DIM}, got {astra_obs.shape}."
      )
    with torch.no_grad():
      astra_action_pkl = self._run_onnx(astra_obs)
      body_action = self._astra_to_mjlab_body_action(astra_action_pkl)
    self.last_astra_action_pkl = astra_action_pkl.detach()
    hand_action = self.hand_base(
      body_action.shape[0], body_action.device, body_action.dtype
    )
    return torch.cat([body_action, hand_action], dim=-1)


class ResidualInteractActorModel(nn.Module):
  """RSL-RL actor with a frozen SONIC base and trainable residual mean.

  PPO sees a Gaussian whose mean is the executed action:

  ``mean = clamp(base_tracker(obs) + residual_gain * residual_head(features))``.
  """

  is_recurrent: bool = False
  residual_mask: torch.Tensor
  residual_action_gain: torch.Tensor

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    base_tracker_kind: str = "checkpoint",
    sonic_encoder_onnx: str = "",
    sonic_decoder_onnx: str = "",
    astra_onnx_path: str = "",
    base_hand_mode: str = "zero",
    enc_input_dim: int = mdp.ENC_INPUT_DIM,
    residual_feature_groups: Sequence[str] | str = (
      "proprio_history",
      "tracker_action",
      "reference_phase",
      "reference_preview",
      "object_state",
      "hand_object_geometry",
      "contact_features",
      "placement_goal",
      "tracking_error",
      "last_residual",
    ),
    residual_feature_dropout: float = 0.0,
    residual_arch: str = "feature_mlp",
    ref_edit_clip: float = 0.12,
    ref_edit_groups: str = "arms",
    ref_edit_init_bias: Sequence[float] | str = (),
    split_hand_net: bool = False,
    frame_hidden_dims: tuple[int, ...] | list[int] = (128, 128),
    token_residual_clip: float = 0.1,
    token_residual_gain: float = 1.0,
    residual_gain: float = 1.0,
    body_residual_gain: float | None = None,
    hand_residual_gain: float | None = None,
    residual_action_clip: float = 0.5,
    residual_mask: str = "all",
    final_action_clip: float | None = None,
    body_init_std: float | None = None,
    hand_init_std: float | None = None,
    disabled_init_std: float | None = None,
    zero_init_residual: bool = True,
    freeze_tracker: bool = True,
    hand_bc_checkpoint: str = "",
    hand_bc_feature_groups: Sequence[str] | str = (),
    freeze_hand_bc: bool = True,
    hand_bc_action_clip: float = 5.0,
    hand_bc_base_start_frame: int = 0,
    hand_residual_start_frame: int | None = None,
    residual_start_frame: int = 0,
    residual_ramp_frames: int = 0,
    residual_lowpass_alpha: float = 1.0,
    body_sample_delta_clip: float | None = None,
    hand_sample_delta_clip: float | None = None,
    fixed_hand_action_frame: int | None = None,
    hand_primitive_mode: str = "none",
    hand_primitive_close_frame: int = 70,
    hand_primitive_open_frame: int = -1,
    hand_primitive_hard_threshold: float = 0.5,
    hand_primitive_init_logit_bias: float = -2.0,
  ) -> None:
    super().__init__()
    if output_dim != mdp.ACTION_DIM:
      raise ValueError(
        "ResidualInteractActorModel expects "
        f"output_dim={mdp.ACTION_DIM}, got {output_dim}."
      )
    self.base_tracker_kind = str(base_tracker_kind).strip().lower()
    if self.base_tracker_kind not in {"checkpoint", "official_onnx", "astra_onnx"}:
      raise ValueError(
        "base_tracker_kind must be 'checkpoint', 'official_onnx', or 'astra_onnx', "
        f"got {base_tracker_kind!r}."
      )
    if self.base_tracker_kind in {"checkpoint", "official_onnx"} and (
      "sonic_obs_or_latent" not in obs
    ):
      raise ValueError(
        f"base_tracker_kind={self.base_tracker_kind!r} requires 'sonic_obs_or_latent'."
      )
    if self.base_tracker_kind == "official_onnx" and "sonic_encoder_obs" not in obs:
      raise ValueError(
        "base_tracker_kind='official_onnx' requires 'sonic_encoder_obs'."
      )
    if self.base_tracker_kind == "astra_onnx" and "astra_obs" not in obs:
      raise ValueError("base_tracker_kind='astra_onnx' requires 'astra_obs'.")

    self.obs_groups = list(obs_groups[obs_set])
    self.obs_dim = sum(int(obs[k].shape[-1]) for k in self.obs_groups)
    self.residual_feature_groups = _as_tuple(residual_feature_groups)
    self.residual_feature_dropout = float(residual_feature_dropout)
    self.residual_arch = str(residual_arch).strip().lower()
    self.ref_edit_clip = float(ref_edit_clip)
    self.ref_edit_groups = str(ref_edit_groups).strip().lower()
    self.ref_edit_init_bias = _as_float_tuple(ref_edit_init_bias)
    if self.ref_edit_groups != "arms":
      raise ValueError("ref_edit_groups currently supports only 'arms'.")
    self.register_buffer(
      "astra_ref_edit_indices",
      torch.arange(108, 122, dtype=torch.long),
      persistent=False,
    )
    if len(self.ref_edit_init_bias) not in {0, 14}:
      raise ValueError("ref_edit_init_bias must contain exactly 14 arm values.")
    self.split_hand_net = bool(split_hand_net)
    self.token_residual_clip = float(token_residual_clip)
    self.token_residual_gain = float(token_residual_gain)
    self.residual_gain = float(residual_gain)
    self.body_residual_gain = (
      None if body_residual_gain is None else float(body_residual_gain)
    )
    self.hand_residual_gain = (
      None if hand_residual_gain is None else float(hand_residual_gain)
    )
    self.residual_action_clip = float(residual_action_clip)
    self.final_action_clip = (
      None if final_action_clip is None else float(final_action_clip)
    )
    self.zero_init_residual = bool(zero_init_residual)
    self.freeze_tracker = bool(freeze_tracker)
    self.hand_bc_checkpoint = str(hand_bc_checkpoint).strip()
    raw_hand_bc_groups = _as_tuple(hand_bc_feature_groups)
    self.hand_bc_feature_groups = (
      raw_hand_bc_groups if raw_hand_bc_groups else self.residual_feature_groups
    )
    self.freeze_hand_bc = bool(freeze_hand_bc)
    self.hand_bc_action_clip = float(hand_bc_action_clip)
    self.hand_bc_base_start_frame = max(int(hand_bc_base_start_frame), 0)
    self.hand_residual_start_frame = (
      self.hand_bc_base_start_frame
      if hand_residual_start_frame is None
      else max(int(hand_residual_start_frame), 0)
    )
    self.residual_start_frame = max(int(residual_start_frame), 0)
    self.residual_ramp_frames = max(int(residual_ramp_frames), 0)
    self.residual_lowpass_alpha = float(residual_lowpass_alpha)
    self.residual_lowpass_alpha = min(max(self.residual_lowpass_alpha, 0.0), 1.0)
    self.body_sample_delta_clip = (
      None
      if body_sample_delta_clip is None or float(body_sample_delta_clip) <= 0.0
      else float(body_sample_delta_clip)
    )
    self.hand_sample_delta_clip = (
      None
      if hand_sample_delta_clip is None or float(hand_sample_delta_clip) <= 0.0
      else float(hand_sample_delta_clip)
    )
    self.fixed_hand_action_frame = (
      None if fixed_hand_action_frame is None else int(fixed_hand_action_frame)
    )
    self.fixed_hand_action_enabled = (
      self.fixed_hand_action_frame is not None and self.fixed_hand_action_frame >= 0
    )
    self.hand_primitive_mode = str(hand_primitive_mode).strip().lower()
    self.hand_primitive_close_frame = int(hand_primitive_close_frame)
    self.hand_primitive_open_frame = int(hand_primitive_open_frame)
    self.hand_primitive_hard_threshold = float(hand_primitive_hard_threshold)
    self.hand_primitive_init_logit_bias = float(hand_primitive_init_logit_bias)
    if self.hand_primitive_mode not in _HAND_PRIMITIVE_MODES:
      raise ValueError(
        "hand_primitive_mode must be one of "
        f"{sorted(_HAND_PRIMITIVE_MODES)}, "
        f"got {hand_primitive_mode!r}."
      )
    self.hand_primitive_enabled = self.hand_primitive_mode != "none"
    self.hand_primitive_grail_enabled = self.hand_primitive_mode.startswith(
      "grail_close_2d"
    )
    self.hand_primitive_hard_enabled = self.hand_primitive_mode.endswith("_hard")

    unknown = sorted(
      {
        g
        for g in (*self.residual_feature_groups, *self.hand_bc_feature_groups)
        if g not in mdp.RESIDUAL_FEATURE_GROUPS
      }
    )
    if unknown:
      raise ValueError(
        f"Unknown residual feature groups: {unknown}. "
        f"Available: {mdp.RESIDUAL_FEATURE_GROUPS}"
      )

    if self.residual_arch not in {
      "feature_mlp",
      "frame_split",
      "latent_token_residual",
      "astra_hidden_residual",
      "astra_ref_edit",
    }:
      raise ValueError(
        "residual_arch must be 'feature_mlp', 'frame_split', or "
        "'latent_token_residual', 'astra_hidden_residual', or 'astra_ref_edit'."
      )
    if self.residual_arch == "frame_split" and "reference_phase" not in obs:
      raise ValueError(
        "frame_split residual actor requires the 'reference_phase' observation group."
      )
    if (
      self.residual_arch == "latent_token_residual"
      and self.base_tracker_kind != "official_onnx"
    ):
      raise ValueError(
        "latent_token_residual requires base_tracker_kind='official_onnx'."
      )
    if (
      self.residual_arch in {"astra_hidden_residual", "astra_ref_edit"}
      and self.base_tracker_kind != "astra_onnx"
    ):
      raise ValueError(
        "ASTRA residual architectures require base_tracker_kind='astra_onnx'."
      )
    if self.split_hand_net and self.residual_arch == "frame_split":
      raise ValueError("split_hand_net does not support residual_arch='frame_split'.")
    if self.hand_primitive_enabled:
      if self.fixed_hand_action_enabled:
        raise ValueError("hand_primitive_mode cannot be combined with fixed hand.")
      if self.split_hand_net:
        raise ValueError("hand_primitive_mode cannot be combined with split_hand_net.")
      if self.hand_bc_checkpoint:
        raise ValueError("hand_primitive_mode cannot be combined with hand BC.")
      if self.residual_arch not in {
        "latent_token_residual",
        "astra_hidden_residual",
        "astra_ref_edit",
      }:
        raise ValueError(
          "hand_primitive_mode requires residual_arch='latent_token_residual' "
          "'astra_hidden_residual', or 'astra_ref_edit'."
        )

    feature_dims: dict[str, int] = {}
    if self.residual_arch in {
      "feature_mlp",
      "latent_token_residual",
      "astra_hidden_residual",
      "astra_ref_edit",
    }:
      for group in self.residual_feature_groups:
        if group == "tracker_action":
          feature_dims[group] = mdp.ACTION_DIM
        elif group not in obs:
          raise ValueError(
            f"Residual feature group '{group}' is unavailable. "
            f"Available observation groups: {list(obs.keys())}"
          )
        else:
          if len(obs[group].shape) != 2:
            raise ValueError(
              f"Residual feature group '{group}' must be 2D, got {obs[group].shape}."
            )
          feature_dims[group] = int(obs[group].shape[-1])
      residual_input_dim = sum(feature_dims.values())
    else:
      feature_dims["frame_t"] = 1
      residual_input_dim = 1
    self.feature_group_dims = feature_dims

    hand_bc_feature_dims: dict[str, int] = {}
    for group in self.hand_bc_feature_groups:
      if group == "tracker_action":
        hand_bc_feature_dims[group] = mdp.ACTION_DIM
      elif group not in obs:
        raise ValueError(
          f"Hand BC feature group '{group}' is unavailable. "
          f"Available observation groups: {list(obs.keys())}"
        )
      else:
        if len(obs[group].shape) != 2:
          raise ValueError(
            f"Hand BC feature group '{group}' must be 2D, got {obs[group].shape}."
          )
        hand_bc_feature_dims[group] = int(obs[group].shape[-1])
    self.hand_bc_feature_group_dims = hand_bc_feature_dims
    hand_bc_input_dim = sum(hand_bc_feature_dims.values())
    if residual_input_dim <= 0:
      raise ValueError("Residual actor input dimension is zero.")
    if self.hand_bc_checkpoint and hand_bc_input_dim <= 0:
      raise ValueError("Hand BC input dimension is zero.")

    if self.base_tracker_kind == "checkpoint":
      # The checkpoint tracker depends on the full GR00T interaction tree.
      # Keep it lazy so official-ONNX runs only need the minimal SONIC assets.
      from mjlab.tasks.apple_eat.sonic_actor import AppleEatSONICActorModel

      tracker_obs_groups = {"actor": ["sonic_obs_or_latent"]}
      self.base_tracker = AppleEatSONICActorModel(
        obs=obs,
        obs_groups=tracker_obs_groups,
        obs_set="actor",
        output_dim=mdp.ACTION_DIM,
        enc_input_dim=enc_input_dim,
        sonic_decoder_onnx=sonic_decoder_onnx,
        init_noise_std=0.08,
      )
    elif self.base_tracker_kind == "official_onnx":
      self.base_tracker = OfficialSonicONNX53Actor(
        obs=obs,
        output_dim=mdp.ACTION_DIM,
        sonic_encoder_onnx=sonic_encoder_onnx,
        sonic_decoder_onnx=sonic_decoder_onnx,
        hand_base_mode=base_hand_mode,
      )
    else:
      self.base_tracker = ASTRAONNX29BodyActor(
        obs=obs,
        output_dim=mdp.ACTION_DIM,
        astra_onnx_path=astra_onnx_path,
        hand_base_mode=base_hand_mode,
      )
    if self.freeze_tracker:
      for param in self.base_tracker.parameters():
        param.requires_grad_(False)
      self.base_tracker.eval()

    obs_device = _first_obs_tensor(obs).device
    ref_for_frame = mdp._ref(str(obs_device))
    self.reference_n_frames = int(ref_for_frame["n_frames"])
    if self.fixed_hand_action_enabled:
      requested_frame = int(cast(int, self.fixed_hand_action_frame))
      frame = min(max(requested_frame, 0), self.reference_n_frames - 1)
      self.fixed_hand_action_frame = frame
      fixed_hand = _reference_hand_action(ref_for_frame, frame)
      fixed_hand = fixed_hand.detach().to(device=obs_device, dtype=torch.float32)
    else:
      fixed_hand = torch.empty(0, device=obs_device, dtype=torch.float32)
    self.register_buffer("fixed_hand_action", fixed_hand)

    if self.hand_primitive_enabled:
      close_frame = min(
        max(int(self.hand_primitive_close_frame), 0), self.reference_n_frames - 1
      )
      self.hand_primitive_close_frame = close_frame
      open_frame = int(self.hand_primitive_open_frame)
      if open_frame >= 0:
        open_frame = min(max(open_frame, 0), self.reference_n_frames - 1)
        self.hand_primitive_open_frame = open_frame
        open_action = _reference_hand_action(ref_for_frame, open_frame)
        open_action = open_action.detach().to(device=obs_device, dtype=torch.float32)
      else:
        self.hand_primitive_open_frame = -1
        open_action = torch.zeros(mdp.NUM_HAND, device=obs_device, dtype=torch.float32)
      close_action = _reference_hand_action(ref_for_frame, close_frame)
      close_action = close_action.detach().to(device=obs_device, dtype=torch.float32)
      # The retargeted reference never opposes the thumb (its
      # right_hand_thumb_rota_joint2 peaks at 0.309 rad over the whole clip,
      # while an enveloping grasp of the sim apple needs ~0.639 rad), so no
      # reference frame can serve as the closed pose.  Allow an explicit
      # override of the right-hand close action, e.g. from an IK-solved grasp.
      _close_override = os.environ.get("MJLAB_HAND_CLOSE_ACTION_RIGHT", "").strip()
      if _close_override:
        _vals = [float(x) for x in _close_override.replace(" ", "").split(",") if x]
        if len(_vals) != mdp.NUM_HAND // 2:
          raise ValueError(
            "MJLAB_HAND_CLOSE_ACTION_RIGHT expects "
            f"{mdp.NUM_HAND // 2} comma-separated values, got {len(_vals)}."
          )
        close_action = close_action.clone()
        close_action[mdp.NUM_HAND // 2 :] = torch.tensor(
          _vals, device=close_action.device, dtype=close_action.dtype
        )
        print(
          "[ResidualInteractActor] right-hand close action overridden by "
          f"MJLAB_HAND_CLOSE_ACTION_RIGHT: {[round(v, 4) for v in _vals]}",
          flush=True,
        )
      delta_action = close_action - open_action
      proxy_ids = _hand_primitive_proxy_ids(delta_action)
      proxy_values = delta_action.index_select(0, proxy_ids).clamp(-5.0, 5.0)
      proxy_values = torch.where(
        proxy_values.abs() > 1.0e-4,
        proxy_values,
        torch.sign(proxy_values + 1.0e-6),
      )
    else:
      open_action = torch.empty(0, device=obs_device, dtype=torch.float32)
      close_action = torch.empty(0, device=obs_device, dtype=torch.float32)
      delta_action = torch.empty(0, device=obs_device, dtype=torch.float32)
      proxy_ids = torch.empty(0, device=obs_device, dtype=torch.long)
      proxy_values = torch.empty(0, device=obs_device, dtype=torch.float32)
    self.register_buffer("hand_primitive_open_action", open_action)
    self.register_buffer("hand_primitive_close_action", close_action)
    self.register_buffer("hand_primitive_delta_action", delta_action)
    self.register_buffer("hand_primitive_proxy_ids", proxy_ids)
    self.register_buffer("hand_primitive_proxy_values", proxy_values)

    self.hand_bc_policy: HandBCPolicy | None = None
    if self.hand_bc_checkpoint and self.fixed_hand_action_enabled:
      raise ValueError("fixed_hand_action_frame cannot be combined with hand BC.")
    if self.hand_bc_checkpoint:
      self.hand_bc_policy = self._load_hand_bc_policy(
        self.hand_bc_checkpoint,
        input_dim=hand_bc_input_dim,
        device=obs_device,
      )

    self.obs_normalization = bool(obs_normalization)
    if self.obs_normalization:
      self.obs_normalizer = EmpiricalNormalization(residual_input_dim)
      if self.split_hand_net:
        self.hand_obs_normalizer = EmpiricalNormalization(residual_input_dim)
    else:
      self.obs_normalizer = nn.Identity()
      if self.split_hand_net:
        self.hand_obs_normalizer = nn.Identity()

    residual_output_dim = mdp.ACTION_DIM
    if self.residual_arch == "latent_token_residual":
      residual_output_dim = OfficialSonicONNX53Actor.token_dim + mdp.NUM_HAND
      if self.fixed_hand_action_enabled:
        residual_output_dim = OfficialSonicONNX53Actor.token_dim
      elif self.hand_primitive_enabled:
        residual_output_dim = OfficialSonicONNX53Actor.token_dim + 2
    elif self.residual_arch == "astra_hidden_residual":
      residual_output_dim = ASTRAONNX29BodyActor.hidden_dim + mdp.NUM_HAND
      if self.fixed_hand_action_enabled:
        residual_output_dim = ASTRAONNX29BodyActor.hidden_dim
      elif self.hand_primitive_enabled:
        residual_output_dim = ASTRAONNX29BodyActor.hidden_dim + 2
    elif self.residual_arch == "astra_ref_edit":
      ref_edit_dim = int(self.astra_ref_edit_indices.numel())
      residual_output_dim = ref_edit_dim + mdp.NUM_HAND
      if self.fixed_hand_action_enabled:
        residual_output_dim = ref_edit_dim
      elif self.hand_primitive_enabled:
        residual_output_dim = ref_edit_dim + 2
    if self.split_hand_net:
      residual_output_dim = (
        OfficialSonicONNX53Actor.token_dim
        if self.residual_arch == "latent_token_residual"
        else (
          ASTRAONNX29BodyActor.hidden_dim
          if self.residual_arch == "astra_hidden_residual"
          else (
            int(self.astra_ref_edit_indices.numel())
            if self.residual_arch == "astra_ref_edit"
            else mdp.NUM_BODY
          )
        )
      )
    self.residual_output_dim = residual_output_dim
    self.residual_mlp = MLP(
      residual_input_dim, residual_output_dim, hidden_dims, activation
    )
    if self.split_hand_net:
      self.hand_mlp = MLP(residual_input_dim, mdp.NUM_HAND, hidden_dims, activation)
    frame_hidden = tuple(int(v) for v in frame_hidden_dims)
    self.body_head = MLP(1, mdp.NUM_BODY, frame_hidden, activation)
    self.hand_head = MLP(1, mdp.NUM_HAND, frame_hidden, activation)
    if distribution_cfg is None:
      distribution_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": 0.08,
        "std_range": (0.01, 1.0),
        "std_type": "scalar",
      }
    dist_cfg = dict(distribution_cfg)
    dist_cfg.setdefault("std_range", (0.01, 1.0))
    class_name = dist_cfg.pop("class_name", "GaussianDistribution")
    if class_name not in {
      "GaussianDistribution",
      "rsl_rl.modules.distribution:GaussianDistribution",
    }:
      raise ValueError(
        "ResidualInteractActorModel only supports GaussianDistribution, "
        f"got {class_name}."
      )
    self.distribution = GaussianDistribution(mdp.ACTION_DIM, **dist_cfg)
    self.register_buffer(
      "residual_mask", _make_residual_mask(residual_mask, device="cpu")
    )
    self.register_buffer(
      "residual_action_gain",
      _make_residual_gain(
        self.residual_gain,
        self.body_residual_gain,
        self.hand_residual_gain,
        device="cpu",
      ),
    )
    self._init_hand_eigen()
    if self.zero_init_residual:
      self._zero_initialize_residual_mean()
    self._set_initial_std(
      body_init_std=body_init_std,
      hand_init_std=hand_init_std,
      disabled_init_std=disabled_init_std,
    )

    self.last_base_action = torch.zeros(1, mdp.ACTION_DIM)
    self.last_residual_action = torch.zeros(1, mdp.ACTION_DIM)
    self.last_final_action = torch.zeros(1, mdp.ACTION_DIM)
    self.last_residual_mean = torch.zeros(1, mdp.ACTION_DIM)
    self.last_action_mean = torch.zeros(1, mdp.ACTION_DIM)
    self.last_hand_sample_delta_pre_clip = torch.zeros(1, mdp.NUM_HAND)
    self.last_hand_sample_delta_post_clip = torch.zeros(1, mdp.NUM_HAND)
    self.last_hand_sample_clip_frac = torch.zeros(1)
    self.last_hand_action_std_mean = torch.zeros(1)
    self.last_body_sample_delta_pre_clip = torch.zeros(1, mdp.NUM_BODY)
    self.last_body_sample_delta_post_clip = torch.zeros(1, mdp.NUM_BODY)
    self.last_body_sample_clip_frac = torch.zeros(1)
    self.last_hand_primitive_close = torch.zeros(1, 2)
    self.last_hand_primitive_delta = torch.zeros(1, mdp.NUM_HAND)
    if self.residual_arch == "astra_hidden_residual":
      token_delta_dim = ASTRAONNX29BodyActor.hidden_dim
    elif self.residual_arch == "astra_ref_edit":
      token_delta_dim = int(self.astra_ref_edit_indices.numel())
    else:
      token_delta_dim = OfficialSonicONNX53Actor.token_dim
    self.last_token_residual = torch.zeros(1, token_delta_dim)
    self.last_previous_token_residual = torch.zeros(1, token_delta_dim)
    self.last_decoder_body_delta = torch.zeros(1, mdp.NUM_BODY)
    self.last_hand_control_gate = torch.ones(1, 1)
    self._prev_body_residual = torch.zeros(0)
    self._prev_hand_residual = torch.zeros(0)
    self._prev_token_residual = torch.zeros(0)

    total_params = sum(p.numel() for p in self.parameters())
    trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    tracker_params = sum(p.numel() for p in self.base_tracker.parameters())
    residual_params = sum(p.numel() for p in self.residual_mlp.parameters())
    hand_params = (
      sum(p.numel() for p in self.hand_mlp.parameters()) if self.split_hand_net else 0
    )
    hand_bc_params = (
      sum(p.numel() for p in self.hand_bc_policy.parameters())
      if self.hand_bc_policy is not None
      else 0
    )
    frame_params = sum(p.numel() for p in self.body_head.parameters()) + sum(
      p.numel() for p in self.hand_head.parameters()
    )
    print(
      "[ResidualInteractActor] active feature groups: "
      + ", ".join(f"{name}={dim}" for name, dim in self.feature_group_dims.items())
    )
    if self.hand_bc_checkpoint:
      print(
        "[ResidualInteractActor] hand BC feature groups: "
        + ", ".join(
          f"{name}={dim}" for name, dim in self.hand_bc_feature_group_dims.items()
        )
      )
    print(
      f"[ResidualInteractActor] residual_input_dim={residual_input_dim}, "
      f"residual_output_dim={residual_output_dim}, "
      f"arch={self.residual_arch}, "
      f"residual_gain={self.residual_gain}, residual_clip={self.residual_action_clip}, "
      f"body_gain={self.body_residual_gain}, hand_gain={self.hand_residual_gain}, "
      f"token_gain={self.token_residual_gain}, "
      f"token_clip={self.token_residual_clip}, "
      f"split_hand_net={self.split_hand_net}, "
      f"fixed_hand_frame={self.fixed_hand_action_frame}, "
      f"hand_bc_base_start_frame={self.hand_bc_base_start_frame}, "
      f"hand_residual_start_frame={self.hand_residual_start_frame}, "
      f"residual_start_frame={self.residual_start_frame}, "
      f"residual_ramp_frames={self.residual_ramp_frames}, "
      f"residual_lowpass_alpha={self.residual_lowpass_alpha}, "
      f"body_sample_delta_clip={self.body_sample_delta_clip}, "
      f"hand_sample_delta_clip={self.hand_sample_delta_clip}, "
      f"hand_primitive={self.hand_primitive_mode}@"
      f"{self.hand_primitive_close_frame}, "
      f"hand_primitive_open_frame={self.hand_primitive_open_frame}, "
      f"hand_primitive_hard_threshold={self.hand_primitive_hard_threshold}, "
      f"hand_primitive_init_logit_bias={self.hand_primitive_init_logit_bias}, "
      f"dropout={self.residual_feature_dropout}, zero_init={self.zero_init_residual}"
    )
    print(
      "[ResidualInteractActor] residual action scale: "
      f"body={float(self.residual_action_gain[: mdp.NUM_BODY].abs().max()):.4f}, "
      f"hand={float(self.residual_action_gain[mdp.NUM_BODY :].abs().max()):.4f}, "
      f"active_dims={int((self.residual_mask > 0).sum().item())}/{mdp.ACTION_DIM}"
    )
    print(
      f"[ResidualInteractActor] params: {total_params:,} total, "
      f"{trainable_params:,} trainable, {tracker_params:,} tracker, "
      f"{residual_params:,} feature_head, {hand_params:,} hand_head, "
      f"{hand_bc_params:,} hand_bc_base, {frame_params:,} frame_split_heads"
    )

  def _zero_initialize_residual_mean(self) -> None:
    linear_layers = [
      module for module in self.residual_mlp.modules() if isinstance(module, nn.Linear)
    ]
    if not linear_layers:
      raise RuntimeError("Residual MLP has no Linear layer to zero-initialize.")
    last = linear_layers[-1]
    nn.init.zeros_(last.weight)
    if last.bias is not None:
      nn.init.zeros_(last.bias)
      if self.residual_arch == "astra_ref_edit" and self.ref_edit_init_bias:
        desired = torch.tensor(
          self.ref_edit_init_bias,
          device=last.bias.device,
          dtype=last.bias.dtype,
        )
        normalized = (desired / max(self.ref_edit_clip, 1.0e-6)).clamp(
          -0.999,
          0.999,
        )
        last.bias.data[: desired.numel()].copy_(torch.atanh(normalized))
      if self.hand_primitive_enabled and self.residual_arch in {
        "latent_token_residual",
        "astra_hidden_residual",
        "astra_ref_edit",
      }:
        if self.residual_arch == "astra_hidden_residual":
          start = ASTRAONNX29BodyActor.hidden_dim
        elif self.residual_arch == "astra_ref_edit":
          start = int(self.astra_ref_edit_indices.numel())
        else:
          start = OfficialSonicONNX53Actor.token_dim
        if last.bias.numel() >= start + 2:
          last.bias.data[start : start + 2].fill_(self.hand_primitive_init_logit_bias)
    for head in (self.body_head, self.hand_head):
      head_linears = [
        module for module in head.modules() if isinstance(module, nn.Linear)
      ]
      if head_linears:
        nn.init.zeros_(head_linears[-1].weight)
        if head_linears[-1].bias is not None:
          nn.init.zeros_(head_linears[-1].bias)
    if self.split_hand_net:
      hand_linears = [
        module for module in self.hand_mlp.modules() if isinstance(module, nn.Linear)
      ]
      if hand_linears:
        nn.init.zeros_(hand_linears[-1].weight)
        if hand_linears[-1].bias is not None:
          nn.init.zeros_(hand_linears[-1].bias)

  def _load_hand_bc_policy(
    self,
    checkpoint: str,
    *,
    input_dim: int,
    device: torch.device,
  ) -> HandBCPolicy:
    path = Path(checkpoint).expanduser()
    if not path.exists():
      raise FileNotFoundError(f"Hand BC checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    compat = ckpt.get("compat", {})
    ckpt_input_dim = int(compat.get("input_dim", -1))
    ckpt_output_dim = int(compat.get("output_dim", -1))
    if ckpt_input_dim != int(input_dim):
      raise RuntimeError(
        f"Hand BC input dim mismatch: checkpoint={ckpt_input_dim}, actor={input_dim}."
      )
    if ckpt_output_dim != mdp.NUM_HAND:
      raise RuntimeError(
        "Hand BC output dim mismatch: "
        f"checkpoint={ckpt_output_dim}, expected={mdp.NUM_HAND}."
      )

    cfg = dict(ckpt.get("config", {}))
    ckpt_groups = tuple(cfg.get("feature_groups", ()))
    if ckpt_groups and ckpt_groups != tuple(self.hand_bc_feature_groups):
      raise RuntimeError(
        "Hand BC feature groups mismatch: "
        f"checkpoint={ckpt_groups}, actor={tuple(self.hand_bc_feature_groups)}."
      )
    hidden_dims = parse_hidden_dims(cfg.get("hidden_dims", (2048, 1024, 512)))
    activation = str(cfg.get("activation", "swish"))
    policy = HandBCPolicy(
      input_dim=input_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      output_dim=mdp.NUM_HAND,
    )
    policy.load_state_dict(ckpt["model_state_dict"], strict=True)
    policy.to(device)
    policy.eval()
    if self.freeze_hand_bc:
      for param in policy.parameters():
        param.requires_grad_(False)
    print(
      "[ResidualInteractActor] loaded hand BC base policy: "
      f"{path} input_dim={input_dim} hidden={hidden_dims} "
      f"freeze={self.freeze_hand_bc}"
    )
    return policy

  def _set_initial_std(
    self,
    *,
    body_init_std: float | None,
    hand_init_std: float | None,
    disabled_init_std: float | None,
  ) -> None:
    if body_init_std is None and hand_init_std is None and disabled_init_std is None:
      return
    if self.distribution.std_type == "scalar":
      current = self.distribution.std_param.detach().clone()
    elif self.distribution.std_type == "log":
      current = self.distribution.log_std_param.detach().exp().clone()
    else:
      raise ValueError(f"Unknown distribution std_type: {self.distribution.std_type}")

    std = current
    if body_init_std is not None:
      std[: mdp.NUM_BODY] = float(body_init_std)
    if hand_init_std is not None:
      std[mdp.NUM_BODY :] = float(hand_init_std)
    if disabled_init_std is not None:
      mask = self.residual_mask.to(dtype=torch.bool)
      std[~mask] = float(disabled_init_std)
    std = std.clamp_min(1e-6)
    with torch.no_grad():
      if self.distribution.std_type == "scalar":
        self.distribution.std_param.copy_(std)
      else:
        self.distribution.log_std_param.copy_(std.log())

  def _sanitize_distribution_std_(self) -> None:
    """Keep learnable scalar std valid after optimizer updates."""
    with torch.no_grad():
      if self.distribution.std_type == "scalar":
        std = torch.nan_to_num(
          self.distribution.std_param,
          nan=float(self.distribution.std_range[0]),
          posinf=float(self.distribution.std_range[1]),
          neginf=float(self.distribution.std_range[0]),
        )
        self.distribution.std_param.copy_(
          std.clamp(
            min=float(self.distribution.std_range[0]),
            max=float(self.distribution.std_range[1]),
          )
        )
      elif self.distribution.std_type == "log":
        log_std = torch.nan_to_num(
          self.distribution.log_std_param,
          nan=float(self.distribution.log_std_range[0]),
          posinf=float(self.distribution.log_std_range[1]),
          neginf=float(self.distribution.log_std_range[0]),
        )
        self.distribution.log_std_param.copy_(
          log_std.clamp(
            min=float(self.distribution.log_std_range[0]),
            max=float(self.distribution.log_std_range[1]),
          )
        )
      else:
        raise ValueError(f"Unknown distribution std_type: {self.distribution.std_type}")

  def _action_std(self) -> torch.Tensor:
    if self.distribution.std_type == "scalar":
      return self.distribution.std_param.detach().clamp(
        self.distribution.std_range[0], self.distribution.std_range[1]
      )
    log_std = self.distribution.log_std_param.detach().clamp(
      self.distribution.log_std_range[0],
      self.distribution.log_std_range[1],
    )
    return log_std.exp()

  def _clamp_sampled_body_delta(self, sampled: torch.Tensor) -> torch.Tensor:
    body_delta = sampled[:, : mdp.NUM_BODY] - self.last_base_action[:, : mdp.NUM_BODY]
    self.last_body_sample_delta_pre_clip = body_delta.detach()
    if self.body_sample_delta_clip is None:
      self.last_body_sample_delta_post_clip = body_delta.detach()
      self.last_body_sample_clip_frac = torch.zeros(
        sampled.shape[0], device=sampled.device, dtype=sampled.dtype
      )
      return sampled

    clip = float(self.body_sample_delta_clip)
    clipped = body_delta.clamp(-clip, clip)
    clip_frac = (body_delta.abs() > clip).float().mean(dim=-1)
    out = sampled.clone()
    out[:, : mdp.NUM_BODY] = self.last_base_action[:, : mdp.NUM_BODY] + clipped
    self.last_body_sample_delta_post_clip = clipped.detach()
    self.last_body_sample_clip_frac = clip_frac.detach()
    return out

  def _clamp_sampled_hand_delta(self, sampled: torch.Tensor) -> torch.Tensor:
    hand_delta = sampled[:, mdp.NUM_BODY :] - self.last_base_action[:, mdp.NUM_BODY :]
    self.last_hand_sample_delta_pre_clip = hand_delta.detach()
    std = self._action_std()
    self.last_hand_action_std_mean = std[mdp.NUM_BODY :].mean().view(1)
    if self.hand_sample_delta_clip is None or self.fixed_hand_action_enabled:
      self.last_hand_sample_delta_post_clip = hand_delta.detach()
      self.last_hand_sample_clip_frac = torch.zeros(
        sampled.shape[0], device=sampled.device, dtype=sampled.dtype
      )
      return sampled

    clip = float(self.hand_sample_delta_clip)
    clipped = hand_delta.clamp(-clip, clip)
    clip_frac = (hand_delta.abs() > clip).float().mean(dim=-1)
    out = sampled.clone()
    out[:, mdp.NUM_BODY :] = self.last_base_action[:, mdp.NUM_BODY :] + clipped
    self.last_hand_sample_delta_post_clip = clipped.detach()
    self.last_hand_sample_clip_frac = clip_frac.detach()
    return out

  def _tracker_action(self, obs: TensorDict) -> torch.Tensor:
    with torch.no_grad():
      base = self.base_tracker(obs, stochastic_output=False)
    return base.detach()

  def train(self, mode: bool = True):
    super().train(mode)
    if self.freeze_tracker:
      self.base_tracker.eval()
    if self.freeze_hand_bc and self.hand_bc_policy is not None:
      self.hand_bc_policy.eval()
    return self

  def _feature_chunks(
    self,
    obs: TensorDict,
    base_action: torch.Tensor,
    *,
    apply_dropout: bool = True,
  ) -> list[torch.Tensor]:
    chunks: list[torch.Tensor] = []
    batch = base_action.shape[0]
    for group in self.residual_feature_groups:
      if group == "tracker_action":
        value = base_action
      else:
        value = cast(torch.Tensor, obs[group])
      if apply_dropout and self.training and self.residual_feature_dropout > 0.0:
        keep_prob = max(1.0 - self.residual_feature_dropout, 0.0)
        keep = torch.rand(batch, 1, device=value.device) < keep_prob
        value = value * keep.to(value.dtype)
      chunks.append(value)
    return chunks

  def _residual_features(
    self, obs: TensorDict, base_action: torch.Tensor
  ) -> torch.Tensor:
    features = torch.cat(self._feature_chunks(obs, base_action), dim=-1)
    return self.obs_normalizer(features)

  def _hand_features(self, obs: TensorDict, base_action: torch.Tensor) -> torch.Tensor:
    if not self.split_hand_net:
      raise RuntimeError("_hand_features requires split_hand_net=True.")
    features = torch.cat(self._feature_chunks(obs, base_action), dim=-1)
    return self.hand_obs_normalizer(features)

  def _hand_bc_feature_chunks(
    self,
    obs: TensorDict,
    base_action: torch.Tensor,
  ) -> list[torch.Tensor]:
    chunks: list[torch.Tensor] = []
    for group in self.hand_bc_feature_groups:
      if group == "tracker_action":
        value = base_action
      else:
        value = cast(torch.Tensor, obs[group])
      chunks.append(value)
    return chunks

  def _frame_t(self, obs: TensorDict) -> torch.Tensor:
    # reference_phase[:, 0] is normalized tracking frame in [0, 1].
    return obs["reference_phase"][:, 0:1].clamp(0.0, 1.0)

  def _reference_frame_gate(self, obs: TensorDict, start_frame: int) -> torch.Tensor:
    if int(start_frame) <= 0:
      ref_obs = _first_obs_tensor(obs)
      return torch.ones(ref_obs.shape[0], 1, device=ref_obs.device, dtype=ref_obs.dtype)
    phase = self._frame_t(obs)
    denom = max(float(self.reference_n_frames - 1), 1.0)
    threshold = min(max(float(start_frame) / denom, 0.0), 1.0)
    return (phase >= threshold).to(dtype=phase.dtype)

  def _reference_frame_ramp(
    self,
    obs: TensorDict,
    start_frame: int,
    ramp_frames: int,
  ) -> torch.Tensor:
    if int(start_frame) <= 0 and int(ramp_frames) <= 0:
      ref_obs = _first_obs_tensor(obs)
      return torch.ones(ref_obs.shape[0], 1, device=ref_obs.device, dtype=ref_obs.dtype)
    phase = self._frame_t(obs)
    denom = max(float(self.reference_n_frames - 1), 1.0)
    frame = phase * denom
    start = float(max(int(start_frame), 0))
    ramp = float(max(int(ramp_frames), 0))
    if ramp <= 0.0:
      return (frame >= start).to(dtype=phase.dtype)
    return ((frame - start) / ramp).clamp(0.0, 1.0)

  def _global_residual_ramp(self, obs: TensorDict) -> torch.Tensor:
    return self._reference_frame_ramp(
      obs, self.residual_start_frame, self.residual_ramp_frames
    )

  def _lowpass_residual(
    self,
    name: str,
    value: torch.Tensor,
    gate: torch.Tensor,
  ) -> torch.Tensor:
    alpha = float(self.residual_lowpass_alpha)
    if alpha >= 1.0:
      return value * gate.to(device=value.device, dtype=value.dtype)
    gate = gate.to(device=value.device, dtype=value.dtype)
    prev = getattr(self, name)
    if (
      not isinstance(prev, torch.Tensor)
      or prev.shape != value.shape
      or prev.device != value.device
    ):
      prev = torch.zeros_like(value)
    out = alpha * value + (1.0 - alpha) * prev.to(dtype=value.dtype)
    out = out * gate
    setattr(self, name, out.detach())
    return out

  def _hand_bc_action(
    self,
    obs: TensorDict,
    tracker_action: torch.Tensor,
  ) -> torch.Tensor | None:
    if self.hand_bc_policy is None:
      return None
    features = torch.cat(self._hand_bc_feature_chunks(obs, tracker_action), dim=-1)
    if self.freeze_hand_bc:
      with torch.no_grad():
        action = self.hand_bc_policy(features).detach()
    else:
      action = self.hand_bc_policy(features)
    action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
    if self.hand_bc_action_clip > 0.0:
      action = action.clamp(-self.hand_bc_action_clip, self.hand_bc_action_clip)
    return action

  def _hand_primitive_delta(self, raw: torch.Tensor) -> torch.Tensor:
    if not self.hand_primitive_enabled:
      raise RuntimeError("_hand_primitive_delta requires hand_primitive_mode.")
    close_buf = cast(torch.Tensor, self.hand_primitive_close_action)
    close = close_buf.to(device=raw.device, dtype=raw.dtype)
    if close.numel() != mdp.NUM_HAND:
      raise RuntimeError("Invalid hand primitive close action buffer.")
    ratio = self._hand_primitive_close(raw)
    left = ratio[:, 0:1] * close[:12].unsqueeze(0)
    right = ratio[:, 1:2] * close[12:].unsqueeze(0)
    delta = torch.cat([left, right], dim=-1)
    self.last_hand_primitive_delta = delta.detach()
    return delta

  def _hand_primitive_close(self, raw: torch.Tensor) -> torch.Tensor:
    if raw.shape[-1] != 2:
      raise RuntimeError(
        f"2D hand primitive expects raw shape (*, 2), got {tuple(raw.shape)}."
      )
    if self.hand_primitive_grail_enabled:
      soft = torch.sigmoid(raw).clamp(0.0, 1.0)
    else:
      soft = torch.sigmoid(raw - 2.0).clamp(0.0, 1.0)
    if self.hand_primitive_hard_enabled:
      hard = (soft >= float(self.hand_primitive_hard_threshold)).to(dtype=soft.dtype)
      close = hard + soft - soft.detach()
      metric_close = hard
    else:
      close = soft
      metric_close = soft
    self.last_hand_primitive_close = metric_close.detach()
    return close

  def _hand_primitive_interpolate(
    self,
    raw: torch.Tensor,
    gate: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if not self.hand_primitive_grail_enabled:
      raise RuntimeError("_hand_primitive_interpolate requires grail_close_2d mode.")
    open_buf = cast(torch.Tensor, self.hand_primitive_open_action)
    grasp_buf = cast(torch.Tensor, self.hand_primitive_close_action)
    open_q = open_buf.to(device=raw.device, dtype=raw.dtype)
    grasp_q = grasp_buf.to(device=raw.device, dtype=raw.dtype)
    if open_q.numel() != mdp.NUM_HAND or grasp_q.numel() != mdp.NUM_HAND:
      raise RuntimeError("Invalid hand primitive open/grasp buffers.")
    close = self._hand_primitive_close(raw)
    close = close * gate.to(device=raw.device, dtype=raw.dtype)
    self.last_hand_primitive_close = close.detach()
    delta = grasp_q - open_q
    left = open_q[:12].unsqueeze(0) + close[:, 0:1] * delta[:12].unsqueeze(0)
    right = open_q[12:].unsqueeze(0) + close[:, 1:2] * delta[12:].unsqueeze(0)
    final_hand = torch.cat([left, right], dim=-1)
    primitive_delta = final_hand - open_q.unsqueeze(0)
    self.last_hand_primitive_delta = primitive_delta.detach()
    return final_hand, primitive_delta

  def _project_sampled_hand_to_eigen(
    self, sampled: torch.Tensor, mean: torch.Tensor
  ) -> torch.Tensor:
    """Keep the *executed* hand action in the eigengrasp subspace, not just the mean.

    PPO samples its exploration noise in all 40 finger dims, so projecting only the mean
    would leave the policy still searching raw joint space -- which is the thing the
    subspace is meant to stop. This mirrors _project_sampled_hand_to_primitive, which
    solves the same problem for the 2-d primitive parameterisation.

    The projection is applied to the deviation from the base hand pose, since the subspace
    describes finger *residuals*. The basis rows are orthonormal, so B.T @ B is the
    orthogonal projector onto their span.
    """
    if self.hand_eigen_k <= 0:
      return sampled
    base = cast(torch.Tensor, self.last_base_action).to(
      device=sampled.device, dtype=sampled.dtype
    )
    if base.shape[0] != sampled.shape[0]:
      return sampled
    basis = self.hand_eigen_basis.to(device=sampled.device, dtype=sampled.dtype)
    base_hand = base[:, mdp.NUM_BODY :]
    delta = sampled[:, mdp.NUM_BODY :] - base_hand
    delta = (delta @ basis.transpose(0, 1)) @ basis
    projected = sampled.clone()
    projected[:, mdp.NUM_BODY :] = base_hand + delta
    # dims the mask disables must keep following the mean, exactly as elsewhere
    active = self.residual_mask[mdp.NUM_BODY :].to(
      device=sampled.device, dtype=torch.bool
    )
    projected[:, mdp.NUM_BODY :] = torch.where(
      active.unsqueeze(0),
      projected[:, mdp.NUM_BODY :],
      mean[:, mdp.NUM_BODY :],
    )
    return projected

  def _project_sampled_hand_to_primitive(
    self, sampled: torch.Tensor, mean: torch.Tensor
  ) -> torch.Tensor:
    if not self.hand_primitive_enabled:
      return sampled
    open_buf = cast(torch.Tensor, self.hand_primitive_open_action)
    close_buf = cast(torch.Tensor, self.hand_primitive_close_action)
    delta_buf = cast(torch.Tensor, self.hand_primitive_delta_action)
    proxy_ids_buf = cast(torch.Tensor, self.hand_primitive_proxy_ids)
    proxy_values_buf = cast(torch.Tensor, self.hand_primitive_proxy_values)
    open_q = open_buf.to(device=sampled.device, dtype=sampled.dtype)
    close = close_buf.to(device=sampled.device, dtype=sampled.dtype)
    delta = delta_buf.to(device=sampled.device, dtype=sampled.dtype)
    proxy_ids = proxy_ids_buf.to(device=sampled.device)
    proxy_values = proxy_values_buf.to(device=sampled.device, dtype=sampled.dtype)
    if close.numel() != mdp.NUM_HAND or proxy_ids.numel() != 2:
      return sampled
    proxy_action = sampled[:, mdp.NUM_BODY :].index_select(1, proxy_ids)
    if self.hand_primitive_grail_enabled and open_q.numel() == mdp.NUM_HAND:
      proxy_open = open_q.index_select(0, proxy_ids)
      proxy_action = proxy_action - proxy_open.unsqueeze(0)
    denom = torch.where(
      proxy_values.abs() > 1.0e-6,
      proxy_values,
      torch.ones_like(proxy_values),
    )
    ratio = (proxy_action / denom.unsqueeze(0)).clamp(0.0, 1.0)
    ratio = torch.nan_to_num(ratio, nan=0.0, posinf=1.0, neginf=0.0)
    if self.hand_primitive_hard_enabled:
      ratio = (ratio >= float(self.hand_primitive_hard_threshold)).to(dtype=ratio.dtype)
    if self.hand_primitive_grail_enabled and open_q.numel() == mdp.NUM_HAND:
      projected_hand = torch.cat(
        [
          open_q[:12].unsqueeze(0) + ratio[:, 0:1] * delta[:12].unsqueeze(0),
          open_q[12:].unsqueeze(0) + ratio[:, 1:2] * delta[12:].unsqueeze(0),
        ],
        dim=-1,
      )
    else:
      projected_hand = torch.cat(
        [
          ratio[:, 0:1] * close[:12].unsqueeze(0),
          ratio[:, 1:2] * close[12:].unsqueeze(0),
        ],
        dim=-1,
      )
    projected = sampled.clone()
    projected[:, mdp.NUM_BODY :] = projected_hand.to(projected.dtype)
    active = self.residual_mask[mdp.NUM_BODY :].to(
      device=sampled.device, dtype=torch.bool
    )
    projected[:, mdp.NUM_BODY :] = torch.where(
      active.unsqueeze(0),
      projected[:, mdp.NUM_BODY :],
      mean[:, mdp.NUM_BODY :],
    )
    return projected

  def _init_hand_eigen(self) -> None:
    """Optionally restrict the 40 finger residuals to a K-dim eigengrasp subspace.

    HAND_EIGEN_K=0 (default) keeps today's behaviour exactly: the head's 40 outputs are the
    40 finger residuals. With K>0 only the first K of those outputs are read, as coordinates
    in the PCA basis fitted over 405,603 retargeted frames, and the remaining 40-K head
    outputs go unused. The head's shape is deliberately untouched so a checkpoint trained
    with one K still loads under another.

    Scaling. Latent k is scaled by ``clip * ||u_k||_1``, which is exactly how far the
    unconstrained +-clip-per-joint cube reaches along PC k, and the decoded residual is then
    clamped per joint to the same +-clip. So the reachable set is the baseline's own action
    set intersected with the subspace -- strictly a restriction, never a change in authority.
    That matters: an authority difference would confound the very comparison this is for.
    """
    self.hand_eigen_k = 0
    self.hand_eigen_scale_mult = 1.0
    raw_k = os.environ.get("HAND_EIGEN_K", "").strip()
    if not raw_k:
      return
    k = int(raw_k)
    if k == 0:
      return
    if not 1 <= k <= mdp.NUM_HAND:
      raise ValueError(f"HAND_EIGEN_K must be within [0, {mdp.NUM_HAND}], got {k}")
    self.hand_eigen_k = k
    self.hand_eigen_scale_mult = float(os.environ.get("HAND_EIGEN_SCALE", "1.0"))
    path = os.environ.get("HAND_EIGEN_NPZ", "").strip() or str(
      Path(__file__).resolve().parent / "wuji_eigengrasp.npz"
    )
    if not Path(path).is_file():
      raise FileNotFoundError(
        f"HAND_EIGEN_K={k} needs the eigengrasp basis but {path} does not exist. "
        "It ships with the mjlab overlay; re-run tools/setup/apply_mjlab_overlay.py."
      )
    data = np.load(path)
    comp = np.asarray(data["components"], dtype=np.float64)
    if comp.shape != (mdp.NUM_HAND, mdp.NUM_HAND):
      raise ValueError(
        f"{path} components has shape {comp.shape}, expected "
        f"({mdp.NUM_HAND}, {mdp.NUM_HAND})"
      )
    basis = comp[:k]
    scale = self.residual_action_clip * np.abs(basis).sum(1) * self.hand_eigen_scale_mult
    # persistent=False: these follow .to(device) but stay out of the state_dict, so turning
    # the projection on or off never makes a saved checkpoint unloadable.
    self.register_buffer(
      "hand_eigen_basis", torch.as_tensor(basis, dtype=torch.float32), persistent=False
    )
    self.register_buffer(
      "hand_eigen_scale", torch.as_tensor(scale, dtype=torch.float32), persistent=False
    )
    evr = float(np.asarray(data["explained_variance_ratio"])[:k].sum())
    n_frames = int(data["n_frames"]) if "n_frames" in data.files else -1
    print(
      f"[ResidualInteractActor] HAND_EIGEN_K {k}/{mdp.NUM_HAND} finger dims -- "
      f"residual projected onto eigengrasp subspace, {evr * 100:.1f}% of the finger "
      f"variance over {n_frames} frames, latent scale "
      f"{self.hand_eigen_scale.min():.3f}..{self.hand_eigen_scale.max():.3f} rad "
      f"(clip {self.residual_action_clip}, mult {self.hand_eigen_scale_mult}), basis {path}"
    )

  def _hand_eigen_residual(self, hand_raw: torch.Tensor) -> torch.Tensor:
    """Decode the first K head outputs as eigengrasp latents into a NUM_HAND residual."""
    basis = self.hand_eigen_basis.to(device=hand_raw.device, dtype=hand_raw.dtype)
    scale = self.hand_eigen_scale.to(device=hand_raw.device, dtype=hand_raw.dtype)
    latent = torch.tanh(hand_raw[:, : self.hand_eigen_k]) * scale.unsqueeze(0)
    hand_residual = latent @ basis
    # Respect the baseline's per-joint clip by shrinking along the direction rather than
    # clamping per joint. Clamping would push the result OUT of the subspace (measured at
    # 0.72 rad off-span at saturation), destroying the property this whole change exists to
    # create; a scalar shrink keeps it exactly in-span. Together with the latent scale above
    # this makes the reachable set the baseline's own +-clip cube intersected with the
    # subspace -- verified to reproduce the cube exactly when k == NUM_HAND.
    peak = hand_residual.abs().amax(dim=-1, keepdim=True)
    shrink = (self.residual_action_clip / peak.clamp_min(1e-9)).clamp(max=1.0)
    return hand_residual * shrink

  def _compute_mean(self, obs: TensorDict) -> torch.Tensor:
    global_residual_gate = self._global_residual_ramp(obs)
    if self.residual_arch == "latent_token_residual":
      if not isinstance(self.base_tracker, OfficialSonicONNX53Actor):
        raise RuntimeError(
          "latent_token_residual requires OfficialSonicONNX53Actor base tracker."
        )
      tokens, body_hist = self.base_tracker.token_and_body_history(obs)
      with torch.no_grad():
        base_body = self.base_tracker.decode_body(tokens, body_hist)
      zero_hand = self.base_tracker.hand_base(
        base_body.shape[0], base_body.device, base_body.dtype
      )
      tracker_action = torch.cat([base_body.detach(), zero_hand], dim=-1)
      bc_hand = self._hand_bc_action(obs, tracker_action)
      hand_base_gate = self._reference_frame_gate(obs, self.hand_bc_base_start_frame)
      hand_residual_gate = self._reference_frame_gate(
        obs, self.hand_residual_start_frame
      )
      hand_residual_gate = hand_residual_gate * global_residual_gate
      self.last_hand_control_gate = hand_residual_gate.detach()
      if self.fixed_hand_action_enabled:
        fixed_hand_action = cast(torch.Tensor, self.fixed_hand_action)
        fixed_hand_action = fixed_hand_action.to(
          device=zero_hand.device, dtype=zero_hand.dtype
        )
        base_hand = fixed_hand_action.unsqueeze(0)
        base_hand = base_hand.expand(base_body.shape[0], -1)
      else:
        if bc_hand is None:
          base_hand = zero_hand
        else:
          bc_hand = bc_hand.to(device=zero_hand.device, dtype=zero_hand.dtype)
          base_hand = torch.where(hand_base_gate > 0.0, bc_hand, zero_hand)
      base_action = torch.cat([base_body.detach(), base_hand], dim=-1)

      features = self._residual_features(obs, base_action)
      raw = self.residual_mlp(features)
      token_dim = OfficialSonicONNX53Actor.token_dim
      token_raw = raw[:, :token_dim]
      if self.fixed_hand_action_enabled:
        hand_raw = torch.zeros(
          base_body.shape[0],
          mdp.NUM_HAND,
          device=base_body.device,
          dtype=base_body.dtype,
        )
      elif self.hand_primitive_enabled:
        hand_raw = raw[:, token_dim : token_dim + 2]
      elif self.split_hand_net:
        hand_raw = self.hand_mlp(self._hand_features(obs, base_action))
      else:
        hand_raw = raw[:, token_dim : token_dim + mdp.NUM_HAND]
      token_residual = torch.tanh(token_raw) * self.token_residual_clip
      token_residual = token_residual * self.token_residual_gain
      token_residual = self._lowpass_residual(
        "_prev_token_residual",
        token_residual,
        global_residual_gate,
      )

      final_body = self.base_tracker.decode_body(tokens + token_residual, body_hist)
      if self.hand_primitive_grail_enabled:
        primitive_hand, primitive_delta = self._hand_primitive_interpolate(
          hand_raw, hand_residual_gate
        )
        hand_mask = self.residual_mask[mdp.NUM_BODY :].to(
          device=primitive_hand.device, dtype=torch.bool
        )
        final_hand = torch.where(
          hand_mask.unsqueeze(0),
          primitive_hand,
          base_hand,
        )
        hand_residual = final_hand - base_hand
        self.last_hand_primitive_delta = (
          final_hand - primitive_hand + primitive_delta
        ).detach()
      elif self.hand_primitive_enabled:
        hand_residual = self._hand_primitive_delta(hand_raw)
        hand_mask = self.residual_mask[mdp.NUM_BODY :].to(
          device=hand_residual.device, dtype=hand_residual.dtype
        )
        hand_gain = self.residual_action_gain[mdp.NUM_BODY :].to(
          device=hand_residual.device, dtype=hand_residual.dtype
        )
        hand_residual = hand_mask.unsqueeze(0) * hand_residual
        hand_residual = hand_residual * hand_residual_gate.to(
          device=hand_residual.device, dtype=hand_residual.dtype
        )
        final_hand = base_hand + hand_gain.unsqueeze(0) * hand_residual
      else:
        if self.hand_eigen_k > 0:
          hand_residual = self._hand_eigen_residual(hand_raw)
        else:
          hand_residual = torch.tanh(hand_raw) * self.residual_action_clip
        hand_mask = self.residual_mask[mdp.NUM_BODY :].to(
          device=hand_residual.device, dtype=hand_residual.dtype
        )
        hand_gain = self.residual_action_gain[mdp.NUM_BODY :].to(
          device=hand_residual.device, dtype=hand_residual.dtype
        )
        hand_residual = hand_mask.unsqueeze(0) * hand_residual
        hand_residual = hand_residual * hand_residual_gate.to(
          device=hand_residual.device, dtype=hand_residual.dtype
        )
        final_hand = base_hand + hand_gain.unsqueeze(0) * hand_residual
      if self.fixed_hand_action_enabled:
        final_hand = base_hand
        hand_residual = torch.zeros_like(hand_residual)

      final = torch.cat([final_body, final_hand], dim=-1)
      decoder_body_delta = final_body - base_body.detach()
      residual = torch.cat([decoder_body_delta, hand_residual], dim=-1)
    elif self.residual_arch in {"astra_hidden_residual", "astra_ref_edit"}:
      if not isinstance(self.base_tracker, ASTRAONNX29BodyActor):
        raise RuntimeError("ASTRA residual architectures require ASTRAONNX29BodyActor.")
      hidden, base_body = self.base_tracker.hidden_and_body_action(obs)
      zero_hand = self.base_tracker.hand_base(
        base_body.shape[0], base_body.device, base_body.dtype
      )
      tracker_action = torch.cat([base_body.detach(), zero_hand], dim=-1)
      bc_hand = self._hand_bc_action(obs, tracker_action)
      hand_base_gate = self._reference_frame_gate(obs, self.hand_bc_base_start_frame)
      hand_residual_gate = self._reference_frame_gate(
        obs, self.hand_residual_start_frame
      )
      hand_residual_gate = hand_residual_gate * global_residual_gate
      self.last_hand_control_gate = hand_residual_gate.detach()
      if self.fixed_hand_action_enabled:
        fixed_hand_action = cast(torch.Tensor, self.fixed_hand_action)
        fixed_hand_action = fixed_hand_action.to(
          device=zero_hand.device, dtype=zero_hand.dtype
        )
        base_hand = fixed_hand_action.unsqueeze(0).expand(base_body.shape[0], -1)
      elif bc_hand is None:
        base_hand = zero_hand
      else:
        bc_hand = bc_hand.to(device=zero_hand.device, dtype=zero_hand.dtype)
        base_hand = torch.where(hand_base_gate > 0.0, bc_hand, zero_hand)
      base_action = torch.cat([base_body.detach(), base_hand], dim=-1)

      features = self._residual_features(obs, base_action)
      raw = self.residual_mlp(features)
      token_dim = (
        ASTRAONNX29BodyActor.hidden_dim
        if self.residual_arch == "astra_hidden_residual"
        else int(self.astra_ref_edit_indices.numel())
      )
      token_raw = raw[:, :token_dim]
      if self.fixed_hand_action_enabled:
        hand_raw = torch.zeros(
          base_body.shape[0],
          mdp.NUM_HAND,
          device=base_body.device,
          dtype=base_body.dtype,
        )
      elif self.hand_primitive_enabled:
        hand_raw = raw[:, token_dim : token_dim + 2]
      elif self.split_hand_net:
        hand_raw = self.hand_mlp(self._hand_features(obs, base_action))
      else:
        hand_raw = raw[:, token_dim : token_dim + mdp.NUM_HAND]

      residual_clip = (
        self.token_residual_clip
        if self.residual_arch == "astra_hidden_residual"
        else self.ref_edit_clip
      )
      token_residual = torch.tanh(token_raw) * residual_clip
      token_residual = token_residual * self.token_residual_gain
      token_residual = self._lowpass_residual(
        "_prev_token_residual",
        token_residual,
        global_residual_gate,
      )
      if self.residual_arch == "astra_hidden_residual":
        candidate_body = self.base_tracker.decode_hidden_to_mjlab_body_action(
          hidden + token_residual
        )
      else:
        ref_edit_full = torch.zeros(
          token_residual.shape[0],
          mdp.ASTRA_OBS_DIM,
          device=token_residual.device,
          dtype=token_residual.dtype,
        )
        edit_indices = cast(torch.Tensor, self.astra_ref_edit_indices).to(
          device=token_residual.device
        )
        ref_edit_full.scatter_(
          1,
          edit_indices.unsqueeze(0).expand(token_residual.shape[0], -1),
          token_residual,
        )
        candidate_body = self.base_tracker.body_action_from_edited_obs(
          obs,
          ref_edit_full,
        )
      final_body = _mask_decoded_body_delta(
        base_body.detach(),
        candidate_body,
        self.residual_mask,
      )

      if self.hand_primitive_grail_enabled:
        primitive_hand, primitive_delta = self._hand_primitive_interpolate(
          hand_raw, hand_residual_gate
        )
        hand_mask = self.residual_mask[mdp.NUM_BODY :].to(
          device=primitive_hand.device, dtype=torch.bool
        )
        final_hand = torch.where(
          hand_mask.unsqueeze(0),
          primitive_hand,
          base_hand,
        )
        hand_residual = final_hand - base_hand
        self.last_hand_primitive_delta = (
          final_hand - primitive_hand + primitive_delta
        ).detach()
      elif self.hand_primitive_enabled:
        hand_residual = self._hand_primitive_delta(hand_raw)
        hand_mask = self.residual_mask[mdp.NUM_BODY :].to(
          device=hand_residual.device, dtype=hand_residual.dtype
        )
        hand_gain = self.residual_action_gain[mdp.NUM_BODY :].to(
          device=hand_residual.device, dtype=hand_residual.dtype
        )
        hand_residual = hand_mask.unsqueeze(0) * hand_residual
        hand_residual = hand_residual * hand_residual_gate.to(
          device=hand_residual.device, dtype=hand_residual.dtype
        )
        final_hand = base_hand + hand_gain.unsqueeze(0) * hand_residual
      else:
        if self.hand_eigen_k > 0:
          hand_residual = self._hand_eigen_residual(hand_raw)
        else:
          hand_residual = torch.tanh(hand_raw) * self.residual_action_clip
        hand_mask = self.residual_mask[mdp.NUM_BODY :].to(
          device=hand_residual.device, dtype=hand_residual.dtype
        )
        hand_gain = self.residual_action_gain[mdp.NUM_BODY :].to(
          device=hand_residual.device, dtype=hand_residual.dtype
        )
        hand_residual = hand_mask.unsqueeze(0) * hand_residual
        hand_residual = self._lowpass_residual(
          "_prev_hand_residual",
          hand_residual,
          hand_residual_gate,
        )
        final_hand = base_hand + hand_gain.unsqueeze(0) * hand_residual
      if self.fixed_hand_action_enabled:
        final_hand = base_hand
        hand_residual = torch.zeros_like(hand_residual)

      final = torch.cat([final_body, final_hand], dim=-1)
      decoder_body_delta = final_body - base_body.detach()
      residual = torch.cat([decoder_body_delta, hand_residual], dim=-1)
    else:
      base_action = self._tracker_action(obs)
      bc_hand = self._hand_bc_action(obs, base_action)
      if bc_hand is not None:
        base_action = torch.cat(
          [
            base_action[:, : mdp.NUM_BODY],
            bc_hand.to(device=base_action.device, dtype=base_action.dtype),
          ],
          dim=-1,
        )

    if self.residual_arch == "frame_split":
      frame_t = self.obs_normalizer(self._frame_t(obs))
      body_residual = torch.tanh(self.body_head(frame_t)) * self.residual_action_clip
      hand_residual = torch.tanh(self.hand_head(frame_t)) * self.residual_action_clip
      body_gain = self.residual_action_gain[: mdp.NUM_BODY].to(
        device=body_residual.device, dtype=body_residual.dtype
      )
      body_mask = self.residual_mask[: mdp.NUM_BODY].to(
        device=body_residual.device, dtype=body_residual.dtype
      )
      hand_gain = self.residual_action_gain[mdp.NUM_BODY :].to(
        device=hand_residual.device, dtype=hand_residual.dtype
      )
      hand_mask = self.residual_mask[mdp.NUM_BODY :].to(
        device=hand_residual.device, dtype=hand_residual.dtype
      )
      body_residual = body_mask.unsqueeze(0) * body_residual
      hand_residual = hand_mask.unsqueeze(0) * hand_residual
      body_residual = self._lowpass_residual(
        "_prev_body_residual",
        body_residual,
        global_residual_gate,
      )
      hand_gate = self._reference_frame_gate(obs, self.hand_residual_start_frame)
      hand_gate = hand_gate * global_residual_gate
      self.last_hand_control_gate = hand_gate.detach()
      hand_residual = self._lowpass_residual(
        "_prev_hand_residual",
        hand_residual,
        hand_gate,
      )
      final_body = (
        base_action[:, : mdp.NUM_BODY] + body_gain.unsqueeze(0) * body_residual
      )
      final_hand = (
        base_action[:, mdp.NUM_BODY :] + hand_gain.unsqueeze(0) * hand_residual
      )
      final = torch.cat([final_body, final_hand], dim=-1)
      residual = torch.cat([body_residual, hand_residual], dim=-1)
      token_residual = torch.zeros(
        base_action.shape[0], OfficialSonicONNX53Actor.token_dim, device=final.device
      )
      decoder_body_delta = final_body - base_action[:, : mdp.NUM_BODY]
    elif self.residual_arch == "feature_mlp":
      features = self._residual_features(obs, base_action)
      if self.split_hand_net:
        body_residual = torch.tanh(self.residual_mlp(features))
        body_residual = body_residual * self.residual_action_clip
        hand_raw = self.hand_mlp(self._hand_features(obs, base_action))
        if self.hand_eigen_k > 0:
          hand_residual = self._hand_eigen_residual(hand_raw)
        else:
          hand_residual = torch.tanh(hand_raw) * self.residual_action_clip
        body_mask = self.residual_mask[: mdp.NUM_BODY].to(
          device=body_residual.device, dtype=body_residual.dtype
        )
        hand_mask = self.residual_mask[mdp.NUM_BODY :].to(
          device=hand_residual.device, dtype=hand_residual.dtype
        )
        body_gain = self.residual_action_gain[: mdp.NUM_BODY].to(
          device=body_residual.device, dtype=body_residual.dtype
        )
        hand_gain = self.residual_action_gain[mdp.NUM_BODY :].to(
          device=hand_residual.device, dtype=hand_residual.dtype
        )
        body_residual = body_mask.unsqueeze(0) * body_residual
        hand_residual = hand_mask.unsqueeze(0) * hand_residual
        body_residual = self._lowpass_residual(
          "_prev_body_residual",
          body_residual,
          global_residual_gate,
        )
        hand_gate = self._reference_frame_gate(obs, self.hand_residual_start_frame)
        hand_gate = hand_gate * global_residual_gate
        self.last_hand_control_gate = hand_gate.detach()
        hand_residual = self._lowpass_residual(
          "_prev_hand_residual",
          hand_residual,
          hand_gate,
        )
        final_body = (
          base_action[:, : mdp.NUM_BODY] + body_gain.unsqueeze(0) * body_residual
        )
        final_hand = (
          base_action[:, mdp.NUM_BODY :] + hand_gain.unsqueeze(0) * hand_residual
        )
        final = torch.cat([final_body, final_hand], dim=-1)
        residual = torch.cat([body_residual, hand_residual], dim=-1)
      else:
        residual = self.residual_mlp(features)
        residual = torch.tanh(residual) * self.residual_action_clip
        residual = residual * self.residual_mask.to(
          device=residual.device, dtype=residual.dtype
        ).unsqueeze(0)
        body_gate = global_residual_gate.to(
          device=residual.device, dtype=residual.dtype
        )
        hand_gate = self._reference_frame_gate(obs, self.hand_residual_start_frame)
        hand_gate = hand_gate.to(device=residual.device, dtype=residual.dtype)
        gate = torch.cat(
          [
            body_gate.expand(-1, mdp.NUM_BODY),
            (body_gate * hand_gate).expand(-1, mdp.NUM_HAND),
          ],
          dim=-1,
        )
        residual = self._lowpass_residual("_prev_body_residual", residual, gate)
        self.last_hand_control_gate = (body_gate * hand_gate).detach()
        action_gain = self.residual_action_gain.to(
          device=residual.device, dtype=residual.dtype
        ).unsqueeze(0)
        final = base_action + action_gain * residual
      token_residual = torch.zeros(
        base_action.shape[0], OfficialSonicONNX53Actor.token_dim, device=final.device
      )
      decoder_body_delta = final[:, : mdp.NUM_BODY] - base_action[:, : mdp.NUM_BODY]
    elif self.residual_arch not in {
      "latent_token_residual",
      "astra_hidden_residual",
      "astra_ref_edit",
    }:
      raise RuntimeError(f"Unknown residual_arch: {self.residual_arch}")
    if self.final_action_clip is not None:
      final = torch.clamp(final, -self.final_action_clip, self.final_action_clip)
      decoder_body_delta = final[:, : mdp.NUM_BODY] - base_action[:, : mdp.NUM_BODY]
      residual = torch.cat(
        [decoder_body_delta, final[:, mdp.NUM_BODY :] - base_action[:, mdp.NUM_BODY :]],
        dim=-1,
      )
    if isinstance(self.base_tracker, ASTRAONNX29BodyActor):
      self.base_tracker.sync_last_astra_action_from_mjlab(final[:, : mdp.NUM_BODY])
    self.last_base_action = base_action.detach()
    self.last_residual_mean = residual
    self.last_residual_action = residual.detach()
    self.last_final_action = final.detach()
    self.last_action_mean = final.detach()
    self.last_previous_token_residual = self.last_token_residual.detach()
    self.last_token_residual = token_residual.detach()
    self.last_decoder_body_delta = decoder_body_delta.detach()
    return final

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    del hidden_state
    obs_td = cast(
      TensorDict,
      unpad_trajectories(obs, masks)
      if masks is not None and not self.is_recurrent
      else obs,
    )
    mean = self._compute_mean(obs_td)
    self.last_action_mean = mean.detach()
    self._sanitize_distribution_std_()
    if stochastic_output:
      self.distribution.update(mean)
      sampled = self.distribution.sample()
      active = self.residual_mask.to(device=sampled.device, dtype=torch.bool)
      if self.fixed_hand_action_enabled:
        active = active.clone()
        active[mdp.NUM_BODY :] = False
      sampled = torch.where(active.unsqueeze(0), sampled, mean)
      gate = getattr(self, "last_hand_control_gate", None)
      if (
        isinstance(gate, torch.Tensor)
        and gate.shape[0] == sampled.shape[0]
        and self.hand_residual_start_frame > 0
      ):
        gate = gate.to(device=sampled.device, dtype=torch.bool)
        sampled_hand = torch.where(
          gate,
          sampled[:, mdp.NUM_BODY :],
          mean[:, mdp.NUM_BODY :],
        )
        sampled = sampled.clone()
        sampled[:, mdp.NUM_BODY :] = sampled_hand
      sampled = self._project_sampled_hand_to_primitive(sampled, mean)
      sampled = self._project_sampled_hand_to_eigen(sampled, mean)
      sampled = self._clamp_sampled_body_delta(sampled)
      sampled = self._clamp_sampled_hand_delta(sampled)
      self.last_final_action = sampled.detach()
      action_gain = self.residual_action_gain.to(
        device=sampled.device, dtype=sampled.dtype
      )
      safe_gain = action_gain.abs().clamp_min(1e-6)
      if self.hand_primitive_grail_enabled:
        safe_gain = safe_gain.clone()
        safe_gain[mdp.NUM_BODY :] = 1.0
      residual = (sampled - self.last_base_action) / safe_gain.unsqueeze(0)
      residual = residual.masked_fill(~active.unsqueeze(0), 0.0)
      self.last_residual_action = residual.detach()
      return sampled
    self.distribution.update(mean)
    return self.distribution.deterministic_output(mean)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    base_action = self._tracker_action(obs)
    if self.residual_arch == "frame_split":
      return self._frame_t(obs)
    return self._residual_features(obs, base_action)

  def reset(
    self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None
  ) -> None:
    del dones, hidden_state

  def get_hidden_state(self) -> HiddenState:
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    del dones

  @property
  def output_mean(self) -> torch.Tensor:
    return self.distribution.mean

  @property
  def output_std(self) -> torch.Tensor:
    return self.distribution.std

  @property
  def output_entropy(self) -> torch.Tensor:
    return self.distribution.entropy

  @property
  def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
    return self.distribution.params

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    return self.distribution.log_prob(outputs)

  def get_kl_divergence(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
  ) -> torch.Tensor:
    return self.distribution.kl_divergence(old_params, new_params)

  def update_normalization(self, obs: TensorDict) -> None:
    if not self.obs_normalization:
      return
    with torch.no_grad():
      if self.residual_arch == "frame_split":
        features = self._frame_t(obs)
      else:
        base_action = self._tracker_action(obs)
        features = torch.cat(self._feature_chunks(obs, base_action), dim=-1)
      cast(Any, self.obs_normalizer).update(features)
      if self.split_hand_net and self.residual_arch != "frame_split":
        cast(Any, self.hand_obs_normalizer).update(features)

  def residual_state_dict(self) -> dict[str, torch.Tensor]:
    prefixes = (
      "residual_mlp.",
      "body_head.",
      "hand_head.",
      "hand_mlp.",
      "hand_obs_normalizer.",
      "distribution.",
      "obs_normalizer.",
    )
    return {
      k: v.detach().cpu()
      for k, v in self.state_dict().items()
      if k.startswith(prefixes)
    }

  def tracker_state_dict(self) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in self.base_tracker.state_dict().items()}
