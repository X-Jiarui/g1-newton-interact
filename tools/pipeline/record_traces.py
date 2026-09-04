"""Record one qpos trace per (run, clip, checkpoint) and write a policy_gallery manifest.

Runs ON a training box. Every job is discovered from /proc rather than from a launcher script, so
a run started by hand is covered too, and -- more importantly -- the rollout inherits the exact
argv and environ the run is TRAINING with. An eval that re-derives those flags scores the policy
in a scene it never saw (see tools/setup/patch_mjlab.py for what that cost last time).

A mixed run carries `--reference-pkls a,b,c`; one trace per clip is recorded by handing the
rollout that clip alone and switching the PMCP scheduler off, which is the only difference from
the training invocation.

  python tools/pipeline/record_traces.py --out /workspace/traces --jobs 4 --with-ckpt0
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, os, re, subprocess, sys, glob

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="directory for the npz traces + manifest.json")
ap.add_argument("--jobs", type=int, default=2, help="rollouts to run concurrently")
ap.add_argument("--with-ckpt0", action="store_true",
                help="also record model_0.pt -- for a resumed round that is the weights the round "
                     "STARTED from, i.e. the before picture")
ap.add_argument("--runs", default="", help="regex; only runs whose name matches are recorded")
ap.add_argument("--steps", type=int, default=0, help="0 = derive the length from the clip")
A = ap.parse_args()

os.makedirs(A.out, exist_ok=True)
MANIFEST = os.path.join(A.out, "manifest.json")


def argv_of(pid: str) -> list[str]:
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace").split("\0")
    except OSError:
        return []


def environ_of(pid: str) -> dict:
    try:
        raw = open(f"/proc/{pid}/environ", "rb").read().decode(errors="replace")
    except OSError:
        return {}
    return dict(x.split("=", 1) for x in raw.split("\0") if "=" in x)


def flag(a: list[str], name: str, default=None):
    for i, x in enumerate(a):
        if x == name and i + 1 < len(a):
            return a[i + 1]
        if x.startswith(name + "="):
            return x.split("=", 1)[1]
    return default


def discover() -> list[dict]:
    runs = []
    for pid in (p for p in os.listdir("/proc") if p.isdigit()):
        a = argv_of(pid)
        if not any("train_newton.py" in x for x in a):
            continue
        if not a or "python" not in os.path.basename(a[0]):
            continue                                  # our own `bash -c` wrapper, not a trainer
        name = flag(a, "--run-name")
        if not name:
            continue
        if A.runs and not re.search(A.runs, name):
            continue
        cwd = os.path.realpath(f"/proc/{pid}/cwd")
        pkls = flag(a, "--reference-pkls") or flag(a, "--reference-pkl") or ""
        stls = flag(a, "--sdf-objects") or flag(a, "--sdf-object") or ""
        pkls = [x for x in pkls.split(",") if x]
        stls = [x for x in stls.split(",") if x]
        if len(stls) == 1 and len(pkls) > 1:
            stls = stls * len(pkls)
        if not pkls:
            continue
        logroot = flag(a, "--log-root", "logs/rsl_rl")
        d = os.path.join(cwd, logroot, "g1_residual_interact", name)
        its = sorted(int(re.search(r"model_(\d+)\.pt", q).group(1))
                     for q in glob.glob(os.path.join(d, "model_*.pt")))
        if not its:
            continue
        runs.append({
            "name": name, "pid": pid, "cwd": cwd, "python": a[0],
            "xml": os.path.join(cwd, flag(a, "--xml", "")),
            "pkls": pkls, "stls": stls, "ckdir": d, "iters": its,
            "reward_cfg": flag(a, "--reward-cfg"),
            "solver_kwargs": flag(a, "--solver-kwargs"),
            "env": environ_of(pid),
        })
    return sorted(runs, key=lambda r: r["name"])


def jobs_for(run: dict) -> list[dict]:
    # The latest checkpoint is the run as it stands; model_0 of a RESUMED round is the weights the
    # round inherited, which is the only honest "before" for a round that never trained from zero.
    want = [run["iters"][-1]]
    if A.with_ckpt0 and run["iters"][0] == 0 and run["iters"][-1] != 0:
        want.insert(0, 0)
    out = []
    for it in want:
        for i, (pkl, stl) in enumerate(zip(run["pkls"], run["stls"])):
            clip = os.path.splitext(os.path.basename(pkl))[0]
            tag = f"{run['name']}__{clip}__it{it}"
            out.append({"run": run["name"], "clip": clip, "it": it, "pkl": pkl, "stl": stl,
                        "npz": os.path.join(A.out, tag + ".npz"),
                        "log": os.path.join(A.out, tag + ".log"),
                        "xml": run["xml"], "cwd": run["cwd"], "python": run["python"],
                        "ck": os.path.join(run["ckdir"], f"model_{it}.pt"),
                        "reward_cfg": run["reward_cfg"], "solver_kwargs": run["solver_kwargs"],
                        "env": run["env"], "n_clips": len(run["pkls"]), "idx": i})
    return out


def record(j: dict) -> dict:
    name = os.path.basename(j["npz"])[:-4]
    if os.path.exists(j["npz"]):
        print(f"[trace] have {name}", flush=True)
    else:
        env = dict(os.environ)
        env.update(j["env"])
        # One clip at a time, so the scheduler that reshuffles envs across clips must be off.
        env["MIX_PMCP_RULE"] = "none"
        env["MIX_PMCP_EVERY"] = "0"
        env.pop("MIX_PMCP_METRIC", None)
        env.pop("MIX_PMCP_QUOTA", None)
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [j["python"], "tools/run/train_newton.py",
               "--xml", j["xml"], "--reference-pkl", j["pkl"], "--sdf-object", j["stl"],
               "--native-contacts", "--rigid-object-table", "--table-under-object",
               "--num-envs", "1", "--iterations", "0", "--seed", "1",
               "--resume", j["ck"], "--rollout-free-run",
               "--dump-qpos", j["npz"],
               "--log-root", "/tmp/record_traces", "--run-name", "TRACE"]
        if A.steps:
            cmd += ["--rollout-steps", str(A.steps)]
        if j["reward_cfg"]:
            cmd += ["--reward-cfg", j["reward_cfg"]]
        if j["solver_kwargs"]:
            cmd += ["--solver-kwargs", j["solver_kwargs"]]
        print(f"[trace] {name}", flush=True)
        with open(j["log"], "w") as fh:
            subprocess.run(cmd, cwd=j["cwd"], env=env, stdout=fh, stderr=subprocess.STDOUT)
        if not os.path.exists(j["npz"]):
            print(f"[trace] FAILED {name}", flush=True)
            print(subprocess.run(["tail", "-6", j["log"]], capture_output=True,
                                 text=True).stdout, flush=True)
            return {}
    # The rollout's own contact/lift is what makes the picture evidence instead of an opinion.
    nums = ""
    try:
        txt = open(j["log"], errors="replace").read()
        for key in ("Stage/physical_contact", "PhaseA/lift_success"):
            m = re.findall(rf"{re.escape(key)}\D*([0-9.]+)", txt)
            if m:
                nums += f"  {key.split('/')[-1]}={m[-1]}"
    except OSError:
        pass
    return {"name": f"{j['clip']}  it{j['it']}  [{j['run']}]",
            "npz": j["npz"], "xml": j["xml"], "stl": j["stl"],
            "info": f"{j['run']}  iter {j['it']}  clip {j['clip']}{nums}"}


runs = discover()
if not runs:
    raise SystemExit("no live train_newton.py runs matched")
jobs = [j for r in runs for j in jobs_for(r)]
print(f"[trace] {len(runs)} run(s) -> {len(jobs)} trace(s) into {A.out}", flush=True)
for r in runs:
    print(f"        {r['name']}: {len(r['pkls'])} clip(s), iters {r['iters'][0]}..{r['iters'][-1]}",
          flush=True)

entries: list[dict] = []
with cf.ThreadPoolExecutor(max_workers=A.jobs) as ex:
    for e in ex.map(record, jobs):
        if e:
            entries.append(e)
            # Written after every trace, so the gallery can be started before the sweep finishes.
            json.dump(sorted(entries, key=lambda x: x["name"]), open(MANIFEST, "w"), indent=1)
print(f"[trace] manifest -> {MANIFEST} ({len(entries)} entries)", flush=True)
