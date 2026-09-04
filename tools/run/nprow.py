"""One row per new-physics run, read from the rsl_rl log tail.

Reports the step size alongside the metrics: runs differ in dt, and a row is not interpretable
without saying which it is.

The progress columns replace ep_len/objfar, both of which stopped meaning what they used to:
  frame   Metric/tracking_frame -- the REFERENCE frame the episode reaches, in resampled 50 fps
          frames. Directly comparable to that clip's contact frame cf (65-95 for our set), which
          ep_len is not: ep_len counts control steps and needs the startup offset subtracted.
  h2o     Metric/hand_to_obj_dist, metres. Starts at ~0.52 on an untrained policy. This is the
          single most direct "how far did the approach get" number in the log.
  u005    Metric/hand_to_obj_under_005_frac -- fraction of steps with the hand within 5 cm.
  phys    Episode_Metrics/physical_contact -- fraction of steps in real contact.
  cfmiss  Episode_Termination/tip_cf_miss -- the cf+10 approach kill. Since it was added it takes
          essentially every episode, which is why og_object_far now reads 0.00 everywhere; objfar
          is kept only so an old run stays comparable.

  tipcfR  Episode_Reward/staged_tip_cf -- a REWARD, renamed from the old `tipcf`. Under the old
          name it was repeatedly read as a distance in centimetres and reported as one. It is not.
  tipcfD  Metric/staged_tip_cf_dist, METRES -- the masked fingertip distance to the reference
          grasp pose. This is the number the approach reward is about; ~0.52 untrained.
  arrF    Metric/staged_tip_cf_arrive_frac -- fraction of env-steps spent within 3 cm of the
          reference grasp pose, i.e. how much of the rollout is spent ARRIVED. 0 until the
          approach works at all.
  arrN    Metric/staged_tip_cf_arrive_frame -- the clip-local frame at which that first happened,
          charging envs that never arrived with their own cf. Read against `cf` (Metric/cf_frame):
          below cf is early, at cf means nobody arrived. This is the only "on time" number; the
          reward itself states only WHERE the hand must be, never WHEN.
  cf      Metric/cf_frame -- the clip's contact frame, so arrN has something to be read against.
  dcf     Metric/staged_tip_cf_dist_at_cf, METRES -- the masked fingertip distance sampled ONCE,
          at the first step an episode reaches its own contact frame. Unlike tipcfD it carries no
          floor from the reference's own motion, so it is what the approach should be judged on.
  atcf    Metric/staged_tip_cf_at_cf_frac -- fraction of envs that reached cf and so contribute
          to dcf. A small dcf over 5 % of envs is a different claim from the same over 90 %.
  wfar    Episode_Termination/wrist_target_far -- envs per step killed for the wrist being over
          0.20 m from its per-frame reference. Compare against num_envs / ep_len, the total
          resets per step.
"""
import re, sys, os, glob, time

KEYS = [("rew", r"Mean reward:\s*([-\d.]+)"),
        ("contact", r"Stage/physical_contact:\s*([\d.]+)"),
        ("lift", r"Episode_Metrics/lift_success:\s*([\d.]+)"),
        ("liftA", r"PhaseA/lift_success:\s*([\d.]+)"),
        ("seq", r"PhaseA/sequence_success:\s*([\d.]+)"),
        # --- how far the episode actually gets -------------------------------------------------
        ("frame", r"Metric/tracking_frame:\s*([\d.]+)"),
        ("h2o", r"Metric/hand_to_obj_dist:\s*([\d.]+)"),
        ("u005", r"Metric/hand_to_obj_under_005_frac:\s*([\d.]+)"),
        ("phys", r"Episode_Metrics/physical_contact:\s*([\d.]+)"),
        ("cfmiss", r"Episode_Termination/tip_cf_miss:\s*([\d.]+)"),
        ("tipcfR", r"Episode_Reward/staged_tip_cf:\s*([\d.]+)"),
        ("tipcfD", r"Metric/staged_tip_cf_dist:\s*([\d.]+)"),
        ("dcf", r"Metric/staged_tip_cf_dist_at_cf:\s*([\d.]+)"),
        ("atcf", r"Metric/staged_tip_cf_at_cf_frac:\s*([\d.]+)"),
        ("wfar", r"Episode_Termination/wrist_target_far:\s*([\d.]+)"),
        # Fraction of environments whose table has been dropped; 0.00 when the run does not ask
        # for removal, so the column is safe to read on every run.
        ("tblrm", r"Metric/table_removed:\s*([\d.-]+)"),
        ("arrF", r"Metric/staged_tip_cf_arrive_frac:\s*([\d.]+)"),
        ("arrN", r"Metric/staged_tip_cf_arrive_frame:\s*([\d.]+)"),
        ("cf", r"Metric/cf_frame:\s*([\d.]+)"),
        # --- kept for continuity with older runs ------------------------------------------------
        ("ep_len", r"Mean episode length:\s*([\d.]+)"),
        ("objfar", r"Episode_Termination/og_object_far:\s*([\d.]+)"),
        ("nonfin", r"Mean kl_nonfinite_frac loss:\s*([\d.]+)"),
        ("pen", r"Penetration/mean_mm:\s*([\d.]+)"),
        ("penmax", r"Penetration/max_mm:\s*([\d.]+)"),
        ("p3mm", r"Penetration/frac_over_3mm:\s*([\d.]+)"),
        ("p4mm", r"Penetration/frac_over_4mm:\s*([\d.]+)"),
        ("psamp", r"Penetration/sampled_contacts:\s*([\d.]+)")]

def parse(path):
    """Everything this reporter knows about one run, from the tail of its log."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 400_000))
            tail = f.read().decode("utf-8", "ignore")
    except OSError:
        return None
    it = re.findall(r"Learning iteration\s+(\d+)/(\d+)", tail)
    try:
        with open(path, "rb") as f:
            head = f.read(200_000).decode("utf-8", "ignore")
    except OSError:
        head = ""
    dt = re.search(r"SIM_TIMESTEP ([\d.]+) ms", head)
    dtxt = f"{float(dt.group(1)):.1f}ms" if dt else "5.0ms"
    vals = {}
    for k, pat in KEYS:
        m = re.findall(pat, tail)
        vals[k] = m[-1] if m else "-"
    cur, tot = (it[-1] if it else ("?", "?"))
    return dtxt, cur, tot, vals


def row(path, name=None, alive=None):
    """One formatted line. `alive` overrides the mtime heuristic when the caller knows better."""
    got = parse(path)
    if got is None:
        return f"{(name or os.path.basename(path)[:-4]):<22s}  (unreadable: {path})"
    dtxt, cur, tot, vals = got
    name = name if name is not None else os.path.basename(path)[:-4]
    if alive is None:
        alive = "" if os.path.getmtime(path) > time.time() - 900 else "  DEAD?"
    return (f"{name:<22s} {dtxt:>6s} it {cur:>5s}/{tot:<5s} rew {vals['rew']:>9s} "
          f"contact {vals['contact']:>7s} lift {vals['lift']:>7s} liftA {vals['liftA']:>7s} "
          f"seq {vals['seq']:>6s} frame {vals['frame']:>7s} h2o {vals['h2o']:>6s} "
          f"u005 {vals['u005']:>6s} phys {vals['phys']:>6s} cfmiss {vals['cfmiss']:>7s} "
          f"tipcfD {vals['tipcfD']:>6s} dcf {vals['dcf']:>6s} atcf {vals['atcf']:>6s} "
          f"arrF {vals['arrF']:>6s} arrN {vals['arrN']:>6s} "
          f"cf {vals['cf']:>5s} tipcfR {vals['tipcfR']:>7s} "
          f"ep_len {vals['ep_len']:>7s} objfar {vals['objfar']:>7s} wfar {vals['wfar']:>6s} "
          f"tblrm {vals['tblrm']:>6s} "
            f"nonfin {vals['nonfin']:>6s} | pen {vals['pen']:>6s}mm max {vals['penmax']:>7s} "
            f">3mm {vals['p3mm']:>6s} >4mm {vals['p4mm']:>6s} n {vals['psamp']:>8s}{alive}")


if __name__ == "__main__":
    for _p in sorted(glob.glob(sys.argv[1])):
        print(row(_p))
