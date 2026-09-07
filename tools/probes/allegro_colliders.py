"""Newton's own Allegro hand example, drawn as the COLLIDERS instead of the visual meshes.

Watching the stock example there is a visible gap between the fingers and the cube, and the picture
alone cannot say why: either the collision geometry is fatter than the mesh you can see -- so the
hand really is touching, just not where it looks like it is -- or the visual is misplaced.

Newton draws shapes flagged VISIBLE, and in a USD robot those are the render meshes; the colliders
are a separate set of shapes carrying COLLIDE_SHAPES. This flips the flags so the window shows what
the solver actually collides. Then the gap either disappears -- it was a visual-versus-collider
offset all along -- or it survives, and it is a real separation the contact never closed.

It also prints the two hulls' sizes side by side, because "the collider is bigger" is a claim that
should come with a number rather than an impression.

    python tools/probes/allegro_colliders.py --viewer viser --world-count 1
    python tools/probes/allegro_colliders.py --viewer viser --world-count 1 --draw both
"""
from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.robot import example_robot_allegro_hand as EX

VISIBLE = int(newton.ShapeFlags.VISIBLE)
COLLIDE = int(newton.ShapeFlags.COLLIDE_SHAPES)


def report(model):
    """Collider size against visual size, per body, measured from the MESH VERTICES.

    Not from `shape_scale`: that is a scale FACTOR and reads 1.0 for every shape here, which makes
    the comparison look identical whatever the geometry is. What decides whether a visible gap is
    real is the extent of the hull the solver collides against the extent of the mesh on screen.
    """
    flags = wp.to_torch(model.shape_flags).cpu().numpy()
    body = wp.to_torch(model.shape_body).cpu().numpy()
    src = model.shape_source

    def extent(s):
        m = src[s] if s < len(src) else None
        v = getattr(m, "vertices", None)
        if v is None:
            return None
        v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
        return None if not len(v) else (float(np.linalg.norm(v, axis=1).max()),
                                        (v.max(0) - v.min(0)))

    rows = {}
    for s in range(len(flags)):
        kind = "collider" if flags[s] & COLLIDE else ("visual" if flags[s] & VISIBLE else "other")
        e = extent(s)
        if e is not None:
            rows.setdefault(int(body[s]), {}).setdefault(kind, []).append(e)

    print("%6s %22s %22s %14s" % ("body", "collider radius / bbox mm", "visual radius / bbox mm",
                                  "collider is"))
    print("-" * 78)
    shown = 0
    for b, d in sorted(rows.items()):
        c, v = d.get("collider", []), d.get("visual", [])
        if not c or not v or shown >= 16:
            continue
        cr = max(x[0] for x in c)
        vr = max(x[0] for x in v)
        cb = max(c, key=lambda x: x[0])[1]
        vb = max(v, key=lambda x: x[0])[1]
        print("%6d %10.2f  %s %10.2f  %s %13s" % (
            b, 1000 * cr, np.round(1000 * cb, 1), 1000 * vr, np.round(1000 * vb, 1),
            ("%+.2f mm" % (1000 * (cr - vr))) if abs(cr - vr) > 1e-6 else "same"))
        shown += 1
    print("\nA collider radius LARGER than the visual is a hand that touches before it looks like "
          "it does:\nthe gap you see is then the difference, not a failure to make contact.")


class Example(EX.Example):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        flags = wp.to_torch(self.model.shape_flags)
        n_coll = int(((flags & COLLIDE) != 0).sum())
        n_vis = int(((flags & VISIBLE) != 0).sum())

        is_coll = (flags & COLLIDE) != 0
        print(f"[allegro-colliders] {n_coll} collider shape(s), {n_vis} visual shape(s); "
              f"drawing '{args.draw}'")
        report(self.model)                      # BEFORE the flags are touched, or every shape reads
                                                # as a collider and the comparison is vacuous
        if args.draw in ("colliders", "both"):
            flags |= bit_where(is_coll, VISIBLE, flags)
        if args.draw == "colliders":
            # Hide anything that is only a render mesh, so nothing occludes what actually collides.
            flags &= ~bit_where(~is_coll, VISIBLE, flags)
        self.viewer.set_model(self.model)      # re-register so the flag change takes effect

    @staticmethod
    def create_parser():
        parser = EX.Example.create_parser()
        parser.set_defaults(world_count=1)
        parser.add_argument("--draw", default="colliders",
                            choices=("colliders", "visuals", "both"),
                            help="which shapes to show. 'colliders' is the point of this script")
        return parser


def bit_where(mask_bool, bit, like):
    """`torch.full_like(bool_tensor, 1)` fills with True, not with the bit value.

    The first version built its masks from a bool tensor, so `~mask` became a logical not and
    `flags &= ~mask` cleared every flag on those shapes instead of one bit. Drawing happened to come
    out right only because VISIBLE == 1; the accounting did not.
    """
    import torch
    return torch.where(mask_bool, torch.full_like(like, bit), torch.zeros_like(like))


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
