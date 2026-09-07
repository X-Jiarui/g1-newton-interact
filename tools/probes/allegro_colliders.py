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


def report(model, label_of):
    """Collider extent against visual extent, per body, so the gap can be attributed."""
    flags = wp.to_torch(model.shape_flags).cpu().numpy()
    body = wp.to_torch(model.shape_body).cpu().numpy()
    stype = wp.to_torch(model.shape_type).cpu().numpy()
    scale = wp.to_torch(model.shape_scale).cpu().numpy()
    rows = {}
    for s in range(len(flags)):
        kind = "collider" if flags[s] & COLLIDE else ("visual" if flags[s] & VISIBLE else "other")
        b = int(body[s])
        rows.setdefault(b, {}).setdefault(kind, []).append(
            (int(stype[s]), float(np.max(np.abs(scale[s])))))
    print(f"{'body':>6}  {'colliders':>10}  {'visuals':>9}   max |scale|: collider vs visual")
    print("-" * 74)
    shown = 0
    for b, d in sorted(rows.items()):
        c, v = d.get("collider", []), d.get("visual", [])
        if not c or shown >= 12:
            continue
        cm = max(x[1] for x in c)
        vm = max((x[1] for x in v), default=float("nan"))
        print(f"{b:>6}  {len(c):>10}  {len(v):>9}   {cm:.5f}  vs  {vm:.5f}"
              f"{'   <-- collider larger' if vm == vm and cm > vm * 1.001 else ''}")
        shown += 1


class Example(EX.Example):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        flags = wp.to_torch(self.model.shape_flags)
        n_coll = int(((flags & COLLIDE) != 0).sum())
        n_vis = int(((flags & VISIBLE) != 0).sum())

        if args.draw in ("colliders", "both"):
            flags |= torch_where(flags & COLLIDE, VISIBLE)
        if args.draw == "colliders":
            # Hide anything that is only a render mesh, so nothing hides the geometry that collides.
            flags &= ~torch_where((flags & COLLIDE) == 0, VISIBLE)

        print(f"[allegro-colliders] {n_coll} collider shape(s), {n_vis} visual shape(s); "
              f"drawing '{args.draw}'")
        report(self.model, None)
        self.viewer.set_model(self.model)      # re-register so the flag change takes effect

    @staticmethod
    def create_parser():
        parser = EX.Example.create_parser()
        parser.set_defaults(world_count=1)
        parser.add_argument("--draw", default="colliders",
                            choices=("colliders", "visuals", "both"),
                            help="which shapes to show. 'colliders' is the point of this script")
        return parser


def torch_where(mask, bit):
    import torch
    return torch.where(mask != 0, torch.full_like(mask, bit), torch.zeros_like(mask))


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
