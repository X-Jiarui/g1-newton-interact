# Staged rewards

The task's default reward is a single term. This is the staged replacement, keyed on the dataset's
own contact frame.

    configs/rewards/staged_cf.yaml

## Why

`docs/` in the data pipeline repo has the measurements; the short version is that the reference
became accurate in two different ways and one term cannot spend both:

| | before cf | after cf |
|---|---|---|
| what is real data | the whole arm and hand: fingertips on the human's own contact points (1.6 cm median), palm within 12.3 deg of the human's, hand closing over 30 frames | the object trajectory |
| what is a guess | -- | the arm, which is one frame of solve carried by a constant joint offset |

So the upper body, the hand shape and the fingertips are tracked before cf and released after; the
object term is gated on real contact and carries the second half; the lower body is tracked
throughout.

## The terms

| term | pre | post | what it reads |
|---|---|---|---|
| `staged_lower_tracking` | 1.0 | 1.0 | reference link distance, lower group |
| `staged_upper_tracking` | 1.0 | 0.0* | reference link distance, upper group |
| `staged_hand_pose` | 1.0 | 0.2 | the reference's 20 hand joints -- the cf grasp itself |
| `staged_tip_object` | 1.0 | 0.3 | fingertip to object, only the tips the reference brings within 25 cm |
| `staged_object_tracking` | 0.0 | 1.0 | object position vs reference, gated on fingertip contact |

\* the term's own `post_weight` param, separate from the reward weight above. Raise it off zero if
the arm starts doing something the tracker will not carry.

## Two things borrowed from Omnigrasp, and one not

**Borrowed: the contact gate.** `staged_object_tracking` pays nothing unless a fingertip is actually
within `contact_threshold` of the object. Without it, a policy collects the object reward for
standing next to an object that is where the reference says because nobody moved it.

**Borrowed: `close_hand_flag`.** `staged_tip_object` scores only the tips the *reference* brings near
the object. Without the mask every finger is asked to reach, including the ones the human never
used, and the policy is pushed toward a five-finger cage on objects that were pinched.

**Not borrowed: the stage boundary.** Omnigrasp's final version switches at a fixed frame 30
(`grasp_start_frame`), compared against *absolute motion time* -- so on a clip whose contact is late,
the pregrasp reward never fires at all. Its earlier `humanoid_omnigrab.py` used each sequence's own
contact time, which is what `staged_mdp.cf_local` does here. cf lands at frame 47 on the median clip
and as early as 22, so a fixed boundary would be wrong for most of the set.
