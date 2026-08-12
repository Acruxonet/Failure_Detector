# Counterexample and task-stage analysis

Stage classification is kinematic and fixed by these thresholds: far approach
uses end-effector/object distance >11.5 cm; close approach ends at 5.5 cm;
grasp contact requires <3 mm object lift; lift/transport continues until the
object is within 6 cm XY of the plate; the remainder is place alignment.

## Main stage result

- All snapshots: 29/50 have perturbed > clean.
- Manipulation-critical stages: 21/22 have perturbed > clean; mean paired delta +0.060015.
- Noncritical stages: 8/28 have perturbed > clean; mean paired delta -0.024197.
- Exploratory one-sided Fisher exact p for enrichment in critical stages: 1.0331945e-06.

This stage split is exploratory / post-hoc and should be confirmed on new tasks and
rollouts before being treated as a general detector result.

## Original 3/10 counterexamples

- State 8: old delta -0.02486; dense run stage `far_approach`, dense delta -0.02486, visibility gain +0.8%.
- State 29: old delta -0.04566; dense run stage `far_approach`, dense delta -0.04566, visibility gain +6.1%.
- State 64: old delta -0.00785; not resampled; neighboring dense deltas state 63: +0.00282, state 65: +0.00570.

## Dense-run counterexamples (21/50)

| state | stage | delta | pert/clean | visibility gain | clean min-vis view | max RGB-change view |
|---:|---|---:|---:|---:|---|---|
| 8 | far_approach | -0.02486 | 0.803 | +0.8% | yaw_p015 | yaw_m030 |
| 9 | far_approach | -0.02035 | 0.845 | +0.8% | yaw_p015 | yaw_m030 |
| 10 | far_approach | -0.02096 | 0.853 | +0.8% | yaw_p015 | yaw_m030 |
| 17 | far_approach | -0.01268 | 0.929 | +0.8% | yaw_p015 | yaw_m030 |
| 18 | far_approach | -0.01284 | 0.924 | +0.8% | yaw_p015 | yaw_m030 |
| 20 | far_approach | -0.03421 | 0.826 | +0.8% | yaw_p015 | yaw_m030 |
| 21 | far_approach | -0.02104 | 0.887 | +0.8% | yaw_p015 | yaw_m030 |
| 25 | far_approach | -0.02190 | 0.858 | +2.8% | yaw_m030 | yaw_m030 |
| 26 | far_approach | -0.03884 | 0.768 | +3.1% | yaw_m030 | yaw_m030 |
| 27 | far_approach | -0.04106 | 0.775 | +3.3% | yaw_m030 | yaw_m030 |
| 29 | far_approach | -0.04566 | 0.769 | +6.1% | yaw_m030 | nominal |
| 30 | far_approach | -0.01976 | 0.895 | +8.8% | yaw_m030 | nominal |
| 31 | far_approach | -0.08935 | 0.642 | +11.3% | yaw_m030 | nominal |
| 32 | far_approach | -0.15908 | 0.492 | +13.4% | yaw_m030 | nominal |
| 34 | far_approach | -0.07553 | 0.799 | +20.8% | yaw_m030 | nominal |
| 35 | far_approach | -0.05532 | 0.857 | +26.1% | yaw_m030 | nominal |
| 39 | close_approach | -0.07901 | 0.700 | +56.7% | yaw_m030 | nominal |
| 66 | place_alignment | -0.00961 | 0.918 | +61.9% | yaw_m030 | nominal |
| 67 | place_alignment | -0.03620 | 0.721 | +63.8% | yaw_m030 | nominal |
| 69 | place_alignment | -0.08054 | 0.528 | +65.4% | yaw_m030 | nominal |
| 70 | place_alignment | -0.03659 | 0.722 | +66.7% | yaw_m030 | nominal |

The segmentation visibility gain measures how many more target-bowl pixels are
visible after perturbation. Large positive gains in a counterexample indicate
that moving the bowl out from under the robot made the anomaly easier, not harder,
to see; a visually obvious anomaly can produce a more consistent recovery action
and therefore lower cross-view disagreement.
