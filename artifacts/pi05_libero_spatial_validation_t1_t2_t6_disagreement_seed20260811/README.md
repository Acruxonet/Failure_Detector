# Deterministic pi0.5 cross-view disagreement experiment

Primary metric: `exec10_mean_pairwise_l2`.
It is the mean pairwise 7D action L2 distance across all 10 camera-view
pairs and the first 10 action steps, after checkpoint postprocessing.

- Clean mean: 0.16391725
- Perturbed mean: 0.17972468
- Mean paired difference: 0.01580743
- Mean perturbed/clean ratio: 1.178646
- Perturbed greater in: 192/300 snapshots
- Paired sign-flip method: fixed_seed_monte_carlo_100000_sign_flips
- Paired sign-flip p (two-sided): 0.00100000
- Paired sign-flip p (perturbed > clean): 0.00059000
- Repeated sample max absolute action difference: 0.0

Raw 50-step and executed 10-step action chunks are in `action_outputs.npz`.
Per-group metrics are in `per_group_metrics.csv`; full settings and all
metric statistics are in `summary.json`.
