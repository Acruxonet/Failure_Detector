# 多任务、多 episode 阶段验证

本次使用 3 个未参与原阶段规律总结的 LIBERO Spatial 任务，每个任务 2 个成功 rollout，
每个 rollout 均匀抽取 50 个 snapshot。每个 snapshot 保持 proprioception、language、
随机种子和推理噪声不变，只比较 clean / 物体轻微扰动后各 5 个相机视角的 action disagreement。

阶段阈值沿用 task 0 的既有规则，没有根据本批结果重新调整。统计上把 6 个 rollout episode
作为独立重复单位；300 个 snapshot 的比例只作描述，避免把同一轨迹内相关帧当作独立样本。

## 汇总结果

- 全部 snapshot：192/300 的扰动 disagreement 更大。
- manipulation-critical：132/165，平均 paired delta +0.045233。
- non-critical：60/135，平均 paired delta -0.020157。
- 6 个 episode 中，critical 平均 delta 为正：6/6；critical 高于 non-critical：6/6。

详细的逐 snapshot、逐阶段、逐 task 和逐 episode 结果分别保存在同目录 CSV；
`paired_scatter_validation.png` 展示三个任务的 paired scatter，
`episode_stage_effects.png` 展示以 episode 为单位的阶段效应。
