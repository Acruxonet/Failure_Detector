# π₀.₅ LIBERO 跨视角不一致性实验

## 实验目的

本实验研究：机器人处于异常状态时，π₀.₅ 在不同相机视角下输出的动作是否会更不一致，以及这种现象是否与任务阶段有关。

## 实验流程

我们在一个 LIBERO pick-and-place 任务中运行了一条成功轨迹，并从轨迹中间 10%–90% 的范围均匀选取 50 个 snapshot。每个 snapshot 包含：

- 原始 clean 状态；
- 将任务黑碗平移 4 cm、旋转 15° 后的 perturbed 状态；
- nominal、±15°、±30° 共 5 个场景相机视角。

所有推理使用同一个冻结的 π₀.₅ checkpoint、相同 seed 和相同扩散噪声。每个配对内除图像外，language、proprioception 和其他推理条件保持一致。

跨视角 disagreement 定义为：5 个视角输出的前 10 步 7D action chunk 之间，两两 L2 距离的平均值。

## 任务阶段划分

阶段根据末端执行器、黑碗和目标盘的几何关系自动划分：

1. `far_approach`：末端仍距离黑碗较远；
2. `close_approach`：末端已接近黑碗，但尚未形成稳定抓取；
3. `grasp_contact`：末端到达黑碗附近，黑碗尚未明显抬升；
4. `lift_transport`：黑碗已经抬起，正在向目标盘搬运；
5. `place_alignment`：黑碗已到达目标盘附近，进入放置和对齐阶段。

其中 `close_approach`、`grasp_contact` 和 `lift_transport` 被合并为 manipulation-critical 阶段。

## 完成的分析

- 绘制 clean disagreement 与 perturbed disagreement 的 paired scatter，并以 `y=x` 为基准；
- 分阶段统计 perturbed disagreement 是否上升；
- 检查 disagreement 下降的反例；
- 使用 MuJoCo segmentation 统计每个视角中黑碗的可见像素，判断遮挡和可见性变化；
- 比较每个相机视角对 action disagreement 的贡献；
- 验证固定 seed、固定噪声下重复推理的 bitwise 一致性。

## 主要发现

- 全部 50 个 snapshot 中，perturbed disagreement 在 29 个 snapshot 上升；
- manipulation-critical 阶段为 21/22 上升，平均配对增量为 `+0.0600`；
- 非关键阶段只有 8/28 上升，平均配对增量为 `-0.0242`；
- 一些反例中，扰动让黑碗从机器人遮挡区域移出，使异常更容易看清，policy 反而产生了更一致的恢复动作。

结果表明，cross-view disagreement 并不是通用的 failure 指标，但在 manipulation-critical 状态上表现出很强的阶段相关信号。

## 多任务验证

随后固定上述阶段规则，在 3 个新的 LIBERO Spatial 任务上各运行 2 个成功 episode，每个 episode 仍取 50 个 snapshot，共 300 对 clean / perturbed 状态。结果如下：

- manipulation-critical 阶段：132/165 个 snapshot 上升，平均配对增量 `+0.04523`；
- non-critical 阶段：60/135 个 snapshot 上升，平均配对增量 `-0.02016`；
- 6/6 个 episode 中，critical 阶段的平均增量均为正，并且均高于同 episode 的 non-critical 阶段；
- 以 episode 为独立重复单位，以上两项单侧精确符号检验均为 `p=0.015625`。

这说明原先观察到的阶段相关规律在本批新任务和新 rollout 上得到一致支持。不过目前仍只有 3 个相近的黑碗放置任务，尚不能视为跨物体、跨任务族的普遍结论。

## 主要结果文件

- `artifacts/pi05_libero_spatial_task0_cross_view_disagreement_50_seed20260811/paired_scatter.png`
- `artifacts/pi05_libero_spatial_task0_cross_view_disagreement_50_seed20260811/paired_scatter_by_stage.png`
- `artifacts/pi05_libero_spatial_task0_cross_view_disagreement_50_seed20260811/counterexample_report.md`
- `artifacts/pi05_libero_spatial_task0_cross_view_disagreement_50_seed20260811/stage_summary.csv`
- `artifacts/pi05_libero_spatial_task0_cross_view_disagreement_50_seed20260811/summary.json`
- `artifacts/pi05_libero_spatial_validation_t1_t2_t6_disagreement_seed20260811/paired_scatter_validation.png`
- `artifacts/pi05_libero_spatial_validation_t1_t2_t6_disagreement_seed20260811/episode_stage_effects.png`
- `artifacts/pi05_libero_spatial_validation_t1_t2_t6_disagreement_seed20260811/VALIDATION_README.md`
- `artifacts/pi05_libero_spatial_validation_t1_t2_t6_disagreement_seed20260811/stage_validation_summary.json`
