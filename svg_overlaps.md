# SVG 内部元素疑似重叠/遮挡清单

> 说明：以下结果通过解析 SVG 元素并计算轴对齐包围盒的相交面积得到。文字宽度按字体大小估算，可能存在少量误报；背景矩形、容器包含、连线/箭头等常见合理重叠已被过滤。

- 共检测到 **94** 种不同画面的 SVG 存在疑似重叠。
- 若算上重复文件，共有 **274** 个文件实例涉及这些画面。
- 按类型统计：图形-图形 **36** 处，文字-文字 **61** 处，文字-图形 **260** 处。

## 排序清单（按疑似重叠数量 / 最大重叠面积）

### 1. `leetgpu/images/prefix_sum_blelloch_detail.svg`

- 元素数：219，重复副本数：3，疑似重叠：19 处
- 所在路径：`leetgpu/images/prefix_sum_blelloch_detail.svg`, `leetgpu/website/images/prefix_sum_blelloch_detail.svg`, `public/leetgpu/images/prefix_sum_blelloch_detail.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (30.0, 582.0, 630.0, 670.0) | text `for (i = 0; i < N; i += 2*stri` | 2425.9 |
| text-shape | text `4 对相加 → 2 元素和` | rect (140.0, 128.0, 190.0, 152.0) | 450.0 |
| text-shape | text `2 对相加 → 4 元素和` | rect (140.0, 172.0, 190.0, 196.0) | 450.0 |
| text-shape | text `1 对相加 → 8 元素和（总和）` | rect (140.0, 216.0, 190.0, 240.0) | 450.0 |
| text-shape | text `swap + add：左=右旧值, 右+=左旧值` | rect (140.0, 260.0, 190.0, 284.0) | 450.0 |
| text-shape | text `活跃线程：1→2→4（逐步倍增）` | rect (350.0, 260.0, 400.0, 284.0) | 450.0 |
| text-shape | text `1 对：a[3]↔a[7]` | rect (140.0, 304.0, 190.0, 328.0) | 450.0 |
| text-shape | text `2 对：a[1]↔a[3], a[5]↔a[7]` | rect (140.0, 348.0, 190.0, 372.0) | 450.0 |
| text-shape | text `4 对 → 完成 exclusive scan` | rect (140.0, 392.0, 190.0, 416.0) | 450.0 |
| text-shape | text `swap + add：左=右旧值, 右+=左旧值` | rect (70.0, 260.0, 120.0, 284.0) | 360.0 |
| ... | ... | ... | 还有 9 处 |

### 2. `leetgpu/images/prefix_sum_hillis_steele_detail.svg`

- 元素数：123，重复副本数：3，疑似重叠：13 处
- 所在路径：`leetgpu/images/prefix_sum_hillis_steele_detail.svg`, `leetgpu/website/images/prefix_sum_hillis_steele_detail.svg`, `public/leetgpu/images/prefix_sum_hillis_steele_detail.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (210.0, 160.0, 260.0, 184.0) | text `a₀+a₁+a₂` | 700.0 |
| text-shape | text `lane i += lane(i-1)` | rect (140.0, 122.0, 190.0, 146.0) | 450.0 |
| text-shape | text `lane i += lane(i-2)` | rect (140.0, 160.0, 190.0, 184.0) | 450.0 |
| text-shape | text `lane i += lane(i-4) → 完成！` | rect (140.0, 198.0, 190.0, 222.0) | 450.0 |
| text-text | text `lane i += lane(i-1)` | text `a₀+a₁` | 365.4 |
| text-text | text `lane i += lane(i-2)` | text `a₀+a₁` | 365.4 |
| text-text | text `lane i += lane(i-4) → 完成！` | text `a₀+a₁` | 365.4 |
| text-shape | text `lane i += lane(i-4) → 完成！` | rect (210.0, 198.0, 260.0, 222.0) | 325.8 |
| text-text | text `lane i += lane(i-4) → 完成！` | text `a₀..a₂` | 308.7 |
| text-text | text `offset=1` | text `a₀` | 157.5 |
| ... | ... | ... | 还有 3 处 |

### 3. `leetgpu/images/reduction_warp_shuffle.svg`

- 元素数：88，重复副本数：5，疑似重叠：13 处
- 所在路径：`aiinfra/week4/website/images/reduction_warp_shuffle.svg`, `leetgpu/images/reduction_warp_shuffle.svg`, `leetgpu/website/images/reduction_warp_shuffle.svg`, `public/leetgpu/images/reduction_warp_shuffle.svg`, `public/week4/images/reduction_warp_shuffle.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `lane 0 收到 lane 4 的值 → 1+5=6` | rect (130.0, 208.0, 170.0, 232.0) | 400.0 |
| text-shape | text `lane 0 收到 lane 2 的值 → 6+10=16` | rect (130.0, 248.0, 170.0, 272.0) | 400.0 |
| text-shape | text `lane 0 收到 lane 2 的值 → 6+10=16` | rect (190.0, 248.0, 230.0, 272.0) | 385.0 |
| text-text | text `lane 0 收到 lane 1 的值 → 16+20=36` | text `→ lane 0 持有最终结果` | 366.0 |
| text-shape | text `lane 0 收到 lane 4 的值 → 1+5=6` | rect (190.0, 208.0, 230.0, 232.0) | 305.0 |
| text-shape | text `lane 0 收到 lane 4 的值 → 1+5=6` | rect (70.0, 208.0, 110.0, 232.0) | 200.0 |
| text-shape | text `lane 0 收到 lane 2 的值 → 6+10=16` | rect (70.0, 248.0, 110.0, 272.0) | 200.0 |
| text-shape | text `lane 0 收到 lane 1 的值 → 16+20=36` | rect (70.0, 288.0, 110.0, 312.0) | 200.0 |
| text-text | text `lane 0 收到 lane 4 的值 → 1+5=6` | text `10` | 80.6 |
| text-text | text `lane 0 收到 lane 2 的值 → 6+10=16` | text `20` | 80.6 |
| ... | ... | ... | 还有 3 处 |

### 4. `leetgpu/images/prefix_sum_warp_inclusive_scan.svg`

- 元素数：116，重复副本数：3，疑似重叠：13 处
- 所在路径：`leetgpu/images/prefix_sum_warp_inclusive_scan.svg`, `leetgpu/website/images/prefix_sum_warp_inclusive_scan.svg`, `public/leetgpu/images/prefix_sum_warp_inclusive_scan.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `lane i 收 lane(i-1)，i≥1 才加` | rect (130.0, 136.0, 170.0, 160.0) | 400.0 |
| text-shape | text `lane i 收 lane(i-2)，i≥2 才加` | rect (130.0, 176.0, 170.0, 200.0) | 400.0 |
| text-shape | text `lane i 收 lane(i-4)，i≥4 才加 → 完成` | rect (130.0, 216.0, 170.0, 240.0) | 400.0 |
| text-shape | text `lane i 收 lane(i-4)，i≥4 才加 → 完成` | rect (190.0, 216.0, 230.0, 240.0) | 400.0 |
| text-shape | text `lane i 收 lane(i-1)，i≥1 才加` | rect (190.0, 136.0, 230.0, 160.0) | 305.0 |
| text-shape | text `lane i 收 lane(i-2)，i≥2 才加` | rect (190.0, 176.0, 230.0, 200.0) | 305.0 |
| text-shape | text `lane i 收 lane(i-4)，i≥4 才加 → 完成` | rect (250.0, 216.0, 290.0, 240.0) | 125.0 |
| text-shape | text `lane i 收 lane(i-1)，i≥1 才加` | rect (70.0, 136.0, 110.0, 160.0) | 120.0 |
| text-shape | text `lane i 收 lane(i-2)，i≥2 才加` | rect (70.0, 176.0, 110.0, 200.0) | 120.0 |
| text-shape | text `lane i 收 lane(i-4)，i≥4 才加 → 完成` | rect (70.0, 216.0, 110.0, 240.0) | 120.0 |
| ... | ... | ... | 还有 3 处 |

### 5. `leetgpu/images/prefix_sum_warp_scan.svg`

- 元素数：121，重复副本数：3，疑似重叠：13 处
- 所在路径：`leetgpu/images/prefix_sum_warp_scan.svg`, `leetgpu/website/images/prefix_sum_warp_scan.svg`, `public/leetgpu/images/prefix_sum_warp_scan.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `lane i 收到 lane (i-1) 的值，i≥1 才加` | rect (130.0, 172.0, 170.0, 196.0) | 400.0 |
| text-shape | text `lane i 收到 lane (i-1) 的值，i≥1 才加` | rect (190.0, 172.0, 230.0, 196.0) | 400.0 |
| text-shape | text `lane i 收到 lane (i-2)，i≥2 才加` | rect (130.0, 212.0, 170.0, 236.0) | 320.0 |
| text-shape | text `lane i 收到 lane (i-4)，i≥4 才加 → ` | rect (130.0, 252.0, 170.0, 276.0) | 320.0 |
| text-shape | text `lane i 收到 lane (i-4)，i≥4 才加 → ` | rect (190.0, 252.0, 230.0, 276.0) | 320.0 |
| text-shape | text `lane i 收到 lane (i-2)，i≥2 才加` | rect (190.0, 212.0, 230.0, 236.0) | 280.0 |
| text-shape | text `lane i 收到 lane (i-1) 的值，i≥1 才加` | rect (70.0, 172.0, 110.0, 196.0) | 200.0 |
| text-shape | text `lane i 收到 lane (i-2)，i≥2 才加` | rect (70.0, 212.0, 110.0, 236.0) | 160.0 |
| text-shape | text `lane i 收到 lane (i-4)，i≥4 才加 → ` | rect (70.0, 252.0, 110.0, 276.0) | 160.0 |
| text-shape | text `lane i 收到 lane (i-4)，i≥4 才加 → ` | rect (250.0, 252.0, 290.0, 276.0) | 136.0 |
| ... | ... | ... | 还有 3 处 |

### 6. `public/images/transpose_tiled_process.svg`

- 元素数：371，重复副本数：2，疑似重叠：12 处
- 所在路径：`aiinfra/week1/website/images/transpose_tiled_process.svg`, `public/images/transpose_tiled_process.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (200.0, 222.0, 228.0, 250.0) | text `Block(1,0) 读取此 tile` | 336.0 |
| text-shape | rect (228.0, 222.0, 256.0, 250.0) | text `Block(1,0) 读取此 tile` | 336.0 |
| text-shape | rect (748.0, 222.0, 776.0, 250.0) | text `Block(1,0) 写入此 tile` | 336.0 |
| text-shape | rect (776.0, 222.0, 804.0, 250.0) | text `Block(1,0) 写入此 tile` | 336.0 |
| text-shape | text `Block(1,0) 写入此 tile` | rect (748.0, 222.0, 776.0, 250.0) | 336.0 |
| text-shape | text `Block(1,0) 写入此 tile` | rect (776.0, 222.0, 804.0, 250.0) | 336.0 |
| text-shape | rect (256.0, 222.0, 284.0, 250.0) | text `Block(1,0) 读取此 tile` | 308.4 |
| text-shape | rect (720.0, 222.0, 748.0, 250.0) | text `Block(1,0) 写入此 tile` | 308.4 |
| text-shape | text `Block(1,0) 写入此 tile` | rect (720.0, 222.0, 748.0, 250.0) | 308.4 |
| text-shape | rect (172.0, 222.0, 200.0, 250.0) | text `Block(1,0) 读取此 tile` | 308.4 |
| ... | ... | ... | 还有 2 处 |

### 7. `leetcode/images/palindrome_manacher_example.svg`

- 元素数：105，重复副本数：3，疑似重叠：12 处
- 所在路径：`leetcode/images/palindrome_manacher_example.svg`, `leetcode/website/images/palindrome_manacher_example.svg`, `public/leetcode/images/palindrome_manacher_example.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `5` | text `6` | 67.2 |
| text-shape | rect (230.0, 78.0, 290.0, 100.0) | text `a` | 17.9 |
| text-shape | rect (290.0, 78.0, 350.0, 100.0) | text `c` | 17.9 |
| text-shape | rect (350.0, 78.0, 410.0, 100.0) | text `a` | 17.9 |
| text-shape | rect (410.0, 78.0, 470.0, 100.0) | text `b` | 17.9 |
| text-shape | rect (470.0, 78.0, 530.0, 100.0) | text `a` | 17.9 |
| text-shape | rect (530.0, 78.0, 590.0, 100.0) | text `c` | 17.9 |
| text-shape | rect (590.0, 78.0, 650.0, 100.0) | text `a` | 17.9 |
| text-shape | rect (650.0, 78.0, 710.0, 100.0) | text `b` | 17.9 |
| text-shape | rect (710.0, 78.0, 770.0, 100.0) | text `a` | 17.9 |
| ... | ... | ... | 还有 2 处 |

### 8. `public/leetcode/images/task_scheduler.svg`

- 元素数：83，重复副本数：3，疑似重叠：10 处
- 所在路径：`leetcode/daily/week7/day2/images/task_scheduler.svg`, `leetcode/website/images/task_scheduler.svg`, `public/leetcode/images/task_scheduler.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `t2` | text `idle` | 134.4 |
| text-text | text `t5` | text `idle` | 134.4 |
| text-text | text `t4` | text `B` | 67.2 |
| text-text | text `t6` | text `A` | 67.2 |
| text-text | text `t7` | text `B` | 67.2 |
| text-text | text `t0` | text `A` | 67.2 |
| text-text | text `t1` | text `B` | 67.2 |
| text-text | text `t3` | text `A` | 67.2 |
| text-shape | rect (580.0, 155.0, 592.0, 165.0) | text `任务 A (max_freq=3)` | 43.5 |
| text-shape | rect (660.0, 155.0, 672.0, 165.0) | text `任务 B (max_freq=3)` | 43.5 |

### 9. `leetgpu/images/prefix_sum_block_exclusive_scan.svg`

- 元素数：160，重复副本数：3，疑似重叠：10 处
- 所在路径：`leetgpu/images/prefix_sum_block_exclusive_scan.svg`, `leetgpu/website/images/prefix_sum_block_exclusive_scan.svg`, `public/leetgpu/images/prefix_sum_block_exclusive_scan.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `③ warp 0 对 warp_sums 做 inclusi` | text `warp 0 inclusive scan` | 111.6 |
| text-shape | text `⋮` | rect (194.0, 348.0, 226.0, 366.0) | 28.7 |
| text-shape | text `⋮  warp 2 – 6 同理（各自 inclusive ` | rect (240.0, 126.0, 276.0, 144.0) | 21.6 |
| text-shape | text `⋮  warp 2 – 6 同理（各自 inclusive ` | rect (278.0, 126.0, 314.0, 144.0) | 21.6 |
| text-shape | text `⋮  warp 2 – 6 同理（各自 inclusive ` | rect (316.0, 126.0, 352.0, 144.0) | 21.6 |
| text-shape | text `⋮  warp 2 – 6 同理（各自 inclusive ` | rect (354.0, 126.0, 390.0, 144.0) | 21.6 |
| text-shape | text `⋮  warp 2 – 6 同理（各自 inclusive ` | rect (138.0, 126.0, 170.0, 144.0) | 19.2 |
| text-shape | text `⋮  warp 2 – 6 同理（各自 inclusive ` | rect (172.0, 126.0, 204.0, 144.0) | 19.2 |
| text-shape | text `⋮  warp 2 – 6 同理（各自 inclusive ` | rect (104.0, 126.0, 136.0, 144.0) | 9.6 |
| text-shape | rect (194.0, 320.0, 226.0, 338.0) | text `⋮` | 4.4 |

### 10. `public/week7/images/mini_ai_infra_architecture.svg`

- 元素数：54，重复副本数：2，疑似重叠：8 处
- 所在路径：`aiinfra/week7/website/images/mini_ai_infra_architecture.svg`, `public/week7/images/mini_ai_infra_architecture.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (140.0, 124.0, 800.0, 204.0) | text `检查过期 → set_exception` | 1540.0 |
| text-shape | rect (160.0, 156.0, 340.0, 194.0) | text `get_batch → 加入 running` | 1260.0 |
| text-shape | rect (380.0, 156.0, 560.0, 194.0) | text `forward（锁外执行）` | 1260.0 |
| text-shape | rect (600.0, 156.0, 780.0, 194.0) | text `检查过期 → set_exception` | 1260.0 |
| text-shape | rect (400.0, 320.0, 530.0, 350.0) | text `FlashAttention` | 910.0 |
| text-shape | rect (380.0, 156.0, 560.0, 194.0) | text `get_batch → 加入 running` | 169.4 |
| text-text | text `• 13 道面试题` | text `→ Week 8` | 102.2 |
| text-shape | rect (540.0, 320.0, 670.0, 350.0) | text `FlashAttention` | 47.6 |

### 11. `public/week2/images/multi_stream_overlap.svg`

- 元素数：61，重复副本数：2，疑似重叠：8 处
- 所在路径：`aiinfra/week2/website/images/multi_stream_overlap.svg`, `public/week2/images/multi_stream_overlap.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (100.0, 196.0, 180.0, 226.0) | rect (140.0, 196.0, 340.0, 358.0) | 1200.0 |
| shape-shape | rect (300.0, 240.0, 380.0, 270.0) | rect (140.0, 196.0, 340.0, 358.0) | 1200.0 |
| shape-shape | rect (300.0, 328.0, 380.0, 358.0) | rect (140.0, 196.0, 340.0, 358.0) | 1200.0 |
| text-shape | text `Stream 1` | rect (100.0, 196.0, 180.0, 226.0) | 220.8 |
| text-shape | text `Default` | rect (100.0, 84.0, 200.0, 116.0) | 163.2 |
| text-shape | text `Comp` | rect (140.0, 196.0, 340.0, 358.0) | 80.0 |
| text-shape | text `H2D` | rect (140.0, 196.0, 340.0, 358.0) | 60.0 |
| text-shape | text `D2H` | rect (140.0, 196.0, 340.0, 358.0) | 60.0 |

### 12. `public/week6/images/continuous_batching_timeline.svg`

- 元素数：72，重复副本数：2，疑似重叠：8 处
- 所在路径：`aiinfra/week6/website/images/continuous_batching_timeline.svg`, `public/week6/images/continuous_batching_timeline.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (300.0, 192.0, 420.0, 220.0) | text `prefill+decode 1` | 840.0 |
| text-shape | rect (540.0, 232.0, 660.0, 260.0) | text `prefill+decode 1` | 840.0 |
| text-shape | rect (420.0, 192.0, 540.0, 220.0) | text `prefill+decode 1` | 414.4 |
| text-shape | rect (660.0, 232.0, 780.0, 260.0) | text `prefill+decode 1` | 414.4 |
| text-shape | text `S2 (gen=8)` | rect (60.0, 152.0, 180.0, 180.0) | 154.0 |
| text-shape | text `S3 (gen=3)` | rect (60.0, 192.0, 300.0, 220.0) | 154.0 |
| text-shape | text `S4 (gen=5)` | rect (60.0, 232.0, 540.0, 260.0) | 154.0 |
| text-shape | text `S1 (gen=4)` | rect (60.0, 112.0, 180.0, 140.0) | 154.0 |

### 13. `leetcode/images/bst_example_walkthrough.svg`

- 元素数：67，重复副本数：4，疑似重叠：8 处
- 所在路径：`leetcode/daily/week3/day5/images/bst_example_walkthrough.svg`, `leetcode/images/bst_example_walkthrough.svg`, `leetcode/website/images/bst_example_walkthrough.svg`, `public/leetcode/images/bst_example_walkthrough.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `3 在 5 的右子树却 < 5 ⚠️` | rect (40.0, 228.0, 520.0, 252.0) | 699.8 |
| shape-shape | circle (136.0, 202.0, 164.0, 230.0) | rect (40.0, 204.0, 520.0, 228.0) | 672.0 |
| shape-shape | circle (216.0, 202.0, 244.0, 230.0) | rect (40.0, 204.0, 520.0, 228.0) | 672.0 |
| text-shape | text `3 在 5 的右子树却 < 5 ⚠️` | rect (40.0, 252.0, 520.0, 276.0) | 175.0 |
| shape-shape | circle (136.0, 202.0, 164.0, 230.0) | rect (40.0, 180.0, 520.0, 204.0) | 56.0 |
| shape-shape | circle (136.0, 202.0, 164.0, 230.0) | rect (40.0, 228.0, 520.0, 252.0) | 56.0 |
| shape-shape | circle (216.0, 202.0, 244.0, 230.0) | rect (40.0, 180.0, 520.0, 204.0) | 56.0 |
| shape-shape | circle (216.0, 202.0, 244.0, 230.0) | rect (40.0, 228.0, 520.0, 252.0) | 56.0 |

### 14. `leetgpu/images/flash_attention_tiling.svg`

- 元素数：57，重复副本数：7，疑似重叠：7 处
- 所在路径：`aiinfra/week4/website/images/flash_attention_kernel_tiling.svg`, `aiinfra/week4/website/images/flash_attention_tiling.svg`, `leetgpu/images/flash_attention_tiling.svg`, `leetgpu/website/images/flash_attention_tiling.svg`, `public/leetgpu/images/flash_attention_tiling.svg` ...

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (220.0, 84.0, 400.0, 324.0) | rect (30.0, 304.0, 610.0, 424.0) | 3600.0 |
| shape-shape | rect (30.0, 84.0, 130.0, 324.0) | rect (30.0, 304.0, 610.0, 424.0) | 2000.0 |
| text-shape | rect (30.0, 84.0, 130.0, 324.0) | text `滑动循环（每个 Q tile）：` | 1200.0 |
| text-shape | rect (30.0, 84.0, 130.0, 324.0) | text `for (kv_start = 0; kv_start < ` | 688.0 |
| text-shape | text `逐块滑入 SRAM` | rect (30.0, 304.0, 610.0, 424.0) | 281.9 |
| text-shape | rect (220.0, 84.0, 400.0, 324.0) | text `for (kv_start = 0; kv_start < ` | 192.0 |
| text-text | text `(整个 tile)` | text `load V[kv_start:kv_start+BN][:` | 81.9 |

### 15. `leetgpu/images/int8_kv_cache_attention_overview.svg`

- 元素数：43，重复副本数：3，疑似重叠：7 处
- 所在路径：`leetgpu/images/int8_kv_cache_attention_overview.svg`, `leetgpu/website/images/int8_kv_cache_attention_overview.svg`, `public/leetgpu/images/int8_kv_cache_attention_overview.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `反量化 V_float = V_int8 × v_scale` | rect (360.0, 250.0, 580.0, 360.0) | 1100.0 |
| text-text | text `反量化 V_float = V_int8 × v_scale` | text `shape: [H,1,d] → squeeze → [H,` | 585.0 |
| text-shape | rect (220.0, 148.0, 242.0, 198.0) | text `1 byte/elem（fp32 的 1/4）→ Decod` | 149.6 |
| text-shape | rect (244.0, 148.0, 266.0, 198.0) | text `1 byte/elem（fp32 的 1/4）→ Decod` | 149.6 |
| text-shape | rect (268.0, 148.0, 290.0, 198.0) | text `1 byte/elem（fp32 的 1/4）→ Decod` | 149.6 |
| text-shape | rect (292.0, 148.0, 314.0, 198.0) | text `1 byte/elem（fp32 的 1/4）→ Decod` | 149.6 |
| text-shape | rect (316.0, 148.0, 338.0, 198.0) | text `1 byte/elem（fp32 的 1/4）→ Decod` | 149.6 |

### 16. `public/images/element_wise_memory_bound.svg`

- 元素数：41，重复副本数：2，疑似重叠：7 处
- 所在路径：`aiinfra/week1/website/images/element_wise_memory_bound.svg`, `public/images/element_wise_memory_bound.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (60.0, 445.0, 660.0, 505.0) | text `GEMM 4096³` | 430.8 |
| text-shape | rect (60.0, 445.0, 660.0, 505.0) | text `Compute` | 271.0 |
| text-shape | rect (60.0, 445.0, 660.0, 505.0) | text `192M` | 154.9 |
| text-shape | rect (60.0, 445.0, 660.0, 505.0) | text `137G` | 154.9 |
| text-shape | rect (60.0, 445.0, 660.0, 505.0) | text `715` | 116.2 |
| shape-shape | circle (87.0, 382.0, 103.0, 398.0) | circle (94.0, 374.0, 106.0, 386.0) | 36.0 |
| text-shape | text `算力峰值` | circle (592.0, 142.0, 608.0, 158.0) | 6.4 |

### 17. `public/week7/images/kernel_integration_overview.svg`

- 元素数：67，重复副本数：2，疑似重叠：7 处
- 所在路径：`aiinfra/week7/website/images/kernel_integration_overview.svg`, `public/week7/images/kernel_integration_overview.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (30.0, 66.0, 370.0, 486.0) | text `→ layernorm_kernel` | 90.0 |
| text-shape | rect (30.0, 66.0, 370.0, 486.0) | text `→ cuBLAS / GEMM kernel` | 90.0 |
| text-shape | rect (30.0, 66.0, 370.0, 486.0) | text `→ flash_attention_kernel` | 90.0 |
| text-shape | rect (30.0, 66.0, 370.0, 486.0) | text `(内含 softmax)` | 90.0 |
| text-shape | rect (30.0, 66.0, 370.0, 486.0) | text `→ cuBLAS / GEMM kernel` | 90.0 |
| text-shape | rect (30.0, 66.0, 370.0, 486.0) | text `→ layernorm_kernel` | 90.0 |
| text-shape | rect (30.0, 66.0, 370.0, 486.0) | text `→ cuBLAS` | 90.0 |

### 18. `leetcode/images/palindrome_manacher_symmetry.svg`

- 元素数：63，重复副本数：3，疑似重叠：6 处
- 所在路径：`leetcode/images/palindrome_manacher_symmetry.svg`, `leetcode/website/images/palindrome_manacher_symmetry.svg`, `public/leetcode/images/palindrome_manacher_symmetry.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (740.0, 82.0, 800.0, 122.0) | rect (745.0, 76.0, 925.0, 168.0) | 2200.0 |
| text-shape | rect (740.0, 82.0, 800.0, 122.0) | text `mir = mirror（镜像）` | 450.0 |
| text-shape | rect (740.0, 82.0, 800.0, 122.0) | text `i 关于 mr 的对称点` | 405.0 |
| text-shape | rect (800.0, 82.0, 860.0, 122.0) | text `mir = mirror（镜像）` | 370.0 |
| text-shape | rect (800.0, 82.0, 860.0, 122.0) | text `i 关于 mr 的对称点` | 202.5 |
| text-text | text `已知最远右端回文 [l, r]，中心 mr` | text `已知回文 [l=2, r=10]` | 196.8 |

### 19. `leetcode/images/min_stack_auxiliary.svg`

- 元素数：49，重复副本数：4，疑似重叠：6 处
- 所在路径：`leetcode/daily/week5/day2/images/min_stack_auxiliary.svg`, `leetcode/images/min_stack_auxiliary.svg`, `leetcode/website/images/min_stack_auxiliary.svg`, `public/leetcode/images/min_stack_auxiliary.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `← 7 > 3，复制当前最小` | rect (660.0, 116.0, 900.0, 376.0) | 620.0 |
| text-text | text `← 7 > 3，复制当前最小` | text `KV Cache：缓存"历史 K/V"` | 450.8 |
| text-shape | text `← 3 < 5，更新最小` | rect (660.0, 116.0, 900.0, 376.0) | 450.0 |
| text-shape | text `← 1 < 3，更新最小` | rect (660.0, 116.0, 900.0, 376.0) | 450.0 |
| text-text | text `← 1 < 3，更新最小` | text `O(L·d²)→O(d²)` | 266.8 |
| text-text | text `← 3 < 5，更新最小` | text `最小栈：缓存"栈内最小值"` | 92.8 |

### 20. `leetgpu/images/presum_intra_block_scan.svg`

- 元素数：189，重复副本数：3，疑似重叠：6 处
- 所在路径：`leetgpu/images/presum_intra_block_scan.svg`, `leetgpu/website/images/presum_intra_block_scan.svg`, `public/leetgpu/images/presum_intra_block_scan.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `input → smem` | rect (50.0, 60.0, 82.0, 82.0) | 320.0 |
| text-shape | text `block scan` | rect (50.0, 246.0, 82.0, 268.0) | 200.0 |
| text-shape | text `warp scan` | rect (50.0, 118.0, 82.0, 140.0) | 160.0 |
| text-text | text `wid_offset=1 → warp_id ≥ 1 读 s` | text `+11` | 77.8 |
| text-text | text `wid_offset=2 → warp_id ≥ 2 读 s` | text `+11` | 77.8 |
| text-shape | text `input → smem` | rect (82.0, 60.0, 114.0, 82.0) | 5.0 |

### 21. `leetcode/images/lru_cache_overview.svg`

- 元素数：50，重复副本数：4，疑似重叠：5 处
- 所在路径：`leetcode/daily/week6/day3/images/lru_cache_overview.svg`, `leetcode/images/lru_cache_overview.svg`, `leetcode/website/images/lru_cache_overview.svg`, `public/leetcode/images/lru_cache_overview.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (40.0, 84.0, 180.0, 264.0) | rect (40.0, 220.0, 820.0, 390.0) | 6160.0 |
| text-text | text `value = 指向链表节点` | text `操作流程：` | 464.1 |
| text-shape | rect (40.0, 84.0, 180.0, 264.0) | text `get(key) / put 命中：` | 369.4 |
| text-shape | rect (40.0, 84.0, 180.0, 264.0) | text `→ node*` | 330.0 |
| text-shape | rect (40.0, 84.0, 180.0, 264.0) | text `→ (淘汰)` | 330.0 |

### 22. `leetcode/images/largest_rect_concept.svg`

- 元素数：46，重复副本数：4，疑似重叠：5 处
- 所在路径：`leetcode/daily/week1/day7/images/largest_rect_concept.svg`, `leetcode/images/largest_rect_concept.svg`, `leetcode/website/images/largest_rect_concept.svg`, `public/leetcode/images/largest_rect_concept.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (225.0, 130.0, 265.0, 280.0) | rect (180.0, 155.0, 265.0, 280.0) | 5000.0 |
| shape-shape | rect (225.0, 130.0, 265.0, 280.0) | rect (180.0, 155.0, 265.0, 280.0) | 5000.0 |
| text-shape | rect (225.0, 130.0, 265.0, 280.0) | text `高=5, 跨 2 根 → 面积=10` | 400.0 |
| text-shape | rect (225.0, 130.0, 265.0, 280.0) | text `高=5, 跨 2 根 → 面积=10` | 400.0 |
| text-shape | rect (450.0, 72.0, 590.0, 136.0) | text `R=右第一个更矮` | 236.0 |

### 23. `leetgpu/images/conv2d_naive_redundant_reads.svg`

- 元素数：47，重复副本数：3，疑似重叠：5 处
- 所在路径：`leetgpu/images/conv2d_naive_redundant_reads.svg`, `leetgpu/website/images/conv2d_naive_redundant_reads.svg`, `public/leetgpu/images/conv2d_naive_redundant_reads.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (70.0, 114.0, 160.0, 204.0) | rect (110.0, 114.0, 200.0, 204.0) | 4500.0 |
| text-shape | rect (70.0, 114.0, 160.0, 204.0) | text `t1 邻域` | 162.5 |
| text-shape | text `t0 邻域` | rect (110.0, 114.0, 200.0, 204.0) | 162.5 |
| text-shape | text `t0 邻域` | rect (110.0, 114.0, 160.0, 204.0) | 162.5 |
| text-shape | text `t1 邻域` | rect (110.0, 114.0, 160.0, 204.0) | 162.5 |

### 24. `leetcode/images/sliding_window_max_deque.svg`

- 元素数：62，重复副本数：4，疑似重叠：5 处
- 所在路径：`leetcode/daily/week2/day7/images/sliding_window_max_deque.svg`, `leetcode/images/sliding_window_max_deque.svg`, `leetcode/website/images/sliding_window_max_deque.svg`, `public/leetcode/images/sliding_window_max_deque.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `一次入队弹掉所有` | text `→ 滑动窗口最值` | 344.4 |
| text-text | text `队头即窗口最大 = 5` | text `队列：两端操作` | 278.5 |
| text-text | text `单调队列 deque（存下标）` | text `队头 ← max` | 94.5 |
| text-text | text `nums[]（窗口 k=3）` | text `[1]` | 63.4 |
| text-text | text `nums[]（窗口 k=3）` | text `[0]` | 63.4 |

### 25. `public/images/roofline_model.svg`

- 元素数：17，重复副本数：2，疑似重叠：5 处
- 所在路径：`aiinfra/week1/website/images/roofline_model.svg`, `public/images/roofline_model.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `Compute-Bound` | text `compute_intensive` | 267.5 |
| text-text | text `Memory-Bound` | text `transpose_optimized` | 253.4 |
| text-shape | text `Memory-Bound` | circle (192.0, 252.0, 208.0, 268.0) | 176.0 |
| text-shape | circle (192.0, 252.0, 208.0, 268.0) | text `transpose_optimized` | 6.4 |
| text-shape | circle (112.0, 302.0, 128.0, 318.0) | text `transpose_naive` | 6.4 |

### 26. `public/week3/images/attention_memory_bound.svg`

- 元素数：40，重复副本数：2，疑似重叠：5 处
- 所在路径：`aiinfra/week3/website/images/attention_memory_bound.svg`, `public/week3/images/attention_memory_bound.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `算力天花板 19.5 TFLOP/s` | text `QK^T GEMM` | 216.0 |
| text-shape | text `QK^T GEMM` | circle (472.0, 152.0, 488.0, 168.0) | 96.0 |
| text-shape | text `AI≈85 → compute-bound` | circle (472.0, 152.0, 488.0, 168.0) | 83.2 |
| text-shape | text `算力天花板 19.5 TFLOP/s` | circle (432.0, 152.0, 448.0, 168.0) | 32.0 |
| text-shape | text `算力天花板 19.5 TFLOP/s` | circle (472.0, 152.0, 488.0, 168.0) | 32.0 |

### 27. `public/week5/images/kv_cache_append_decode.svg`

- 元素数：53，重复副本数：2，疑似重叠：5 处
- 所在路径：`aiinfra/week5/website/images/kv_cache_append_decode.svg`, `public/week5/images/kv_cache_append_decode.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (40.0, 200.0, 440.0, 280.0) | text `填入 K₁..K_N` | 214.0 |
| text-shape | rect (520.0, 200.0, 920.0, 280.0) | text `已有 K₁..K_N` | 214.0 |
| text-shape | rect (520.0, 200.0, 920.0, 280.0) | text `追加 K_{t+1}` | 196.0 |
| text-shape | rect (40.0, 200.0, 440.0, 280.0) | text `空` | 34.0 |
| text-shape | rect (520.0, 200.0, 920.0, 280.0) | text `空` | 34.0 |

### 28. `public/images/cache_hierarchy_comparison.svg`

- 元素数：87，重复副本数：2，疑似重叠：5 处
- 所在路径：`aiinfra/week1/website/images/cache_hierarchy_comparison.svg`, `public/images/cache_hierarchy_comparison.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (430.0, 256.0, 770.0, 296.0) | text `~200 cyc` | 60.0 |
| text-shape | rect (430.0, 208.0, 770.0, 248.0) | text `(全局共享)` | 54.4 |
| text-shape | rect (430.0, 256.0, 770.0, 296.0) | text `(全局互联)` | 54.4 |
| text-shape | rect (430.0, 112.0, 770.0, 152.0) | text `❌ 自动` | 47.5 |
| text-shape | rect (430.0, 160.0, 770.0, 200.0) | text `(所有 SM)` | 38.4 |

### 29. `leetcode/images/climbing_stairs_state_transition.svg`

- 元素数：21，重复副本数：4，疑似重叠：4 处
- 所在路径：`leetcode/daily/week2/day2/images/climbing_stairs_state_transition.svg`, `leetcode/images/climbing_stairs_state_transition.svg`, `leetcode/website/images/climbing_stairs_state_transition.svg`, `public/leetcode/images/climbing_stairs_state_transition.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (180.0, 180.0, 250.0, 230.0) | rect (175.0, 180.0, 235.0, 202.0) | 1210.0 |
| text-shape | text `i-1 阶` | rect (175.0, 180.0, 235.0, 202.0) | 237.2 |
| text-text | text `i-1 阶` | text `爬 2 阶` | 68.3 |
| shape-shape | rect (310.0, 110.0, 380.0, 160.0) | rect (265.0, 158.0, 325.0, 180.0) | 30.0 |

### 30. `leetgpu/images/presum_inter_block_addback.svg`

- 元素数：104，重复副本数：3，疑似重叠：4 处
- 所在路径：`leetgpu/images/presum_inter_block_addback.svg`, `leetgpu/website/images/presum_inter_block_addback.svg`, `public/leetgpu/images/presum_inter_block_addback.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (280.0, 192.0, 336.0, 214.0) | text `(单值，无需 reduce)` | 497.9 |
| text-shape | rect (220.0, 192.0, 276.0, 214.0) | text `(单值，无需 reduce)` | 101.9 |
| text-shape | text `g_block_sum[]（来自 Kernel 1）` | rect (160.0, 60.0, 216.0, 84.0) | 46.2 |
| text-shape | text `—` | rect (280.0, 192.0, 336.0, 214.0) | 6.9 |

### 31. `public/week5/images/paged_attention_block_table.svg`

- 元素数：75，重复副本数：2，疑似重叠：4 处
- 所在路径：`aiinfra/week5/website/images/paged_attention_block_table.svg`, `public/week5/images/paged_attention_block_table.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (286.0, 250.0, 366.0, 300.0) | text `← 逻辑 block 3` | 398.5 |
| text-shape | rect (286.0, 304.0, 366.0, 354.0) | text `← 逻辑 block 0` | 398.5 |
| text-shape | rect (40.0, 412.0, 120.0, 462.0) | text `← 逻辑 block 2` | 398.5 |
| text-shape | rect (122.0, 250.0, 202.0, 300.0) | text `← 逻辑 block 1` | 398.5 |

### 32. `leetcode/images/lca_walkthrough.svg`

- 元素数：53，重复副本数：4，疑似重叠：4 处
- 所在路径：`leetcode/daily/week1/day5/images/lca_walkthrough.svg`, `leetcode/images/lca_walkthrough.svg`, `leetcode/website/images/lca_walkthrough.svg`, `public/leetcode/images/lca_walkthrough.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | circle (202.0, 112.0, 238.0, 148.0) | rect (206.0, 106.0, 234.0, 120.0) | 224.0 |
| shape-shape | circle (382.0, 112.0, 418.0, 148.0) | rect (386.0, 106.0, 414.0, 120.0) | 224.0 |
| text-shape | circle (382.0, 112.0, 418.0, 148.0) | text `q` | 20.9 |
| text-shape | circle (202.0, 112.0, 238.0, 148.0) | text `p` | 20.9 |

### 33. `leetgpu/images/prefix_sum_srunning_schunktotal.svg`

- 元素数：177，重复副本数：3，疑似重叠：4 处
- 所在路径：`leetgpu/images/prefix_sum_srunning_schunktotal.svg`, `leetgpu/website/images/prefix_sum_srunning_schunktotal.svg`, `public/leetgpu/images/prefix_sum_srunning_schunktotal.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `thread 255 写` | rect (100.0, 360.0, 600.0, 374.0) | 109.2 |
| text-shape | text `thread 255 写` | rect (100.0, 546.0, 600.0, 560.0) | 109.2 |
| text-shape | text `thread 255 写` | rect (100.0, 174.0, 600.0, 188.0) | 109.2 |
| text-shape | rect (530.0, 54.0, 542.0, 63.0) | text `s_chunk_total / s_running` | 49.2 |

### 34. `leetcode/images/sliding_window_max_example_walkthrough.svg`

- 元素数：117，重复副本数：4，疑似重叠：4 处
- 所在路径：`leetcode/daily/week2/day7/images/sliding_window_max_example_walkthrough.svg`, `leetcode/images/sliding_window_max_example_walkthrough.svg`, `leetcode/website/images/sliding_window_max_example_walkthrough.svg`, `public/leetcode/images/sliding_window_max_example_walkthrough.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `-3<-1 入队；front=1>0 不过期；收集 3` | rect (424.0, 142.0, 540.0, 162.0) | 55.3 |
| text-shape | rect (104.0, 142.0, 224.0, 162.0) | text `-3<-1 入队；front=1>0 不过期；收集 3` | 55.3 |
| text-shape | text `nums[i]` | rect (104.0, 56.0, 224.0, 74.0) | 22.4 |
| text-shape | rect (44.0, 56.0, 68.0, 74.0) | text `nums[i]` | 22.4 |

### 35. `public/week6/images/scheduling_strategy_decision_tree.svg`

- 元素数：44，重复副本数：2，疑似重叠：3 处
- 所在路径：`aiinfra/week6/website/images/scheduling_strategy_decision_tree.svg`, `public/week6/images/scheduling_strategy_decision_tree.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (400.0, 220.0, 640.0, 260.0) | rect (520.0, 220.0, 760.0, 260.0) | 4800.0 |
| text-shape | rect (400.0, 220.0, 640.0, 260.0) | text `Continuous Batching ✓` | 535.4 |
| text-shape | text `Dynamic Batching` | rect (520.0, 220.0, 760.0, 260.0) | 387.2 |

### 36. `leetgpu/images/prefix_sum_phase1_block_scan.svg`

- 元素数：69，重复副本数：3，疑似重叠：3 处
- 所在路径：`leetgpu/images/prefix_sum_phase1_block_scan.svg`, `leetgpu/website/images/prefix_sum_phase1_block_scan.svg`, `public/leetgpu/images/prefix_sum_phase1_block_scan.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (215.0, 156.0, 315.0, 196.0) | text `0, a₂₅₆, +a₂₅₇, …` | 1400.0 |
| text-shape | rect (345.0, 156.0, 445.0, 196.0) | text `0, a₅₁₂, +a₅₁₃, …` | 1400.0 |
| text-text | text `0, a₂₅₆, +a₂₅₇, …` | text `0, a₅₁₂, +a₅₁₃, …` | 130.2 |

### 37. `public/week7/images/request_lifecycle_states.svg`

- 元素数：44，重复副本数：2，疑似重叠：3 处
- 所在路径：`aiinfra/week7/website/images/request_lifecycle_states.svg`, `public/week7/images/request_lifecycle_states.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (482.0, 52.0, 642.0, 96.0) | rect (482.0, 88.0, 642.0, 132.0) | 1280.0 |
| shape-shape | rect (482.0, 88.0, 642.0, 132.0) | rect (482.0, 128.0, 642.0, 172.0) | 640.0 |
| text-shape | text `set_result() → Future+Callback` | rect (482.0, 88.0, 642.0, 132.0) | 201.7 |

### 38. `public/week7/images/profiling_toolchain.svg`

- 元素数：66，重复副本数：2，疑似重叠：3 处
- 所在路径：`aiinfra/week7/website/images/profiling_toolchain.svg`, `public/week7/images/profiling_toolchain.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (340.0, 76.0, 620.0, 216.0) | text `ncu --metrics sm__throughput,d` | 1166.4 |
| text-shape | rect (650.0, 76.0, 910.0, 216.0) | text `timer.start("forward") / timer` | 933.1 |
| text-shape | rect (30.0, 76.0, 310.0, 216.0) | text `nsys profile --trace=cuda,nvtx` | 777.6 |

### 39. `leetcode/images/daily_temperatures_monotonic_stack.svg`

- 元素数：50，重复副本数：4，疑似重叠：3 处
- 所在路径：`leetcode/daily/week2/day6/images/daily_temperatures_monotonic_stack.svg`, `leetcode/images/daily_temperatures_monotonic_stack.svg`, `leetcode/website/images/daily_temperatures_monotonic_stack.svg`, `public/leetcode/images/daily_temperatures_monotonic_stack.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (200.0, 200.0, 280.0, 228.0) | text `[5] = 72° ← 栈顶` | 1120.0 |
| text-shape | rect (200.0, 262.0, 280.0, 290.0) | text `[2] = 75° ← 栈顶` | 1120.0 |
| text-text | text `temperatures[]` | text `[0]` | 63.4 |

### 40. `leetcode/images/container_water_example_walkthrough.svg`

- 元素数：135，重复副本数：4，疑似重叠：3 处
- 所在路径：`leetcode/daily/week3/day1/images/container_water_example_walkthrough.svg`, `leetcode/images/container_water_example_walkthrough.svg`, `leetcode/website/images/container_water_example_walkthrough.svg`, `public/leetcode/images/container_water_example_walkthrough.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (44.0, 56.0, 94.0, 74.0) | text `left, right` | 700.0 |
| text-shape | text `left, right` | rect (94.0, 56.0, 154.0, 74.0) | 81.2 |
| text-shape | rect (20.0, 56.0, 44.0, 74.0) | text `left, right` | 81.2 |

### 41. `leetcode/images/minimum_window_substring.svg`

- 元素数：46，重复副本数：4，疑似重叠：3 处
- 所在路径：`leetcode/daily/week6/day7/images/minimum_window_substring.svg`, `leetcode/images/minimum_window_substring.svg`, `leetcode/website/images/minimum_window_substring.svg`, `public/leetcode/images/minimum_window_substring.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (60.0, 90.0, 840.0, 140.0) | text `② 已覆盖(A,B,C)！收缩左指针找最小` | 285.0 |
| text-shape | rect (60.0, 156.0, 840.0, 206.0) | text `③ 最终最小：BANC（长度4）` | 218.0 |
| text-text | text `s = A D O B E C O D E B A N C ` | text `① 扩张右指针找覆盖` | 161.9 |

### 42. `public/leetcode/images/trie_structure.svg`

- 元素数：50，重复副本数：3，疑似重叠：3 处
- 所在路径：`leetcode/daily/week7/day4/images/trie_structure.svg`, `leetcode/website/images/trie_structure.svg`, `public/leetcode/images/trie_structure.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `→ a→p→p→l→e, is_end=T →` | text `True` | 254.8 |
| text-text | text `→ a→p→p, is_end=T →` | text `True` | 184.8 |
| text-text | text `→ a→p 存在 →` | text `True` | 127.4 |

### 43. `public/week6/images/framework_comparison.svg`

- 元素数：36，重复副本数：2，疑似重叠：3 处
- 所在路径：`aiinfra/week6/website/images/framework_comparison.svg`, `public/week6/images/framework_comparison.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `Batching` | rect (40.0, 100.0, 240.0, 128.0) | 167.2 |
| text-shape | text `KV Cache` | rect (40.0, 168.0, 240.0, 196.0) | 167.2 |
| text-shape | text `调度器` | rect (40.0, 134.0, 240.0, 162.0) | 88.5 |

### 44. `leetcode/images/three_sum_example_walkthrough.svg`

- 元素数：55，重复副本数：4，疑似重叠：3 处
- 所在路径：`leetcode/daily/week2/day3/images/three_sum_example_walkthrough.svg`, `leetcode/images/three_sum_example_walkthrough.svg`, `leetcode/website/images/three_sum_example_walkthrough.svg`, `public/leetcode/images/three_sum_example_walkthrough.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `步骤 1：left=2(nums[2]=-1), right` | text `left` | 84.5 |
| text-text | text `步骤 2：left=3(nums[3]=0), right=` | text `left` | 84.5 |
| text-text | text `步骤 1：left=2(nums[2]=-1), right` | text `[0]` | 63.4 |

### 45. `leetcode/images/bst_bounds_recursion.svg`

- 元素数：34，重复副本数：4，疑似重叠：3 处
- 所在路径：`leetcode/daily/week3/day5/images/bst_bounds_recursion.svg`, `leetcode/images/bst_bounds_recursion.svg`, `leetcode/website/images/bst_bounds_recursion.svg`, `public/leetcode/images/bst_bounds_recursion.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | circle (250.0, 70.0, 290.0, 110.0) | text `(−∞, +∞)` | 72.0 |
| text-shape | circle (132.0, 152.0, 168.0, 188.0) | text `(−∞, 10)` | 64.8 |
| text-shape | circle (372.0, 152.0, 408.0, 188.0) | text `(10, +∞)` | 59.1 |

### 46. `public/week5/images/kv_cache_allocation_strategies.svg`

- 元素数：78，重复副本数：2，疑似重叠：2 处
- 所在路径：`aiinfra/week5/website/images/kv_cache_allocation_strategies.svg`, `public/week5/images/kv_cache_allocation_strategies.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (330.0, 64.0, 610.0, 424.0) | rect (500.0, 120.0, 620.0, 180.0) | 6600.0 |
| text-shape | rect (50.0, 256.0, 80.0, 316.0) | text `req3: 2/10` | 200.0 |

### 47. `public/images/grid_block_thread.svg`

- 元素数：30，重复副本数：2，疑似重叠：2 处
- 所在路径：`aiinfra/week1/website/images/grid_block_thread.svg`, `public/images/grid_block_thread.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (80.0, 70.0, 640.0, 450.0) | text `总线程数 = gridDim.x × gridDim.y ×` | 4387.2 |
| text-shape | rect (110.0, 130.0, 350.0, 270.0) | text `blockDim = 4 × 2 = 8 threads` | 1342.1 |

### 48. `public/week3/images/decode_memory_bound.svg`

- 元素数：52，重复副本数：2，疑似重叠：2 处
- 所在路径：`aiinfra/week3/website/images/decode_memory_bound.svg`, `public/week3/images/decode_memory_bound.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (30.0, 64.0, 410.0, 264.0) | text `AI ≈ 384 FLOP/Byte → Compute-b` | 945.0 |
| text-shape | rect (450.0, 64.0, 830.0, 264.0) | text `AI ≈ 2 FLOP/Byte → Memory-boun` | 873.0 |

### 49. `public/images/optimization_decision_tree.svg`

- 元素数：33，重复副本数：2，疑似重叠：2 处
- 所在路径：`aiinfra/week1/website/images/optimization_decision_tree.svg`, `public/images/optimization_decision_tree.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (320.0, 360.0, 500.0, 450.0) | text `• 减少数据读写` | 614.7 |
| text-shape | rect (530.0, 360.0, 710.0, 450.0) | text `• 提高指令吞吐` | 614.7 |

### 50. `leetgpu/images/segmented_prefix_sum_overview.svg`

- 元素数：41，重复副本数：3，疑似重叠：2 处
- 所在路径：`leetgpu/images/segmented_prefix_sum_overview.svg`, `leetgpu/website/images/segmented_prefix_sum_overview.svg`, `public/leetgpu/images/segmented_prefix_sum_overview.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `② 输出 output[]（段内 exclusive pre` | text `段B 重新从 0 开始` | 485.6 |
| text-text | text `① 输入 input[]（两段：A | B）` | text `段A` | 85.5 |

### 51. `public/images/occupancy_concept.svg`

- 元素数：21，重复副本数：2，疑似重叠：2 处
- 所在路径：`aiinfra/week1/website/images/occupancy_concept.svg`, `public/images/occupancy_concept.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (80.0, 80.0, 640.0, 200.0) | text `Active Warps = 8` | 449.3 |
| text-shape | rect (80.0, 80.0, 640.0, 200.0) | text `Empty Slots = 4` | 421.2 |

### 52. `public/leetcode/images/trap_algorithm_flow_v2.svg`

- 元素数：32，重复副本数：3，疑似重叠：2 处
- 所在路径：`leetcode/daily/week1/day1/images/trap_algorithm_flow_v2.svg`, `leetcode/website/images/trap_algorithm_flow_v2.svg`, `public/leetcode/images/trap_algorithm_flow_v2.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (350.0, 314.0, 450.0, 394.0) | text `→ right−−` | 396.0 |
| text-shape | rect (10.0, 314.0, 130.0, 394.0) | text `left++ ←` | 292.0 |

### 53. `leetcode/images/bst_inorder_monotonic.svg`

- 元素数：59，重复副本数：4，疑似重叠：2 处
- 所在路径：`leetcode/daily/week3/day5/images/bst_inorder_monotonic.svg`, `leetcode/images/bst_inorder_monotonic.svg`, `leetcode/website/images/bst_inorder_monotonic.svg`, `public/leetcode/images/bst_inorder_monotonic.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (285.0, 58.0, 520.0, 354.0) | rect (506.0, 262.0, 534.0, 286.0) | 336.0 |
| text-shape | rect (285.0, 58.0, 520.0, 354.0) | text `6` | 39.2 |

### 54. `public/leetgpu/images/gemm_three_level_reuse.svg`

- 元素数：56，重复副本数：2，疑似重叠：2 处
- 所在路径：`leetgpu/website/images/gemm_three_level_reuse.svg`, `public/leetgpu/images/gemm_three_level_reuse.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `算术强度演进（FLOP / Byte）：` | rect (90.0, 416.0, 110.0, 476.0) | 240.0 |
| text-shape | text `算术强度演进（FLOP / Byte）：` | rect (150.0, 408.0, 170.0, 476.0) | 240.0 |

### 55. `public/week5/images/kv_cache_memory_layout.svg`

- 元素数：42，重复副本数：2，疑似重叠：2 处
- 所在路径：`aiinfra/week5/website/images/kv_cache_memory_layout.svg`, `public/week5/images/kv_cache_memory_layout.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (172.0, 196.0, 196.0, 218.0) | rect (100.0, 196.0, 180.0, 240.0) | 176.0 |
| shape-shape | rect (172.0, 218.0, 196.0, 240.0) | rect (100.0, 196.0, 180.0, 240.0) | 176.0 |

### 56. `public/week3/images/prefill_vs_decode.svg`

- 元素数：59，重复副本数：2，疑似重叠：2 处
- 所在路径：`aiinfra/week3/website/images/prefill_vs_decode.svg`, `public/week3/images/prefill_vs_decode.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (30.0, 64.0, 430.0, 444.0) | text `生成1个token后` | 153.9 |
| text-shape | rect (450.0, 64.0, 850.0, 444.0) | text `生成1个token后` | 135.9 |

### 57. `leetcode/images/longest_palindrome_center_expand.svg`

- 元素数：33，重复副本数：4，疑似重叠：2 处
- 所在路径：`leetcode/daily/week3/day3/images/longest_palindrome_center_expand.svg`, `leetcode/images/longest_palindrome_center_expand.svg`, `leetcode/website/images/longest_palindrome_center_expand.svg`, `public/leetcode/images/longest_palindrome_center_expand.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `中心 i=1,i+1=2` | text `s[1]==s[2] ✓ → "bb" len=2，向外不对` | 153.9 |
| text-text | text `中心 i=1` | text `s[0]==s[2] ✓ → "bab" len=3` | 89.1 |

### 58. `public/week6/images/dynamic_batcher_flow.svg`

- 元素数：42，重复副本数：2，疑似重叠：2 处
- 所在路径：`aiinfra/week6/website/images/dynamic_batcher_flow.svg`, `public/week6/images/dynamic_batcher_flow.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (40.0, 310.0, 440.0, 440.0) | text `实际系统按 SLA 调参：latency SLO → 短 t` | 109.1 |
| text-shape | rect (500.0, 310.0, 900.0, 440.0) | text `实际系统按 SLA 调参：latency SLO → 短 t` | 109.1 |

### 59. `public/week6/images/request_lifecycle_v1.svg`

- 元素数：32，重复副本数：2，疑似重叠：2 处
- 所在路径：`aiinfra/week6/website/images/request_lifecycle_v1.svg`, `public/week6/images/request_lifecycle_v1.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (322.0, 90.0, 562.0, 170.0) | text `gen 满 max_new_tokens` | 72.2 |
| text-shape | text `gen 满 max_new_tokens` | rect (624.0, 90.0, 844.0, 170.0) | 54.2 |

### 60. `public/week6/images/throughput_latency_curve.svg`

- 元素数：25，重复副本数：2，疑似重叠：2 处
- 所在路径：`aiinfra/week6/website/images/throughput_latency_curve.svg`, `public/week6/images/throughput_latency_curve.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | circle (544.0, 279.0, 556.0, 291.0) | rect (100.0, 80.0, 550.0, 360.0) | 72.0 |
| shape-shape | circle (544.0, 279.0, 556.0, 291.0) | rect (550.0, 80.0, 860.0, 360.0) | 72.0 |

### 61. `public/week5/images/continuous_vs_static_batching.svg`

- 元素数：67，重复副本数：2，疑似重叠：2 处
- 所在路径：`aiinfra/week5/website/images/continuous_vs_static_batching.svg`, `public/week5/images/continuous_vs_static_batching.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (860.0, 158.0, 900.0, 186.0) | text `B...` | 64.8 |
| text-shape | text `i5` | rect (740.0, 222.0, 780.0, 250.0) | 32.4 |

### 62. `leetgpu/images/vector_addition_grid_stride.svg`

- 元素数：54，重复副本数：3，疑似重叠：2 处
- 所在路径：`leetgpu/images/vector_addition_grid_stride.svg`, `leetgpu/website/images/vector_addition_grid_stride.svg`, `public/leetgpu/images/vector_addition_grid_stride.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `数组下标 i:` | text `0` | 61.6 |
| text-shape | rect (40.0, 300.0, 560.0, 376.0) | text `}` | 17.9 |

### 63. `leetcode/images/longest_palindrome_example_walkthrough.svg`

- 元素数：86，重复副本数：4，疑似重叠：2 处
- 所在路径：`leetcode/daily/week3/day3/images/longest_palindrome_example_walkthrough.svg`, `leetcode/images/longest_palindrome_example_walkthrough.svg`, `leetcode/website/images/longest_palindrome_example_walkthrough.svg`, `public/leetcode/images/longest_palindrome_example_walkthrough.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (56.0, 148.0, 176.0, 166.0) | text `偶数扩展 expand(i,i+1)` | 42.0 |
| text-shape | text `偶数扩展 expand(i,i+1)` | rect (296.0, 148.0, 396.0, 166.0) | 42.0 |

### 64. `public/week5/images/decode_breakdown.svg`

- 元素数：30，重复副本数：2，疑似重叠：1 处
- 所在路径：`aiinfra/week5/website/images/decode_breakdown.svg`, `public/week5/images/decode_breakdown.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (40.0, 266.0, 880.0, 416.0) | text `→ CUDA Graph（把 decode 循环录制成图，消` | 3238.8 |

### 65. `public/week2/images/default_stream_sync.svg`

- 元素数：40，重复副本数：2，疑似重叠：1 处
- 所在路径：`aiinfra/week2/website/images/default_stream_sync.svg`, `public/week2/images/default_stream_sync.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (140.0, 82.0, 260.0, 112.0) | rect (180.0, 82.0, 420.0, 156.0) | 2400.0 |

### 66. `leetgpu/images/vector_addition_coalesced.svg`

- 元素数：47，重复副本数：3，疑似重叠：1 处
- 所在路径：`leetgpu/images/vector_addition_coalesced.svg`, `leetgpu/website/images/vector_addition_coalesced.svg`, `public/leetgpu/images/vector_addition_coalesced.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (90.0, 144.0, 226.0, 170.0) | text `A[0] A[1] A[2] A[3] ... A[31] ` | 1496.0 |

### 67. `public/leetcode/images/lru_cache.svg`

- 元素数：56，重复副本数：3，疑似重叠：1 处
- 所在路径：`leetcode/daily/week7/day7/images/lru_cache.svg`, `leetcode/website/images/lru_cache.svg`, `public/leetcode/images/lru_cache.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `key=3 → Node(3,3)` | text `操作示例（capacity=2）` | 932.5 |

### 68. `public/week6/images/scheduler_state_machine.svg`

- 元素数：40，重复副本数：2，疑似重叠：1 处
- 所在路径：`aiinfra/week6/website/images/scheduler_state_machine.svg`, `public/week6/images/scheduler_state_machine.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (40.0, 244.0, 160.0, 284.0) | rect (140.0, 244.0, 260.0, 284.0) | 800.0 |

### 69. `public/week6/images/inflight_batching_flow.svg`

- 元素数：75，重复副本数：2，疑似重叠：1 处
- 所在路径：`aiinfra/week6/website/images/inflight_batching_flow.svg`, `public/week6/images/inflight_batching_flow.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (380.0, 260.0, 560.0, 304.0) | rect (40.0, 300.0, 880.0, 400.0) | 720.0 |

### 70. `leetcode/images/house_robber_state_transition.svg`

- 元素数：24，重复副本数：4，疑似重叠：1 处
- 所在路径：`leetcode/daily/week3/day2/images/house_robber_state_transition.svg`, `leetcode/images/house_robber_state_transition.svg`, `leetcode/website/images/house_robber_state_transition.svg`, `public/leetcode/images/house_robber_state_transition.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (200.0, 100.0, 360.0, 136.0) | rect (60.0, 80.0, 230.0, 120.0) | 600.0 |

### 71. `leetcode/images/merge_intervals_greedy.svg`

- 元素数：32，重复副本数：4，疑似重叠：1 处
- 所在路径：`leetcode/daily/week6/day1/images/merge_intervals_greedy.svg`, `leetcode/images/merge_intervals_greedy.svg`, `leetcode/website/images/merge_intervals_greedy.svg`, `public/leetcode/images/merge_intervals_greedy.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (60.0, 116.0, 120.0, 144.0) | rect (100.0, 116.0, 220.0, 144.0) | 560.0 |

### 72. `public/leetgpu/images/gemm_thread_tile_layout.svg`

- 元素数：151，重复副本数：2，疑似重叠：1 处
- 所在路径：`leetgpu/website/images/gemm_thread_tile_layout.svg`, `public/leetgpu/images/gemm_thread_tile_layout.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `累加器 acc[8][8] 驻留寄存器，K 维循环累加` | text `← TN=8 →  rB[0..7]` | 524.9 |

### 73. `public/week3/images/block_reduce_two_level.svg`

- 元素数：54，重复副本数：2，疑似重叠：1 处
- 所在路径：`aiinfra/week3/website/images/block_reduce_two_level.svg`, `public/week3/images/block_reduce_two_level.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (40.0, 92.0, 880.0, 182.0) | text `if (lane==0) smem[wid] = val;` | 464.0 |

### 74. `public/week7/images/troubleshooting_guide.svg`

- 元素数：89，重复副本数：2，疑似重叠：1 处
- 所在路径：`aiinfra/week7/website/images/troubleshooting_guide.svg`, `public/week7/images/troubleshooting_guide.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `• 监控 torch.cuda.` | text `memory_allocated()` | 434.0 |

### 75. `public/week6/images/dynamic_vs_continuous_batching.svg`

- 元素数：67，重复副本数：2，疑似重叠：1 处
- 所在路径：`aiinfra/week6/website/images/dynamic_vs_continuous_batching.svg`, `public/week6/images/dynamic_vs_continuous_batching.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `R1 (5 tok)` | rect (560.0, 100.0, 620.0, 124.0) | 300.0 |

### 76. `leetgpu/images/reduction_two_level.svg`

- 元素数：66，重复副本数：3，疑似重叠：1 处
- 所在路径：`leetgpu/images/reduction_two_level.svg`, `leetgpu/website/images/reduction_two_level.svg`, `public/leetgpu/images/reduction_two_level.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (30.0, 82.0, 590.0, 104.0) | text `input[N-1]` | 270.0 |

### 77. `leetcode/images/two_sum_flow.svg`

- 元素数：35，重复副本数：4，疑似重叠：1 处
- 所在路径：`leetcode/daily/week2/day1/images/two_sum_flow.svg`, `leetcode/images/two_sum_flow.svg`, `leetcode/website/images/two_sum_flow.svg`, `public/leetcode/images/two_sum_flow.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (200.0, 400.0, 360.0, 430.0) | text `每个元素只查一次、存一次 → O(n)` | 269.0 |

### 78. `leetcode/images/two_sum_walkthrough.svg`

- 元素数：45，重复副本数：4，疑似重叠：1 处
- 所在路径：`leetcode/daily/week2/day1/images/two_sum_walkthrough.svg`, `leetcode/images/two_sum_walkthrough.svg`, `leetcode/website/images/two_sum_walkthrough.svg`, `public/leetcode/images/two_sum_walkthrough.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (80.0, 250.0, 200.0, 276.0) | text `查 7？无；查 2？有！` | 220.0 |

### 79. `public/week5/images/request_lifecycle.svg`

- 元素数：45，重复副本数：2，疑似重叠：1 处
- 所在路径：`aiinfra/week5/website/images/request_lifecycle.svg`, `public/week5/images/request_lifecycle.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `显存恢复 → swap in 回 RUNNING` | rect (640.0, 180.0, 900.0, 300.0) | 215.0 |

### 80. `public/leetcode/images/trap_two_pointers_v5.svg`

- 元素数：23，重复副本数：3，疑似重叠：1 处
- 所在路径：`leetcode/daily/week1/day1/images/trap_two_pointers_v5.svg`, `leetcode/website/images/trap_two_pointers_v5.svg`, `public/leetcode/images/trap_two_pointers_v5.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `right →` | text `R_max=4` | 203.3 |

### 81. `public/leetcode/images/trap_two_pointers_v4.svg`

- 元素数：23，重复副本数：3，疑似重叠：1 处
- 所在路径：`leetcode/daily/week1/day1/images/trap_two_pointers_v4.svg`, `leetcode/website/images/trap_two_pointers_v4.svg`, `public/leetcode/images/trap_two_pointers_v4.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `right →` | text `R_max=4` | 203.3 |

### 82. `leetcode/images/palindrome_example_walkthrough.svg`

- 元素数：98，重复副本数：3，疑似重叠：1 处
- 所在路径：`leetcode/images/palindrome_example_walkthrough.svg`, `leetcode/website/images/palindrome_example_walkthrough.svg`, `public/leetcode/images/palindrome_example_walkthrough.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `pref` | rect (75.0, 128.0, 125.0, 150.0) | 152.3 |

### 83. `leetcode/images/largest_rect_walkthrough.svg`

- 元素数：108，重复副本数：4，疑似重叠：1 处
- 所在路径：`leetcode/daily/week1/day7/images/largest_rect_walkthrough.svg`, `leetcode/images/largest_rect_walkthrough.svg`, `leetcode/website/images/largest_rect_walkthrough.svg`, `public/leetcode/images/largest_rect_walkthrough.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (332.0, 218.0, 362.0, 236.0) | text `→ 栈顶=1, w=4−1−1=2, 面积=5×2=10 ⭐` | 120.0 |

### 84. `leetcode/images/reverse_k_group.svg`

- 元素数：63，重复副本数：4，疑似重叠：1 处
- 所在路径：`leetcode/daily/week5/day7/images/reverse_k_group.svg`, `leetcode/images/reverse_k_group.svg`, `leetcode/website/images/reverse_k_group.svg`, `public/leetcode/images/reverse_k_group.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `第 2 组（4,5）不足 k=3 → 保持原序` | rect (440.0, 100.0, 900.0, 320.0) | 117.6 |

### 85. `leetgpu/images/presum_overview.svg`

- 元素数：114，重复副本数：3，疑似重叠：1 处
- 所在路径：`leetgpu/images/presum_overview.svg`, `leetgpu/website/images/presum_overview.svg`, `public/leetgpu/images/presum_overview.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `(__device__ 静态)` | rect (80.0, 256.0, 120.0, 280.0) | 108.9 |

### 86. `leetcode/images/permutations_backtrack_tree.svg`

- 元素数：54，重复副本数：4，疑似重叠：1 处
- 所在路径：`leetcode/daily/week1/day6/images/permutations_backtrack_tree.svg`, `leetcode/images/permutations_backtrack_tree.svg`, `leetcode/website/images/permutations_backtrack_tree.svg`, `public/leetcode/images/permutations_backtrack_tree.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| shape-shape | rect (22.0, 242.0, 78.0, 266.0) | rect (74.0, 242.0, 130.0, 266.0) | 96.0 |

### 87. `public/leetcode/images/median_finder.svg`

- 元素数：35，重复副本数：3，疑似重叠：1 处
- 所在路径：`leetcode/daily/week7/day6/images/median_finder.svg`, `leetcode/website/images/median_finder.svg`, `public/leetcode/images/median_finder.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (280.0, 316.0, 660.0, 368.0) | text `💡 与 P50 延迟统计同构：P50 = 中位数，双堆 O(` | 90.1 |

### 88. `leetcode/images/arithmetic_sum.svg`

- 元素数：26，重复副本数：3，疑似重叠：1 处
- 所在路径：`leetcode/images/arithmetic_sum.svg`, `leetcode/website/images/arithmetic_sum.svg`, `public/leetcode/images/arithmetic_sum.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `阈值 T` | rect (400.0, 230.0, 470.0, 270.0) | 84.5 |

### 89. `leetgpu/images/relu_branchless.svg`

- 元素数：29，重复副本数：3，疑似重叠：1 处
- 所在路径：`leetgpu/images/relu_branchless.svg`, `leetgpu/website/images/relu_branchless.svg`, `public/leetgpu/images/relu_branchless.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `单条指令，最优（条越短越好）` | text `→ elementwise kernel 通用经验：能用数学` | 72.6 |

### 90. `public/week7/images/week7_knowledge_map.svg`

- 元素数：74，重复副本数：2，疑似重叠：1 处
- 所在路径：`aiinfra/week7/website/images/week7_knowledge_map.svg`, `public/week7/images/week7_knowledge_map.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-text | text `稳定性` | text `• 500+ 请求 100% 成功` | 71.4 |

### 91. `public/week3/images/kernel_fusion_opportunities.svg`

- 元素数：51，重复副本数：2，疑似重叠：1 处
- 所在路径：`aiinfra/week3/website/images/kernel_fusion_opportunities.svg`, `public/week3/images/kernel_fusion_opportunities.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `y→HBM (2MB)` | ellipse (246.0, 130.0, 314.0, 170.0) | 68.9 |

### 92. `public/week2/images/thread_tile_mapping.svg`

- 元素数：56，重复副本数：2，疑似重叠：1 处
- 所在路径：`aiinfra/week2/website/images/thread_tile_mapping.svg`, `public/week2/images/thread_tile_mapping.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | text `16×16 = 256 个 Thread Tile，每线程负` | rect (60.0, 400.0, 820.0, 456.0) | 44.3 |

### 93. `leetcode/images/container_water_algorithm_flow.svg`

- 元素数：24，重复副本数：4，疑似重叠：1 处
- 所在路径：`leetcode/daily/week3/day1/images/container_water_algorithm_flow.svg`, `leetcode/images/container_water_algorithm_flow.svg`, `leetcode/website/images/container_water_algorithm_flow.svg`, `public/leetcode/images/container_water_algorithm_flow.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | rect (40.0, 128.0, 480.0, 248.0) | text `}` | 32.0 |

### 94. `leetcode/images/linked_list_cycle_example_walkthrough.svg`

- 元素数：53，重复副本数：4，疑似重叠：1 处
- 所在路径：`leetcode/daily/week3/day4/images/linked_list_cycle_example_walkthrough.svg`, `leetcode/images/linked_list_cycle_example_walkthrough.svg`, `leetcode/website/images/linked_list_cycle_example_walkthrough.svg`, `public/leetcode/images/linked_list_cycle_example_walkthrough.svg`

| 类型 | 元素 A | 元素 B | 重叠面积 |
|------|--------|--------|----------|
| text-shape | circle (250.0, 250.0, 290.0, 290.0) | text `[3]` | 3.8 |

