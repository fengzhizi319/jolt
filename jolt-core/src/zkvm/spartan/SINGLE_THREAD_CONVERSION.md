# 单线程版本改造文档

## 改造概述

将 `compute_first_quadratic_evals_and_bound_polys` 函数从并行版本改造为单线程版本，以便更容易理解代码逻辑。

## 改动位置

**文件**: `jolt-core/src/zkvm/spartan/product.rs`  
**函数**: `ProductVirtualRemainderProver::compute_first_quadratic_evals_and_bound_polys`  
**行数**: 第 721-794 行

## 主要改动

### 1. 移除并行迭代器

**改动前** (并行版本):
```rust
let (t0_acc_unr, t_inf_acc_unr) = left_bound
    .par_chunks_exact_mut(2 * num_x_in_vals)
    .zip(right_bound.par_chunks_exact_mut(2 * num_x_in_vals))
    .enumerate()
    .fold(
        || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
        |(mut acc0, mut acci), (x_out_val, (left_chunk, right_chunk))| {
            // ... 计算逻辑 ...
            (acc0, acci)
        },
    )
    .reduce(
        || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
        |a, b| (a.0 + b.0, a.1 + b.1),
    );
```

**改动后** (单线程版本):
```rust
// 初始化全局累加器
let mut t0_acc_unr = F::Unreduced::<9>::zero();
let mut t_inf_acc_unr = F::Unreduced::<9>::zero();

// 外部循环：遍历高位变量 x_out
for (x_out_val, (left_chunk, right_chunk)) in left_bound
    .chunks_exact_mut(2 * num_x_in_vals)
    .zip(right_bound.chunks_exact_mut(2 * num_x_in_vals))
    .enumerate()
{
    // ... 计算逻辑 ...
}
```

### 2. 简化累加器结构

**并行版本的累加器模型**：
- 每个线程维护独立的累加器 `(acc0, acci)`
- 最后通过 `reduce` 将所有线程的结果合并

**单线程版本的累加器模型**：
- 使用单个全局累加器 `t0_acc_unr` 和 `t_inf_acc_unr`
- 每次迭代直接累加到全局累加器

### 3. 移除未使用的导入

```rust
// 删除这一行
use rayon::prelude::*;
```

## 代码对比

### 并行版本的计算模式

```rust
// 1. 使用 fold 为每个线程创建局部累加器
.fold(
    || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
    |(mut acc0, mut acci), (x_out_val, (left_chunk, right_chunk))| {
        // 计算...
        acc0 += e_out.mul_unreduced::<9>(reduced0);
        acci += e_out.mul_unreduced::<9>(reduced_inf);
        (acc0, acci)  // 返回线程局部累加器
    },
)
// 2. 使用 reduce 合并所有线程的结果
.reduce(
    || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
    |a, b| (a.0 + b.0, a.1 + b.1),
);
```

### 单线程版本的计算模式

```rust
// 1. 初始化全局累加器
let mut t0_acc_unr = F::Unreduced::<9>::zero();
let mut t_inf_acc_unr = F::Unreduced::<9>::zero();

// 2. 直接在循环中累加
for (x_out_val, (left_chunk, right_chunk)) in ... {
    // 计算...
    t0_acc_unr += e_out.mul_unreduced::<9>(reduced0);
    t_inf_acc_unr += e_out.mul_unreduced::<9>(reduced_inf);
}
```

## 核心计算逻辑保持不变

以下计算逻辑在两个版本中**完全相同**：

```rust
// 内循环累加器
let mut inner_sum0 = F::Unreduced::<9>::zero();
let mut inner_sum_inf = F::Unreduced::<9>::zero();

// 遍历低位变量 x_in
for x_in_val in 0..num_x_in_vals {
    // 1. 索引计算
    let base_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
    let idx_lo = base_idx << 1;
    let idx_hi = idx_lo + 1;

    // 2. 数据获取
    let row_lo = ProductCycleInputs::from_trace::<F>(trace, idx_lo);
    let row_hi = ProductCycleInputs::from_trace::<F>(trace, idx_hi);

    // 3. UniSkip 投影
    let (left0, right0) = ProductVirtualEval::fused_left_right_at_r::<F>(&row_lo, &weights_at_r0[..]);
    let (left1, right1) = ProductVirtualEval::fused_left_right_at_r::<F>(&row_hi, &weights_at_r0[..]);

    // 4. 二次多项式计算
    let p0 = left0 * right0;
    let slope = (left1 - left0) * (right1 - right0);

    // 5. Eq 权重
    let e_in = split_eq_poly.E_in_current()[x_in_val];

    // 6. 累加
    inner_sum0 += e_in.mul_unreduced::<9>(p0);
    inner_sum_inf += e_in.mul_unreduced::<9>(slope);

    // 7. 保存数据
    let off = 2 * x_in_val;
    left_chunk[off] = left0;
    left_chunk[off + 1] = left1;
    right_chunk[off] = right0;
    right_chunk[off + 1] = right1;
}

// 8. 结合高位权重
let e_out = split_eq_poly.E_out_current()[x_out_val];
let reduced0 = F::from_montgomery_reduce::<9>(inner_sum0);
let reduced_inf = F::from_montgomery_reduce::<9>(inner_sum_inf);
```

## 算法解释（带详细注释）

### 嵌套循环结构

```rust
// 外层循环：遍历 x_out (高位变量)
for x_out_val in 0..num_x_out_vals {
    
    // 内层累加器：累加当前 x_out 的所有 x_in 贡献
    let mut inner_sum0 = ...;
    let mut inner_sum_inf = ...;
    
    // 内层循环：遍历 x_in (低位变量)
    for x_in_val in 0..num_x_in_vals {
        // 计算当前点 (x_out, x_in) 的贡献
        // 累加到 inner_sum
        inner_sum0 += e_in * p0;
        inner_sum_inf += e_in * slope;
    }
    
    // 将 inner_sum 乘以 e_out，累加到全局累加器
    t0_acc_unr += e_out * inner_sum0;
    t_inf_acc_unr += e_out * inner_sum_inf;
}
```

### 数学等价性

这个嵌套循环计算的是：

```
t0 = Σ_{x_out} E_out[x_out] · (Σ_{x_in} E_in[x_in] · P(x_out, x_in, 0))
t_inf = Σ_{x_out} E_out[x_out] · (Σ_{x_in} E_in[x_in] · Slope(x_out, x_in))
```

其中：
- `E_out[x_out]` 和 `E_in[x_in]` 是 Eq 多项式的分解形式
- `P(x, 0)` 是二次多项式在 0 点的值：`Left(x, 0) * Right(x, 0)`
- `Slope(x)` 是二次多项式的二次项系数：`(Left(x, 1) - Left(x, 0)) * (Right(x, 1) - Right(x, 0))`

## 性能影响

### 单线程版本

**优点**：
- ✅ 代码更清晰，更容易理解
- ✅ 更容易调试和分析
- ✅ 没有线程同步开销
- ✅ 数据访问模式更可预测

**缺点**：
- ❌ 无法利用多核并行
- ❌ 在大规模 trace 上性能较差

### 并行版本

**优点**：
- ✅ 充分利用多核 CPU
- ✅ 大规模数据处理速度快

**缺点**：
- ❌ 代码逻辑较复杂
- ❌ 需要理解 fold/reduce 模式
- ❌ 有线程同步开销

## 测试验证

所有单元测试均通过，验证了单线程版本的正确性：

```bash
running 4 tests
test zkvm::spartan::product::tests::test_grand_product_argument_sumcheck ... ok
test zkvm::spartan::product::tests::test_sumcheck_for_grand_product ... ok
test zkvm::spartan::product::tests::test_polynomial_evaluation_for_product ... ok
test zkvm::spartan::product::tests::test_sumcheck_protocol_simulation ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured
```

## 推荐使用场景

### 使用单线程版本

- 🎓 **学习和理解算法**：适合初学者理解代码逻辑
- 🐛 **调试问题**：更容易设置断点和跟踪执行
- 📊 **小规模测试**：trace 长度 < 10000 时性能差异不大
- 📖 **代码审查**：更容易验证正确性

### 使用并行版本

- ⚡ **生产环境**：需要高性能处理大规模 trace
- 🏭 **批量处理**：处理多个大型证明任务
- 💻 **多核服务器**：充分利用硬件资源

## 总结

单线程改造成功完成，核心计算逻辑保持不变，仅简化了并行控制流程。这个版本更适合学习和理解 Grand Product Argument 的实现细节。

---

**改造日期**: 2026-01-27  
**改造范围**: 第 721-794 行  
**测试状态**: ✅ 全部通过
