# C++ 面试一周学习计划

> **适用对象**：有 C/C++ 基础、准备 AI Infra / 系统 / 后端方向 C++ 面试的开发者；建议已完成本仓库 [Week 1](../../daily/week1/README.md) CUDA 基础教程，对内存层级有初步认知
> **本周目标**：系统梳理 C++ 面试高频考点，覆盖内存模型、RAII/智能指针、移动语义、模板泛型、多态底层、并发编程、现代 C++ 新特性，每讲配可编译代码与面试问答，最终能独立应对 C++ 一二面技术问答
> **时间投入**：工作日每天 2.5h（早间 1.5h 精读 + 晚间 1h 跑代码），周末每天 4h，周计 20.5h
> **周日里程碑**：完成 40+ 道高频面试题的问答整理，覆盖 7 大主题，产出个人面试 cheat sheet 与可编译代码集

---

## 本周总览

| 维度 | 内容 |
|------|------|
| **整体目标** | 掌握 C++ 内存模型与对象生命周期、RAII 与智能指针、移动语义与完美转发、模板与 SFINAE/concepts、虚函数表与多态底层、并发原语与 memory order、C++11~23 新特性 |
| **核心产出** | ① 7 份可编译 C++ 示例代码（`kernels/*.cpp`）② 40+ 道高频面试问答 ③ 内存布局/虚函数表/移动语义的手推笔记 ④ 个人面试 cheat sheet |
| **验收标准** | ① 能画出 C++ 对象内存布局（含虚函数表、多重继承）② 能手写 thread-safe singleton 与 RAII 资源管理类 ③ 能解释 `std::move`/`std::forward` 的区别与引用折叠规则 ④ 能说出 `shared_ptr` 的控制块结构与线程安全性 ⑤ 能解释 6 种 memory order 的语义与适用场景 |
| **面试覆盖** | 覆盖大厂 C++ 一二面 80%+ 高频题：内存管理、智能指针、移动语义、模板、多态、并发、现代特性 |

### 本专题与每日教程的区别

| 维度 | 每日教程（daily） | 本 C++ 面试专题 |
|------|-------------------|-----------------|
| **焦点** | CUDA kernel 与 GPU 优化 | C++ 语言本身的高频面试点 |
| **代码** | `.cu` CUDA kernel | 纯 C++（`.cpp`，`g++` 编译） |
| **目的** | 学会写 GPU kernel | 应对 C++ 技术面试 |
| **联系** | CUDA 是 C++ 的扩展，本专题为理解 CUTLASS 模板、推理框架 C++ 源码打基础 | Day 4 模板直接联系 [CUTLASS 专题](../cutlass/README.md) 的模板设计 |

> 💡 **一句话总结**：CUDA 是 C++ 的方言，CUTLASS 是 C++ 模板的极致运用，推理框架（vLLM/TensorRT）的 host 侧全是 C++。本专题补齐"语言层"的面试基本功，让你在读懂 [DeepGEMM 专题](../deepgemm/README.md) 的 `csrc/` C++ 代码、[CUTLASS 专题](../cutlass/README.md) 的模板抽象时不再有语言障碍。

### 前置准备清单

#### 环境

- [ ] C++ 编译器支持 C++20（`g++ --version` ≥ 11 或 `clang++ --version` ≥ 14）
- [ ] CMake ≥ 3.18（可选，部分示例用 CMake 组织）
- [ ] 能运行 `g++ -std=c++20 -pthread -o demo demo.cpp && ./demo`

#### 验证命令
```bash
# 验证编译器
g++ --version
# 预期输出：g++ (Ubuntu 11.x / 13.x) 11.4 / 13.0 ...

# 验证 C++20 支持
g++ -std=c++20 -x c++ -E - <<'EOF' < /dev/null
#include <version>
int main() { return 0; }
EOF
# 预期：无报错

# 验证线程支持
g++ -std=c++20 -pthread -x c++ -o /tmp/thread_test - <<'EOF'
#include <thread>
#include <iostream>
int main() { std::thread t([]{ std::cout << "ok\n"; }); t.join(); return 0; }
EOF
/tmp/thread_test
# 预期输出：ok
```

#### 必读资源（本周会反复用到）
- ⭐ [cppreference.com](https://zh.cppreference.com/) — C++ 标准库权威参考
- ⭐ [Effective Modern C++](https://www.oreilly.com/library/view/effective-modern-c/9781491908419/) — Scott Meyers，C++11/14 最佳实践
- 📌 [cpppatterns.com](https://cpppatterns.com/) — 现代 C++ 模式速查
- 📌 [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) — Bjarne Stroustrup & Herb Sutter
- 📎 [左值右值详解](https://en.cppreference.com/w/cpp/language/value_category) — 值类别（lvalue/xvalue/prvalue）

---

## Day 1（周一）：内存模型与基础语义

> **今日目标**：理解 C++ 内存区域划分（栈/堆/全局/常量）、对象生命周期、指针 vs 引用的本质区别、`const` 与 `constexpr` 语义、类型系统与值类别（lvalue/rvalue/xvalue）
> **考察度**：⭐⭐⭐⭐ 高频考点，"栈和堆的区别""指针和引用的区别"几乎是必问题

详见 [day1.md](day1.md)

---

## Day 2（周二）：RAII 与智能指针

> **今日目标**：掌握 RAII 资源管理原则、`unique_ptr`/`shared_ptr`/`weak_ptr` 的实现原理与使用场景、循环引用问题、自定义 deleter、`make_unique`/`make_shared` 的优势
> **考察度**：⭐⭐⭐⭐⭐ 核心考点，智能指针是 C++ 面试的"送分题"也是"送命题"

详见 [day2.md](day2.md)

---

## Day 3（周三）：移动语义与完美转发

> **今日目标**：理解右值引用、`std::move` 的本质（只是类型转换不移动任何东西）、移动构造/赋值、`std::forward` 与完美转发、引用折叠规则、RVO/NRVO 返回值优化
> **考察度**：⭐⭐⭐⭐⭐ 核心考点，"`std::move` 做了什么"是区分 C++ 水平的分水岭

详见 [day3.md](day3.md)

---

## Day 4（周四）：模板与泛型编程

> **今日目标**：掌握函数模板/类模板、模板特化与偏特化、变参模板（parameter pack）、SFINAE 与 `if constexpr`、C++20 concepts，联系 CUTLASS 的模板设计哲学
> **考察度**：⭐⭐⭐⭐ 进阶考点，AI Infra 方向必问（CUTLASS/DeepGEMM 全是模板）

详见 [day4.md](day4.md)

---

## Day 5（周五）：面向对象与多态底层

> **今日目标**：深入虚函数表（vtable）与虚表指针（vptr）、虚析构函数的必要性、多重继承的内存布局、纯虚函数与抽象类、Rule of 3/5/0、对象切片
> **考察度**：⭐⭐⭐⭐⭐ 核心考点，"虚函数表怎么实现的"是经典面试题

详见 [day5.md](day5.md)

---

## Day 6（周六）：并发编程基础

> **今日目标**：掌握 `std::thread` 基础、`mutex`/`lock_guard`/`unique_lock`、`condition_variable`、`std::atomic` 与 6 种 memory order、死锁预防、thread-safe singleton
> **考察度**：⭐⭐⭐⭐ 实战考点，并发是系统岗位必问

详见 [day6.md](day6.md)

---

## Day 7（周日）：现代 C++ 新特性与高频题复盘

> **今日目标**：汇总 C++11/14/17/20/23 关键新特性、整理 40+ 道高频面试题答案、完成个人面试 cheat sheet、全周知识串联
> **考察度**：⭐⭐⭐⭐ 总结复盘，查漏补缺

详见 [day7.md](day7.md)

---

## 目录结构

```
aiinfra/topics/cpp/
├── README.md                         # 本文件（一周学习计划）
├── day1.md                           # 内存模型与基础语义
├── day2.md                           # RAII 与智能指针
├── day3.md                           # 移动语义与完美转发
├── day4.md                           # 模板与泛型编程
├── day5.md                           # 面向对象与多态底层
├── day6.md                           # 并发编程基础
├── day7.md                           # 现代 C++ 新特性与高频题复盘
└── kernels/                          # 可编译代码示例
    ├── memory_model_basics.cpp       # Day 1: 内存模型与值类别
    ├── smart_pointers.cpp            # Day 2: 智能指针全演示
    ├── move_semantics.cpp            # Day 3: 移动语义与完美转发
    ├── templates_and_sfinae.cpp      # Day 4: 模板/SFINAE/concepts
    ├── vtable_and_polymorphism.cpp   # Day 5: 虚函数表与多态
    ├── concurrency_basics.cpp        # Day 6: 并发原语
    └── modern_cpp_features.cpp       # Day 7: C++11~23 新特性
```

> 💡 **后续延伸**：完成本专题后，建议带着 C++ 模板知识回到 [CUTLASS 专题](../cutlass/README.md) Day 4（三层抽象源码精读）——你会发现自己能看懂 `template <typename Shape, typename WarpShape, ...>` 的每个参数含义。再读 [DeepGEMM 专题](../deepgemm/README.md) 的 `csrc/jit/compiler.hpp` 时，`std::unordered_map`、`std::filesystem`、`std::atomic` 等也不再陌生。C++ 是 AI Infra 的"母语"，掌握它后再读任何框架源码都会更顺畅。
