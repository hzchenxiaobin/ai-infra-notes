# Day 7（周日）：现代 C++ 新特性与高频题复盘

> **本周定位**：C++ 面试系统化准备的收官日，汇总新特性 + 全周高频题复盘
> **前置要求**：已完成 Day 1-6
> **今日目标**：汇总 C++11/14/17/20/23 关键新特性、整理 40+ 道高频面试题速查表、完成个人面试 cheat sheet、全周知识串联
> **时间投入**：4h（早间 1.5h 新特性汇总 + 下午 1.5h 高频题复盘 + 晚间 1h cheat sheet 整理）
> **考察度**：⭐⭐⭐⭐ 总结复盘，查漏补缺

---

## 本日在本周知识图谱中的位置

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| C++11~23 新特性速查表 | 全周知识串联 |
| 40+ 道高频面试题速查表 | 面试覆盖度验证 |
| 个人面试 cheat sheet | 最终产出 |
| 7 天知识点关联图 | 知识体系化 |

---

### 学习任务 1：C++11~23 新特性速览（60 分钟）

#### C++ 各版本关键特性

| 版本 | 年份 | 关键特性 | 面试频率 |
|------|------|----------|----------|
| **C++11** | 2011 | auto / lambda / 智能指针 / 移动语义 / 右值引用 / `nullptr` / range-for / `constexpr` / 变参模板 / `thread` / `atomic` / `override`/`final` | ⭐⭐⭐⭐⭐ |
| **C++14** | 2014 | 函数返回类型推导 / 泛型 lambda / `make_unique` / 数字字面量分隔符 / `decltype(auto)` | ⭐⭐⭐ |
| **C++17** | 2017 | `if constexpr` / 结构化绑定 / `std::optional`/`variant`/`any` / `string_view` / `filesystem` / 折叠表达式 / CTAD / `[[nodiscard]]`/`[[maybe_unused]]` | ⭐⭐⭐⭐ |
| **C++20** | 2020 | Concepts / Ranges / Coroutines / Modules / `<=>` 飞船运算符 / `consteval`/`constinit` / `std::format` / `std::span` / 指定初始化 | ⭐⭐⭐⭐ |
| **C++23** | 2023 | `std::expected` / `std::print` / Deducing this / `std::mdspan` / `if consteval` / `std::flat_map` | ⭐⭐ |

#### C++11 核心特性（面试必问）

```cpp
// modern_cpp_features.cpp —— C++11~23 新特性演示
// 编译: g++ -std=c++20 -o modern modern_cpp_features.cpp && ./modern

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <string_view>
#include <format>
#include <algorithm>
#include <numeric>

void demo_cpp11() {
    std::cout << "=== C++11 ===" << std::endl;

    // 1. auto 类型推导
    auto x = 42;        // int
    auto y = 3.14;      // double
    auto s = "hello"s;  // std::string

    // 2. lambda 表达式
    auto add = [](int a, int b) { return a + b; };
    int captured = 10;
    auto add_capture = [captured](int a) { return a + captured; };  // 值捕获
    auto add_ref = [&captured](int a) { return a + captured; };     // 引用捕获

    // 3. nullptr（取代 NULL / 0）
    int* p = nullptr;

    // 4. range-for
    std::vector<int> v = {1, 2, 3, 4, 5};
    for (const auto& elem : v) {
        std::cout << "  " << elem;
    }
    std::cout << std::endl;

    // 5. override / final
    // 在派生类中标记 override 确保正确重写虚函数

    // 6. uniform initialization（统一初始化）
    std::vector<int> v2{1, 2, 3};  // 列表初始化
    int arr[]{10, 20, 30};         // 数组初始化
}
```

#### C++14 增量特性

```cpp
void demo_cpp14() {
    std::cout << "\n=== C++14 ===" << std::endl;

    // 1. 函数返回类型推导
    auto get_size() -> size_t;  // C++11 trailing return
    auto get_value() { return 42; }  // C++14 自动推导

    // 2. 泛型 lambda
    auto generic = [](auto a, auto b) { return a + b; };
    std::cout << "  generic(1, 2) = " << generic(1, 2) << std::endl;
    std::cout << "  generic(1.5, 2.5) = " << generic(1.5, 2.5) << std::endl;

    // 3. make_unique（C++11 只有 make_shared）
    auto ptr = std::make_unique<int>(42);

    // 4. 数字字面量分隔符
    long big_num = 1'000'000;  // 提高可读性
    std::cout << "  1'000'000 = " << big_num << std::endl;
}
```

#### C++17 核心特性

```cpp
void demo_cpp17() {
    std::cout << "\n=== C++17 ===" << std::endl;

    // 1. 结构化绑定
    std::pair p = {1, "hello"s};
    auto [num, str] = p;
    std::cout << "  结构化绑定: " << num << ", " << str << std::endl;

    // 2. std::optional（可能没有值）
    std::optional<int> find_even(const std::vector<int>& v) {
        for (auto x : v) if (x % 2 == 0) return x;
        return std::nullopt;
    }
    auto opt = find_even({1, 3, 5, 8});
    if (opt) std::cout << "  找到偶数: " << *opt << std::endl;

    // 3. std::variant（类型安全的 union）
    std::variant<int, double, std::string> var = "hello";
    std::visit([](auto&& v) { std::cout << "  variant: " << v << std::endl; }, var);

    // 4. std::string_view（不拷贝的字符串视图）
    std::string long_str = "a very long string";
    std::string_view sv = long_str.substr(2, 5);  // 不拷贝
    std::cout << "  string_view: " << sv << std::endl;

    // 5. if constexpr（Day 4 详讲）

    // 6. CTAD（类模板参数推导）
    std::pair pr = {1, 2.0};   // 推导为 pair<int, double>
    std::vector vec = {1, 2, 3}; // 推导为 vector<int>
    std::cout << "  CTAD pair: (" << pr.first << ", " << pr.second << ")" << std::endl;
}
```

#### C++20 核心特性

```cpp
void demo_cpp20() {
    std::cout << "\n=== C++20 ===" << std::endl;

    // 1. Concepts（Day 4 详讲）

    // 2. Ranges（函数式管道）
    std::vector<int> nums = {1, 2, 3, 4, 5, 6};
    auto result = nums
        | std::views::filter([](int n) { return n % 2 == 0; })
        | std::views::transform([](int n) { return n * n; });
    std::cout << "  Ranges: ";
    for (int n : result) std::cout << n << " ";
    std::cout << std::endl;

    // 3. <=> 飞船运算符（三路比较）
    auto cmp = (1 <=> 2);  // std::strong_ordering::less
    std::cout << "  1 <=> 2: " << (cmp < 0 ? "less" : cmp > 0 ? "greater" : "equal") << std::endl;

    // 4. std::format（取代 printf / stringstream）
    std::string formatted = std::format("  format: x={}, y={:.2f}", 42, 3.14159);
    std::cout << formatted << std::endl;

    // 5. consteval（必须编译期求值，vs constexpr 可选）
    // consteval int square(int n) { return n * n; }
    // int s = square(5);  // 保证编译期求值

    // 6. std::span（数组视图，类似 string_view）
    int arr[] = {10, 20, 30};
    std::span<int> sp = arr;
    std::cout << "  span size: " << sp.size() << std::endl;
}
```

#### C++23 速览

```cpp
void demo_cpp23() {
    std::cout << "\n=== C++23 ===" << std::endl;

    // 1. std::expected（错误处理，替代异常/可选值）
    // std::expected<int, std::string> divide(int a, int b) {
    //     if (b == 0) return std::unexpected("除零错误");
    //     return a / b;
    // }

    // 2. std::print（直接格式化输出，取代 cout << ）
    // std::print("x = {}, y = {}\n", 42, 3.14);

    // 3. Deducing this（显式 this 参数）
    // struct Widget {
    //     void foo(this Widget& self) { }  // 显式 this
    // };

    // 4. std::mdspan（多维数组视图，AI Infra 场景有用）
    // 类似 CUDA 中的 tensor view

    // 注意：C++23 需要较新编译器支持，面试中了解即可
    std::cout << "  (C++23 特性需较新编译器支持)" << std::endl;
}
```

### 学习任务 2：40+ 高频面试题速查表（60 分钟）

#### Day 1：内存模型与基础语义

| # | 题目 | 核心答案 |
|---|------|----------|
| 1 | 栈和堆的区别 | 栈自动管理/快/小/无碎片；堆手动管理/慢/大/有碎片 |
| 2 | 指针和引用的区别 | 引用必须初始化/不可重新绑定/不能为空/无算术/不占独立存储 |
| 3 | `const int*` vs `int* const` | const 在 `*` 左修饰数据，右修饰指针 |
| 4 | `const` vs `constexpr` | const 只读（可运行时），constexpr 编译期常量 |
| 5 | 左值和右值 | 能取地址是左值，不能是右值；`std::move` 把 lvalue 转 xvalue |
| 6 | 返回局部变量引用 | UB（悬空引用）；用返回值（RVO）或智能指针 |

#### Day 2：RAII 与智能指针

| # | 题目 | 核心答案 |
|---|------|----------|
| 7 | RAII 是什么 | 资源获取即初始化，绑定资源生命周期到对象生命周期 |
| 8 | `unique_ptr` vs `shared_ptr` | 独占/零开销/不可拷贝 vs 共享/引用计数/有控制块 |
| 9 | `shared_ptr` 控制块 | 强计数 + 弱计数 + deleter + allocator |
| 10 | `make_shared` vs `new` | 1 次分配/缓存友好 vs 2 次分配；但 make_shared 内存延迟释放 |
| 11 | 循环引用与 `weak_ptr` | 互相 shared_ptr 引用导致泄漏；用 weak_ptr 打破，lock() 提升 |
| 12 | `shared_ptr` 线程安全 | 引用计数原子安全；对象本身不安全；同一 shared_ptr 变量并发读写不安全 |
| 13 | 手写 `unique_ptr` | 禁拷贝 + 移动语义(noexcept) + 析构 + operator*/->/get |

#### Day 3：移动语义与完美转发

| # | 题目 | 核心答案 |
|---|------|----------|
| 14 | `std::move` 做了什么 | 只是 `static_cast` 到右值引用，不移动任何东西 |
| 15 | `std::move` vs `std::forward` | move 无条件转右值；forward 条件转（保持原始值类别） |
| 16 | 引用折叠规则 | `&& &&` → `&&`，其余 → `&` |
| 17 | 转发引用 `T&&` | 模板中的 T&&，根据实参推导为 T& 或 T&& |
| 18 | 移动构造为什么 `noexcept` | vector 扩容用移动(noexcept) vs 拷贝(安全) |
| 19 | 完美转发 | `forward<T>` + 引用折叠，保持参数原始类别 |
| 20 | RVO/NRVO | RVO(C++17 强制) / NRVO(编译器优化)；不要 `return std::move(local)` |

#### Day 4：模板与泛型编程

| # | 题目 | 核心答案 |
|---|------|----------|
| 21 | 模板为什么在头文件 | 编译器需完整定义才能实例化 |
| 22 | 全特化 vs 偏特化 | 全特化无模板参数；偏特化保留部分；函数模板不支持偏特化（用重载） |
| 23 | 变参模板 | `typename... Args`；C++17 折叠表达式简化展开 |
| 24 | SFINAE | 替换失败不是错误，从重载候选移除 |
| 25 | `if constexpr` vs `if` | 编译期求值，false 分支不编译 |
| 26 | C++20 Concepts | 命名约束模板参数，取代 SFINAE，报错友好 |

#### Day 5：面向对象与多态底层

| # | 题目 | 核心答案 |
|---|------|----------|
| 27 | 虚函数实现 | vtable(函数地址表) + vptr(每对象) + 间接调用 |
| 28 | 虚析构必要性 | 非 virtual 析构通过基类指针 delete 派生类 → 派生类析构不调用 → 泄漏 |
| 29 | 构造函数能 virtual 吗 | 不能，vptr 在构造时才设置 |
| 30 | 多重继承内存布局 | 多个 vptr + 基类子对象按序排列；指针转换需偏移调整 |
| 31 | Rule of 3/5/0 | 3: 析构+拷贝×2；5: +移动×2；0: 用智能指针，都不定义 |
| 32 | 对象切片 | 按值传派生类给基类参数 → 派生部分丢失；用引用/指针避免 |

#### Day 6：并发编程

| # | 题目 | 核心答案 |
|---|------|----------|
| 33 | `join` vs `detach` | join 等待完成；detach 后台分离；必须处理其一 |
| 34 | `lock_guard` vs `unique_lock` | lock_guard 不可中途解锁(简单)；unique_lock 灵活(支持 cv) |
| 35 | 虚假唤醒 | wait 无 notify 也可能返回；用谓词版 wait 或 while 循环 |
| 36 | 6 种 memory order | relaxed/acquire/release/acq_rel/seq_cst/consume；默认 seq_cst |
| 37 | 线程安全 Singleton | Meyers' Singleton（局部 static，C++11 保证）或 call_once |
| 38 | 死锁预防 | 固定顺序 / scoped_lock / try_lock+回退 / 避免嵌套锁 |

#### 综合高频题

| # | 题目 | 核心答案 |
|---|------|----------|
| 39 | `new`/`delete` vs `malloc`/`free` | new 调构造+返回类型化指针；malloc 只分配内存；delete 调析构 |
| 40 | `delete` vs `delete[]` | delete 用于单对象（调析构+释放）；delete[] 用于数组（逐个析构+释放） |
| 41 | 虚函数能内联吗 | 智能指针直接调函数体时可内联(devirtualization)；通过引用/指针动态调用时不可内联 |
| 42 | `emplace_back` vs `push_back` | emplace 直接在容器内存中构造(完美转发)；push_back 先构造临时对象再移动/拷贝 |
| 43 | `override` vs `final` | override 标记重写虚函数(编译器检查)；final 禁止进一步重写/继承 |
| 44 | 浅拷贝 vs 深拷贝 | 浅拷贝复制指针(共享数据/double free 风险)；深拷贝复制指向的数据 |

### 学习任务 3：7 天知识点关联图（30 分钟）

```
Day 1: 内存模型与基础语义
  │
  ├─ 栈/堆 → Day 2: RAII（栈对象确定性析构管理堆资源）
  │
  ├─ 值类别(lvalue/rvalue) → Day 3: 移动语义（右值引用 + std::move）
  │                            │
  │                            └─ 引用折叠 → Day 4: 完美转发 + 变参模板
  │                                              │
  │                                              └─ 模板 → CUTLASS/DeepGEMM 源码
  │
  ├─ 对象生命周期 → Day 5: 虚函数表 + Rule of 5
  │                    │
  │                    └─ 析构 → Day 2: 智能指针(虚析构保证正确释放)
  │
  └─ 存储期(thread_local) → Day 6: 并发(thread/atomic/memory order)
                               │
                               └─ shared_ptr 线程安全 → Day 6: atomic 引用计数
```

> 💡 **知识串联**：Day 1 的内存模型是所有后续内容的基础——Day 2 的 RAII 用栈管理堆，Day 3 的移动语义优化堆对象的传递，Day 4 的模板生成类型安全的代码，Day 5 的虚函数表是对象内存布局的延伸，Day 6 的并发原子操作是内存模型的进阶。

### 学习任务 4：个人面试 Cheat Sheet 模板（30 分钟）

整理一份面试速查表，建议按以下结构：

```markdown
# C++ 面试 Cheat Sheet

## 1. 内存管理
- 栈 vs 堆：[5 个区别]
- RAII：[核心原则]
- 智能指针选择：unique_ptr(默认) > shared_ptr(共享) > weak_ptr(观察)

## 2. 移动语义
- std::move = static_cast<T&&>（不移动）
- std::forward = 条件转换（保持值类别）
- 引用折叠：只有 && && → &&
- 移动操作标 noexcept

## 3. 模板
- 函数模板不支持偏特化（用重载）
- SFINAE → if constexpr → Concepts
- 变参模板 + 折叠表达式

## 4. 多态
- vtable + vptr → 间接调用
- 虚析构必须（基类指针 delete 派生类）
- Rule of 0（用智能指针）

## 5. 并发
- lock_guard(简单) / unique_lock(灵活)
- condition_variable + 谓词(防虚假唤醒)
- memory order: seq_cst(默认) / relaxed(计数) / release-acquire(同步)
- Singleton: Meyers' (局部 static)

## 6. 高频陷阱
- 不要 return std::move(local)（阻止 NRVO）
- 不要返回局部变量引用（悬空引用）
- 不要忘记虚析构（资源泄漏）
- 不要手动 lock/unlock（用 RAII）
- shared_ptr 循环引用（用 weak_ptr）
```

### 学习任务 5：AI Infra 方向 C++ 面试额外准备（20 分钟）

AI Infra / CUDA 方向的 C++ 面试除了语言基础，还可能涉及：

| 额外考点 | 关联本专题 | 说明 |
|----------|-----------|------|
| C++ 模板元编程 | Day 4 | CUTLASS 的 `CollectiveBuilder` 全是模板 |
| RAII 管理 CUDA 资源 | Day 2 | `cudaMalloc`/`cudaFree` 用 RAII 封装 |
| 移动语义避免 GPU 数据拷贝 | Day 3 | Tensor/Buffer 对象的移动 |
| `std::atomic` 与 CUDA 原子操作对比 | Day 6 | 概念类似，API 不同 |
| 虚函数在 CUDA 中的限制 | Day 5 | `__device__` 代码中虚函数支持有限 |

```cpp
// AI Infra 场景：RAII 管理 CUDA 资源
class CudaBuffer {
    void* ptr_;
    size_t size_;
public:
    explicit CudaBuffer(size_t n) : size_(n) {
        cudaMalloc(&ptr_, n);  // 获取资源
    }
    ~CudaBuffer() {
        cudaFree(ptr_);  // RAII 自动释放
    }
    // Rule of 5...
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
        o.ptr_ = nullptr;
    }
    void* get() const { return ptr_; }
    size_t size() const { return size_; }
};
```

### 今日检查清单

- [ ] 能列出 C++11 的 5 个以上核心特性
- [ ] 能说出 C++17 的结构化绑定 / `optional` / `variant` / `string_view` 的用途
- [ ] 能说出 C++20 的 Concepts / Ranges / `std::format` 的用途
- [ ] 完成了 40+ 道高频面试题的速查表
- [ ] 能画出 7 天知识点关联图
- [ ] 整理了个人面试 cheat sheet
- [ ] 能说出 AI Infra 方向 C++ 面试的额外考点
- [ ] `modern_cpp_features.cpp` 编译运行通过

#### 本周总结

Day 7 我们完成了 C++ 面试的系统化复盘：

1. **C++11~23 新特性**：从 auto/lambda/智能指针到 Concepts/Ranges/`std::format`，掌握每个版本的关键特性
2. **40+ 高频面试题**：覆盖 7 大主题，每题有核心答案速查
3. **知识体系化**：7 天的知识点串联成完整体系——内存模型是基础，RAII/移动语义/模板/多态/并发是上层
4. **AI Infra 联系**：C++ 是 CUDA/CUTLASS/推理框架的底层语言，掌握 C++ 后再读源码事半功倍

> 💡 **后续建议**：① 每天挑 3-5 道题口述练习，模拟面试场景；② 回到 [CUTLASS 专题](../cutlass/README.md) 或 [DeepGEMM 专题](../deepgemm/README.md) 读源码，检验 C++ 知识的实际应用；③ 针对薄弱主题（如模板元编程、memory order）做专项练习。

---
