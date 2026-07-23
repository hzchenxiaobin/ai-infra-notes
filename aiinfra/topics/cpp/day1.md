# Day 1（周一）：内存模型与基础语义

> **本周定位**：本专题是 C++ 面试的系统化准备，覆盖语言核心高频考点。本周目标是每天吃透一个主题，配可编译代码与面试问答，最终能应对大厂 C++ 一二面。
> **前置要求**：有 C/C++ 基础语法知识，能独立编写简单 C++ 程序
> **今日目标**：理解 C++ 内存区域划分（栈/堆/全局/常量/代码段）、对象生命周期与存储期、指针 vs 引用的本质区别、`const`/`constexpr` 语义、值类别（lvalue/rvalue/xvalue/prvalue），能回答"栈和堆的区别""指针和引用的区别"等高频题
> **时间投入**：2.5h（早间 1.5h 精读内存模型 + 晚间 1h 跑代码与值类别实验）
> **考察度**：⭐⭐⭐⭐ 高频考点，几乎每场 C++ 面试都会涉及

---

## 本日在本周知识图谱中的位置

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| C++ 五大内存区域划分表 | ① 能画出 C++ 对象内存布局（基础） |
| 指针 vs 引用对比表 | ① 同上 |
| 值类别（lvalue/rvalue/xvalue）判断练习 | ③ 为 Day 3 移动语义打基础 |
| `const`/`constexpr` 语义对比 | ④ 为 Day 4 模板非类型参数打基础 |

---

### 学习任务 1：C++ 内存区域划分（45 分钟）

#### 五大内存区域

C++ 程序运行时的内存分为五个区域，每个区域的生命周期和管理方式不同：

| 内存区域 | 存储内容 | 分配方式 | 生命周期 | 访问速度 |
|----------|----------|----------|----------|----------|
| **栈（Stack）** | 局部变量、函数参数、返回地址 | 编译器自动分配/释放 | 函数返回即销毁 | 最快（移动栈指针） |
| **堆（Heap）** | `new`/`malloc` 分配的对象 | 手动分配/释放（或智能指针） | 手动释放或 RAII | 较慢（需要搜索空闲块） |
| **全局/静态区** | 全局变量、静态变量 | 程序启动分配 | 程序结束销毁 | 中等 |
| **常量区** | 字符串字面量、`const` 全局变量 | 程序启动分配 | 程序结束销毁 | 中等 |
| **代码区** | 编译后的机器指令 | 程序加载时 | 程序结束 | 只读 |

> 💡 **一句话总结**：栈是编译器管理的"自动挡"，堆是程序员管理的"手动挡"——智能指针（Day 2）就是给堆装上"自动挡"。

#### 栈 vs 堆的深度对比

| 维度 | 栈 | 堆 |
|------|------|------|
| 分配/释放 | 编译器自动，函数返回即释放 | 手动 `new`/`delete`，或智能指针 |
| 空间大小 | 有限（Linux 默认 8MB，`ulimit -s`） | 大（受物理内存限制） |
| 碎片 | 无（连续分配/释放） | 有（外部碎片） |
| 速度 | O(1)，只需移动栈指针 | 较慢，需搜索空闲块 + 可能触发系统调用 |
| 线程安全 | 是（每线程独立栈） | 否（需同步，或用线程局部存储） |
| 生长方向 | 向下（高地址 → 低地址） | 向上（低地址 → 高地址） |

#### 代码验证（`kernels/memory_model_basics.cpp`）

```cpp
// memory_model_basics.cpp —— C++ 内存区域与生命周期演示
// 编译: g++ -std=c++20 -o memory_model_basics memory_model_basics.cpp && ./memory_model_basics

#include <iostream>
#include <string>

int g_global = 42;              // 全局/静态区
static int s_static = 100;      // 全局/静态区
const char* g_str = "hello";    // "hello" 在常量区，g_str 指针在全局区

void demo_memory_regions() {
    int stack_var = 1;          // 栈
    static int func_static = 2; // 全局/静态区（只初始化一次）
    int* heap_ptr = new int(3); // 堆

    std::cout << "=== 内存区域地址 ===" << std::endl;
    std::cout << "栈变量 stack_var:       " << &stack_var << std::endl;
    std::cout << "堆指针 heap_ptr 指向:   " << heap_ptr << std::endl;
    std::cout << "堆指针 heap_ptr 本身:   " << &heap_ptr << " (在栈上)" << std::endl;
    std::cout << "全局变量 g_global:      " << &g_global << std::endl;
    std::cout << "静态变量 s_static:      " << &s_static << std::endl;
    std::cout << "函数静态 func_static:   " << &func_static << std::endl;
    std::cout << "字符串字面量 g_str 指向:" << (const void*)g_str << std::endl;

    // 栈地址 vs 堆地址 vs 全局地址：观察地址范围差异
    // 栈地址通常最大（高地址），全局区居中，堆在中间

    delete heap_ptr;  // 手动释放堆内存
}

int main() {
    demo_memory_regions();
    return 0;
}
```

```bash
g++ -std=c++20 -o memory_model_basics memory_model_basics.cpp && ./memory_model_basics
```

```text
=== 内存区域地址 ===
栈变量 stack_var:       0x7ffd3a2b1c3c
堆指针 heap_ptr 指向:   0x55a1e8c2a2e0
堆指针 heap_ptr 本身:   0x7ffd3a2b1c30 (在栈上)
全局变量 g_global:      0x55a1e8b9d010
静态变量 s_static:      0x55a1e8b9d014
函数静态 func_static:   0x55a1e8b9d018
字符串字面量 g_str 指向:0x55a1e8b98214
```

> ⚠️ **注意**：`heap_ptr` 本身是一个指针变量，它在**栈**上（8 字节），它指向的 `new int(3)` 才在**堆**上。面试中常被问"指针本身存在哪里"——答案是指针变量本身在栈上（如果它是局部变量），它指向的对象在堆上。

#### 存储期（Storage Duration）

C++ 定义了四种存储期，对应对象的生存时间：

| 存储期 | 关键字/特征 | 生命周期 |
|--------|------------|----------|
| **自动存储期** | 局部变量（无 `static`） | 所在代码块结束 |
| **静态存储期** | 全局变量、`static` 变量 | 程序结束 |
| **动态存储期** | `new` 分配的对象 | `delete` 时 |
| **线程存储期** | `thread_local` 变量 | 线程结束 |

```cpp
void demo_storage_duration() {
    int auto_var = 1;               // 自动存储期：函数返回销毁
    static int static_var = 0;      // 静态存储期：程序结束销毁，只初始化一次
    thread_local int tl_var = 0;    // 线程存储期：线程结束销毁
    int* dyn_var = new int(1);      // 动态存储期：delete 时销毁
    // ...
    delete dyn_var;
}
```

### 学习任务 2：指针 vs 引用（45 分钟）

这是 C++ 面试**最经典的对比题**，必须能脱口而出 5 个以上区别。

#### 核心区别对比

| 维度 | 指针（Pointer） | 引用（Reference） |
|------|-----------------|-------------------|
| **本质** | 存储地址的变量（有自己的内存） | 已有对象的别名（不占独立内存） |
| **初始化** | 可以不初始化（野指针风险） | **必须**在声明时初始化 |
| **重新绑定** | 可以指向不同对象 | 绑定后不可改变 |
| **空值** | 可以是 `nullptr` | 不能为空（无"空引用"） |
| **算术** | 支持指针算术（`ptr++`, `ptr+1`） | 不支持 |
| **多级** | 有多级指针（`int**`） | 无多级引用 |
| **自增含义** | `ptr++` 移动到下一个元素 | `ref++` 是对引用对象的 `++` |
| **sizeof** | 指针大小（64 位系统 8 字节） | 被引用对象的大小 |
| **访问成员** | `ptr->member` | `ref.member` |

> 💡 **面试标准答案**：引用是对象的别名，必须初始化且不可重新绑定，不能为空，不占独立存储；指针是存储地址的独立变量，可以为空、可以重新赋值、支持算术运算。引用更安全（不会空悬），指针更灵活。

#### 什么时候用指针，什么时候用引用？

| 场景 | 推荐 | 原因 |
|------|------|------|
| 函数参数（只读） | `const T&` | 避免拷贝，语法简洁 |
| 函数参数（需修改实参） | `T&` | 比指针更直观 |
| 函数参数（可选参数） | `T*`（可为 `nullptr`） | 表达"可能不存在" |
| 函数参数（所有权转移） | `T*` 或 `std::unique_ptr<T>` | 明确所有权语义 |
| 类成员（生命周期独立） | `T*` / 智能指针 | 需要动态管理 |
| 类成员（生命周期依赖外部） | `T&` 或 `T*` | 不持有所有权 |
| 返回值（可能无效） | `T*`（返回 `nullptr`） | 引用不能返回空 |

> ⚠️ **注意**：引用底层实现通常也是指针，但语义上不允许为空和重新绑定。编译器优化时可能完全消除引用（直接用原对象地址）。

#### 常见陷阱

```cpp
// 陷阱 1：返回局部变量的引用（悬空引用）
int& dangerous() {
    int x = 42;
    return x;  // UB！x 在函数返回后销毁
}

// 陷阱 2：引用绑定到临时对象（延长生命周期，但有限制）
const std::string& s = std::string("temp");  // 合法：const 引用延长临时对象生命周期
// std::string& s2 = std::string("temp");   // 非法：非 const 引用不能绑定临时对象

// 陷阱 3：引用与指针混合使用
int a = 1, b = 2;
int& ref = a;   // ref 绑定 a
ref = b;        // 这是赋值！a 变成 2，不是重新绑定 ref 到 b
```

### 学习任务 3：const 与 constexpr 语义（30 分钟）

#### const 的多层语义

`const` 在不同位置含义不同，是面试常考的"读代码"题：

```cpp
// 指针与 const 的四种组合
const int* p1;        // 指向 const int 的指针：不能通过 p1 修改数据
int const* p2;        // 同上（等价写法）
int* const p3 = &a;   // const 指针指向 int：指针本身不可变，数据可改
const int* const p4 = &a;  // 都不可变

// 记忆口诀：const 在 * 左边修饰数据，在 * 右边修饰指针
```

| 声明 | 数据可变 | 指针可变 |
|------|----------|----------|
| `int* p` | ✅ | ✅ |
| `const int* p` | ❌ | ✅ |
| `int* const p` | ✅ | ❌ |
| `const int* const p` | ❌ | ❌ |

#### const 成员函数

```cpp
class Vector {
    int* data_;
    size_t size_;
public:
    // const 成员函数：保证不修改对象状态
    size_t size() const { return size_; }

    // 非 const 版本：可以修改对象
    int& operator[](size_t i) { return data_[i]; }

    // const 版本：返回 const 引用，不能修改
    const int& operator[](size_t i) const { return data_[i]; }

    // mutable：即使 const 函数也可修改
    mutable size_t access_count_ = 0;
    int at(size_t i) const {
        ++access_count_;  // 合法：mutable 成员
        return data_[i];
    }
};
```

> ⚠️ **注意**：`mutable` 允许在 const 成员函数中修改成员变量，常用于缓存、计数器等"逻辑 const"场景。但不要滥用——它打破了 const 保证。

#### const vs constexpr

| 维度 | `const` | `constexpr` |
|------|---------|-------------|
| 含义 | 只读（运行时也可初始化） | 编译期常量表达式 |
| 初始化 | 运行时值 | **必须**编译期可求值 |
| 用于函数 | 只是 const 成员函数 | 编译期可求值的函数 |
| 用于变量 | 运行期只读 | 编译期常量 |
| 数组大小 | 不行（除非也是编译期常量） | 可以 |

```cpp
const int x = get_runtime_value();  // 合法：运行时初始化，之后只读
// constexpr int y = get_runtime_value();  // 编译错误：必须编译期可求值
constexpr int z = 42;               // 合法：编译期常量

constexpr int factorial(int n) {    // constexpr 函数
    return n <= 1 ? 1 : n * factorial(n - 1);
}
constexpr int f5 = factorial(5);    // 编译期求值：120
int runtime_f = factorial(10);      // 也可以运行时调用
```

### 学习任务 4：值类别（lvalue/rvalue/xvalue）（30 分钟）

值类别是 Day 3 移动语义的前置知识，面试中"什么是左值什么是右值"也是高频题。

#### C++11 值类别体系

C++11 将表达式分为三大类：

| 类别 | 全称 | 含义 | 示例 |
|------|------|------|------|
| **lvalue** | left value | 有身份、不可移动 | `int a; a` —— 变量名 |
| **xvalue** | expiring value | 有身份、可移动 | `std::move(a)` —— 即将销毁 |
| **prvalue** | pure rvalue | 无身份、可移动 | `42`, `a + b`, `std::string("tmp")` |
| **glvalue** | generalized lvalue | 有身份（lvalue + xvalue） | `a`, `std::move(a)` |
| **rvalue** | right value | 可移动（xvalue + prvalue） | `42`, `std::move(a)` |

> 💡 **简化记忆**：能取地址的是 lvalue，不能取地址的是 rvalue。`std::move(x)` 把 lvalue 转成 xvalue（一种 rvalue）。

#### 代码验证

```cpp
// memory_model_basics.cpp（续）—— 值类别演示

void demo_value_categories() {
    int a = 10;          // a 是 lvalue
    int b = 20;          // b 是 lvalue
    int c = a + b;       // a + b 是 prvalue，c 是 lvalue

    // a 的地址可取 → lvalue
    std::cout << "&a = " << &a << std::endl;

    // a + b 的地址不可取 → prvalue
    // &(a + b);  // 编译错误

    int& lref = a;       // lvalue 引用绑定 lvalue
    // int& lref2 = 42;  // 编译错误：lvalue 引用不能绑定 prvalue
    const int& cref = 42; // 合法：const lvalue 引用可绑定 prvalue（延长生命周期）

    int&& rref = std::move(a);  // rvalue 引用绑定 xvalue
    int&& rref2 = 42;    // 合法：rvalue 引用可绑定 prvalue

    std::cout << "a = " << a << ", rref = " << rref << std::endl;
}
```

#### 面试高频辨析

| 表达式 | 类别 | 能否取地址 |
|--------|------|-----------|
| `int a; a` | lvalue | ✅ |
| `42` | prvalue | ❌ |
| `a + 1` | prvalue | ❌ |
| `std::move(a)` | xvalue | ❌ |
| `a[0]` | lvalue | ✅ |
| `*ptr` | lvalue | ✅ |
| `std::string("hi")` | prvalue | ❌ |
| `func()` （返回非引用） | prvalue | ❌ |
| `func()` （返回 `T&`） | lvalue | ✅ |
| `func()` （返回 `T&&`） | xvalue | ❌ |

### 面试题积累（今日 6 道）

**Q1：栈和堆有什么区别？什么时候用栈，什么时候用堆？**
> 答：栈由编译器自动分配释放，速度快、无碎片、空间有限（~8MB），适合局部变量；堆由程序员手动管理（`new`/`delete`），空间大但分配慢、有碎片，适合动态大小的对象或生命周期超出函数作用域的对象。实际开发中优先用栈，必须用堆时用智能指针管理。

**Q2：指针和引用有什么区别？**
> 答：① 引用必须初始化且不可重新绑定，指针可以不初始化、可以重新赋值；② 引用不能为空，指针可以是 `nullptr`；③ 引用不支持算术运算，指针支持；④ `sizeof` 引用得到的是被引用对象大小，指针得到的是指针大小（8 字节）；⑤ 引用是别名不占独立存储，指针是独立变量占 8 字节。引用更安全，指针更灵活。

**Q3：`const int* p`、`int* const p`、`const int* const p` 有什么区别？**
> 答：`const int* p`：指向 const int 的指针，不能通过 p 修改数据，但指针本身可变；`int* const p`：const 指针指向 int，指针本身不可变但数据可改；`const int* const p`：两者都不可变。口诀：const 在 `*` 左边修饰数据，在右边修饰指针。

**Q4：`const` 和 `constexpr` 有什么区别？**
> 答：`const` 表示只读，可在运行时初始化；`constexpr` 表示编译期常量表达式，必须在编译期可求值。`constexpr` 变量隐式是 `const`，但 `const` 变量不一定是 `constexpr`。`constexpr` 还可用于函数，表示该函数在输入为编译期常量时可编译期求值。

**Q5：什么是左值和右值？`std::move` 做了什么？**
> 答：左值是有身份、能取地址的表达式（如变量名）；右值是不能取地址的表达式（如字面量 `42`、临时对象）。C++11 进一步细分为 lvalue/xvalue/prvalue。`std::move` 本质是一个 `static_cast` 到右值引用，它不做任何移动操作，只是把表达式标记为"可移动"——真正的移动发生在移动构造函数/赋值运算符中。

**Q6：返回局部变量的引用会发生什么？**
> 答：返回局部变量的引用是未定义行为（UB），因为局部变量在函数返回后销毁，引用变成悬空引用（dangling reference）。如果编译器优化，可能"看似"正常但随时可能崩溃。解决方案：返回值（依赖 RVO）、返回 `std::unique_ptr`、或返回静态/堆上的对象。

### 今日检查清单

- [ ] 能列出 C++ 五大内存区域及其特点
- [ ] 能说出栈和堆的 5 个以上区别
- [ ] 理解四种存储期（自动/静态/动态/线程）
- [ ] 能说出指针和引用的 5 个以上区别
- [ ] 能区分 `const int*`、`int* const`、`const int* const`
- [ ] 能解释 `const` 成员函数与 `mutable` 的关系
- [ ] 能说出 `const` 与 `constexpr` 的区别
- [ ] 能判断常见表达式是 lvalue 还是 rvalue
- [ ] `memory_model_basics.cpp` 编译运行通过

#### 明日预告

Day 2 将深入 **RAII 与智能指针**——C++ 面试的核心考点。会讲 `unique_ptr`/`shared_ptr`/`weak_ptr` 的实现原理、控制块结构、循环引用问题与解决方案。今天理解了内存区域和对象生命周期，明天就要学习如何用 RAII 自动管理堆上的对象。建议今晚先扫一眼 `<memory>` 头文件的智能指针接口。

---
