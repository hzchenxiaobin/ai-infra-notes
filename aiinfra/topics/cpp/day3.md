# Day 3（周三）：移动语义与完美转发

> **本周定位**：C++ 面试系统化准备，今日聚焦 C++11 最重要的特性——移动语义
> **前置要求**：已完成 Day 1（值类别）与 Day 2（智能指针用到了移动语义）
> **今日目标**：理解右值引用、`std::move` 的本质（只是类型转换）、移动构造/赋值的实现、`std::forward` 与完美转发、引用折叠规则、RVO/NRVO 返回值优化，能回答"`std::move` 做了什么"这一分水岭问题
> **时间投入**：2.5h（早间 1.5h 精读移动语义 + 晚间 1h 跑代码与引用折叠实验）
> **考察度**：⭐⭐⭐⭐⭐ 核心考点，区分 C++ 水平的分水岭

---

## 本日在本周知识图谱中的位置

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| `std::move` 本质理解（`static_cast` 到右值引用） | ③ 能解释 `std::move`/`std::forward` 区别 |
| 移动构造/赋值实现与 `noexcept` 重要性 | ③ 同上 |
| 引用折叠规则表 | ③ 同上（完美转发的语言基础） |
| `std::forward` 实现与完美转发 demo | ③ 同上 |
| RVO/NRVO 机制与移动的协作 | ③ 同上 |

---

### 学习任务 1：右值引用与 std::move 的本质（45 分钟）

#### 右值引用是什么

C++11 引入右值引用 `T&&`，用于绑定**即将销毁的对象**（右值），是实现移动语义的语言基础：

| 引用类型 | 绑定对象 | 用途 |
|----------|----------|------|
| `T&`（左值引用） | 左值 | 普通引用 |
| `const T&` | 左值 + 右值 | 避免拷贝（旧方案） |
| `T&&`（右值引用） | 右值 | 移动语义 |

#### std::move 到底做了什么

**面试标准答案**：`std::move` 不移动任何东西，它只是一个到右值引用的 `static_cast`。

```cpp
// move_semantics.cpp —— 移动语义与完美转发
// 编译: g++ -std=c++20 -o move_semantics move_semantics.cpp && ./move_semantics

#include <iostream>
#include <string>
#include <utility>
#include <vector>

// std::move 的等价实现（C++11 起）
template <typename T>
constexpr typename std::remove_reference<T>::type&& my_move(T&& arg) noexcept {
    return static_cast<typename std::remove_reference<T>::type&&>(arg);
}
// C++14 简化版：
// template <typename T>
// constexpr std::remove_reference_t<T>&& my_move(T&& arg) noexcept {
//     return static_cast<std::remove_reference_t<T>&&>(arg);
// }

void demo_move_essence() {
    std::cout << "=== std::move 的本质 ===" << std::endl;
    std::string s = "hello world";
    std::cout << "  移动前: s = \"" << s << "\"" << std::endl;

    // std::move(s) 只是产生一个 std::string&& 类型的表达式
    // 真正的"移动"发生在 string 的移动构造函数中
    std::string s2 = std::move(s);

    std::cout << "  移动后: s = \"" << s << "\" (处于有效但未指定状态)" << std::endl;
    std::cout << "         s2 = \"" << s2 << "\"" << std::endl;
}
```

> ⚠️ **关键**：`std::move` 之后，源对象处于**有效但未指定（valid but unspecified）**的状态——可以对它赋值或析构，但不应该读取它的值。这是移动语义的约定。

#### 为什么需要移动语义

| 方案 | 拷贝开销 | 说明 |
|------|----------|------|
| 值传递（拷贝构造） | 深拷贝所有数据 | 昂贵（如 `std::vector` 拷贝整个数组） |
| 左值引用传递 | 无拷贝 | 但不能接受临时对象（非 const 引用） |
| `const` 引用 | 无拷贝 | 接受临时对象但不能修改 |
| **右值引用（移动）** | **无深拷贝** | **"偷走"源对象的数据，源变空壳** |

### 学习任务 2：移动构造与移动赋值（45 分钟）

#### 移动构造函数实现

```cpp
// move_semantics.cpp（续）—— 移动构造/赋值

class StringVector {
    int* data_;
    size_t size_;
    size_t cap_;
public:
    // 构造
    explicit StringVector(size_t n) : data_(new int[n]()), size_(0), cap_(n) {}

    // 拷贝构造（深拷贝，昂贵）
    StringVector(const StringVector& other)
        : data_(new int[other.cap_]), size_(other.size_), cap_(other.cap_) {
        std::copy(other.data_, other.data_ + size_, data_);
        std::cout << "  拷贝构造（深拷贝 " << size_ << " 元素）" << std::endl;
    }

    // 移动构造（偷取资源，O(1)）
    StringVector(StringVector&& other) noexcept
        : data_(other.data_), size_(other.size_), cap_(other.cap_) {
        other.data_ = nullptr;  // 关键：置空源对象，防止 double free
        other.size_ = 0;
        other.cap_ = 0;
        std::cout << "  移动构造（O(1) 偷取）" << std::endl;
    }

    // 拷贝赋值
    StringVector& operator=(const StringVector& other) {
        if (this != &other) {
            delete[] data_;
            data_ = new int[other.cap_];
            size_ = other.size_;
            cap_ = other.cap_;
            std::copy(other.data_, other.data_ + size_, data_);
            std::cout << "  拷贝赋值" << std::endl;
        }
        return *this;
    }

    // 移动赋值
    StringVector& operator=(StringVector&& other) noexcept {
        if (this != &other) {
            delete[] data_;       // 释放自己的旧资源
            data_ = other.data_;  // 偷取
            size_ = other.size_;
            cap_ = other.cap_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.cap_ = 0;
            std::cout << "  移动赋值" << std::endl;
        }
        return *this;
    }

    ~StringVector() { delete[] data_; }

    void push(int v) { if (size_ < cap_) data_[size_++] = v; }
    size_t size() const { return size_; }
};

void demo_move_constructor() {
    std::cout << "\n=== 移动构造/赋值 ===" << std::endl;
    StringVector v1(100);
    for (int i = 0; i < 100; i++) v1.push(i);

    StringVector v2 = std::move(v1);  // 调用移动构造
    std::cout << "  v2.size = " << v2.size() << std::endl;

    StringVector v3(10);
    v3 = std::move(v2);  // 调用移动赋值
    std::cout << "  v3.size = " << v3.size() << std::endl;
}
```

#### 为什么移动操作要 noexcept

这是面试高频追问点——`std::vector` 扩容时，如果元素类型的移动构造是 `noexcept`，则用移动；否则用拷贝（保证异常安全）。

```cpp
// move_semantics.cpp（续）—— noexcept 的重要性

void demo_noexcept_matters() {
    std::cout << "\n=== noexcept 对 vector 扩容的影响 ===" << std::endl;

    // 带 noexcept 移动构造的类 → vector 扩容用移动
    std::vector<StringVector> vec;
    for (int i = 0; i < 10; i++) {
        vec.emplace_back(1000);  // 扩容时会用移动构造（noexcept）
    }

    // 如果移除 noexcept，vector 会用拷贝构造（更安全但更慢）
    // 可以用 noexcept(false) 版本对比
}
```

> 💡 **面试要点**：移动构造/赋值**应该**标记为 `noexcept`。因为移动操作通常只是交换指针，不会抛异常。标记 `noexcept` 后，`std::vector` 扩容时才会使用移动而非拷贝，大幅提升性能。

### 学习任务 3：引用折叠规则（30 分钟）

引用折叠是完美转发的语言基础，面试中"引用折叠规则"是移动语义的进阶题。

#### 四条折叠规则

当模板参数推导产生"引用的引用"时，C++ 规定**右值引用的右值引用折叠为右值引用，其余折叠为左值引用**：

| 组合 | 折叠结果 |
|------|----------|
| `T& &` | `T&` |
| `T& &&` | `T&` |
| `T&& &` | `T&` |
| `T&& &&` | `T&&` |

> 💡 **记忆口诀**：只有 `&&` + `&&` = `&&`，其余都是 `&`。或者"只要有一个 `&` 就折叠成 `&`"。

#### 转发引用（Forwarding Reference）

模板中的 `T&&` 不是右值引用，而是**转发引用**（旧称万能引用 universal reference），它能根据实参类型推导为左值引用或右值引用：

```cpp
// move_semantics.cpp（续）—— 转发引用与引用折叠

template <typename T>
void show_type(T&& param) {
    // T&& 是转发引用，不是右值引用
    // 传入左值 → T 推导为 T&，param 类型为 T&（引用折叠）
    // 传入右值 → T 推导为 T，param 类型为 T&&
}

void demo_forwarding_reference() {
    std::cout << "\n=== 转发引用与引用折叠 ===" << std::endl;
    int x = 42;

    show_type(x);            // 传入左值 → T = int&, param = int& && → int&
    show_type(std::move(x)); // 传入右值 → T = int,  param = int&&
    show_type(42);           // 传入右值 → T = int,  param = int&&
}
```

> ⚠️ **注意**：`T&&` 是转发引用**当且仅当**在模板参数推导上下文中。以下情况 `T&&` 是真正的右值引用，不是转发引用：
> - `void f(std::string&& s)` —— 非模板，是右值引用
> - `template <typename T> class C { void f(T&&); }` —— T 已确定，不是推导，是右值引用

### 学习任务 4：std::forward 与完美转发（30 分钟）

#### 完美转发的问题

转发引用会丢失原始值类别——传入左值后 `param` 变成左值引用，再传递给别人时始终是左值：

```cpp
void target(int& x)       { std::cout << "  左值引用" << std::endl; }
void target(int&& x)      { std::cout << "  右值引用" << std::endl; }

template <typename T>
void bad_forward(T&& param) {
    target(param);  // param 是左值（有名字），永远调用 target(int&)
}

void demo_bad_forward() {
    bad_forward(42);  // 期望调用 target(int&&)，实际调用 target(int&)
}
```

#### std::forward 解决方案

`std::forward` 根据模板参数 T 恢复原始值类别：

```cpp
// move_semantics.cpp（续）—— 完美转发

template <typename T>
void perfect_forward(T&& param) {
    target(std::forward<T>(param));  // 恢复原始值类别
}

// std::forward 的简化实现
template <typename T>
constexpr T&& my_forward(typename std::remove_reference<T>::type& arg) noexcept {
    return static_cast<T&&>(arg);
}
// 如果 T = int&   → static_cast<int& &&> → int&  （折叠）→ 左值
// 如果 T = int    → static_cast<int&&>   → int&&       → 右值

void demo_perfect_forward() {
    std::cout << "\n=== 完美转发 ===" << std::endl;
    int x = 42;
    perfect_forward(x);            // 传入左值 → target(int&)
    perfect_forward(std::move(x)); // 传入右值 → target(int&&)
    perfect_forward(42);           // 传入右值 → target(int&&)
}
```

> 💡 **一句话总结**：`std::move` 无条件转右值，`std::forward` 条件转右值（保持原始值类别）。`std::move` 用于"我想移动它"，`std::forward` 用于"我想转发它，保持它原来的左值/右值属性"。

#### 完美转发的实际应用

```cpp
// make_shared 的完美转发
template <typename T, typename... Args>
std::shared_ptr<T> my_make_shared(Args&&... args) {
    return std::shared_ptr<T>(new T(std::forward<Args>(args)...));
}

// emplace_back 的完美转发
// std::vector<T>::emplace_back(Args&&... args) 把参数完美转发给 T 的构造函数

// 应用：避免不必要的临时对象
struct Widget {
    Widget(const std::string& s) { /* 拷贝 */ }
    Widget(std::string&& s) { /* 移动 */ }
};

void demo_emplace_vs_push() {
    std::vector<Widget> vec;
    std::string s = "hello";

    // push_back 先构造临时 Widget 再移动/拷贝入 vector
    vec.push_back(Widget(s));

    // emplace_back 直接在 vector 内存中构造，完美转发参数
    vec.emplace_back(s);          // 转发左值 → 拷贝构造
    vec.emplace_back(std::move(s)); // 转发右值 → 移动构造
}
```

### 学习任务 5：RVO/NRVO 返回值优化（15 分钟）

| 优化 | 全称 | 说明 |
|------|------|------|
| **RVO** | Return Value Optimization | 返回临时对象（prvalue）时省略拷贝 |
| **NRVO** | Named Return Value Optimization | 返回局部变量（具名）时省略拷贝 |
| **C++17 强制 RVO** | Guaranteed Copy Elision | prvalue 返回**保证**省略拷贝（不是优化，是标准要求） |

```cpp
// move_semantics.cpp（续）—— RVO/NRVO

StringVector create_vector() {
    StringVector v(100);  // 局部变量
    for (int i = 0; i < 100; i++) v.push(i);
    return v;  // NRVO：可能省略拷贝/移动，直接在调用者栈上构造
}

StringVector create_vector_rvo() {
    return StringVector(100);  // RVO：C++17 起保证省略，无拷贝无移动
}

void demo_rvo() {
    std::cout << "\n=== RVO/NRVO ===" << std::endl;
    auto v1 = create_vector();       // NRVO（编译器可能省略）
    auto v2 = create_vector_rvo();   // RVO（C++17 保证省略）
    std::cout << "  v1.size = " << v1.size() << ", v2.size = " << v2.size() << std::endl;
}
```

> ⚠️ **不要对返回值用 `std::move`**：`return std::move(local_var)` 会**阻止** NRVO！因为 `std::move` 把它变成右值，编译器只能用移动而非省略。正确写法是 `return local_var;`，让编译器自动应用 NRVO。

```cpp
StringVector bad_return() {
    StringVector v(100);
    return std::move(v);  // 坏！阻止 NRVO，强制移动
}

StringVector good_return() {
    StringVector v(100);
    return v;  // 好！NRVO 可能完全省略拷贝/移动
}
```

### 面试题积累（今日 6 道）

**Q1：`std::move` 做了什么？它移动了什么？**
> 答：`std::move` 不移动任何东西，它只是一个到右值引用的无条件 `static_cast`。它把表达式标记为"可移动"（右值），真正的移动发生在移动构造函数/赋值运算符中。`std::move` 之后源对象处于"有效但未指定"状态——可以赋值或析构，但不应读取其值。

**Q2：`std::move` 和 `std::forward` 有什么区别？**
> 答：`std::move` 无条件转换为右值引用（用于"我想移动它"）；`std::forward` 条件转换——根据模板参数 T 恢复原始值类别（用于"我想转发它，保持它原来是左值还是右值"）。`std::move` 用于不需要保留值类别的场景，`std::forward` 用于完美转发（如 `make_shared`、`emplace_back`）。

**Q3：什么是引用折叠？**
> 答：当模板参数推导产生"引用的引用"时，C++ 按四条规则折叠：`T& &`→`T&`、`T& &&`→`T&`、`T&& &`→`T&`、`T&& &&`→`T&&`。口诀："只有右值引用的右值引用还是右值引用，其余都折叠成左值引用"。引用折叠是转发引用（`T&&`）和完美转发的语言基础。

**Q4：为什么移动构造函数要标记 `noexcept`？**
> 答：`std::vector` 扩容时需要把旧元素迁移到新内存。如果移动构造是 `noexcept`，vector 用移动（快）；否则用拷贝（安全——如果中途抛异常，已拷贝的可以回滚）。不标 `noexcept` 的移动构造可能反而导致 vector 用更慢的拷贝。所以移动操作通常应标 `noexcept`（移动只交换指针，不会失败）。

**Q5：什么是完美转发？为什么需要它？**
> 答：完美转发是指函数模板将参数传递给另一个函数时，保持参数的原始值类别（左值/右值）和 `const` 属性不变。需要它是因为转发引用 `T&&` 会丢失值类别——传入右值后 `param` 变成具名变量（左值），再传递时变成左值。`std::forward<T>(param)` 通过引用折叠恢复原始类别。典型应用：`make_shared`、`emplace_back`、工厂函数。

**Q6：`return std::move(local_var)` 有什么问题？**
> 答：它会阻止 NRVO（命名返回值优化）。`std::move` 把局部变量变成右值，编译器只能用移动构造；而直接 `return local_var` 时，编译器可以完全省略拷贝/移动（NRVO），直接在调用者栈上构造对象。C++17 起，返回 prvalue（如 `return T(args)`）保证省略（RVO），但返回具名局部变量（NRVO）仍由编译器决定，`std::move` 会阻止这个优化。

### 今日检查清单

- [ ] 能解释 `std::move` 的本质（`static_cast` 到右值引用，不移动任何东西）
- [ ] 能写出移动构造函数（偷取资源 + 置空源对象 + `noexcept`）
- [ ] 能解释为什么移动操作要标 `noexcept`（vector 扩容用移动而非拷贝）
- [ ] 能说出引用折叠的四条规则
- [ ] 能区分转发引用 `T&&` 和右值引用 `T&&`
- [ ] 能解释 `std::forward` 的工作原理（条件转换 + 引用折叠）
- [ ] 能说出 `std::move` vs `std::forward` 的区别
- [ ] 能解释 RVO/NRVO 以及为什么不要 `return std::move(local)`
- [ ] `move_semantics.cpp` 编译运行通过

#### 明日预告

Day 4 将深入**模板与泛型编程**——函数/类模板、模板特化与偏特化、变参模板、SFINAE 与 C++20 concepts。今天的完美转发用到了变参模板（`Args&&... args`），明天从模板基础讲起。模板是 [CUTLASS 专题](../cutlass/README.md) 的核心——CUTLASS 的三层抽象全是模板参数。建议今晚先看看 `std::vector` 的声明，感受模板的复杂度。

---
