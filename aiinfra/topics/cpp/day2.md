# Day 2（周二）：RAII 与智能指针

> **本周定位**：C++ 面试系统化准备，今日聚焦资源管理的核心范式——RAII 与智能指针
> **前置要求**：已完成 Day 1（内存模型与基础语义），理解栈/堆的区别与对象生命周期
> **今日目标**：掌握 RAII 资源管理原则、`unique_ptr`/`shared_ptr`/`weak_ptr` 的实现原理与使用场景、循环引用问题、自定义 deleter、`make_unique`/`make_shared` 的优势，能手写一个简化版 `unique_ptr`
> **时间投入**：2.5h（早间 1.5h 精读智能指针原理 + 晚间 1h 跑代码与手写 unique_ptr）
> **考察度**：⭐⭐⭐⭐⭐ 核心考点，智能指针是 C++ 面试的必问题

---

## 本日在本周知识图谱中的位置

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| RAII 原则与手写资源管理类 | ② 能手写 RAII 资源管理类 |
| `unique_ptr`/`shared_ptr`/`weak_ptr` 对比表 | ④ 能说出 `shared_ptr` 控制块结构 |
| 循环引用 demo 与 `weak_ptr` 解决方案 | ④ 同上（`weak_ptr` 打破循环） |
| `shared_ptr` 线程安全性分析 | ④ 同上（控制块原子操作） |

---

### 学习任务 1：RAII 原则（30 分钟）

#### 什么是 RAII

RAII（Resource Acquisition Is Initialization）是 C++ 最核心的资源管理范式：**资源的获取就是对象的初始化，资源的释放就是对象的析构**。

| 要点 | 说明 |
|------|------|
| **资源获取** | 在构造函数中获取资源（分配内存、打开文件、加锁等） |
| **资源释放** | 在析构函数中释放资源（释放内存、关闭文件、解锁等） |
| **自动释放** | 对象离开作用域时，析构函数自动调用（即使抛异常） |
| **栈语义** | 利用栈对象的确定性析构，保证资源总是被释放 |

> 💡 **一句话总结**：RAII 把"资源生命周期"绑定到"对象生命周期"——对象构造时获取资源，析构时释放资源。因为 C++ 保证栈对象离开作用域时必定析构（即使异常展开），所以资源总是被正确释放。

#### RAII 的经典应用

```cpp
// smart_pointers.cpp —— RAII 经典应用
// 编译: g++ -std=c++20 -o smart_pointers smart_pointers.cpp && ./smart_pointers

#include <iostream>
#include <memory>
#include <mutex>
#include <fstream>

// 1. 内存管理：智能指针（今天的主角）
// 2. 锁管理：std::lock_guard / std::unique_lock（Day 6 详讲）
// 3. 文件管理：std::fstream 构造时打开，析构时关闭

// 手写 RAII 内存管理类
class MyArray {
    int* data_;
    size_t size_;
public:
    explicit MyArray(size_t n) : data_(new int[n]()), size_(n) {
        std::cout << "  MyArray(" << n << ") 分配 " << n * sizeof(int) << " 字节" << std::endl;
    }
    ~MyArray() {
        delete[] data_;
        std::cout << "  ~MyArray() 释放 " << size_ * sizeof(int) << " 字节" << std::endl;
    }

    // 禁止拷贝（避免 double free）
    MyArray(const MyArray&) = delete;
    MyArray& operator=(const MyArray&) = delete;

    // 允许移动（Day 3 详讲）
    MyArray(MyArray&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    int& operator[](size_t i) { return data_[i]; }
    size_t size() const { return size_; }
};

void demo_raii() {
    std::cout << "=== RAII 演示 ===" << std::endl;
    {
        MyArray arr(10);
        arr[0] = 42;
        std::cout << "  arr[0] = " << arr[0] << std::endl;
        // arr 离开作用域时自动调用 ~MyArray()，无需手动 delete
    }
    std::cout << "  arr 已自动释放" << std::endl;
}
```

> ⚠️ **注意**：RAII 的关键是析构函数的**确定性调用**——与 Java/C# 的 GC 不同，C++ 对象离开作用域时析构**必定**发生，时间点确定。这是 C++ 资源管理比 GC 语言更精确的原因。

### 学习任务 2：unique_ptr——独占所有权（45 分钟）

`std::unique_ptr` 是 C++11 引入的独占式智能指针，取代了 C++98 的 `std::auto_ptr`。

#### 核心特征

| 特征 | 说明 |
|------|------|
| **独占所有权** | 同一时刻只有一个 `unique_ptr` 拥有对象 |
| **不可拷贝** | 拷贝构造和拷贝赋值被 `delete` |
| **可移动** | 通过移动语义转移所有权 |
| **零开销** | 大小 = 原始指针（默认 deleter），无引用计数开销 |
| **自定义 deleter** | 可指定自定义释放逻辑（如 `fclose`、`cudaFree`） |

#### 基本用法

```cpp
// smart_pointers.cpp（续）—— unique_ptr 演示

void demo_unique_ptr() {
    std::cout << "\n=== unique_ptr 演示 ===" << std::endl;

    // 1. 创建
    std::unique_ptr<int> p1 = std::make_unique<int>(42);
    std::cout << "  *p1 = " << *p1 << std::endl;

    // 2. 转移所有权（移动）
    std::unique_ptr<int> p2 = std::move(p1);  // p1 变为 nullptr
    std::cout << "  移动后: p1 = " << (p1 ? "非空" : "空")
              << ", *p2 = " << *p2 << std::endl;

    // 3. 自定义 deleter
    auto file_deleter = [](FILE* f) {
        if (f) { std::cout << "  fclose() 被调用" << std::endl; fclose(f); }
    };
    {
        std::unique_ptr<FILE, decltype(file_deleter)> fp(fopen("/tmp/test.txt", "w"), file_deleter);
        if (fp) fprintf(fp.get(), "hello");
        // fp 离开作用域，自动调用 file_deleter
    }

    // 4. 数组版本
    std::unique_ptr<int[]> arr = std::unique_ptr<int[]>(new int[5]);
    arr[0] = 1;  // unique_ptr<T[]> 支持 operator[]
}
```

#### 手写 unique_ptr（面试高频手撕题）

```cpp
// smart_pointers.cpp（续）—— 手写简化版 unique_ptr

template <typename T>
class MyUniquePtr {
    T* ptr_;
public:
    explicit MyUniquePtr(T* p = nullptr) : ptr_(p) {}

    // 禁止拷贝
    MyUniquePtr(const MyUniquePtr&) = delete;
    MyUniquePtr& operator=(const MyUniquePtr&) = delete;

    // 允许移动
    MyUniquePtr(MyUniquePtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    MyUniquePtr& operator=(MyUniquePtr&& other) noexcept {
        if (this != &other) {
            delete ptr_;        // 释放旧资源
            ptr_ = other.ptr_;  // 接管新资源
            other.ptr_ = nullptr;
        }
        return *this;
    }

    // 析构释放
    ~MyUniquePtr() { delete ptr_; }

    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    T* get() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
};

void demo_handwritten_unique_ptr() {
    std::cout << "\n=== 手写 unique_ptr 演示 ===" << std::endl;
    MyUniquePtr<int> p1(new int(100));
    std::cout << "  *p1 = " << *p1 << std::endl;

    MyUniquePtr<int> p2 = std::move(p1);
    std::cout << "  移动后: p1 = " << (p1 ? "非空" : "空")
              << ", *p2 = " << *p2 << std::endl;
    // p2 离开作用域自动释放
}
```

> 💡 **面试要点**：手写 `unique_ptr` 时注意三点——① 禁止拷贝（`= delete`）；② 移动构造/赋值要 `noexcept`（否则 `std::vector` 扩容时会 fallback 到拷贝）；③ 移动赋值要先释放旧资源再接管新资源，并处理自赋值。

### 学习任务 3：shared_ptr——共享所有权（45 分钟）

`std::shared_ptr` 是引用计数的共享式智能指针，多个 `shared_ptr` 可以指向同一对象。

#### 控制块结构

`shared_ptr` 的核心是**控制块（Control Block）**，面试必问：

| 控制块字段 | 说明 |
|-----------|------|
| **强引用计数** | 有多少个 `shared_ptr` 共享对象 |
| **弱引用计数** | 有多少个 `weak_ptr` 观察对象 |
| **自定义 deleter** | 释放对象的函数指针 |
| **分配器** | 内存分配器（可选） |

> 💡 **关键洞察**：`shared_ptr` 有两次内存分配——① 对象本身的内存；② 控制块的内存。`std::make_shared` 把两者合并为一次分配（更高效，但内存延迟释放——只要 `weak_ptr` 还在，控制块就不释放，对象内存也跟着不释放）。

#### 基本用法

```cpp
// smart_pointers.cpp（续）—— shared_ptr 演示

void demo_shared_ptr() {
    std::cout << "\n=== shared_ptr 演示 ===" << std::endl;

    // 1. 创建
    auto p1 = std::make_shared<int>(42);
    auto p2 = p1;  // 拷贝：引用计数 +1
    std::cout << "  *p1 = " << *p1 << ", use_count = " << p1.use_count() << std::endl;  // 2

    // 2. p2 离开作用域，引用计数 -1
    {
        auto p3 = p1;
        std::cout << "  进入内部块: use_count = " << p1.use_count() << std::endl;  // 3
    }
    std::cout << "  离开内部块: use_count = " << p1.use_count() << std::endl;  // 2

    // 3. shared_ptr 的线程安全性
    // 控制块的引用计数操作是原子的（线程安全）
    // 但指向同一对象的 shared_ptr 的非 const 操作不是线程安全的
    auto sp = std::make_shared<int>(0);
    // 多线程同时读 sp（const 操作）是安全的
    // 多线程同时写 sp（如 sp = ...）不是线程安全的，需要外部同步
}
```

#### shared_ptr 的线程安全性（面试高频）

| 操作 | 线程安全？ | 说明 |
|------|-----------|------|
| 引用计数增减 | ✅ | 控制块用原子操作 |
| 不同线程拷贝**不同** `shared_ptr` 指向同一对象 | ✅ | 各自操作各自的引用计数副本 |
| 不同线程拷贝**同一个** `shared_ptr` 对象 | ❌ | 需要外部同步 |
| 不同线程通过 `shared_ptr` 读指向的对象 | ✅ | 只要没人写 |
| 不同线程通过 `shared_ptr` 写指向的对象 | ❌ | 对象本身不是线程安全的 |

> ⚠️ **注意**：`shared_ptr` 的"线程安全"仅指**引用计数操作**是原子的，不意味着指向的对象是线程安全的。这是面试中最容易混淆的点。

### 学习任务 4：weak_ptr 与循环引用（30 分钟）

#### 循环引用问题

```cpp
// smart_pointers.cpp（续）—— 循环引用问题

struct BadNode {
    std::shared_ptr<BadNode> next;
    ~BadNode() { std::cout << "  ~BadNode()" << std::endl; }
};

void demo_circular_reference() {
    std::cout << "\n=== 循环引用问题 ===" << std::endl;
    auto a = std::make_shared<BadNode>();
    auto b = std::make_shared<BadNode>();
    a->next = b;  // b 的强引用计数 = 2
    b->next = a;  // a 的强引用计数 = 2
    std::cout << "  a.use_count = " << a.use_count() << std::endl;  // 2
    std::cout << "  b.use_count = " << b.use_count() << std::endl;  // 2
    // a, b 离开作用域：各自引用计数从 2 → 1
    // 但两者互相引用，引用计数永远不归零 → 内存泄漏！
    // 不会打印 ~BadNode()
}
```

#### weak_ptr 解决方案

`std::weak_ptr` 是不增加强引用计数的"观察者"指针：

| 特征 | `shared_ptr` | `weak_ptr` |
|------|-------------|-----------|
| 引用计数 | 增加强引用计数 | 增加**弱**引用计数 |
| 访问对象 | 直接访问 | 需 `lock()` 提升为 `shared_ptr` |
| 影响释放 | 强计数归零才释放 | 不影响对象释放 |
| 用途 | 共享所有权 | 打破循环引用、观察者模式、缓存 |

```cpp
// smart_pointers.cpp（续）—— weak_ptr 解决循环引用

struct GoodNode {
    std::shared_ptr<GoodNode> next;
    std::weak_ptr<GoodNode> prev;  // 用 weak_ptr 打破循环
    ~GoodNode() { std::cout << "  ~GoodNode()" << std::endl; }
};

void demo_weak_ptr_solution() {
    std::cout << "\n=== weak_ptr 解决循环引用 ===" << std::endl;
    auto a = std::make_shared<GoodNode>();
    auto b = std::make_shared<GoodNode>();
    a->next = b;        // b 强引用 = 2
    b->prev = a;        // a 强引用仍 = 1（weak_ptr 不增加强计数）
    std::cout << "  a.use_count = " << a.use_count() << std::endl;  // 1
    std::cout << "  b.use_count = " << b.use_count() << std::endl;  // 2

    // 使用 weak_ptr 前需要 lock() 提升为 shared_ptr
    if (auto sp = b->prev.lock()) {
        std::cout << "  b->prev 指向的对象存活: use_count = " << sp.use_count() << std::endl;
    }
    // a, b 离开作用域后正确释放，会打印 ~GoodNode()
}
```

> 💡 **规则**：当两个对象互相引用时，一个用 `shared_ptr`，另一个用 `weak_ptr`。通常"父持有子用 `shared_ptr`，子引用父用 `weak_ptr`"。

### 学习任务 5：make_unique vs make_shared vs new（15 分钟）

| 方式 | 内存分配次数 | 异常安全 | 缓存局部性 | 备注 |
|------|-------------|---------|-----------|------|
| `new` + 构造 `shared_ptr` | 2 次（对象 + 控制块） | ❌ 可能泄漏 | 差 | 不推荐 |
| `std::make_shared` | 1 次（合并分配） | ✅ | 好 | 推荐 |
| `std::make_unique` | 1 次 | ✅ | 好 | 推荐 |

```cpp
// 异常安全问题
void func(std::shared_ptr<A> a, std::shared_ptr<B> b);

// 可能泄漏：C++17 前参数求值顺序未指定
// func(std::shared_ptr<A>(new A()), std::shared_ptr<B>(new B()));
// 如果 new A() 成功、new B() 抛异常，A 泄漏

// 安全：make_ 保证不会泄漏
// func(std::make_shared<A>(), std::make_shared<B>());
```

> ⚠️ **`make_shared` 的缺点**：对象内存和控制块合并分配，所以只要 `weak_ptr` 还在，控制块就不释放，对象内存也跟着不释放（即使强引用已归零）。如果对象很大且有 `weak_ptr` 长期存在，用 `shared_ptr(new T)` 更合适。

### 面试题积累（今日 6 道）

**Q1：RAII 是什么？为什么 C++ 需要 RAII？**
> 答：RAII（Resource Acquisition Is Initialization）把资源生命周期绑定到对象生命周期——构造函数获取资源，析构函数释放资源。C++ 需要它因为：① 栈对象的析构是确定性的（离开作用域必定调用，即使异常展开）；② 自动管理，避免忘记释放；③ 异常安全——即使中间抛异常，已构造的 RAII 对象都会正确析构。智能指针、`lock_guard`、`fstream` 都是 RAII 的应用。

**Q2：`unique_ptr` 和 `shared_ptr` 有什么区别？各自的使用场景？**
> 答：`unique_ptr` 独占所有权，不可拷贝只能移动，零开销（无引用计数），大小等于原始指针；`shared_ptr` 共享所有权，引用计数，有控制块开销。使用场景：独占资源用 `unique_ptr`（默认首选），需要共享所有权时用 `shared_ptr`。性能敏感场景优先 `unique_ptr`——它可以隐式转换为 `shared_ptr`，反之不行。

**Q3：`shared_ptr` 的控制块包含什么？`make_shared` 和 `new` 构造 `shared_ptr` 有什么区别？**
> 答：控制块包含强引用计数、弱引用计数、自定义 deleter、分配器。`make_shared` 把对象内存和控制块合并为一次分配（更高效、缓存友好、异常安全）；`new` 方式分两次分配。但 `make_shared` 的代价是：只要 `weak_ptr` 还在，对象内存就不释放（控制块和对象内存在一起）。大对象 + 长期 `weak_ptr` 场景不适合 `make_shared`。

**Q4：什么是循环引用？如何解决？**
> 答：两个对象互相用 `shared_ptr` 指向对方，导致引用计数永远不归零，内存泄漏。解决方案：把其中一个改为 `weak_ptr`——`weak_ptr` 不增加强引用计数，不影响对象释放。使用时通过 `lock()` 提升为 `shared_ptr`。规则：父持有子用 `shared_ptr`，子引用父用 `weak_ptr`。

**Q5：`shared_ptr` 是线程安全的吗？**
> 答：引用计数操作是线程安全的（原子操作），但指向的对象本身不是线程安全的。具体来说：不同线程拷贝**不同的** `shared_ptr`（即使指向同一对象）是安全的；但不同线程读写**同一个** `shared_ptr` 变量需要外部同步。多线程读写 `shared_ptr` 指向的对象也需要自己加锁。

**Q6：手写一个简化版 `unique_ptr`。**
> 答：核心要点——① 构造函数接收裸指针；② 禁止拷贝（`= delete`）；③ 移动构造/赋值（`noexcept`，转移指针后置源为 `nullptr`）；④ 析构 `delete` 指针；⑤ 重载 `operator*`/`operator->`/`get()`/`operator bool()`。移动赋值要注意自赋值检查和先释放旧资源。详见 `kernels/smart_pointers.cpp` 的 `MyUniquePtr`。

### 今日检查清单

- [ ] 能解释 RAII 原则及其优势（确定性析构、异常安全）
- [ ] 能说出 `unique_ptr` 的核心特征（独占、不可拷贝、零开销）
- [ ] 能手写简化版 `unique_ptr`（含移动语义）
- [ ] 能说出 `shared_ptr` 控制块的四字段（强计数/弱计数/deleter/allocator）
- [ ] 能解释 `make_shared` vs `new` 构造 `shared_ptr` 的区别
- [ ] 能解释循环引用问题并用 `weak_ptr` 解决
- [ ] 能说清 `shared_ptr` 的线程安全性（计数安全 vs 对象不安全）
- [ ] `smart_pointers.cpp` 编译运行通过

#### 明日预告

Day 3 将深入**移动语义与完美转发**——`std::move` 到底做了什么、引用折叠规则、`std::forward` 如何实现完美转发。今天的智能指针大量使用了移动语义（`unique_ptr` 靠移动转移所有权），明天从语言层面理解移动的本质。建议今晚先想清楚：`std::move` 之后原来的变量变成了什么？

---
