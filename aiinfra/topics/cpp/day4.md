# Day 4（周四）：模板与泛型编程

> **本周定位**：C++ 面试系统化准备，今日聚焦模板——AI Infra 方向必问（CUTLASS/DeepGEMM 全是模板）
> **前置要求**：已完成 Day 3（移动语义与完美转发用到了变参模板）
> **今日目标**：掌握函数模板/类模板、模板特化与偏特化、变参模板（parameter pack）、SFINAE 与 `if constexpr`、C++20 concepts，联系 CUTLASS 的模板设计哲学
> **时间投入**：2.5h（早间 1.5h 精读模板机制 + 晚间 1h 跑代码与 SFINAE 实验）
> **考察度**：⭐⭐⭐⭐ 进阶考点，AI Infra / 系统方向必问

---

## 本日在本周知识图谱中的位置

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| 函数/类模板基础与实例化机制 | ④ 为读懂 CUTLASS 模板代码打基础 |
| 特化/偏特化对比表 | ④ 同上 |
| 变参模板与 `sizeof...(pack)` | ④ 同上 |
| SFINAE → `if constexpr` → concepts 演进 | ④ 同上 |
| CUTLASS 模板设计案例 | 联系 [CUTLASS 专题](../cutlass/README.md) Day 4 |

---

### 学习任务 1：函数模板与类模板基础（40 分钟）

#### 函数模板

```cpp
// templates_and_sfinae.cpp —— 模板与泛型编程
// 编译: g++ -std=c++20 -o templates templates_and_sfinae.cpp && ./templates

#include <iostream>
#include <type_traits>
#include <string>
#include <vector>

// 函数模板
template <typename T>
T my_max(const T& a, const T& b) {
    return a < b ? b : a;
}

// 模板显式实例化（少用，用于强制生成特定版本）
// template int my_max<int>(const int&, const int&);

void demo_function_template() {
    std::cout << "=== 函数模板 ===" << std::endl;
    std::cout << "  max(3, 5) = " << my_max(3, 5) << std::endl;          // T=int
    std::cout << "  max(1.5, 2.5) = " << my_max(1.5, 2.5) << std::endl; // T=double
    // my_max(3, 2.5);  // 编译错误：T 推导冲突
    std::cout << "  max<double>(3, 2.5) = " << my_max<double>(3, 2.5) << std::endl; // 显式指定
}
```

#### 类模板

```cpp
// templates_and_sfinae.cpp（续）—— 类模板

template <typename T, size_t N>
class FixedArray {
    T data_[N];
public:
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    constexpr size_t size() const { return N; }

    // 类模板中的友元函数
    friend std::ostream& operator<<(std::ostream& os, const FixedArray& arr) {
        os << "[";
        for (size_t i = 0; i < N; i++) {
            if (i > 0) os << ", ";
            os << arr.data_[i];
        }
        return os << "]";
    }
};

void demo_class_template() {
    std::cout << "\n=== 类模板 ===" << std::endl;
    FixedArray<int, 5> arr;
    for (size_t i = 0; i < arr.size(); i++) arr[i] = i * 10;
    std::cout << "  " << arr << std::endl;
    std::cout << "  size = " << arr.size() << std::endl;
}
```

#### 模板编译模型

| 要点 | 说明 |
|------|------|
| **两阶段编译** | ① 模板定义检查（不依赖 T 的部分）；② 实例化时检查（依赖 T 的部分） |
| **头文件实现** | 模板通常写在头文件中（编译器需要看到完整定义才能实例化） |
| **显式实例化** | 可在 `.cpp` 中 `template class Foo<int>;` 强制实例化，避免头文件暴露实现 |

> ⚠️ **注意**：模板不是代码，而是生成代码的"蓝图"。模板本身不产生机器码，只有实例化（指定具体类型）后才生成代码。这就是为什么模板必须写在头文件里——编译器在每个使用点都需要看到完整定义来生成实例化代码。

### 学习任务 2：模板特化与偏特化（40 分钟）

#### 全特化 vs 偏特化

| 类型 | 函数模板 | 类模板 |
|------|----------|--------|
| **全特化** | ✅ 支持 | ✅ 支持 |
| **偏特化** | ❌ 不支持 | ✅ 支持 |
| **重载** | ✅ 支持（替代偏特化） | ❌ 不支持 |

```cpp
// templates_and_sfinae.cpp（续）—— 特化与偏特化

// 类模板：通用版本
template <typename T>
class TypeName {
public:
    static std::string get() { return "unknown"; }
};

// 类模板全特化：int
template <>
class TypeName<int> {
public:
    static std::string get() { return "int"; }
};

// 类模板全特化：double
template <>
class TypeName<double> {
public:
    static std::string get() { return "double"; }
};

// 类模板偏特化：指针类型
template <typename T>
class TypeName<T*> {
public:
    static std::string get() { return "pointer to " + TypeName<T>::get(); }
};

// 类模板偏特化：数组类型
template <typename T, size_t N>
class TypeName<T[N]> {
public:
    static std::string get() { return "array[" + std::to_string(N) + "] of " + TypeName<T>::get(); }
};

void demo_specialization() {
    std::cout << "\n=== 特化与偏特化 ===" << std::endl;
    std::cout << "  int:        " << TypeName<int>::get() << std::endl;
    std::cout << "  double:     " << TypeName<double>::get() << std::endl;
    std::cout << "  int*:       " << TypeName<int*>::get() << std::endl;
    std::cout << "  int[5]:     " << TypeName<int[5]>::get() << std::endl;
    std::cout << "  char:       " << TypeName<char>::get() << std::endl;
}
```

#### 函数模板：用重载代替偏特化

```cpp
// 函数模板不能偏特化，用重载代替
template <typename T>
void process(T val) {
    std::cout << "  通用: " << val << std::endl;
}

// 重载（不是特化）：指针类型
template <typename T>
void process(T* val) {
    std::cout << "  指针: *ptr = " << *val << std::endl;
}

void demo_function_overload() {
    std::cout << "\n=== 函数模板重载 ===" << std::endl;
    int x = 42;
    process(42);      // 通用版本
    process(&x);      // 指针重载版本
}
```

### 学习任务 3：变参模板（30 分钟）

变参模板（Variadic Templates）是 C++11 引入的，允许接受任意数量、任意类型的参数。

```cpp
// templates_and_sfinae.cpp（续）—— 变参模板

// 递归展开 parameter pack
void print() {}  // 终止函数

template <typename T, typename... Args>
void print(T first, Args... rest) {
    std::cout << "  " << first << std::endl;
    print(rest...);  // 递归展开
}

// sizeof... 获取参数包大小
template <typename... Args>
constexpr size_t count_args() {
    return sizeof...(Args);
}

void demo_variadic() {
    std::cout << "\n=== 变参模板 ===" << std::endl;
    print(1, "hello", 3.14, 'a');
    std::cout << "  参数个数: " << count_args<int, double, char>() << std::endl;
}
```

#### C++17 折叠表达式（Fold Expression）

C++17 前需要递归展开 parameter pack，C++17 引入折叠表达式大幅简化：

```cpp
// C++17 折叠表达式
template <typename... Args>
auto sum(Args... args) {
    return (args + ...);  // 一元右折叠：arg1 + (arg2 + (arg3 + ...))
}

template <typename... Args>
void print_fold(Args... args) {
    ((std::cout << "  " << args << std::endl), ...);  // 逗号折叠
}

void demo_fold_expression() {
    std::cout << "\n=== C++17 折叠表达式 ===" << std::endl;
    std::cout << "  sum(1,2,3,4) = " << sum(1, 2, 3, 4) << std::endl;
    print_fold(1, "hello", 3.14);
}
```

| 折叠形式 | 语法 | 含义 |
|----------|------|------|
| 一元右折叠 | `(pack op ...)` | `arg1 op (arg2 op (... op argN))` |
| 一元左折叠 | `(... op pack)` | `((arg1 op arg2) op ...) op argN` |
| 二元右折叠 | `(pack op ... op init)` | `arg1 op (arg2 op (... op (argN op init)))` |
| 二元左折叠 | `(init op ... op pack)` | `(((init op arg1) op arg2) op ...) op argN` |

> 💡 **联系 CUTLASS**：CUTLASS 的 `CollectiveBuilder` 和 CuTe 的 `make_layout` 大量使用变参模板。例如 `make_layout(make_shape(_64{}, _2{}), make_stride(_64{}, _4096{}))` 底层就是变参模板展开。理解变参模板是读懂 CUTLASS 源码的前提。

### 学习任务 4：SFINAE 与 if constexpr（40 分钟）

#### SFINAE 原理

SFINAE（Substitution Failure Is Not An Error）——模板参数替换失败不是错误，而是从重载候选中移除。这是 C++11/14 中实现条件编译的主要手段。

```cpp
// templates_and_sfinae.cpp（续）—— SFINAE

// SFINAE：enable_if 根据 T 是否为整型选择不同重载
template <typename T,
          typename = std::enable_if_t<std::is_integral_v<T>>>
std::string classify(T val) {
    return "整型: " + std::to_string(val);
}

template <typename T,
          typename = std::enable_if_t<std::is_floating_point_v<T>>>
std::string classify(T val) {  // 注意：签名不能只靠默认模板参数区分
    return "浮点型";  // 实际需要用不同技巧避免歧义
}

// 更实用的 SFINAE 模式：检测类型是否有 size() 方法
template <typename T>
class has_size {
    template <typename U>
    static auto test(int) -> decltype(std::declval<U>().size(), std::true_type{});
    template <typename U>
    static auto test(...) -> std::false_type;
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

void demo_sfinae() {
    std::cout << "\n=== SFINAE ===" << std::endl;
    std::cout << "  int 有 size(): " << has_size<int>::value << std::endl;        // 0
    std::cout << "  vector 有 size(): " << has_size<std::vector<int>>::value << std::endl; // 1
    std::cout << "  string 有 size(): " << has_size<std::string>::value << std::endl;     // 1
}
```

#### if constexpr（C++17，SFINAE 的简化替代）

```cpp
// templates_and_sfinae.cpp（续）—— if constexpr

template <typename T>
std::string classify_modern(const T& val) {
    if constexpr (std::is_integral_v<T>) {
        return "整型: " + std::to_string(val);
    } else if constexpr (std::is_floating_point_v<T>) {
        return "浮点型";
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "字符串: " + val;
    } else {
        return "其他类型";
    }
}

void demo_if_constexpr() {
    std::cout << "\n=== if constexpr（C++17）===" << std::endl;
    std::cout << "  " << classify_modern(42) << std::endl;
    std::cout << "  " << classify_modern(3.14) << std::endl;
    std::cout << "  " << classify_modern(std::string("hi")) << std::endl;
}
```

> 💡 **面试要点**：`if constexpr` 与普通 `if` 的区别——`if constexpr` 在编译期求值，false 分支的代码**不会被编译**（可以包含对当前类型无效的操作）。普通 `if` 两个分支都会编译，类型不匹配会报错。

### 学习任务 5：C++20 Concepts（20 分钟）

Concepts 是 C++20 引入的，对模板参数的约束机制，取代了繁琐的 SFINAE：

```cpp
// templates_and_sfinae.cpp（续）—— C++20 Concepts

// 定义 concept
template <typename T>
concept Numeric = std::is_integral_v<T> || std::is_floating_point_v<T>;

template <typename T>
concept HasSize = requires(T t) {
    { t.size() } -> std::convertible_to<size_t>;
};

// 使用 concept 约束模板
template <Numeric T>
T add(T a, T b) { return a + b; }

// requires 子句
template <typename T>
    requires HasSize<T>
size_t get_size(const T& val) { return val.size(); }

// 简化语法
void demo_concepts() {
    std::cout << "\n=== C++20 Concepts ===" << std::endl;
    std::cout << "  add(1, 2) = " << add(1, 2) << std::endl;
    std::cout << "  add(1.5, 2.5) = " << add(1.5, 2.5) << std::endl;
    // add("a", "b");  // 编译错误：不满足 Numeric concept
    // 错误信息会清晰指出"不满足 Numeric"，而非 SFINAE 的几百行报错

    std::vector<int> v = {1, 2, 3};
    std::cout << "  get_size(vector) = " << get_size(v) << std::endl;
}
```

#### SFINAE → if constexpr → Concepts 演进

| 时代 | 技术 | 优点 | 缺点 |
|------|------|------|------|
| C++11/14 | SFINAE (`enable_if`) | 功能完整 | 语法繁琐、报错难读 |
| C++17 | `if constexpr` | 简洁 | 只适用于函数体内分支 |
| C++20 | Concepts | 语法清晰、报错友好 | 需要 C++20 支持 |

#### 联系 CUTLASS

CUTLASS 大量使用 SFINAE 和 `if constexpr`（因为要兼容 C++14/17），C++20 Concepts 在 CUTLASS 4.x 开始引入。理解模板是读懂 CUTLASS 的前提：

```cpp
// CUTLASS 的 CollectiveBuilder 简化示意
template <typename ArchTag, typename OpClass, /* ... */>
struct CollectiveBuilder {
    // SFINAE 选择不同架构的实现
    using Type = std::conditional_t<
        std::is_same_v<ArchTag, cutlass::arch::Sm90>,
        Sm90Mainloop<OpClass, /*...*/>,
        Sm80Mainloop<OpClass, /*...*/>
    >;
};
```

### 面试题积累（今日 5 道）

**Q1：函数模板和类模板有什么区别？模板为什么通常写在头文件里？**
> 答：函数模板由编译器根据实参自动推导类型参数；类模板需要显式指定类型参数（C++17 CTAD 除外）。模板必须写在头文件中，因为模板是生成代码的"蓝图"，编译器在每个使用点都需要看到完整定义才能实例化。如果定义在 `.cpp` 中，其他翻译单元无法实例化。显式实例化可以解决这个问题但会增加编译耦合。

**Q2：模板的特化和偏特化有什么区别？函数模板能偏特化吗？**
> 答：全特化是为特定类型参数提供完整实现；偏特化是为部分类型参数（如指针类型 `T*`）提供实现。类模板支持全特化和偏特化；函数模板只支持全特化，不支持偏特化——需要偏特化时用函数重载代替。全特化后不再有模板参数，偏特化仍保留部分模板参数。

**Q3：什么是变参模板？C++17 的折叠表达式解决了什么问题？**
> 答：变参模板用 `typename... Args` 接受任意数量参数。C++17 前需要递归展开（定义终止函数 + 递归版本），代码冗长；C++17 折叠表达式用 `(pack op ...)` 一行展开，如 `(args + ...)` 求和、`((cout << args), ...)` 逐个打印。折叠有四种形式：一元/二元 × 左/右折叠。CUTLASS 的 CuTe Layout/Shape 大量使用变参模板。

**Q4：SFINAE 是什么？`if constexpr` 和 SFINAE 有什么区别？**
> 答：SFINAE（Substitution Failure Is Not An Error）——模板参数替换失败不是错误，而是从重载候选中移除。用于在编译期根据类型特征选择不同实现。`if constexpr`（C++17）更简洁——在函数体内编译期分支，false 分支不编译。区别：SFINAE 作用于函数签名/重载选择层面；`if constexpr` 作用于函数体内分支。C++20 Concepts 进一步简化，提供清晰的约束语法和友好的报错。

**Q5：C++20 Concepts 解决了什么问题？**
> 答：Concepts 对模板参数提供命名约束，解决 SFINAE 的两个痛点——① 语法繁琐（`enable_if` 嵌套）；② 报错信息不可读（几百行模板展开错误）。Concepts 让约束声明式化（`template <Numeric T>`），报错时直接指出"不满足 Numeric concept"。还支持 `requires` 表达式检测类型是否有特定成员函数，如 `requires(T t) { t.size(); }`。

### 今日检查清单

- [ ] 能解释模板的两阶段编译（定义检查 vs 实例化检查）
- [ ] 知道模板为什么通常写在头文件里
- [ ] 能区分全特化和偏特化，知道函数模板不能偏特化
- [ ] 能用递归或折叠表达式展开 parameter pack
- [ ] 能解释 SFINAE 原理并写出 `enable_if` 的简单示例
- [ ] 能说出 `if constexpr` 与普通 `if` 的区别
- [ ] 能用 C++20 concept 约束模板参数
- [ ] 能说出 SFINAE → `if constexpr` → Concepts 的演进
- [ ] `templates_and_sfinae.cpp` 编译运行通过

#### 明日预告

Day 5 将深入**面向对象与多态底层**——虚函数表的内存布局、虚析构函数为什么必要、多重继承的对象布局、Rule of 3/5/0。今天学的模板加上明天的多态，就构成了理解 CUTLASS/DeepGEMM 源码的完整 C++ 基础。建议今晚先想想：虚函数是怎么实现"运行时多态"的？

---
