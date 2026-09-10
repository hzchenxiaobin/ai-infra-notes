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

## 今日术语速查表

> 先建立词汇表，再进入正文；每个术语在对应学习任务里有完整展开。

| 术语 | 一句话定义 |
|------|-----------|
| **模板（template）** | 以类型/编译期常量为参数的代码生成"蓝图"，本身不产生机器码 |
| **实例化（instantiation）** | 编译器把模板实参代入模板、生成具体类/函数的过程 |
| **模板实参推导（deduction）** | 编译器按调用实参类型反推模板参数，只看签名不看函数体 |
| **非类型模板参数（NTTP）** | 编译期常量形参（如 `size_t N`），实参必须是常量表达式 |
| **CTAD** | 类模板实参推导（C++17），`std::vector v{1,2,3}` 免写 `<int>` |
| **依赖名（dependent name）** | 语义依赖模板参数的名字，实例化前编译器不知道它是类型/函数/模板 |
| **两阶段查找** | ①定义处检查非依赖代码；②实例化处检查依赖代码——模板报错时机由它决定 |
| **全特化（full specialization）** | 为一组**具体**模板实参提供的完整替代实现，`template <>` 后不再有模板参数 |
| **偏特化（partial specialization）** | 为一类**参数形态**（`T*`、`T[N]`）提供的替代实现，本身仍是模板 |
| **参数包（parameter pack）** | 接受零个或多个实参的形参：`typename... Args` |
| **包展开（pack expansion）** | 按"模式 + `...`"把包展开成零个或多个逗号分隔的实参 |
| **SFINAE** | Substitution Failure Is Not An Error：替换失败不是错误，只是候选出局 |
| **替换（substitution）** | 重载决议中把推导出的实参代回候选**签名**的动作 |
| **直接语境（immediate context）** | SFINAE 的失败边界：仅限签名本身；函数体内/连带实例化的失败是硬错误 |
| **detection idiom** | 用 SFINAE 探测"类型是否拥有某接口"的惯用法（`void_t`） |
| **被丢弃语句（discarded statement）** | `if constexpr` 的 false 分支：实例化时不实例化其中代码 |
| **concept** | 类型上的具名布尔谓词（C++20）：`template <typename T> concept X = ...;` |
| **requires 子句 / 表达式** | 子句=挂在声明上的约束；表达式=就地探测接口合法性的 bool 表达式 |
| **subsumption** | 约束蕴涵：约束更紧的重载更特化、优先入选——概念参与重载排序 |

---

### 学习任务 1：函数模板与类模板基础（40 分钟）

#### 函数模板

**函数模板（function template）**：以类型（或编译期常量）为参数的函数生成器。调用时编译器做**模板实参推导（argument deduction）**——按实参类型反推 `T`，只看函数签名、不看函数体；多个实参**各自独立推导**，同一个 `T` 被推出不同类型即报错。这正是 `my_max(3, 2.5)` 失败（`int` vs `double` 冲突）而 `my_max<double>(3, 2.5)` 可行（显式指定模板实参、关闭推导）的原因。推导不出的场景（如无参模板）也必须显式指定。

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

三个语言点：

- **非类型模板参数（NTTP）**：`size_t N` 是编译期常量形参，实参必须是常量表达式。`FixedArray<int, 5>` 的 `5` 在编译期定死，成员 `T data_[N]` 因此可以是零开销栈数组——`std::array` 就是这么实现的
- **CTAD（类模板实参推导，C++17）**：类模板也能从构造实参推导参数——`std::vector v{1, 2, 3}` 免写 `<int>`、`std::lock_guard lk(mtx)` 免写 `<std::mutex>`（这也是 Q1 里"类模板需要显式指定"的例外）
- **模板参数的三种形态**：类型参数（`typename T`）、非类型参数（`size_t N`，C++20 起还允许浮点/类类型）、模板模板参数（`template <typename> class C`，接一个模板本身作实参）

#### 模板编译模型

| 要点 | 说明 |
|------|------|
| **两阶段编译** | ① 模板定义检查（不依赖 T 的部分）；② 实例化时检查（依赖 T 的部分） |
| **头文件实现** | 模板通常写在头文件中（编译器需要看到完整定义才能实例化） |
| **显式实例化** | 可在 `.cpp` 中 `template class Foo<int>;` 强制实例化，避免头文件暴露实现 |

> ⚠️ **注意**：模板不是代码，而是生成代码的"蓝图"。模板本身不产生机器码，只有实例化（指定具体类型）后才生成代码。这就是为什么模板必须写在头文件里——编译器在每个使用点都需要看到完整定义来生成实例化代码。

#### 依赖名与两阶段查找

**依赖名（dependent name）**：语义依赖模板参数的名字，如 `T::iterator`——实例化前编译器不知道它是类型、函数还是模板。模板编译因此分两个阶段，即**两阶段查找（two-phase lookup）**：

| 阶段 | 时机 | 检查什么 |
|------|------|----------|
| 第一阶段 | 模板**定义**处 | 语法 + 非依赖代码的语义 + 非依赖名字查找 |
| 第二阶段 | **实例化**处 | 把实参代入后的依赖代码 + 依赖名字查找 |

这解释了两个常见现象：模板函数体内的类型错误**报错在实例化处**（第二阶段才检查）；`typename`/`template` 消歧关键字必须加——编译器默认把依赖名当作**不是类型**、后面跟 `<` 当作**小于号**：

```cpp
template <typename T>
void f(T& container) {
    typename T::iterator it = container.begin();  // 不加 typename 编译报错
    container.template get<int>();                // 不加 template 会被当作比较运算
}
```

（C++20 起 P0634 放宽了 `typename` 的多数场合，但读老代码/面试仍要求能解释。）

### 学习任务 2：模板特化与偏特化（40 分钟）

#### 概念定义：特化与偏特化

- **全特化（full specialization）**：为**一组具体模板实参**提供完整替代实现。`template <>` 后尖括号为空——它不再是模板，是独立的类/函数
- **偏特化（partial specialization）**：为**一类参数形态**（所有指针 `T*`、定长数组 `T[N]`）提供替代实现，仍保留部分模板参数——它本身还是模板
- **匹配规则**：实例化时编译器在主模板与所有偏特化中选**最特化**（most specialized）的匹配者，无人匹配则用主模板。这套偏序比较与函数重载决议是**两套独立机制**
- **特化的独立性**：类模板的全特化是一个全新的类，不会"继承"主模板的任何成员，必须自带全部接口；且特化必须在首次实例化**之前**声明，否则行为未定义

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

#### ⚠️ 函数模板全特化的经典陷阱

函数模板全特化**不参与重载决议**。决议先在"普通函数 + 主模板（及其重载）"中选出一个，选中了主模板之后才会去看有没有匹配的特化——主模板没被选中，它的特化自然也没机会：

```cpp
template <typename T> void f(T)  { /* 主模板 */ }
template <> void f<int*>(int*)   { /* f<int*> 全特化 */ }
void f(int*)                      { /* 普通重载 */ }

f((int*)nullptr);       // → 普通重载！非模板优先于模板，特化根本没被考虑
f<int*>((int*)nullptr); // → 全特化（显式指定模板实参，只在模板里选）
```

> 💡 **选择原则**（Effective C++ 条款 46 的精神）：想为特定类型定制行为——类模板用特化/偏特化；**函数模板用重载**（或 C++17 `if constexpr` / C++20 concepts），不要用全特化。

### 学习任务 3：变参模板（30 分钟）

变参模板（Variadic Templates）是 C++11 引入的，允许接受任意数量、任意类型的参数。三个核心概念：

| 概念 | 定义 |
|------|------|
| **参数包（parameter pack）** | 接受零个或多个实参的形参：`typename... Args` 是模板参数包，`Args... args` 是函数参数包 |
| **包展开（pack expansion）** | 把"模式 + `...`"展开成零个或多个逗号分隔的实参：`print(rest...)`、`std::forward<Args>(args)...` |
| **模式（pattern）** | 展开时每个包元素代入的部分——`std::forward<Args>(args)...` 的模式是 `std::forward<Args>(args)`，`Args` 与 `args` 两个包同步展开 |

参数包没有"长度"概念：不能 `args[0]` 取值、不能 `sizeof(args)`——只能整体展开、用 `sizeof...(args)`（编译期常量）计数，或递归剥离。

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

递归展开的三要素：**终止重载**（空参包版本 `void print()`）+ **递归重载**（剥离一个具名参数 `first`，剩余打包成新的 `rest`）+ **递归调用点展开** `print(rest...)`。注意终止函数是**重载**而不是特化——函数模板不支持偏特化，这里靠重载决议：参数越少越优先匹配终止版。

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

> ⚠️ **空参数包规则**：一元折叠作用于空包时，`&&` 得 `true`、`||` 得 `false`、`,` 得 `void()`，其余运算符非法；二元折叠有初值兜底，空包直接返回初值。所以"全部满足"的习惯写法是二元形式 `(args && ... && true)`，规避空包非法。

> 💡 **联系 CUTLASS**：CUTLASS 的 `CollectiveBuilder` 和 CuTe 的 `make_layout` 大量使用变参模板。例如 `make_layout(make_shape(_64{}, _2{}), make_stride(_64{}, _4096{}))` 底层就是变参模板展开。理解变参模板是读懂 CUTLASS 源码的前提。

### 学习任务 4：SFINAE 与 if constexpr（40 分钟）

#### SFINAE 原理与概念定义

SFINAE（**S**ubstitution **F**ailure **I**s **N**ot **A**n **E**rror，"替换失败不是错误"）不是某个库特性，而是模板重载决议中的一条基础规则。先立好三个定义：

| 概念 | 定义 |
|------|------|
| **替换（substitution）** | 重载决议中，编译器把推导出的模板实参代回**候选模板的签名**（模板参数列表、函数形参类型、返回类型）的动作 |
| **替换失败** | 代回后签名中出现非法类型/表达式（如 `T = int` 时的 `typename T::type` 不存在） |
| **直接语境（immediate context）** | SFINAE 的失败边界：只有签名**本身**的失败才算数。函数体内的错误、替换引发的连带实例化失败都不属于直接语境——那些是硬错误 |

规则一句话：**替换在直接语境中失败 → 该候选被移出候选集，继续尝试其他候选；全部出局才报 no matching function**。反过来说：函数体内的错误发生在重载决议**之后**的实例化阶段，无法"回传"成替换失败——这就是为什么必须用 `enable_if` 把"类型不满足要求"**预先写进签名**。

`std::enable_if` 是对这条规则的标准化封装，定义只有两行：

```cpp
template <bool B, typename T = void> struct enable_if {};            // B=false：空壳，没有 type
template <typename T> struct enable_if<true, T> { using type = T; }; // B=true ：有 type
```

条件为假时 `enable_if_t<false>` 没有 `::type`，在签名里引用它即制造"条件性替换失败"。正确写法是把条件放进**额外非类型模板参数的类型**里：

```cpp
// templates_and_sfinae.cpp（续）—— SFINAE
// 形态：enable_if_t<cond, int> = 0 —— 条件为假时第二个模板参数的类型不存在 → 替换失败
template <typename T,
          std::enable_if_t<std::is_integral_v<T>, int> = 0>
std::string classify(T val) {
    return "整型: " + std::to_string(val);
}

template <typename T,
          std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
std::string classify(T val) {
    return "浮点型";
}

// classify(42)   → 整型重载替换成功；浮点重载 enable_if<false, int> 无 type → 被剔除
// classify(3.14) → 镜像过程；两条路径互不干扰，各自只实例化胜者
```

> ⚠️ **经典陷阱**：若写成 `typename = std::enable_if_t<cond>`（默认类型模板参数形态），两个 `classify` 会**重定义冲突**——默认模板实参**不参与函数模板签名**，两个重载在编译器眼里是同一个模板。推荐形态把条件放进第二个模板参数的**类型本身**，各重载才是不同模板，且对任一具体类型至多一个替换成功、天然互斥。

enable_if 的三种常见摆法：

| 形态 | 写法 | 问题 |
|------|------|------|
| 返回类型 | `typename enable_if_t<cond, T> f(...)` | 构造函数、类型转换运算符没有返回类型，用不了 |
| **冗余非类型模板参数（推荐）** | `template <typename T, enable_if_t<cond, int> = 0>` | 无 |
| 默认类型模板参数 | `template <typename T, typename = enable_if_t<cond>>` | 多个重载会撞签名（上面的陷阱） |

```cpp
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

`has_size` 是 **detection idiom（接口探测惯用法）**的标准形态：用一对重载的 `test` 函数 + 尾置返回类型里的 `decltype` 表达式制造"有 `size()` → 替换成功、没有 → 替换失败"，再把结果装进 `true_type`/`false_type`。C++17 的 `std::void_t` 把它压缩到四行：

```cpp
// C++17 detection idiom：void_t 版 has_size
template <typename, typename = void>
struct detect_size : std::false_type {};                 // 主模板：默认"没有"

template <typename T>
struct detect_size<T, std::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};                                 // 偏特化：表达式合法则"有"
```

原理：`void_t<...>` 恒等于 `void`，但它给了偏特化一个"把**表达式合法性**变成**特化选择**"的机会——`decltype(declval<T>().size())` 非法时偏特化替换失败，落回主模板得 `false`。注意这是 SFINAE 用于**类模板偏特化选择**的形态（enable_if 形态则用于函数重载决议）——同一条规则的两副面孔。

#### if constexpr（C++17，SFINAE 的简化替代）

**定义**：`if constexpr` 的条件必须是**编译期常量表达式**（上下文可转 `bool`）；条件为假的分支成为**被丢弃语句（discarded statement）**——外围模板实例化时**不实例化**被丢弃语句中的代码，因此其中依赖模板参数、对当前类型非法的操作不会报错。

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

> 💡 **面试要点**：`if constexpr` 与普通 `if` 的本质区别——

| 维度 | 普通 `if` | `if constexpr` |
|------|-----------|----------------|
| 条件求值 | 运行期 | 编译期（必须是常量表达式） |
| false 分支 | 照常编译（两个分支都必须对当前类型合法） | **被丢弃、不实例化**（可含对当前类型非法的操作） |
| 生成代码 | 两分支都生成 + 运行期跳转 | 只生成选中分支 |
| 模板内按类型分路径 | 必须绕道重载 / SFINAE | 直接分支 |

> ⚠️ **丢弃 ≠ 不编译**：被丢弃语句只是**不实例化**（两阶段查找的第二阶段被跳过），语法分析和**非依赖**语义检查照常执行——被丢弃分支里写 `static_assert(sizeof(int) == 8)`（非依赖、恒假）照样报错。判断标准是**依赖性**：`if constexpr` 不是预处理器的 `#if`。

> 💡 两个实用细节：被丢弃分支的 `return` **不参与 `auto` 返回类型推导**，所以各分支可以返回不同类型；最经典的用法是**递归终止**——递归调用写在条件不满足的分支里，实例化自然停止，不再需要终止重载。

> 📖 **深入阅读**：SFINAE 的编译器流水线逐步拆解、if constexpr 与 SFINAE 在**作用层面**（接口层 vs 实现层）与**编译行为**（替换阶段 vs 实例化阶段）上的本质区别，见模拟面试精讲 [SFINAE 与 if constexpr](../interview/mock_interview/sfinae_vs_if_constexpr.md)；SFINAE → Concepts 的完整改造见 [从 enable_if 到 C++20 Concepts](../interview/mock_interview/enable_if_to_concepts.md)。

### 学习任务 5：C++20 Concepts（20 分钟）

Concepts 是 C++20 引入的模板参数约束机制，取代了繁琐的 SFINAE。四个核心定义：

| 概念 | 定义 |
|------|------|
| **concept** | 类型上的**具名布尔谓词**：`template <typename T> concept X = 常量表达式;`——约束终于有了名字，可组合（`&&`/`\|\|`/`!`）、可复用 |
| **requires 子句** | 挂在模板声明上的约束：`template <typename T> requires X<T> ...`，约束成为签名的一部分 |
| **requires 表达式** | 就地探测"这些表达式是否合法"并产出 bool 的表达式：`requires(const T& t) { t.size(); }`——detection idiom 的官方替代 |
| **subsumption（约束蕴涵）** | 约束被归一化成**原子约束**集合后，若 A 的原子集 ⊆ B 的原子集，则 B **更特化、重载优先入选**——概念第一次参与重载排序，enable_if 永远做不到 |

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

#### requires 子句 vs requires 表达式（易混点）

两个 `requires` 长得一样但角色不同：**子句**把约束挂到声明上（跟在 template 列表之后），**表达式**产出一个 bool 值。两者可组合出 `requires requires (T t) { ... }`（第一个是子句，第二个是表达式）。requires 表达式还能直接当 `if constexpr` 的条件用：

```cpp
if constexpr (requires { val.size(); }) {  // C++20：detection idiom 一行版
    // 有 size() 的路径
}
```

#### 简写函数模板（constrained auto）

约束了 `auto` 参数的函数自动成为模板。单参数场景三种写法语义等价：

```cpp
template <Numeric T> void incr(T& x);        // 写法 A：concept 代替 typename
template <typename T> requires Numeric<T>    // 写法 B：requires 子句
void incr(T& x);
void incr(Numeric auto& x);                  // 写法 C：constrained auto（简写模板）
```

（注意：`incr(Numeric auto a, Numeric auto b)` 的两个参数是**两个独立**模板参数，不等价于 `template <Numeric T> void incr(T a, T b)`——多参数要共享类型时只能用 A/B。）

#### subsumption 实例：约束参与重载排序

```cpp
template <typename T> concept HasSize = requires(const T& t) { t.size(); };
template <typename T> concept SizedNumeric = HasSize<T> && std::integral<typename T::value_type>;

template <HasSize T>      void dump(const T& v);  // 一般容器
template <SizedNumeric T> void dump(const T& v);  // 整型元素容器：约束更紧、更特化

// dump(vector<int>)    → 第二个（SizedNumeric 蕴涵 HasSize → 优先入选，不 ambiguous）
// dump(vector<string>) → 第一个
```

用 enable_if 写同样的分派必须手工构造互斥条件（`has_size && !is_integral<value_type>` ……），每加一层就要回改所有已有条件——subsumption 让重载集合可以**自然扩展**，这是 Concepts 最实质的能力提升。

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

### 面试题积累（今日 7 道）

**Q1：函数模板和类模板有什么区别？模板为什么通常写在头文件里？**
> 答：函数模板由编译器根据实参自动推导类型参数；类模板需要显式指定类型参数（C++17 CTAD 除外）。模板必须写在头文件中，因为模板是生成代码的"蓝图"，编译器在每个使用点都需要看到完整定义才能实例化。如果定义在 `.cpp` 中，其他翻译单元无法实例化。显式实例化可以解决这个问题但会增加编译耦合。

**Q2：模板的特化和偏特化有什么区别？函数模板能偏特化吗？**
> 答：全特化是为特定类型参数提供完整实现；偏特化是为部分类型参数（如指针类型 `T*`）提供实现。类模板支持全特化和偏特化；函数模板只支持全特化，不支持偏特化——需要偏特化时用函数重载代替。全特化后不再有模板参数，偏特化仍保留部分模板参数。

**Q3：什么是变参模板？C++17 的折叠表达式解决了什么问题？**
> 答：变参模板用 `typename... Args` 接受任意数量参数。C++17 前需要递归展开（定义终止函数 + 递归版本），代码冗长；C++17 折叠表达式用 `(pack op ...)` 一行展开，如 `(args + ...)` 求和、`((cout << args), ...)` 逐个打印。折叠有四种形式：一元/二元 × 左/右折叠。CUTLASS 的 CuTe Layout/Shape 大量使用变参模板。

**Q4：SFINAE 是什么？`if constexpr` 和 SFINAE 有什么区别？**
> 答：SFINAE（Substitution Failure Is Not An Error）——把模板实参代回候选**签名**时失败不是错误，而是该候选从重载候选集中移除；失败只算**直接语境**（签名本身）里的，函数体内的错误无法回传成替换失败，所以约束必须预先写进签名（enable_if）。`if constexpr`（C++17）作用于**函数体内**——false 分支成为被丢弃语句、不实例化（但非依赖代码仍被检查）。区别：SFINAE 在**接口层**做选择——决定哪个重载对这组实参存在，发生于重载决议的替换阶段；`if constexpr` 在**实现层**做选择——决定函数体内哪段代码被实例化，发生于实例化阶段。C++20 Concepts 接管接口层约束，语法更清晰、报错更友好。

**Q5：C++20 Concepts 解决了什么问题？**
> 答：Concepts 对模板参数提供命名约束，解决 SFINAE 的两个痛点——① 语法繁琐（`enable_if` 嵌套）；② 报错信息不可读（几百行模板展开错误）。Concepts 让约束声明式化（`template <Numeric T>`），报错时直接指出"不满足 Numeric concept"。还支持 `requires` 表达式检测类型是否有特定成员函数，如 `requires(T t) { t.size(); }`。

**Q6：什么是两阶段查找？模板里的 `typename` 关键字是干什么的？**
> 答：模板编译分两阶段——①定义处检查语法与非依赖代码、查找非依赖名字；②实例化处把实参代入后检查依赖代码、查找依赖名字。`typename` 用于依赖名消歧：`T::iterator` 里的 `iterator` 是依赖名，编译器默认假定它不是类型，要当类型用必须写 `typename T::iterator`（C++20 起多数场合可省略）。同理，依赖对象后跟 `<` 要当模板用必须写 `obj.template get<int>()`。这也解释了为什么模板体内的类型错误报错在实例化处而非定义处。

**Q7：两个 `typename = std::enable_if_t<...>` 形态的重载为什么编译失败？正确写法是什么？**
> 答：默认模板实参**不参与函数模板签名**，两个只差默认模板实参的"重载"是同一个模板，直接重定义错误。正确写法是把条件放进额外**非类型模板参数的类型**里：`template <typename T, std::enable_if_t<cond, int> = 0>`——两个重载的第二个模板参数类型不同，才是不同的模板；且对任一具体类型至多一个条件为真、替换成功，天然互斥无歧义。

### 今日检查清单

- [ ] 能解释模板的两阶段编译（定义检查 vs 实例化检查）
- [ ] 知道模板为什么通常写在头文件里
- [ ] 能说出两阶段查找、依赖名，以及 `typename`/`template` 消歧的用法
- [ ] 能区分全特化和偏特化，知道函数模板不能偏特化、全特化不参与重载决议
- [ ] 能用递归或折叠表达式展开 parameter pack，知道空包的折叠规则
- [ ] 能解释 SFINAE 原理（替换 / 直接语境）并写出 `enable_if` 的正确形态
- [ ] 知道 `typename = enable_if_t<...>` 多重载为什么冲突，会写推荐形态
- [ ] 能说出 `if constexpr` 与普通 `if` 的区别，解释"丢弃 ≠ 不编译"
- [ ] 能用 C++20 concept 约束模板参数，区分 requires 子句与 requires 表达式
- [ ] 能说出 subsumption 解决了什么（约束参与重载排序）以及 SFINAE → `if constexpr` → Concepts 的演进
- [ ] `templates_and_sfinae.cpp` 编译运行通过

#### 明日预告

Day 5 将深入**面向对象与多态底层**——虚函数表的内存布局、虚析构函数为什么必要、多重继承的对象布局、Rule of 3/5/0。今天学的模板加上明天的多态，就构成了理解 CUTLASS/DeepGEMM 源码的完整 C++ 基础。建议今晚先想想：虚函数是怎么实现"运行时多态"的？

---
