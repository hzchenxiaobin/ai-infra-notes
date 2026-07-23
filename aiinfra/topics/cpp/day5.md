# Day 5（周五）：面向对象与多态底层

> **本周定位**：C++ 面试系统化准备，今日深入虚函数表与多态的底层实现
> **前置要求**：已完成 Day 1-4，理解内存布局与对象生命周期
> **今日目标**：深入虚函数表（vtable）与虚表指针（vptr）、虚析构函数的必要性、多重继承的内存布局、纯虚函数与抽象类、Rule of 3/5/0、对象切片，能手推单继承与多重继承的对象内存布局
> **时间投入**：2.5h（早间 1.5h 精读虚函数表 + 晚间 1h 跑代码验证内存布局）
> **考察度**：⭐⭐⭐⭐⭐ 核心考点，"虚函数表怎么实现的"是经典面试题

---

## 本日在本周知识图谱中的位置

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| 虚函数表与 vptr 内存模型 | ① 能画出 C++ 对象内存布局（含 vtable） |
| 虚析构函数必要性 demo | ① 同上 |
| 多重继承内存布局图 | ① 同上 |
| Rule of 3/5/0 对比表 | ② 能手写 RAII 资源管理类 |
| 对象切片 demo | ① 同上 |

---

### 学习任务 1：虚函数与虚函数表（45 分钟）

#### 静态绑定 vs 动态绑定

| 维度 | 普通函数（静态绑定） | 虚函数（动态绑定） |
|------|---------------------|-------------------|
| 绑定时机 | 编译期 | 运行时 |
| 决定因素 | 静态类型（指针/引用类型） | 动态类型（实际对象类型） |
| 实现方式 | 直接调用地址 | 通过 vtable 间接调用 |
| 开销 | 无 | 一次间接跳转 + 可能的 cache miss |

#### 虚函数表（vtable）的工作原理

每个**含有虚函数的类**都有一个虚函数表（vtable），存放该类所有虚函数的地址。每个**对象**都有一个虚表指针（vptr），指向所属类的 vtable。

```cpp
// vtable_and_polymorphism.cpp —— 虚函数表与多态
// 编译: g++ -std=c++20 -o vtable vtable_and_polymorphism.cpp && ./vtable

#include <iostream>
#include <string>

class Animal {
public:
    virtual ~Animal() = default;           // 虚析构（今天重点）
    virtual void speak() const {            // 虚函数
        std::cout << "  Animal::speak()" << std::endl;
    }
    virtual int legs() const { return 0; } // 虚函数
    void eat() {                            // 非虚函数
        std::cout << "  Animal::eat()" << std::endl;
    }
protected:
    int age_ = 0;  // 成员变量
};

class Dog : public Animal {
public:
    void speak() const override {           // 重写虚函数
        std::cout << "  Dog::speak() — Woof!" << std::endl;
    }
    int legs() const override { return 4; }
    void fetch() {                           // 非虚函数
        std::cout << "  Dog::fetch()" << std::endl;
    }
private:
    std::string breed_ = "unknown";  // 派生类成员
};

void demo_virtual_dispatch() {
    std::cout << "=== 虚函数动态绑定 ===" << std::endl;
    Dog dog;
    Animal* p = &dog;  // 静态类型 Animal*，动态类型 Dog

    p->speak();  // 动态绑定 → Dog::speak()
    p->eat();    // 静态绑定 → Animal::eat()（非虚函数）
    std::cout << "  legs = " << p->legs() << std::endl;  // 动态绑定 → Dog::legs()
}
```

#### vtable 内存模型

| 内存区域 | 内容 |
|----------|------|
| **Animal 的 vtable** | `[{~Animal, &Animal::speak, &Animal::legs}]` |
| **Dog 的 vtable** | `[{~Dog, &Dog::speak, &Dog::legs}]` |
| **Dog 对象内存布局** | `[vptr → Dog::vtable] [age_] [breed_]` |

> 💡 **关键洞察**：虚函数调用分两步——① 通过对象的 vptr 找到 vtable；② 在 vtable 中按偏移找到函数地址。因为 vptr 指向实际类型的 vtable，所以即使指针类型是基类，调用的也是派生类的函数。这就是"运行时多态"的本质。

#### 虚函数的代价

| 开销 | 说明 |
|------|------|
| 每对象 +8 字节 | vptr 占一个指针大小（64 位系统 8 字节） |
| 调用间接跳转 | 一次额外内存访问（读 vptr → 读 vtable[slot]） |
| 可能 cache miss | vtable 可能不在 cache 中 |
| 编译器优化受限 | 编译器难以内联虚函数（Devirtualization 优化除外） |

> ⚠️ **注意**：不是所有函数都该声明为 `virtual`。只有需要运行时多态的才用虚函数。性能敏感场景（如 CUDA kernel 的 host 端调用链）要特别注意虚函数开销。

### 学习任务 2：虚析构函数（30 分钟）

#### 为什么析构函数要声明为 virtual

这是 C++ 面试的**经典必问题**——如果基类析构不是 virtual，通过基类指针 delete 派生类对象时只调用基类析构，派生类析构不被调用，导致资源泄漏。

```cpp
// vtable_and_polymorphism.cpp（续）—— 虚析构函数

class BadBase {
public:
    ~BadBase() { std::cout << "  ~BadBase()" << std::endl; }  // 非虚析构！
};

class BadDerived : public BadBase {
    int* data_;
public:
    BadDerived() : data_(new int[100]) {
        std::cout << "  BadDerived() 分配 400 字节" << std::endl;
    }
    ~BadDerived() {
        delete[] data_;
        std::cout << "  ~BadDerived() 释放 400 字节" << std::endl;
    }
};

class GoodBase {
public:
    virtual ~GoodBase() { std::cout << "  ~GoodBase()" << std::endl; }  // 虚析构！
};

class GoodDerived : public GoodBase {
    int* data_;
public:
    GoodDerived() : data_(new int[100]) {
        std::cout << "  GoodDerived() 分配 400 字节" << std::endl;
    }
    ~GoodDerived() override {
        delete[] data_;
        std::cout << "  ~GoodDerived() 释放 400 字节" << std::endl;
    }
};

void demo_virtual_destructor() {
    std::cout << "\n=== 虚析构函数 ===" << std::endl;

    std::cout << "  [BadBase — 非虚析构]" << std::endl;
    BadBase* bad = new BadDerived();
    delete bad;  // 只调用 ~BadBase()！~BadDerived() 不调用 → 内存泄漏！

    std::cout << "\n  [GoodBase — 虚析构]" << std::endl;
    GoodBase* good = new GoodDerived();
    delete good;  // 先 ~GoodDerived() 再 ~GoodBase() → 正确释放
}
```

> 💡 **规则**：只要类打算被继承且通过基类指针删除，析构函数**必须**声明为 `virtual`。反过来，如果不打算被继承（如 `final` 类或工具类），不需要虚析构——虚析构会引入 vtable 开销。

### 学习任务 3：多重继承的内存布局（30 分钟）

#### 单继承布局

| 布局 | 内容 |
|------|------|
| `Dog` 对象 | `[vptr → Dog::vtable] [Animal::age_] [Dog::breed_]` |

#### 多重继承布局

```cpp
// vtable_and_polymorphism.cpp（续）—— 多重继承

class Drawable {
public:
    virtual ~Drawable() = default;
    virtual void draw() const { std::cout << "  Drawable::draw()" << std::endl; }
protected:
    int x_ = 0, y_ = 0;
};

class Clickable {
public:
    virtual ~Clickable() = default;
    virtual void onClick() { std::cout << "  Clickable::onClick()" << std::endl; }
protected:
    int click_count_ = 0;
};

class Button : public Drawable, public Clickable {
public:
    void draw() const override { std::cout << "  Button::draw()" << std::endl; }
    void onClick() override { std::cout << "  Button::onClick()" << std::endl; }
private:
    std::string label_ = "OK";
};

void demo_multiple_inheritance() {
    std::cout << "\n=== 多重继承 ===" << std::endl;

    std::cout << "  sizeof(Drawable) = " << sizeof(Drawable) << std::endl;    // 16 (vptr+8 + x+y=8)
    std::cout << "  sizeof(Clickable) = " << sizeof(Clickable) << std::endl;  // 16 (vptr+8 + click_count=4+pad4)
    std::cout << "  sizeof(Button) = " << sizeof(Button) << std::endl;        // 48 (两个vptr + 两个基类成员 + label)

    Button btn;
    Drawable* d = &btn;
    Clickable* c = &btn;

    std::cout << "  Button 地址:  " << &btn << std::endl;
    std::cout << "  as Drawable: " << d << std::endl;    // 同一地址（第一个基类）
    std::cout << "  as Clickable:" << c << std::endl;     // 偏移后的地址（第二个基类）

    d->draw();     // Button::draw()
    c->onClick();  // Button::onClick()
}
```

**Button 对象内存布局（多重继承）：**

| 偏移 | 内容 | 来源 |
|------|------|------|
| 0 | vptr₁ → Button::vtable_Drawable | Drawable 的虚表 |
| 8 | x_ (4) + y_ (4) | Drawable 成员 |
| 16 | vptr₂ → Button::vtable_Clickable | Clickable 的虚表 |
| 24 | click_count_ (4) + padding (4) | Clickable 成员 |
| 32 | label_ (std::string, ~32 字节) | Button 成员 |

> ⚠️ **注意**：多重继承时，派生类有**多个 vptr**——每个有虚函数的基类各一个。`static_cast<Clickable*>(&btn)` 会调整指针偏移，指向 Clickable 子对象的位置。这就是为什么多重继承的指针转换不是简单的地址复制。

#### 虚继承（菱形继承）

菱形继承（A → B,C → D）中，D 有两个 A 子对象。虚继承 `virtual public A` 让 D 只有一个 A 子对象：

| 普通菱形继承 | 虚继承 |
|-------------|--------|
| D 有两个 A 子对象（冗余） | D 只有一个 A 子对象 |
| 访问 A 成员有歧义 | 无歧义 |
| 不需要额外指针 | 需要 vbase pointer/table |

> 💡 **面试建议**：菱形继承在实际工程中极少使用，面试主要考"知不知道有这个问题"。CUTLASS / CUDA 工程中几乎不出现虚继承——性能敏感代码避免多重继承的开销。

### 学习任务 4：Rule of 3/5/0（20 分钟）

| 规则 | 版本 | 需要定义的函数 | 适用场景 |
|------|------|---------------|----------|
| **Rule of 3** | C++98 | 析构、拷贝构造、拷贝赋值 | 管理资源的类 |
| **Rule of 5** | C++11 | Rule of 3 + 移动构造、移动赋值 | 管理资源且需要移动的类 |
| **Rule of 0** | 现代 C++ | 都不定义，用智能指针管理 | 大多数类（推荐） |

```cpp
// vtable_and_polymorphism.cpp（续）—— Rule of 5

class ResourceManager {
    int* data_;
    size_t size_;
public:
    // 构造
    explicit ResourceManager(size_t n) : data_(new int[n]()), size_(n) {}

    // 1. 析构
    ~ResourceManager() { delete[] data_; }

    // 2. 拷贝构造
    ResourceManager(const ResourceManager& other)
        : data_(new int[other.size_]), size_(other.size_) {
        std::copy(other.data_, other.data_ + size_, data_);
    }

    // 3. 拷贝赋值
    ResourceManager& operator=(const ResourceManager& other) {
        if (this != &other) {
            int* new_data = new int[other.size_];  // 先分配（异常安全）
            std::copy(other.data_, other.data_ + other.size_, new_data);
            delete[] data_;
            data_ = new_data;
            size_ = other.size_;
        }
        return *this;
    }

    // 4. 移动构造
    ResourceManager(ResourceManager&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // 5. 移动赋值
    ResourceManager& operator=(ResourceManager&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
};

// Rule of 0：用智能指针，让编译器自动生成
class ModernResource {
    std::unique_ptr<int[]> data_;  // 自动管理
    size_t size_;
public:
    explicit ModernResource(size_t n) : data_(std::make_unique<int[]>(n)), size_(n) {}
    // 不需要写析构/拷贝/移动——unique_ptr 自动处理
    // unique_ptr 不可拷贝 → ModernResource 也不可拷贝
    // unique_ptr 可移动 → ModernResource 自动获得移动语义
};
```

> 💡 **Rule of 0 是现代 C++ 的推荐实践**：用智能指针/STL 容器管理资源，让编译器自动生成特殊成员函数。只有当你直接管理裸资源（如 `new`/`delete`、`fopen`/`fclose`、`cudaMalloc`/`cudaFree`）时才需要 Rule of 5。

### 学习任务 5：对象切片（15 分钟）

```cpp
// vtable_and_polymorphism.cpp（续）—— 对象切片

void process_by_value(Animal a) {  // 按值传递 → 切片！
    a.speak();  // 永远调用 Animal::speak()，Dog 部分被切掉
}

void process_by_ref(const Animal& a) {  // 按引用传递 → 不切片
    a.speak();  // 动态绑定 → Dog::speak()
}

void demo_object_slicing() {
    std::cout << "\n=== 对象切片 ===" << std::endl;
    Dog dog;

    std::cout << "  按值传递（切片）:" << std::endl;
    process_by_value(dog);  // Dog 的 Animal 部分拷贝到 a，Dog 特有部分丢失

    std::cout << "  按引用传递（不切片）:" << std::endl;
    process_by_ref(dog);    // 引用指向完整 Dog 对象
}
```

> ⚠️ **注意**：对象切片发生在按值传递派生类对象给基类参数时——派生类部分被"切掉"，只保留基类部分。vptr 也变成基类的 vptr，虚函数调用基类版本。解决方案：用指针或引用传递。

### 面试题积累（今日 6 道）

**Q1：虚函数是怎么实现的？虚函数表是什么？**
> 答：每个含有虚函数的类有一个虚函数表（vtable），存放该类所有虚函数的地址。每个对象有一个虚表指针（vptr），在构造时指向所属类的 vtable。虚函数调用时：① 通过对象的 vptr 找到 vtable；② 在 vtable 中按函数偏移找到地址；③ 间接调用。因为 vptr 指向实际类型的 vtable，所以通过基类指针也能调用派生类的虚函数。代价：每对象 +8 字节 vptr，每次调用一次间接跳转。

**Q2：为什么基类的析构函数要声明为 virtual？**
> 答：如果不声明 virtual，通过基类指针 delete 派生类对象时，只调用基类析构，派生类析构不被调用，导致派生类资源泄漏。声明 virtual 后，析构调用是动态绑定的——先调用派生类析构，再调用基类析构。规则：只要类可能被继承且通过基类指针删除，析构必须 virtual。不打算被继承的类不需要（避免 vtable 开销）。

**Q3：构造函数能是虚函数吗？为什么？**
> 答：不能。虚函数调用依赖 vptr，而 vptr 在构造函数中才被设置——构造函数执行前对象还没有 vptr（或指向基类 vptr）。构造函数的作用就是初始化对象（包括设置 vptr），所以它本身不能是虚函数。析构函数可以是虚函数，因为析构时 vptr 已经存在。

**Q4：什么是对象切片？如何避免？**
> 答：对象切片发生在按值传递派生类对象给基类参数时——派生类特有的部分被"切掉"，只拷贝基类部分，vptr 也变成基类的。之后虚函数调用基类版本，丢失多态性。避免方法：用指针或引用传递，不要按值传递多态对象。

**Q5：Rule of 3/5/0 是什么？什么时候用哪个？**
> 答：Rule of 3（C++98）：如果类需要自定义析构、拷贝构造、拷贝赋值中的任一个，通常三个都需要。Rule of 5（C++11）：在 Rule of 3 基础上加移动构造和移动赋值。Rule of 0（现代 C++）：用智能指针/STL 管理资源，不定义任何特殊成员函数，让编译器自动生成。推荐 Rule of 0——只有直接管理裸资源时才用 Rule of 5。

**Q6：多重继承的对象内存布局是怎样的？有什么问题？**
> 答：多重继承的派生类有多个 vptr（每个有虚函数的基类各一个），对象内存按声明顺序排列各基类子对象。指针类型转换时需要调整偏移（`static_cast<Base2*>(&derived)` 不是同一地址）。菱形继承会导致基类子对象重复，用虚继承解决但增加 vbase 开销。性能敏感代码应避免多重继承。

### 今日检查清单

- [ ] 能解释虚函数表的工作原理（vtable + vptr + 间接调用）
- [ ] 能说出虚函数的代价（vptr 开销 + 间接跳转 + 优化受限）
- [ ] 能解释为什么基类析构要声明 virtual
- [ ] 知道构造函数不能是虚函数（vptr 在构造时才设置）
- [ ] 能画出单继承和多重继承的对象内存布局
- [ ] 理解多重继承时指针转换的偏移调整
- [ ] 能说出 Rule of 3/5/0 的区别和推荐用法
- [ ] 能解释对象切片及其避免方法
- [ ] `vtable_and_polymorphism.cpp` 编译运行通过

#### 明日预告

Day 6 将深入**并发编程基础**——`std::thread`、`mutex`/`lock_guard`/`unique_lock`、`condition_variable`、`std::atomic` 与 6 种 memory order。今天的虚函数和对象生命周期知识在理解并发同步原语时同样重要。建议今晚先想想：`shared_ptr` 的引用计数是怎么做到线程安全的？

---
