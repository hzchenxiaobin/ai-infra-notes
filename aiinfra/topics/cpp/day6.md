# Day 6（周六）：并发编程基础

> **本周定位**：C++ 面试系统化准备，今日聚焦多线程并发——系统岗位必问
> **前置要求**：已完成 Day 1-5，理解 RAII（`lock_guard` 也是 RAII）、移动语义（`unique_lock` 可移动）
> **今日目标**：掌握 `std::thread` 基础、`mutex`/`lock_guard`/`unique_lock`、`condition_variable`、`std::atomic` 与 6 种 memory order、死锁预防、thread-safe singleton，能手写生产者-消费者模型
> **时间投入**：2.5h（早间 1.5h 精读并发原语 + 晚间 1h 跑代码与死锁实验）
> **考察度**：⭐⭐⭐⭐ 实战考点，系统/后端/AI Infra 方向必问

---

## 本日在本周知识图谱中的位置

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| `std::thread` 创建与 join/detach | ⑤ 能解释并发原语 |
| `mutex` + RAII 锁管理 | ② 能手写 RAII 资源管理类（`lock_guard`） |
| `condition_variable` 生产者-消费者 | ⑤ 同上 |
| `std::atomic` 与 6 种 memory order | ⑤ 同上（核心难点） |
| Thread-safe singleton 实现 | ⑤ 同上（面试高频手撕） |

---

### 学习任务 1：std::thread 基础（30 分钟）

#### 创建与管理线程

```cpp
// concurrency_basics.cpp —— 并发编程基础
// 编译: g++ -std=c++20 -pthread -o concurrency concurrency_basics.cpp && ./concurrency

#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <vector>
#include <chrono>
#include <string>

void demo_thread_basics() {
    std::cout << "=== std::thread 基础 ===" << std::endl;

    // 1. 用函数指针创建线程
    auto func = [](int id) {
        std::cout << "  线程 " << id << " 运行中" << std::endl;
    };
    std::thread t1(func, 1);
    t1.join();  // 等待线程结束

    // 2. 用 lambda 创建线程
    std::thread t2([]() {
        std::cout << "  lambda 线程运行中" << std::endl;
    });
    t2.join();

    // 3. detach：分离线程（后台运行，不等待）
    std::thread t3([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // 注意：detach 后线程在后台运行，主线程结束时可能被强制终止
    });
    t3.detach();

    // 4. 获取硬件并发数
    std::cout << "  硬件并发线程数: " << std::thread::hardware_concurrency() << std::endl;
}
```

> ⚠️ **注意**：`std::thread` 对象必须在 `join()` 或 `detach()` 之前销毁——否则程序终止（调用 `std::terminate`）。`join` 等待线程完成，`detach` 分离到后台。通常优先 `join`；`detach` 要确保线程不访问已销毁的变量。

#### 线程传参的陷阱

```cpp
void demo_thread_params() {
    std::cout << "\n=== 线程传参陷阱 ===" << std::endl;

    // 陷阱 1：传引用需要 std::ref
    int x = 0;
    // std::thread t([&x] { x = 42; });  // 用 lambda 捕获引用更安全
    std::thread t([](int& ref) { ref = 42; }, std::ref(x));  // 显式 ref
    t.join();
    std::cout << "  x = " << x << std::endl;  // 42

    // 陷阱 2：传指针到局部变量（悬空指针）
    // auto bad = []() {
    //     int local = 42;
    //     std::thread t([](int* p) { /* use p */ }, &local);
    //     t.detach();  // local 可能已销毁！
    // };

    // 安全做法：传值或用智能指针
    auto safe = [](std::shared_ptr<int> sp) {
        std::cout << "  *sp = " << *sp << std::endl;
    };
    std::thread t2(safe, std::make_shared<int>(100));
    t2.join();
}
```

### 学习任务 2：mutex 与 RAII 锁管理（40 分钟）

#### 三种锁管理器

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| `std::lock_guard<M>` | 构造加锁、析构解锁，不可中途解锁 | 简单临界区 |
| `std::unique_lock<M>` | 可中途解锁、可移动、支持延迟加锁/条件变量 | 灵活场景、`condition_variable` |
| `std::scoped_lock<M...>` | 可同时锁多个 mutex（避免死锁） | 多锁场景 |

```cpp
// concurrency_basics.cpp（续）—— mutex 与 RAII

class ThreadSafeCounter {
    mutable std::mutex mtx_;  // mutable：const 函数中也能锁
    int count_ = 0;
public:
    void increment() {
        std::lock_guard<std::mutex> lock(mtx_);  // RAII：构造加锁，析构解锁
        ++count_;
    }

    int get() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return count_;
    }

    // 需要灵活控制时用 unique_lock
    int get_and_reset() {
        std::unique_lock<std::mutex> lock(mtx_);
        int val = count_;
        count_ = 0;
        lock.unlock();  // 可以中途解锁（lock_guard 不行）
        // 做一些不需要锁的操作...
        return val;
    }
};

void demo_mutex() {
    std::cout << "\n=== mutex 与 RAII 锁 ===" << std::endl;
    ThreadSafeCounter counter;
    std::vector<std::thread> threads;

    for (int i = 0; i < 10; i++) {
        threads.emplace_back([&counter]() {
            for (int j = 0; j < 1000; j++) counter.increment();
        });
    }
    for (auto& t : threads) t.join();

    std::cout << "  最终计数: " << counter.get() << " (期望 10000)" << std::endl;
}
```

#### 死锁与预防

```cpp
// concurrency_basics.cpp（续）—— 死锁预防

// 死锁场景：两个线程以不同顺序锁两个 mutex
class DeadlockDemo {
    std::mutex mtx1_, mtx2_;
public:
    void func_a() {
        std::lock_guard<std::mutex> l1(mtx1_);  // 先锁 1
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::lock_guard<std::mutex> l2(mtx2_);  // 再锁 2 ← 可能死锁
    }
    void func_b() {
        std::lock_guard<std::mutex> l2(mtx2_);  // 先锁 2
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::lock_guard<std::mutex> l1(mtx1_);  // 再锁 1 ← 可能死锁
    }

    // 解决方案 1：std::scoped_lock 同时锁多个（原子操作，不会死锁）
    void func_safe() {
        std::scoped_lock lock(mtx1_, mtx2_);  // 同时锁，避免死锁
        // ...
    }

    // 解决方案 2：固定加锁顺序（所有线程都按同一顺序锁）
    void func_ordered() {
        std::lock_guard<std::mutex> l1(mtx1_);  // 总是先锁 1
        std::lock_guard<std::mutex> l2(mtx2_);  // 再锁 2
        // ...
    }
};
```

| 死锁预防策略 | 说明 |
|-------------|------|
| **固定顺序** | 所有线程按相同顺序获取锁 |
| **scoped_lock** | `std::scoped_lock` 原子地锁多个 mutex |
| **try_lock + 回退** | `std::try_lock` 尝试加锁，失败则释放已有锁重试 |
| **避免嵌套锁** | 持有锁时不调用未知代码（可能间接获取其他锁） |

> 💡 **RAII 与锁**：`lock_guard` 是 RAII 的经典应用——构造时加锁，析构时自动解锁。即使临界区内抛异常，锁也会正确释放。永远不要手动 `lock()`/`unlock()`，用 RAII 包装器。

### 学习任务 3：condition_variable（30 分钟）

`condition_variable` 用于线程间通知，经典应用是生产者-消费者模型：

```cpp
// concurrency_basics.cpp（续）—— 生产者-消费者

template <typename T>
class ThreadSafeQueue {
    std::queue<T> queue_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    bool done_ = false;

public:
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            queue_.push(std::move(value));
        }
        cv_.notify_one();  // 通知一个等待的消费者
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !queue_.empty() || done_; });  // 等待条件
        if (queue_.empty() && done_) return false;  // 队列空且生产结束
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void set_done() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            done_ = true;
        }
        cv_.notify_all();  // 通知所有等待的消费者
    }
};

void demo_producer_consumer() {
    std::cout << "\n=== 生产者-消费者 ===" << std::endl;
    ThreadSafeQueue<int> queue;

    // 生产者
    auto producer = [&queue]() {
        for (int i = 0; i < 5; i++) {
            queue.push(i);
            std::cout << "  生产: " << i << std::endl;
        }
        queue.set_done();
    };

    // 消费者
    auto consumer = [&queue]() {
        int val;
        while (queue.pop(val)) {
            std::cout << "  消费: " << val << std::endl;
        }
    };

    std::thread p(producer);
    std::thread c(consumer);
    p.join();
    c.join();
}
```

> ⚠️ **注意**：`cv_.wait(lock, predicate)` 内部用循环防止**虚假唤醒（spurious wakeup）**——即使没有 `notify`，`wait` 也可能返回。带谓词的 `wait` 会自动处理虚假唤醒。如果用无谓词版本 `cv_.wait(lock)`，需要手动 `while (!condition) cv_.wait(lock);`。

### 学习任务 4：std::atomic 与 memory order（40 分钟）

#### atomic 基础

`std::atomic` 提供无锁原子操作，避免 mutex 开销：

```cpp
// concurrency_basics.cpp（续）—— atomic

void demo_atomic() {
    std::cout << "\n=== std::atomic ===" << std::endl;

    // atomic 计数器（无锁线程安全）
    std::atomic<int> counter{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 10; i++) {
        threads.emplace_back([&counter]() {
            for (int j = 0; j < 10000; j++) {
                counter.fetch_add(1, std::memory_order_relaxed);  // 原子自增
            }
        });
    }
    for (auto& t : threads) t.join();
    std::cout << "  atomic 计数: " << counter.load() << " (期望 100000)" << std::endl;

    // 常用操作
    std::atomic<int> a{10};
    a.store(20);              // 原子写
    int v = a.load();         // 原子读
    int old = a.exchange(30); // 原子交换，返回旧值
    int expected = 30;
    bool ok = a.compare_exchange_strong(expected, 40);  // CAS：如果 a==30 则 a=40
    std::cout << "  CAS 结果: " << (ok ? "成功" : "失败")
              << ", a = " << a.load() << std::endl;
}
```

#### 6 种 memory order（核心难点）

| Memory Order | 语义 | 保证 | 适用场景 |
|--------------|------|------|----------|
| `memory_order_relaxed` | 无排序约束 | 仅原子性 | 计数器（不需同步） |
| `memory_order_acquire` | 读操作 | 后续读写不能重排到此操作之前 | 读取同步标志 |
| `memory_order_release` | 写操作 | 之前的读写不能重排到此操作之后 | 写入同步标志 |
| `memory_order_acq_rel` | 读写操作 | acquire + release | CAS 操作 |
| `memory_order_seq_cst` | 全局顺序一致 | acquire + release + 全局顺序 | 默认（最强） |
| `memory_order_consume` | 数据依赖排序 | 类似 acquire（已不推荐） | 几乎不用 |

```cpp
// concurrency_basics.cpp（续）—— memory order 与同步

// 经典模式：release-acquire 配对实现同步
int payload = 0;
std::atomic<bool> ready{false};

void producer_release() {
    payload = 42;  // 非原子写
    // release：确保 payload 的写在此操作之前完成
    ready.store(true, std::memory_order_release);
}

void consumer_acquire() {
    // acquire：确保后续对 payload 的读在此操作之后
    while (!ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    // 这里读到的 payload 一定是 42（release-acquire 同步保证）
    std::cout << "  payload = " << payload << std::endl;
}

void demo_memory_order() {
    std::cout << "\n=== memory order（release-acquire 同步）===" << std::endl;
    payload = 0;
    ready.store(false);

    std::thread c(consumer_acquire);
    std::thread p(producer_release);
    p.join();
    c.join();
}
```

> 💡 **面试要点**：`release` 和 `acquire` 配对可以建立**happens-before**关系——release 之前的所有写操作对 acquire 之后的读操作可见。`seq_cst`（默认）最强但最慢，`relaxed` 最快但无同步。大多数场景用默认 `seq_cst`，性能敏感的计数器用 `relaxed`，同步标志用 `release`/`acquire`。

### 学习任务 5：Thread-safe Singleton（20 分钟）

这是面试高频手撕题，考察双重检查锁、`call_once`、Meyers' Singleton 三种实现：

```cpp
// concurrency_basics.cpp（续）—— Thread-safe Singleton

// 方式 1：Meyers' Singleton（C++11 起线程安全，推荐）
class Singleton1 {
public:
    static Singleton1& instance() {
        static Singleton1 inst;  // C++11 保证局部 static 初始化线程安全
        return inst;
    }
    Singleton1(const Singleton1&) = delete;
    Singleton1& operator=(const Singleton1&) = delete;
private:
    Singleton1() { std::cout << "  Singleton1 构造" << std::endl; }
    ~Singleton1() = default;
};

// 方式 2：双重检查锁（DCLP，C++11 前，现已不推荐）
class Singleton2 {
    static std::atomic<Singleton2*> instance_;
    static std::mutex mtx_;
public:
    static Singleton2* instance() {
        Singleton2* p = instance_.load(std::memory_order_acquire);
        if (!p) {
            std::lock_guard<std::mutex> lock(mtx_);
            p = instance_.load(std::memory_order_relaxed);
            if (!p) {
                p = new Singleton2();
                instance_.store(p, std::memory_order_release);
            }
        }
        return p;
    }
private:
    Singleton2() { std::cout << "  Singleton2 构造" << std::endl; }
};
std::atomic<Singleton2*> Singleton2::instance_{nullptr};
std::mutex Singleton2::mtx_;

// 方式 3：std::call_once
class Singleton3 {
    static std::once_flag flag_;
    static Singleton3* instance_;
public:
    static Singleton3* instance() {
        std::call_once(flag_, []() { instance_ = new Singleton3(); });
        return instance_;
    }
private:
    Singleton3() { std::cout << "  Singleton3 构造" << std::endl; }
};
std::once_flag Singleton3::flag_;
Singleton3* Singleton3::instance_ = nullptr;

void demo_singleton() {
    std::cout << "\n=== Thread-safe Singleton ===" << std::endl;
    std::vector<std::thread> threads;
    for (int i = 0; i < 3; i++) {
        threads.emplace_back([]() { Singleton1::instance(); });
    }
    for (auto& t : threads) t.join();
}
```

| 实现 | 线程安全 | 推荐度 | 备注 |
|------|----------|--------|------|
| Meyers' Singleton | ✅（C++11 保证） | ⭐ 推荐 | 最简洁，编译器保证线程安全 |
| 双重检查锁 (DCLP) | ✅（需 atomic） | 不推荐 | C++11 前容易写错 |
| `std::call_once` | ✅ | ⭐ 推荐 | 明确、可读 |
| 简单加锁 | ✅ | 可用 | 每次调用都加锁，性能差 |

### 面试题积累（今日 6 道）

**Q1：`std::thread` 的 `join` 和 `detach` 有什么区别？**
> 答：`join` 等待线程完成并回收资源（阻塞调用线程）；`detach` 将线程分离到后台运行（不等待，线程独立执行）。`std::thread` 对象必须在 `join` 或 `detach` 之前处理，否则析构时调用 `std::terminate`。通常优先 `join`；`detach` 需确保线程不访问已销毁的局部变量。

**Q2：`lock_guard` 和 `unique_lock` 有什么区别？**
> 答：`lock_guard` 在构造时加锁、析构时解锁，不可中途解锁，不可移动，开销最小；`unique_lock` 更灵活——支持延迟加锁（`defer_lock`）、中途解锁/重新加锁、可以移动、可以配合 `condition_variable`。简单临界区用 `lock_guard`，需要灵活控制或条件变量用 `unique_lock`。

**Q3：什么是虚假唤醒？如何处理？**
> 答：虚假唤醒是指 `condition_variable::wait` 在没有 `notify` 的情况下也可能返回。处理方法：① 用带谓词的 `wait(lock, predicate)`，内部自动循环检查；② 手动 `while (!condition) cv.wait(lock)`。永远不要用 `if` 判断条件后直接 `wait`——虚假唤醒会导致错误。

**Q4：`std::atomic` 的 6 种 memory order 是什么？默认是哪个？**
> 答：① `relaxed`：只保证原子性，无排序约束；② `acquire`：后续读写不能重排到此之前；③ `release`：之前读写不能重排到此之后；④ `acq_rel`：acquire + release；⑤ `seq_cst`：全局顺序一致（最强，默认）；⑥ `consume`：数据依赖排序（已不推荐）。默认是 `seq_cst`。`release`/`acquire` 配对可以建立 happens-before 关系；计数器用 `relaxed`；同步标志用 `release`/`acquire`。

**Q5：如何实现线程安全的单例模式？**
> 答：推荐 Meyers' Singleton——局部 `static` 变量，C++11 标准保证其初始化是线程安全的（编译器内部用 `call_once` 或等价机制）。代码简洁：`static T& instance() { static T inst; return inst; }`。其他方案：`std::call_once`（明确可控）或双重检查锁（DCLP，C++11 前使用，需 atomic 保证正确性，现已不推荐）。

**Q6：如何预防死锁？**
> 答：四种策略——① 固定加锁顺序：所有线程按相同顺序获取锁；② `std::scoped_lock`：原子地锁多个 mutex，避免循环等待；③ `std::try_lock` + 回退：尝试加锁，失败则释放已有锁重试；④ 避免嵌套锁：持有锁时不调用未知代码（可能间接获取其他锁）。最常用的是 `scoped_lock` 和固定顺序。

### 今日检查清单

- [ ] 能创建 `std::thread` 并正确 `join`/`detach`
- [ ] 知道线程传引用需要 `std::ref`
- [ ] 能用 `lock_guard` 管理临界区
- [ ] 能说出 `lock_guard` vs `unique_lock` 的区别
- [ ] 能用 `condition_variable` 实现生产者-消费者
- [ ] 知道虚假唤醒及其处理方法
- [ ] 能说出 6 种 memory order 的语义
- [ ] 能用 `release`/`acquire` 配对实现线程间同步
- [ ] 能实现线程安全的单例模式（Meyers' Singleton）
- [ ] 能说出 3 种以上死锁预防策略
- [ ] `concurrency_basics.cpp` 编译运行通过

#### 明日预告

Day 7 将做**全周总结与高频题复盘**——汇总 C++11~23 关键新特性、整理 40+ 道高频面试题答案、完成个人面试 cheat sheet。今天学完并发，明天把 7 天的知识串联起来，查漏补缺。建议今晚先回顾本周前 6 天的检查清单，标记还不熟悉的知识点。

---
