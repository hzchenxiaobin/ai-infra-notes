// concurrency_basics.cpp —— 并发编程基础
// Day 6: 并发编程基础
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
#include <memory>

void demo_thread_basics() {
    std::cout << "=== std::thread 基础 ===" << std::endl;

    auto func = [](int id) {
        std::cout << "  线程 " << id << " 运行中" << std::endl;
    };
    std::thread t1(func, 1);
    t1.join();

    std::thread t2([]() {
        std::cout << "  lambda 线程运行中" << std::endl;
    });
    t2.join();

    std::thread t3([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    });
    t3.detach();

    std::cout << "  硬件并发线程数: " << std::thread::hardware_concurrency() << std::endl;
}

void demo_thread_params() {
    std::cout << "\n=== 线程传参 ===" << std::endl;
    int x = 0;
    std::thread t([](int& ref) { ref = 42; }, std::ref(x));
    t.join();
    std::cout << "  x = " << x << std::endl;

    auto safe = [](std::shared_ptr<int> sp) {
        std::cout << "  *sp = " << *sp << std::endl;
    };
    std::thread t2(safe, std::make_shared<int>(100));
    t2.join();
}

class ThreadSafeCounter {
    mutable std::mutex mtx_;
    int count_ = 0;
public:
    void increment() {
        std::lock_guard<std::mutex> lock(mtx_);
        ++count_;
    }
    int get() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return count_;
    }
    int get_and_reset() {
        std::unique_lock<std::mutex> lock(mtx_);
        int val = count_;
        count_ = 0;
        lock.unlock();
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
        cv_.notify_one();
    }
    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !queue_.empty() || done_; });
        if (queue_.empty() && done_) return false;
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }
    void set_done() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            done_ = true;
        }
        cv_.notify_all();
    }
};

void demo_producer_consumer() {
    std::cout << "\n=== 生产者-消费者 ===" << std::endl;
    ThreadSafeQueue<int> queue;

    auto producer = [&queue]() {
        for (int i = 0; i < 5; i++) {
            queue.push(i);
            std::cout << "  生产: " << i << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        queue.set_done();
    };

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

void demo_atomic() {
    std::cout << "\n=== std::atomic ===" << std::endl;
    std::atomic<int> counter{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 10; i++) {
        threads.emplace_back([&counter]() {
            for (int j = 0; j < 10000; j++) {
                counter.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& t : threads) t.join();
    std::cout << "  atomic 计数: " << counter.load() << " (期望 100000)" << std::endl;

    std::atomic<int> a{10};
    a.store(20);
    int v = a.load();
    int old = a.exchange(30);
    int expected = 30;
    bool ok = a.compare_exchange_strong(expected, 40);
    std::cout << "  CAS 结果: " << (ok ? "成功" : "失败")
              << ", a = " << a.load() << std::endl;
}

int payload = 0;
std::atomic<bool> ready{false};

void producer_release() {
    payload = 42;
    ready.store(true, std::memory_order_release);
}

void consumer_acquire() {
    while (!ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
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

class Singleton1 {
public:
    static Singleton1& instance() {
        static Singleton1 inst;
        return inst;
    }
    Singleton1(const Singleton1&) = delete;
    Singleton1& operator=(const Singleton1&) = delete;
private:
    Singleton1() { std::cout << "  Singleton1 构造" << std::endl; }
    ~Singleton1() = default;
};

void demo_singleton() {
    std::cout << "\n=== Thread-safe Singleton ===" << std::endl;
    std::vector<std::thread> threads;
    for (int i = 0; i < 3; i++) {
        threads.emplace_back([]() { Singleton1::instance(); });
    }
    for (auto& t : threads) t.join();
    std::cout << "  (Meyers' Singleton — 局部 static, C++11 保证线程安全)" << std::endl;
}

int main() {
    demo_thread_basics();
    demo_thread_params();
    demo_mutex();
    demo_producer_consumer();
    demo_atomic();
    demo_memory_order();
    demo_singleton();
    return 0;
}
