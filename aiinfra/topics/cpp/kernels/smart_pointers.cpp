// smart_pointers.cpp —— RAII 与智能指针全演示
// Day 2: RAII 与智能指针
// 编译: g++ -std=c++20 -o smart_pointers smart_pointers.cpp && ./smart_pointers

#include <iostream>
#include <memory>
#include <string>
#include <utility>

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
    MyArray(const MyArray&) = delete;
    MyArray& operator=(const MyArray&) = delete;
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
    }
    std::cout << "  arr 已自动释放" << std::endl;
}

void demo_unique_ptr() {
    std::cout << "\n=== unique_ptr 演示 ===" << std::endl;

    std::unique_ptr<int> p1 = std::make_unique<int>(42);
    std::cout << "  *p1 = " << *p1 << std::endl;

    std::unique_ptr<int> p2 = std::move(p1);
    std::cout << "  移动后: p1 = " << (p1 ? "非空" : "空")
              << ", *p2 = " << *p2 << std::endl;

    auto file_deleter = [](FILE* f) {
        if (f) { std::cout << "  fclose() 被调用" << std::endl; fclose(f); }
    };
    {
        std::unique_ptr<FILE, decltype(file_deleter)> fp(fopen("/tmp/test.txt", "w"), file_deleter);
        if (fp) fprintf(fp.get(), "hello");
    }

    std::unique_ptr<int[]> arr = std::unique_ptr<int[]>(new int[5]);
    arr[0] = 1;
    std::cout << "  arr[0] = " << arr[0] << std::endl;
}

template <typename T>
class MyUniquePtr {
    T* ptr_;
public:
    explicit MyUniquePtr(T* p = nullptr) : ptr_(p) {}
    MyUniquePtr(const MyUniquePtr&) = delete;
    MyUniquePtr& operator=(const MyUniquePtr&) = delete;
    MyUniquePtr(MyUniquePtr&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    MyUniquePtr& operator=(MyUniquePtr&& other) noexcept {
        if (this != &other) {
            delete ptr_;
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    ~MyUniquePtr() { delete ptr_; }
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    T* get() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
};

void demo_handwritten_unique_ptr() {
    std::cout << "\n=== 手写 unique_ptr ===" << std::endl;
    MyUniquePtr<int> p1(new int(100));
    std::cout << "  *p1 = " << *p1 << std::endl;
    MyUniquePtr<int> p2 = std::move(p1);
    std::cout << "  移动后: p1 = " << (p1 ? "非空" : "空")
              << ", *p2 = " << *p2 << std::endl;
}

void demo_shared_ptr() {
    std::cout << "\n=== shared_ptr 演示 ===" << std::endl;
    auto p1 = std::make_shared<int>(42);
    auto p2 = p1;
    std::cout << "  *p1 = " << *p1 << ", use_count = " << p1.use_count() << std::endl;
    {
        auto p3 = p1;
        std::cout << "  进入内部块: use_count = " << p1.use_count() << std::endl;
    }
    std::cout << "  离开内部块: use_count = " << p1.use_count() << std::endl;
}

struct BadNode {
    std::shared_ptr<BadNode> next;
    ~BadNode() { std::cout << "  ~BadNode()" << std::endl; }
};

void demo_circular_reference() {
    std::cout << "\n=== 循环引用问题 ===" << std::endl;
    auto a = std::make_shared<BadNode>();
    auto b = std::make_shared<BadNode>();
    a->next = b;
    b->next = a;
    std::cout << "  a.use_count = " << a.use_count() << std::endl;
    std::cout << "  b.use_count = " << b.use_count() << std::endl;
    std::cout << "  (a, b 离开作用域后不会释放 — 内存泄漏)" << std::endl;
}

struct GoodNode {
    std::shared_ptr<GoodNode> next;
    std::weak_ptr<GoodNode> prev;
    ~GoodNode() { std::cout << "  ~GoodNode()" << std::endl; }
};

void demo_weak_ptr_solution() {
    std::cout << "\n=== weak_ptr 解决循环引用 ===" << std::endl;
    auto a = std::make_shared<GoodNode>();
    auto b = std::make_shared<GoodNode>();
    a->next = b;
    b->prev = a;
    std::cout << "  a.use_count = " << a.use_count() << std::endl;
    std::cout << "  b.use_count = " << b.use_count() << std::endl;
    if (auto sp = b->prev.lock()) {
        std::cout << "  b->prev 指向的对象存活: use_count = " << sp.use_count() << std::endl;
    }
    std::cout << "  (a, b 离开作用域后正确释放)" << std::endl;
}

int main() {
    demo_raii();
    demo_unique_ptr();
    demo_handwritten_unique_ptr();
    demo_shared_ptr();
    demo_circular_reference();
    demo_weak_ptr_solution();
    return 0;
}
