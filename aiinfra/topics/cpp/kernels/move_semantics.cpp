// move_semantics.cpp —— 移动语义与完美转发
// Day 3: 移动语义与完美转发
// 编译: g++ -std=c++20 -o move_semantics move_semantics.cpp && ./move_semantics

#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <type_traits>

template <typename T>
constexpr typename std::remove_reference<T>::type&& my_move(T&& arg) noexcept {
    return static_cast<typename std::remove_reference<T>::type&&>(arg);
}

void demo_move_essence() {
    std::cout << "=== std::move 的本质 ===" << std::endl;
    std::string s = "hello world";
    std::cout << "  移动前: s = \"" << s << "\"" << std::endl;
    std::string s2 = std::move(s);
    std::cout << "  移动后: s = \"" << s << "\" (有效但未指定状态)" << std::endl;
    std::cout << "         s2 = \"" << s2 << "\"" << std::endl;
}

class StringVector {
    int* data_;
    size_t size_;
    size_t cap_;
public:
    explicit StringVector(size_t n) : data_(new int[n]()), size_(0), cap_(n) {}

    StringVector(const StringVector& other)
        : data_(new int[other.cap_]), size_(other.size_), cap_(other.cap_) {
        std::copy(other.data_, other.data_ + size_, data_);
        std::cout << "  拷贝构造（深拷贝 " << size_ << " 元素）" << std::endl;
    }

    StringVector(StringVector&& other) noexcept
        : data_(other.data_), size_(other.size_), cap_(other.cap_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.cap_ = 0;
        std::cout << "  移动构造（O(1) 偷取）" << std::endl;
    }

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

    StringVector& operator=(StringVector&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
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

    StringVector v2 = std::move(v1);
    std::cout << "  v2.size = " << v2.size() << std::endl;

    StringVector v3(10);
    v3 = std::move(v2);
    std::cout << "  v3.size = " << v3.size() << std::endl;
}

void demo_noexcept_matters() {
    std::cout << "\n=== noexcept 对 vector 扩容的影响 ===" << std::endl;
    std::vector<StringVector> vec;
    for (int i = 0; i < 10; i++) {
        vec.emplace_back(1000);
    }
    std::cout << "  vector 扩容时使用了移动构造（noexcept）" << std::endl;
}

void target(int& x)       { std::cout << "  左值引用" << std::endl; }
void target(int&& x)      { std::cout << "  右值引用" << std::endl; }

template <typename T>
void bad_forward(T&& param) {
    target(param);
}

template <typename T>
void perfect_forward(T&& param) {
    target(std::forward<T>(param));
}

void demo_perfect_forward() {
    std::cout << "\n=== 完美转发 ===" << std::endl;
    int x = 42;

    std::cout << "  bad_forward(x): ";
    bad_forward(x);
    std::cout << "  bad_forward(42): ";
    bad_forward(42);

    std::cout << "  perfect_forward(x): ";
    perfect_forward(x);
    std::cout << "  perfect_forward(42): ";
    perfect_forward(42);
}

StringVector create_vector() {
    StringVector v(100);
    for (int i = 0; i < 100; i++) v.push(i);
    return v;
}

StringVector create_vector_rvo() {
    return StringVector(100);
}

void demo_rvo() {
    std::cout << "\n=== RVO/NRVO ===" << std::endl;
    auto v1 = create_vector();
    std::cout << "  v1.size = " << v1.size() << std::endl;
    auto v2 = create_vector_rvo();
    std::cout << "  v2.size = " << v2.size() << std::endl;
}

int main() {
    demo_move_essence();
    demo_move_constructor();
    demo_noexcept_matters();
    demo_perfect_forward();
    demo_rvo();
    return 0;
}
