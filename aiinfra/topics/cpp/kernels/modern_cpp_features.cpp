// modern_cpp_features.cpp —— C++11~23 新特性演示
// Day 7: 现代 C++ 新特性与高频题复盘
// 编译: g++ -std=c++20 -o modern modern_cpp_features.cpp && ./modern
// 注意: std::format 需 GCC 13+, 部分特性可能需要更新编译器

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <string_view>
#include <algorithm>
#include <numeric>
#include <ranges>
#include <span>
#include <compare>

using namespace std::string_literals;

#if __has_include(<format>)
#include <format>
#define HAS_FORMAT 1
#else
#define HAS_FORMAT 0
#endif

void demo_cpp11() {
    std::cout << "=== C++11 ===" << std::endl;

    auto x = 42;
    auto y = 3.14;
    auto s = "hello"s;

    auto add = [](int a, int b) { return a + b; };
    int captured = 10;
    auto add_capture = [captured](int a) { return a + captured; };
    auto add_ref = [&captured](int a) { return a + captured; };

    int* p = nullptr;

    std::vector<int> v = {1, 2, 3, 4, 5};
    std::cout << "  range-for: ";
    for (const auto& elem : v) std::cout << elem << " ";
    std::cout << std::endl;

    std::cout << "  add(1,2) = " << add(1, 2) << std::endl;
    std::cout << "  add_capture(5) = " << add_capture(5) << std::endl;
}

void demo_cpp14() {
    std::cout << "\n=== C++14 ===" << std::endl;

    auto generic = [](auto a, auto b) { return a + b; };
    std::cout << "  generic(1, 2) = " << generic(1, 2) << std::endl;
    std::cout << "  generic(1.5, 2.5) = " << generic(1.5, 2.5) << std::endl;

    auto ptr = std::make_unique<int>(42);
    std::cout << "  *ptr = " << *ptr << std::endl;

    long big_num = 1'000'000;
    std::cout << "  1'000'000 = " << big_num << std::endl;
}

void demo_cpp17() {
    std::cout << "\n=== C++17 ===" << std::endl;

    std::pair p = {1, "hello"s};
    auto [num, str] = p;
    std::cout << "  结构化绑定: " << num << ", " << str << std::endl;

    auto find_even = [](const std::vector<int>& v) -> std::optional<int> {
        for (auto x : v) if (x % 2 == 0) return x;
        return std::nullopt;
    };
    auto opt = find_even({1, 3, 5, 8});
    if (opt) std::cout << "  找到偶数: " << *opt << std::endl;

    std::variant<int, double, std::string> var = "hello";
    std::visit([](auto&& v) { std::cout << "  variant: " << v << std::endl; }, var);

    std::string long_str = "a very long string";
    std::string_view sv = std::string_view(long_str).substr(2, 5);
    std::cout << "  string_view: " << sv << std::endl;

    std::pair pr = {1, 2.0};
    std::cout << "  CTAD pair: (" << pr.first << ", " << pr.second << ")" << std::endl;
}

void demo_cpp20() {
    std::cout << "\n=== C++20 ===" << std::endl;

    std::vector<int> nums = {1, 2, 3, 4, 5, 6};
    auto result = nums
        | std::views::filter([](int n) { return n % 2 == 0; })
        | std::views::transform([](int n) { return n * n; });
    std::cout << "  Ranges: ";
    for (int n : result) std::cout << n << " ";
    std::cout << std::endl;

    auto cmp = (1 <=> 2);
    std::cout << "  1 <=> 2: " << (cmp < 0 ? "less" : cmp > 0 ? "greater" : "equal") << std::endl;

#if HAS_FORMAT
    std::string formatted = std::format("  format: x={}, y={:.2f}", 42, 3.14159);
    std::cout << formatted << std::endl;
#else
    std::cout << "  (std::format 需 GCC 13+)" << std::endl;
#endif

    int arr[] = {10, 20, 30};
    std::span<int> sp = arr;
    std::cout << "  span size: " << sp.size() << std::endl;
}

void demo_cpp23() {
    std::cout << "\n=== C++23 ===" << std::endl;
    std::cout << "  (C++23 特性需较新编译器支持)" << std::endl;
    std::cout << "  - std::expected (错误处理)" << std::endl;
    std::cout << "  - std::print (格式化输出)" << std::endl;
    std::cout << "  - Deducing this (显式 this 参数)" << std::endl;
    std::cout << "  - std::mdspan (多维数组视图)" << std::endl;
}

int main() {
    demo_cpp11();
    demo_cpp14();
    demo_cpp17();
    demo_cpp20();
    demo_cpp23();
    return 0;
}
