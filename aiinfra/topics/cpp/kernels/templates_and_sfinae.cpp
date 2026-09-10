// templates_and_sfinae.cpp —— 模板与泛型编程
// Day 4: 模板与泛型编程
// 编译: g++ -std=c++20 -o templates templates_and_sfinae.cpp && ./templates

#include <iostream>
#include <concepts>
#include <type_traits>
#include <string>
#include <vector>
#include <utility>

template <typename T>
T my_max(const T& a, const T& b) {
    return a < b ? b : a;
}

void demo_function_template() {
    std::cout << "=== 函数模板 ===" << std::endl;
    std::cout << "  max(3, 5) = " << my_max(3, 5) << std::endl;
    std::cout << "  max(1.5, 2.5) = " << my_max(1.5, 2.5) << std::endl;
    std::cout << "  max<double>(3, 2.5) = " << my_max<double>(3, 2.5) << std::endl;
}

template <typename T, size_t N>
class FixedArray {
    T data_[N];
public:
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    constexpr size_t size() const { return N; }
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
    for (size_t i = 0; i < arr.size(); i++) arr[i] = static_cast<int>(i * 10);
    std::cout << "  " << arr << std::endl;
    std::cout << "  size = " << arr.size() << std::endl;
}

template <typename T>
class TypeName {
public:
    static std::string get() { return "unknown"; }
};

template <>
class TypeName<int> {
public:
    static std::string get() { return "int"; }
};

template <>
class TypeName<double> {
public:
    static std::string get() { return "double"; }
};

template <typename T>
class TypeName<T*> {
public:
    static std::string get() { return "pointer to " + TypeName<T>::get(); }
};

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

void print() {}

template <typename T, typename... Args>
void print(T first, Args... rest) {
    std::cout << "  " << first << std::endl;
    print(rest...);
}

template <typename... Args>
constexpr size_t count_args() {
    return sizeof...(Args);
}

void demo_variadic() {
    std::cout << "\n=== 变参模板 ===" << std::endl;
    print(1, "hello", 3.14, 'a');
    std::cout << "  参数个数: " << count_args<int, double, char>() << std::endl;
}

template <typename... Args>
auto sum(Args... args) {
    return (args + ...);
}

template <typename... Args>
void print_fold(Args... args) {
    ((std::cout << "  " << args << std::endl), ...);
}

void demo_fold_expression() {
    std::cout << "\n=== C++17 折叠表达式 ===" << std::endl;
    std::cout << "  sum(1,2,3,4) = " << sum(1, 2, 3, 4) << std::endl;
    print_fold(1, "hello", 3.14);
}

template <typename T>
class has_size {
    template <typename U>
    static auto test(int) -> decltype(std::declval<U>().size(), std::true_type{});
    template <typename U>
    static auto test(...) -> std::false_type;
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

// C++17 detection idiom：void_t 版 has_size（四行）
template <typename, typename = void>
struct detect_size : std::false_type {};

template <typename T>
struct detect_size<T, std::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

// enable_if 重载选择：条件放进额外非类型模板参数的类型里
// （typename = enable_if_t<cond> 形态的两个重载会因默认模板实参不参与签名而重定义冲突）
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

void demo_sfinae() {
    std::cout << "\n=== SFINAE ===" << std::endl;
    std::cout << "  int 有 size(): " << has_size<int>::value << std::endl;        // 0
    std::cout << "  vector 有 size(): " << has_size<std::vector<int>>::value << std::endl; // 1
    std::cout << "  string 有 size(): " << has_size<std::string>::value << std::endl;     // 1
    std::cout << "  detect_size<int>: " << detect_size<int>::value << std::endl;           // 0（void_t 版）
    std::cout << "  detect_size<vector<int>>: " << detect_size<std::vector<int>>::value << std::endl; // 1
    std::cout << "  classify(42): " << classify(42) << std::endl;   // → 整型重载
    std::cout << "  classify(3.14): " << classify(3.14) << std::endl; // → 浮点重载
}

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

template <typename T>
concept Numeric = std::is_integral_v<T> || std::is_floating_point_v<T>;

template <typename T>
concept HasSize = requires(T t) {
    { t.size() } -> std::convertible_to<size_t>;
};

// subsumption：SizedNumeric 蕴涵 HasSize → 更特化的重载优先入选
template <typename T>
concept SizedNumeric = HasSize<T> && std::integral<typename T::value_type>;

template <Numeric T>
T add(T a, T b) { return a + b; }

template <typename T>
    requires HasSize<T>
size_t get_size(const T& val) { return val.size(); }

template <HasSize T>
void dump(const T& v) { std::cout << "  一般容器" << std::endl; }

template <SizedNumeric T>
void dump(const T& v) { std::cout << "  整型元素容器（subsumption 优先）" << std::endl; }

void demo_concepts() {
    std::cout << "\n=== C++20 Concepts ===" << std::endl;
    std::cout << "  add(1, 2) = " << add(1, 2) << std::endl;
    std::cout << "  add(1.5, 2.5) = " << add(1.5, 2.5) << std::endl;

    std::vector<int> v = {1, 2, 3};
    std::cout << "  get_size(vector) = " << get_size(v) << std::endl;

    dump(std::vector<std::string>{"a"});  // → 一般容器
    dump(v);                              // → 整型元素容器（SizedNumeric 更特化）
}

int main() {
    demo_function_template();
    demo_class_template();
    demo_specialization();
    demo_variadic();
    demo_fold_expression();
    demo_sfinae();
    demo_if_constexpr();
    demo_concepts();
    return 0;
}
