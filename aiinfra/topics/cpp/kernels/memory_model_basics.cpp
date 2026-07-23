// memory_model_basics.cpp —— C++ 内存区域与值类别演示
// Day 1: 内存模型与基础语义
// 编译: g++ -std=c++20 -o memory_model_basics memory_model_basics.cpp && ./memory_model_basics

#include <iostream>
#include <string>
#include <utility>

int g_global = 42;
static int s_static = 100;
const char* g_str = "hello";

void demo_memory_regions() {
    int stack_var = 1;
    static int func_static = 2;
    int* heap_ptr = new int(3);

    std::cout << "=== 内存区域地址 ===" << std::endl;
    std::cout << "栈变量 stack_var:       " << &stack_var << std::endl;
    std::cout << "堆指针 heap_ptr 指向:   " << heap_ptr << std::endl;
    std::cout << "堆指针 heap_ptr 本身:   " << &heap_ptr << " (在栈上)" << std::endl;
    std::cout << "全局变量 g_global:      " << &g_global << std::endl;
    std::cout << "静态变量 s_static:      " << &s_static << std::endl;
    std::cout << "函数静态 func_static:   " << &func_static << std::endl;
    std::cout << "字符串字面量 g_str 指向:" << (const void*)g_str << std::endl;

    delete heap_ptr;
}

void demo_const_semantics() {
    std::cout << "\n=== const 语义 ===" << std::endl;
    int a = 10, b = 20;

    const int* p1 = &a;       // 指向 const int 的指针
    int const* p2 = &a;       // 同上
    int* const p3 = &a;       // const 指针指向 int
    const int* const p4 = &a; // 都不可变

    std::cout << "  const int* p1: 可改指针, 不可改数据" << std::endl;
    p1 = &b;                  // OK: 指针可变
    // *p1 = 5;               // 编译错误: 数据不可变

    std::cout << "  int* const p3: 可改数据, 不可改指针" << std::endl;
    *p3 = 5;                  // OK: 数据可变
    // p3 = &b;               // 编译错误: 指针不可变

    constexpr int z = 42;
    std::cout << "  constexpr int z = " << z << std::endl;
}

void demo_value_categories() {
    std::cout << "\n=== 值类别 ===" << std::endl;
    int a = 10;
    int b = 20;
    int c = a + b;

    std::cout << "  &a = " << &a << " (lvalue, 可取地址)" << std::endl;
    std::cout << "  a + b 是 prvalue (不可取地址)" << std::endl;

    int& lref = a;
    const int& cref = 42;
    int&& rref = std::move(a);

    std::cout << "  lref = " << lref << std::endl;
    std::cout << "  cref = " << cref << std::endl;
    std::cout << "  rref = " << rref << std::endl;
}

int main() {
    demo_memory_regions();
    demo_const_semantics();
    demo_value_categories();
    return 0;
}
