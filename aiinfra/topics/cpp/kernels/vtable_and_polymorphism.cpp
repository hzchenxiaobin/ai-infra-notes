// vtable_and_polymorphism.cpp —— 虚函数表与多态底层
// Day 5: 面向对象与多态底层
// 编译: g++ -std=c++20 -o vtable vtable_and_polymorphism.cpp && ./vtable

#include <iostream>
#include <string>
#include <memory>
#include <algorithm>

class Animal {
public:
    virtual ~Animal() = default;
    virtual void speak() const {
        std::cout << "  Animal::speak()" << std::endl;
    }
    virtual int legs() const { return 0; }
    void eat() {
        std::cout << "  Animal::eat()" << std::endl;
    }
protected:
    int age_ = 0;
};

class Dog : public Animal {
public:
    void speak() const override {
        std::cout << "  Dog::speak() — Woof!" << std::endl;
    }
    int legs() const override { return 4; }
    void fetch() {
        std::cout << "  Dog::fetch()" << std::endl;
    }
private:
    std::string breed_ = "unknown";
};

void demo_virtual_dispatch() {
    std::cout << "=== 虚函数动态绑定 ===" << std::endl;
    Dog dog;
    Animal* p = &dog;

    p->speak();
    p->eat();
    std::cout << "  legs = " << p->legs() << std::endl;
}

class BadBase {
public:
    ~BadBase() { std::cout << "  ~BadBase()" << std::endl; }
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
    virtual ~GoodBase() { std::cout << "  ~GoodBase()" << std::endl; }
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
    delete bad;

    std::cout << "\n  [GoodBase — 虚析构]" << std::endl;
    GoodBase* good = new GoodDerived();
    delete good;
}

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

    std::cout << "  sizeof(Drawable) = " << sizeof(Drawable) << std::endl;
    std::cout << "  sizeof(Clickable) = " << sizeof(Clickable) << std::endl;
    std::cout << "  sizeof(Button) = " << sizeof(Button) << std::endl;

    Button btn;
    Drawable* d = &btn;
    Clickable* c = &btn;

    std::cout << "  Button 地址:  " << &btn << std::endl;
    std::cout << "  as Drawable: " << d << std::endl;
    std::cout << "  as Clickable:" << c << std::endl;

    d->draw();
    c->onClick();
}

class ResourceManager {
    int* data_;
    size_t size_;
public:
    explicit ResourceManager(size_t n) : data_(new int[n]()), size_(n) {}
    ~ResourceManager() { delete[] data_; }

    ResourceManager(const ResourceManager& other)
        : data_(new int[other.size_]), size_(other.size_) {
        std::copy(other.data_, other.data_ + size_, data_);
    }

    ResourceManager& operator=(const ResourceManager& other) {
        if (this != &other) {
            int* new_data = new int[other.size_];
            std::copy(other.data_, other.data_ + other.size_, new_data);
            delete[] data_;
            data_ = new_data;
            size_ = other.size_;
        }
        return *this;
    }

    ResourceManager(ResourceManager&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

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

class ModernResource {
    std::unique_ptr<int[]> data_;
    size_t size_;
public:
    explicit ModernResource(size_t n) : data_(std::make_unique<int[]>(n)), size_(n) {}
};

void demo_rule_of_five() {
    std::cout << "\n=== Rule of 5 vs Rule of 0 ===" << std::endl;
    ResourceManager r1(10);
    ResourceManager r2 = std::move(r1);
    std::cout << "  Rule of 5: 手动管理 5 个特殊成员函数" << std::endl;

    ModernResource m1(10);
    ModernResource m2 = std::move(m1);
    std::cout << "  Rule of 0: unique_ptr 自动管理" << std::endl;
}

void process_by_value(Animal a) {
    a.speak();
}

void process_by_ref(const Animal& a) {
    a.speak();
}

void demo_object_slicing() {
    std::cout << "\n=== 对象切片 ===" << std::endl;
    Dog dog;

    std::cout << "  按值传递（切片）:" << std::endl;
    process_by_value(dog);

    std::cout << "  按引用传递（不切片）:" << std::endl;
    process_by_ref(dog);
}

int main() {
    demo_virtual_dispatch();
    demo_virtual_destructor();
    demo_multiple_inheritance();
    demo_rule_of_five();
    demo_object_slicing();
    return 0;
}
