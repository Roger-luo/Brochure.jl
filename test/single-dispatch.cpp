#include <iostream>

struct A {};
struct B : public A {};

class Foo {
public:
    virtual void wooo(A *a, A *b) { std::cout << "A/A" << std::endl; };
    virtual void wooo(A *a, B *b) { std::cout << "A/B" << std::endl; };
};

void CallMyFn(Foo *p, A *arg1, A *arg2) {
    p->wooo(arg1, arg2);
}

int main(int argc, char const *argv[]) {
    Foo *f = new Foo();
    A *a = new A(); B *b = new B();
    CallMyFn(f, a, b);
    return 0;
}
