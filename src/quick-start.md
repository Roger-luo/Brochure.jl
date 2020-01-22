# 快速入门

尽管本书不会过多关注基础入门知识，但是为了让我们的读者能够站在同一起跑线上，我们在这一节快速过一遍Julia的各种语法。
如果你已经对Julia很熟练了，那么不要浪费时间请跳过本节。如果你对Julia还不了解，我依然**建议你完整阅读Julia官方文档的手册部分（Manual）**，
这部分我**删去了很多细节**，而**着重在相对深入的概念解释**和针对一些有学习过其它语言的人的**常见误区的解释**。如果你只是刚刚上手，
我依然非常推荐你阅读这一部分，它会帮助你避免很多坑。

## 变量（Variable）

Julia语言中最基础的元素是变量，简单的来说变量就是一个绑定了一个值（value）的名字（name），例如

```@repl
x = 1
```

这里 `x` 是一个变量，它的名字叫 `x`，它的值是一个数字 `1`。但是一个名字并不需要唯一的绑定这个值，

```@repl
x = "我是一个变量的值"
```

!!! note
    在 C/C++ 里，一个变量是唯一绑定一个类型的，并且对应内存地址。Julia的变量类似于 C++ 的引用（reference），
    但依然在语义上不完全一样。具体的一个体现就是Julia不能重载赋值运算符，因为由于变量只是绑定一个名字，所以不存在
    需要重载赋值运算的场景。单独使用 `=` 永远只表示给右边的值绑定左边的名字。

## 类型（Type）

类型是一种标记，它存在的目的是为了告诉编译器部分关于你代码的信息，或者简单来说，它是对变量的“分类”。Julia语言是一个
强类型（strong typed）语言，这意味着所有的变量/值都有一个类型。比如 `1` 的类型是 `Int`，你可以用 `typeof(1)`
来查看它。

当我们把一个值绑定给一个变量之后，这个变量也自动具有了对应的类型

```@repl
x = 1
typeof(x)
```

如果你给这个变量赋予了新的值，那么变量的类型也会改变

```@repl
x = "我是一个变量的值"
typeof(x)
```

而类型的类型是数据类型 `DataType`

```@repl
typeof(Int)
```

`DataType` 的类型还是 `DataType`

```@repl
typeof(DataType)
```

在Julia里所有的变量都有一个类型，你可以用 `typeof` 来获取它。

## 函数
Julia是一个构建在多重派发（multiple dispatch）基础上的语言，
它的函数实际上是一般的，有些翻译里也称为**范型**的函数（generic function）。
而我们每一个具体的函数定义都会为这个函数增加一种方法（method），方法
是一种具体执行这个范型函数的方式。所以很自热的，我们可以声明没有任何方法的
函数

```@repl func
function foo
end
```

而每一种不同的输入类型的组合都将创建一个新的方法，

```@repl func
function foo(x)
   x + 1
   return x
end
```

这里 `function` 和 `end` 关键字作为标识函数声明的符号。
有人也许会有疑问，为什么Julia里要使用 `end` 作为结尾，而不是 `{}` 或者像 Python一样使用缩进呢？当你阅读到
[表达式](#表达式-1)这一部分我想你就能够得到部分答案，当你掌握元编程之后，你便会体会到 `end` 的妙处。

或者，你也可以使用更简洁的写法。但是注意由于这个声明和上一个声明的类型标签相同（都是一个 `Any` 类型的函数参数）

```@repl func
foo(x) = x + 1
```

#### 类型标注

#### 带类型标注的方法声明

我们还可以对函数参数加上类型标注（type annotation），这将具体规定这个方法所适配的类型，例如

```@repl func
foo(x::Int) = 2x
```

或者是更多的变量，不同类型的组合等等

```@repl func
foo(x::Int, y::Float32) = y
foo(x::Float32, y::Int) = x + y
foo(x, y) = 2x + y
```

## 控制流

一般来说，控制流包含两种。一种是[循环表达式](#循环表达式-1)，一种是[条件表达式](#条件表达式-1)。

### 循环表达式

Julia中可以使用常见的两种方式定义循环，一种是 `for` 一种是 `while`。`for` 关键字后跟 `in` 关键字表示需要循环的区间

```@repl
function foo(start)
    for k in start:10
        @show k
    end
end

foo(2)
```

另外一种则是 `while` 

```@repl
function foo(start)
    k = start
    while k < 10
        @show k
        k += 1
    end
end

foo(2)
```

### 条件表达式
条件表达式即 `if ... else ... end`，例如

```@repl
function foo(x)
    if x > 0
        println("x 比 0 大")
    elseif x > -1
        println("x > -1")
    else
        print("x <= -1")
    end
end

foo(1)
foo(-0.5)
foo(-1)
```

## 自定义类型
一般来说我们会使用两种组合类型（composite type），这些类型由其它的数据类型组合而来。而在Julia里有两种，一种是成员的值在定义之后可变的
类型，另外一种是成员的值在定义之后不可变的类型。类型在Julia中主要有两个作用：一是用来派发方法（method），二是用来包装数据结构。

### 不可变类型
不可变类型使用 `struct` 关键字进行声明（也就是说我们默认一个类型是不可变的），格式如下

```julia
struct Cat
    name::String
end
```

### 可变类型

可变类型需要使用 `mutable` 关键字进行标注

```julia
mutable struct Cat
    name::String
end
```

!!! note
    在Julia里，从语义上讲（semantically）我们不区分这两种类型对应的内存分配方式。但是在优化层面，尽管Julia没有提供
    像C++一样的显式声明栈上分配的内存（stack allocated）的语义，但是通过对不可变等性质的推导，它依然可以和C++达到相近
    的内存分配大小 [^相关讨论]。

[^相关讨论]:
    相关讨论可以参见discourse上的帖子:
    - [Why mutable structs are allocated on the heap?](https://discourse.julialang.org/t/why-mutable-structs-are-allocated-on-the-heap/12992/25?u=roger-luo)
    - [Clarification about memory management of immutable and mutable struct](https://discourse.julialang.org/t/clarification-about-memory-management-of-immutable-and-mutable-struct/31064)

## 参数类型

在很多情况下，一些类型有着相近的含义和数据结构，但是它们需要派发的方法可能有所不同。这个时候我们往往会需要用到参数类型。
Julia中类型参数可以使用大括号 `{}` 声明。类型参数本身在编译时期也是有类型的，统一为 `TypeVar` 类型。例如下面这个文档
中也用到了的复数类型的例子。对于类型参数，我们可以使用 `<:` 来声明它的上界。

```julia
struct Complex{T <: Number}
    real::T
    imag::T
end
```

## 数组
数组是一种特别的类型，和其它语言不同的是，在Julia我们的数组是多维数组（multi-dimensional array）。所谓数组，
实际上它是一种对内存的抽象模型。在Julia里一个数组类型（`Array`）的实例代表了一块连续的内存

在Julia里它扮演了两种角色：**计算机意义上的数组**以及**数学意义上的多维张量**。
在很多机器学习框架中，也往往实现了多维数组或者张量（Tensor，例如PyTorch）。而这些
实现本质上只是一种对一块**连续内存**的查看方式。

一般来说多维数组的实际数据结构如下

```julia
struct MyArray{T, N}
    storage::Ptr{T}
    size::NTuple{N, Int}
    strides::NTuple{N, Int}
end
```

其中 `storage` 是一个指向某块内存的指针，这块内存上存了一些 `T` 类型构成的元素 ，`size` 记录了这个多维数组的大小，`strides` 则
用来表示每个维度之间间隔的元素个数，什么意思呢？例如下表是一个有20个浮点类型（双精度）的内存块，它可能存储了一个4x5矩阵的值，也有可能存储了一个2x5x2的三阶张量的值。

| 内存地址 | 0xf21010 | 0xf21018 | 0xf21020 | ``\cdots`` | 0xf210a0 | 0xf210a8 |
| ------  | -------- | -------- | -------- | -------- | -------- | -------- |
| 值      | 0.0      | 1.0      |2.0       | ``\cdots``| 18.0    | 19.0     |

向系统申请这个内存块，在不再使用之后删除所分配的内存并不需要知道对应张量的大小。甚至有可能几个元素数目不同但是总数相同的张量（比如4x4,2x2x2x2,1x16的不同大小张量）可以通过用不同的 `MyArray` 共享一块内存。但当我们需要完成张量的一些运算，例如对于矩阵，他们的乘积（matrix product），点积（dot product）等运算会需要使用维度的信息（各个维度的大小）并且这个时候我们将按照维度来访问不同位置的元素，这使得我们首先需要存储各个维度的大小 `size` ，但是这还不够，我们实际上在访问一块连续内存的时候实际上使用的是不同维度上的间隔，例如第一个维度上的间隔一般是0，第二个维度上的间隔是第一个维度的大小`size[0]`，依次类推，但也有可能由于是由某个较大的张量分割来的，并不满足上述间隔分配方式，所以我们有必要再用一个数组存储各个维度的间隔大小 `strides`。这样在访问某个角标ijk对应的内存地址时就可以用

```julia
(i - 1) * strides[0] + (j - 1) * strides[1] + (k - 1) * strides[2]
```

作为一个元素的真实内存地址了。当然Julia已经为我们做好了这些事情，在平时使用的时候我们不需要去在意它到底是怎么实现的。但是在我们后面的章节里，我们还会用到这些性质和定义。

### 数组的view
我们上面讲了多维数组其实是一种对一块内存的查看方式，而在对数组进行分块的时候Julia和MATLAB以及Python的numpy有一个很大的不同是**在Julia里对诸如`A[1, :]`之类的分块操作都是创建一个新的内存，然后将这部分内容拷贝到新的内存空间上去**，从而保证对 `A[1, :]` 的原地（in-place）操作不会影响到 `A` ，这能够保证你的程序是正确并且安全的。但是在很多情况下，我们是不需要拷贝的，这会让你的代码变慢，**很多从MATLAB或者Python转到Julia的用户都会犯这个错误**，而如果我们只是要查看这部分元素，则要借助 `view` ，它会产生一个 `SubArray`类型的实例，不会进行拷贝。如果你希望将某一整块代码中的分块操作都变成这样的行为，你可以用 `@views` 这个宏将这部分代码包裹起来。我们曾经在中文论坛上遇到过一个很典型例子 [^参见discourse]。

[^参见discourse]:
    你可以通过这个例子了解到这一技巧是如何被具体的使用的：[Julia内存分配随着循环的进行越来越大](https://discourse.juliacn.com/t/topic/2763)。

!!! note
    目前 `view` 暂时创建的 `SubArray` 依然会在堆（heap）上分配一个很小的内存。这会造成一定的额外开销，这是一个已知的问题[^issue14995]，在这个问题被彻底从编译器层面解决前，你可以使用 [UnsafeArrays.jl](https://github.com/oschulz/UnsafeArrays.jl) 来暂时解决这个问题。

[^issue14995]:
    参见 [JuliaLang/julia#14955](https://github.com/JuliaLang/julia/issues/14955)

### 数组的接口（interface）
Julia在标准库里提供了大量的一般性的方法，这意味着通过重载少量必要的接口，你就可以使用大部分的功能。数组也不例外，这也是为什么Julia里多维数组的生态如此丰富的原因之一。我们往往可以通过为特定结构的，使用特定硬件的数组定义单独的类型的和部分特化的方法（method），而剩余的部分则会通过这些一般性的方法自动被编译器推导出来。

## 表达式
在Julia里，任何一段程序都首先是一段表达式（expression）。所谓的表达式是一种数据结构，它存储了
一段程序的抽象语法树（abstract syntax tree）。我们可以用引用（quote）语法来获取一段程序的表达式

```@repl expr
ex = :(1 + 2 * b)
typeof(ex)
```

而 `ex` 的类型则是 `Expr` 类型。而表达式中暂时没有值的变量则会被解析为一个符号，它的类型是 `Symbol`，
你可以通过 `:` + `一段合法的变量名` 获得这样一个符号。它相当于一种特别的字符串。

```@repl expr
typeof(:a)
```

`Expr` 类型里的结构很简单，它包括一个符号类型的头，作为这段表达式的标签，以及一个一维的 `Any` 类型数组，
这个数组将存储这个表达式的子表达式。这里 `dump` 将会打印出一个Julia对象的内部结构。

```@repl expr
dump(Expr)
```

我们不妨看看上面的表达式是什么样子的

```@repl expr
dump(:(1 + 2 * b))
```

这是什么意思呢？首先我们说抽象语法数会把这样一段表达式存储成如下的形式

```@example
# using GraphRecipes # hide
# using Plots # hide
# default(size=(300, 300)) # hide
# pyplot() # hide
# code = :(1 + 2 * b) # hide
# plot(code, fontsize=12, shorten=0.01, axis_buffer=0.15, nodeshape=:rect) # hide
```

而上面的 `Expr` 就存储了这样一颗树，它的根结点是最低优先级的 `+` 函数的函数调用，

```@repl expr
ex.head
ex.args[1]
```

这个函数调用有两个输入，分别是

```@repl expr
ex.args[2]
ex.args[3]
```

而第二个输入则是另外一段表达式（的结果），这段表达式是`*` 函数的函数调用

```@repl expr
ex.args[3].head
ex.args[3].args[1]
ex.args[3].args[2]
ex.args[3].args[3]
```

于是最终通过嵌套 `Expr` 这样的一个节点和 `Symbol` 类型这样的叶子节点，我们构成了一颗树。

### xxx ... end
到了这里，我们就会发现在Julia里使用 `end` 作为代码块的结尾的好处了：它可以很自然地和不同的标记组合从而用较少的标记产生不同的语义。当我们写 `begin ... end` 的时候，我们定义了一个运行时的代码块，而当我们写 `quote ... end` 的时候我们将这一整段表达式单独拿了出来，到了函数定义，函数的AST其实只是一个由函数签名定义的表达式+一个代码块（code block）

```@example
dump(:(
    function foo(x)
        x + 1
    end
))
```

其中函数体实际上相当于用 `quote ... end` 将函数体引用起来

```@example
dump(
    quote
        x + 1
    end
)
```

## 再谈谈函数
在Julia里，函数声明是可以不用写 `return` 关键字的，所有的函数在没有显式（explicitly）
声明 `return` 关键字的时候都将默认返回整段表达式中的最后一个表达式。在有了上面关于表达式的概念以后
其实这样的设计有时候是很自然的。

## 变量的作用域
变量的作用域决定了一个变量能够被访问的区域。在Julia里作用域分为全局作用域（global scope）和局部作用域（local scope）。

## 语法糖（Syntax Sugar）以及一些语法的等价性
