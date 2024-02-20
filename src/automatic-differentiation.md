# 自动微分

自动微分（automatic differentiation）技术在机器学习里也叫做后向传播，它的原理实际上是通过记录运算顺序，利用已经定义好的导数规则，生成一个正常计算程序对偶的程序。一般来说有两种自动微分方式，一种是前向自动微分（Forward Automatic Differentiation）另外一种是后向自动微分（Reverse Automatic Differentiation），后者更加适合多参数的情况（算法细节就不详述了，多参数的时候后向自动微分的时间复杂度更低，一次传播可以计算所有的参数）。后向自动微分会将所有的操作以一张图的方式存储下来，这张图称为计算图。这也是各大深度学习框架的核心所在——如何干净地产生一个计算图，然后高效计算它。为了展示计算图是什么，我从 Cornell，CS5740，2017sp 这门课的课件里搬运了一些图，然后把他们做成了动画。动画使用纯 Julia 的框架 Luxor 制作。

我们以计算下面这个表达式为例：

```math
y = \mathbf{x}^T \mathbf{A} \mathbf{x} + \mathbf{b} \cdot \mathbf{x} + c
```

我们将会调用这样一些 Julia 函数来计算它：

1. ``z_1 = x^T``
2. ``z_2 = z_1 A``
3. ``y_1 = z_2 x``
4. ``y_2 = b \cdot x``
5. ``y_1 + y_2 + c``

而实际上我们可以把这个过程画成下面的这个图

![forward]([http://blog.rogerluo.me/images/comput-graph-forward.gif](https://blog.rogerluo.dev/images/comput-graph-forward.gif))

而计算这样一张图，我们将先从叶子结点开始赋值（绿色），然后依次计算相邻的节点，不断向前传播。这个过程称为前向传播过程。接下来我们按照链式法则来计算导数，每个节点返回的导数都和输入的数量相同，我们从最上面的节点开始向后传播，将当前有导数的节点标记为红色。

![backward]([https://blog.rogerluo.dev/images/comput-graph-backward.gif](https://blog.rogerluo.dev/images/comput-graph-backward.gif))

当红色传播到变量处时，我们就获得了变量的导数。

## 动态图 VS 静态图
按照构建计算图的方式不同，我们可以将计算图分为动态图和静态图两种，尽管在算法上并没有很大区别，但是在实现上我们可以选择在前向传播的过程中构建计算图（比如 PyTorch），也可以选择先构建计算图再计算各个节点的值（比如 tensorflow）。

就我个人而言，我比较喜欢 PyTorch，所以这里我将实现一个动态图。

## 定义计算图中的节点
在我们开始写具体的实现之前，先来为所有的节点类型定义一个抽象类型（类似于基类）：

```julia
abstract type AbstractNode end
```

在 PyTorch 里，能够拥有导数的称为变量（Variable），尽管在 0.4 版本之后 Tensor 默认就是一个 Variable 了（有 requires_grad 为 True），在后端依然还有这个类型。它是对计算图构建过程中不可或缺的类型。接下来我们来定义变量（Variable）

```julia
mutable struct Variable{T} <: AbstractNode
    value::T
    grad::T

    Variable(val::T) where T = new{T}(val, zero(val))
end
```

类似 PyTorch 一样，变量存储了值（value）和它的梯度（grad），在每一次后向传播的过程中我们将会不断地将梯度累加到这个变量的梯度上去。这里 `zero` 是几乎所有 Julia 数值类型都有的一个接口，它将放回对应的零元素，例如对 `Float64` 类型的 Julia 变量，将返回 0.0，对`Array{Float64}`将返回一个充满 0.0 的 `Array{Float64}`。

## 其它节点
我们现在有了变量了，也就是计算图的叶子结点，接下来还需要有中间的节点。它们将存储一个函数和它们的输入

```julia
struct Node{FT <: Function, ArgsT <: Tuple, KwargsT <: NamedTuple} <: AbstractNode
    f::FT
    args::ArgsT
    kwargs::KwargsT
end
```

我们这里使用参数类型，这样在将来进行分发的时候，编译器能够自己通过类型推导出要分发的函数从而提高运行时的性能。但我们应当要考虑 broadcast（广播）和正常的函数调用的区别，由于 Julia 能够对任意函数进行广播，广播时所调用的实际上是 broadcast 函数，所以我们不妨实现两个 trait 来区分这种情况：

```julia
abstract type Operator end

module Trait
import YAAD: Operator

struct Method{FT} <: Operator
    f::FT
end

struct Broadcasted{FT} <: Operator
    f::FT
end
end # Trait
```

这里我将这两个 trait 实现在一个 module 里面是为了能够显示地体现出他们俩是 trait，因为之后调用的时候将会写为 `Trait.Method` 和 `Trait.Broadcasted` ，他们各自存储了一个函数（注意 Julia 里每个函数都是一个 callable 的类型）。然后我们把原先 Node 类型的参数约束 Function 改成 Operator

```julia
struct Node{FT <: Operator, ArgsT <: Tuple, KwargsT <: NamedTuple} <: AbstractNode
    f::FT
    args::ArgsT
    kwargs::KwargsT
end
```

接下来为了方便我们来定义一些构造函数

```julia
# wrap function to Method
Node(f::Function, args, kwargs) = Node(Trait.Method(f), args, kwargs)
Node(op, args) = Node(op, args, NamedTuple())
```

第一个是因为大部分时间，我们要记录的函数就是它本身而不是一个广播，第二个是因为大部分涉及数值计算的函数都没有关键字（keyword）。实际上，Node 类型本身也只是函数和它的输入类型的一个 trait，它在计算的过程中也只是负责（静态地）分发方法。在更加高级的实现里，我们实际上有更加漂亮的实现，利用 Cassette.jl 对 Julia 代码进行非侵入式地自动微分（意思是无需给源码重载运算符，增加 Variable 类型，编译器将直接在 JIT 期间对前向传播的代码进行变换，从而直接得到计算梯度的代码）。最后，我们还需要定义一个缓存函数输出的对象，这个缓存的值将会被一些函数的导数用到

```julia
mutable struct CachedNode{NT <: AbstractNode, OutT} <: AbstractNode
    node::NT
    output::OutT
end
```

而这个节点将在前向传播的同时被构建出来（否则我们无法知道输出的类型是什么）

```julia
function CachedNode(f, args...; kwargs...)
    node = Node(f, args, kwargs.data) # this constructs a Node
    output = forward(node)
    CachedNode(node, output)
end
```

我们暂且把这个接口定义为 forward（与 PyTorch 一致）

## 求值
求值是最重要的部分，因为我们需要将我们的自动微分设计地可扩展，尽量不要在扩展的时候编写冗余的代码。而在 Julia 里，我们可以利用多重派发（multiple dispatch）实现这一点。

### 前向传播
那么如何进行前向传播呢？这取决于对于 forward 这个抽象函数（generic function），实现了什么方法（method）：

1. 如果输入是一个 Node 类型，我们将其展开

```julia
forward(node.f, map(forward, node.args)...; map(forward, node.kwargs)...)
```

2. 这将使得我们多了一层插入自定义方法的接口，如果我们有一个自定义的算符，它并非一个函数，我们只需要实现对应的方法即可，例如

```julia
struct Linear <: Operator
  w::Matrix{Float64}
  b::Vector{Float64}
end


forward(op::Linear, x::Vector{Float64}) = op.w * x + op.b
```

3. 然而对于简单的函数调用，我们并不想每次都写

```julia
function forward(::Method{typeof(sin)}, x)
  sin(x)
end
```

所以我们再实现一个默认展开 Operator 的方法

```julia
forward(op::Operator, args...; kwargs...) = op(args...; kwargs...)
```

这意味着只要 Operator 实现了自己的 call 方法（如果这个 Operator 类型是 callable 的），那么就无需去写别的东西，自动调用这个方法。当然我们现在要回去给 Method Trait 实现一下它的 call 方法

```julia
(op::Trait.Method)(args...; kwargs...) = op.f(args...; kwargs...)
```

例如，我们现在只需要定义 Linear 的 call 方法就够了

```julia
(op::Linear)(x::Vector) = op.w * x + op.b
```

4. 此外，除了变量，还有一些常数例如

```julia
Variable(2.0) + 3.0
```

这里的 3.0 就是一个不需要求导的常数，我们原封不动地返回它，这样我们只要实现一个 value 接口来获取值即可

```julia
value(x) = x
value(x::Variable) = x.value
value(x::CachedNode) = x.output
```

然后直接调用 value

```julia
forward(x) = x
forward(x::Variable) = value(x)
```

然后别忘了，对于其它类型我们返回一个友好一些的报错

```julia
forward(x::NT) where {NT <: AbstractNode} = error("forward method is not implemented for node type: $NT")

function value(x::T) where {T <: AbstractNode}
    error(
        "Expected value in this node $x of type $T ",
        "check if you defined a non-cached node",
        " or overload value function for your node."
    )
end
```

然后对于 `Variable` 和 `CachedNode` 我们要返回它们存储的值，好的👌，到目前为止，我们已经搞定前向传播部分了，接下来是后向传播部分。


## 后向传播
后向传播实际上和前向传播几乎是一样的，我们只要不断地在不同的类型标签下迭代 backward 接口即可（注意我不打算在这里实现关键词的后向传播，尽管这并不难）

首先，对 Variable 来说，这很简单直接加接收到的梯度就好了

```julia
function backward(x::Variable, grad)
    x.grad += grad
    nothing
end
```

然后我们现在定义 `CachedNode` 的后向传播规则。我们先从一个叫 `gradient` 的方法里获得各个输入的导数，然后再把这些导数依次输入到输入类型对应的 `backward` 函数里去

```julia
function backward(node::CachedNode, f, grad)
    grad_inputs = gradient(node, grad)
    for (each, each_grad) in zip(args(node), grad_inputs)
        backward(each, each_grad)
    end
    nothing
end
```

等等，我们要在这里加一些友好的报错信息，免得以后我们自己抓狂。首先是类型的检查，这完全是静态的，所以不同担心会影响性能

```julia
backward_type_assert(node::CachedNode{<:AbstractNode, T}, grad::T) where T = true
backward_type_assert(node::CachedNode{<:AbstractNode, T1}, grad::T2) where {T1, T2} =
    error("Gradient is expected to have the same",
          " type with outputs, expected $T1",
          " got $T2")
```

我们在这里要求输出和梯度的类型要一样，但是对于多维数组（`AbstractArray`）我们只要求它们的数据类型和维度相同即可，因为有可能一些函数会返回特别优化的数组（例如稀疏数组，或者一些懒惰求值的中间结果）。

```julia
# exclude arrays
backward_type_assert(node::CachedNode{<:AbstractNode, T1}, grad::T2) where
    {T, N, T1 <: AbstractArray{T, N}, T2 <: AbstractArray{T, N}} = true
```

然后我们还要检查梯度和输出的大小是否匹配

```julia
function backward_size_assert(node::CachedNode, grad)
    size(node.output) == size(grad) ||
        error(
            "gradient should have the same size with output,",
            " expect size $(size(node.output)), got $(size(grad))"
        )
end
```

在 Julia 里，可以通过编译选项把边界检查关掉，因为我们有时候完全不需要边界检查，你可以通过增加 @boundscheck 这个宏来实现这一点，最后我们的 backward 函数如下：

```julia
function backward(node::CachedNode, f, grad)
    backward_type_assert(node, grad)
    @boundscheck backward_size_assert(node, grad)

    grad_inputs = gradient(node, grad)
    for (each, each_grad) in zip(args(node), grad_inputs)
        backward(each, each_grad)
    end
    nothing
end
```

现在我们来考虑如何定义梯度，也就是 gradient 方法，我们依然希望不要写冗余的代码，同时保证性能和扩展性。比如，实现 sin 的导数只需要定义

```julia
gradient(::typeof(sin), grad, output, x) = grad * cos(x)
```

我们还是利用多重派发来实现这一点，先把 `CachedNode` 展开

```julia
gradient(x::CachedNode, grad) = gradient(x.node.f, grad, x.output, map(value, x.node.args)...; map(value, x.node.kwargs)...)
```

然后把 Operator 展开到函数上去

```julia
gradient(x::Trait.Method, grad, output, args...; kwargs...) =
    gradient(x.f, grad, output, args...; kwargs...)
```

最后定义一个报错信息

```julia
gradient(fn, grad, output, args...; kwargs...) =
    error(
        "gradient of operator $fn is not defined\n",
        "Possible Fix:\n",
        "define one of the following:\n",
        "1. gradient(::typeof($fn), grad, output, args...; kwargs...)\n",
        "2. gradient(op::Trait.Method{typeof($fn)}, grad, output, args...; kwargs...)\n",
        "3. gradient(op::Trait.Broadcasted{typeof($fn)}, grad, output, args...; kwargs...)\n"
    )
```

这样，我们就可以选择不同的 gradient 接口来实现导数，Julia 将自动派发你实现的这个方法，例如

```julia
# I re-define the concrete type `Linear` here in order to store the gradient
struct Linear <: Operator
  w::Variable{Matrix{Float64}}
  b::Variable{Vector{Float64}}
end

function gradient(op::Linear, grad, output, x)
  grad_w, grad_b = # some gradient expression to calculate the gradient of w and b
  backward(op.w, grad_w) # update gradient of w
  backward(op.w, grad_b) # update gradient of b

  grad_input = # calculate the gradient of input
  grad_input # return the gradient of input
end
```

最后我们定义一个 `register` 的接口用来产生 `CachedNode`

```julia
register(f, args...; kwargs...) = CachedNode(f, args...; kwargs...)
```

这样我们就可以通过重载函数/运算符来构建计算图了

```julia
Base.sin(x::AbstractNode) = register(Base.sin, x)
gradient(::typeof(Base.sin), grad, output, x) = (grad * cos(x), )
```

不过等等，似乎这里有时候需要判断一下输入是什么类型比较好，我们不妨为 Variable 和 CachedNode 定义一个抽象类型 Value

```julia
abstract type Value{T} <: AbstractNode end
```

Value 类型将带有其子类型的值的类型 T 作为其参数。现在先回去修改 `Variable` 和 `CachedNode`

```julia
mutable struct Variable{T} <: Value{T}
    value::T
    grad::T

    Variable(val::T) where T = new{T}(val, zero(grad))
end

mutable struct CachedNode{NT <: AbstractNode, OutT} <: Value{OutT}
    node::NT
    output::OutT
end
```

### 广播
然而上面的定义还只能给标量用，对于数组我们还需要广播才行。Julia 自己实现了一套广播系统，它能够广播任何 Julia 函数到数组上，会融合多个被广播的函数（从而产生更优质的向量化 SIMD 代码），同时还允许定义广播的行为。这恰好就是我们需要的：我们要在广播的同时产生一个计算图，记录这个操作。首先我们定义我们自己的广播风格（BroadcastStyle）：

```julia
struct ComputGraphStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:AbstractNode}) = ComputGraphStyle()
Broadcast.BroadcastStyle(s::ComputGraphStyle, x::Broadcast.BroadcastStyle) = s
```

这还不够，Julia 的 broadcast 是懒惰求值的，它先通过 broadcasted 方法构建中间类型，然后再在最后通过 materialize 方法进行求值。我们还需要让它们也被记录在计算图里

```julia
function Broadcast.broadcasted(::ComputGraphStyle, f, args...)
    mt = Trait.Broadcasted(f)
    register(mt, args...)
end

Broadcast.materialize(x::AbstractNode) = register(Broadcast.materialize, x)
```

然后我们让 `materialize` 在后向传播的时候直接返回梯度

```julia
function backward(node::CachedNode, ::typeof(Broadcast.materialize), grad)
    backward_type_assert(node, grad)
    @boundscheck backward_size_assert(node, grad)
    backward(node.node.args[1], grad) # materialize only has one arguments, we don't need the for loop
end
```

然而这时，Broadcasted 类型的 backward 会调用默认的 CachedNode 的 backward 方法，有时就会因为类型不同报错（因为我们之前这么定义了）我们为这个类型开个后门

```julia
function backward(node::CachedNode, ::Trait.Broadcasted, grad)
    grad_inputs = gradient(node, grad)
    for (each, each_grad) in zip(args(node), grad_inputs)
        backward(each, each_grad)
    end
    nothing
end
```

### 免费获得更多的算符
Julia 有一个包叫做 DiffRules.jl，它记录了大量常用算符的导数规则，并且这些导数规则都以 Julia 表达式的方式记录，这意味着我们可以利用元编程批量生产算符。这些导数规则都在一个常数列表里，名为`DiffRules.DEFINED_DIFFRULES`，我们遍历它即可

```julia
for (mod, name, nargs) in keys(DiffRules.DEFINED_DIFFRULES)
  # code generation
end
```

这里 mod 是 module 的名字，name 是函数的名字，nargs 是函数输入变量的个数，然后我们就可以用如下的方式来批量产生这些导数的定义

```julia
for (mod, name, nargs) in keys(DiffRules.DEFINED_DIFFRULES)
    f_ex_head = Expr(:., mod, QuoteNode(name))

    if nargs == 1
        df_ex = DiffRules.diffrule(mod, name, :x)

        name === :abs && continue # exclude abs, it cannot be directly broadcasted

        @eval begin
            $(f_ex_head)(x::AbstractNode) = register($(f_ex_head), x)
            gradient(::typeof($(f_ex_head)), grad, output, x) = (grad * $df_ex, )
            gradient(mt::Trait.Broadcasted{typeof($f_ex_head)}, grad, output, x) = (@.(grad * $(df_ex)), )
        end
    elseif nargs == 2
        df_ex = DiffRules.diffrule(mod, name, :x, :y)

        @eval begin

            $(f_ex_head)(x1::AbstractNode, x2) = register($f_ex_head, x1, x2)
            $(f_ex_head)(x1, x2::AbstractNode) = register($f_ex_head, x1, x2)
            $(f_ex_head)(x1::AbstractNode, x2::AbstractNode) = register($f_ex_head, x1, x2)

            gradient(::typeof($f_ex_head), grad, output, x, y) =
                (grad * $(df_ex[1]), grad * $(df_ex[2]))
            gradient(::Trait.Broadcasted{typeof($f_ex_head)}, grad, output, x, y) =
                (@.(grad * ($(df_ex[1]))), @.(grad * $(df_ex[2])))
        end
    else
        @info "unknown operator $name"
    end
end
```

对如何使用代码生成，我建议你阅读 Julia 的文档：[元编程 · Julia中文文档](http://docs.juliacn.com/latest/manual/metaprogramming/) 。我在这里跳过了 `abs` 函数是因为批量广播的宏不能对 `if else` 进行广播。我们需要单独去定义 `abs` 的导数，但是剩下几乎所有的数学函数都用 Diffrules 生成了。


## 代码修饰
之后我又花了一些时间实现仿照 PyTorch 了一个计算 Jacobbian 的函数用来做单元测试。然后利用 Trait 将数组类型的 `Variable` 重新插入 `AbstractArray` 的类型树中以实现更好的打印信息。

## 性能对比
好了！到此我们就写完了这个自动微分库了，它的性能怎么样呢？我起初以为这么简单的一个实现只是一个玩具，但实际上它的性能非常不错！我需要计算一个称为 MPS 的东西（Matrix product state），所以我在这里使用了我使用最频繁的操作进行 benchmark，这个操作是 tr(x1 * x2) ，这里 x1 和 x2 是矩阵，然后对其求迹。

所以我首先为 YAAD 实现了这两个算符：

```julia
# 这一部分其实已经在 DiffRules 进行代码生成的时候定义过了
Base.:(*)(lhs::Value, rhs) = register(Base.:(*), lhs, rhs)
Base.:(*)(lhs, rhs::Value) = register(Base.:(*), lhs, rhs)
Base.:(*)(lhs::Value, rhs::Value) = register(Base.:(*), lhs, rhs)

# 这里开始是新的定义
using LinearAlgebra

LinearAlgebra.tr(x::Value) = register(LinearAlgebra.tr, x)
gradient(::typeof(tr), grad, output, x) = (grad * Matrix(I, size(x)), )

function gradient(::typeof(*), grad, output, lhs::AbstractVecOrMat, rhs::AbstractVecOrMat)
    grad * transpose(rhs), transpose(lhs) * grad
end
```julia

然后我选取了几个 Julia 的库（Zygote，Flux，YAAD 是我的），还有 PyTorch 在 CPU 上进行了一下比较

```julia
Zygote.@grad LinearAlgebra.tr(x) = LinearAlgebra.tr(x), Δ-> (Δ * Matrix(I, size(x)), )

function bench_tr_mul_yaad(x1, x2)
    z = tr(x1 * x2)
    YAAD.backward(z)
    x1.grad, x2.grad
end

function bench_tr_mul_autograd(x1, x2)
    z = AutoGrad.@diff tr(x1 * x2)
    AutoGrad.grad(z, x1), AutoGrad.grad(z, x2)
end

function bench_tr_mul_zygote(x1, x2)
    Zygote.gradient((x1, x2)->tr(x1 * x2), x1, x2)
end

function bench_tr_mul_flux(x1, x2)
    z = tr(x1 * x2)
    Flux.Tracker.back!(z, 1)
    x1.grad, x2.grad
end
```

然后在 Python 里测试 PyTorch（我们的接口和 PyTorch 非常相似不是吗？）

```python
def bench_tr_mul_torch(x1, x2):
    z = torch.trace(torch.matmul(x1, x2))
    z.backward()
    return x1.grad, x2.grad
```

然后输入定义如下：

```julia
xv, yv = rand(30, 30), rand(30, 30)
yaad_x, yaad_y = YAAD.Variable(xv), YAAD.Variable(yv)
autograd_x, autograd_y = AutoGrad.Param(xv), AutoGrad.Param(yv)
flux_x, flux_y = Flux.param(xv), Flux.param(yv)
```

此外，在进行测试之前，我们实现一个手动计算梯度的版本作为基准：

```julia
function bench_tr_mul_base(x1, x2)
    z1 = x1 * x2
    z2 = tr(z1)

    grad_z1 = Matrix{eltype(z1)}(I, size(z1))
    grad_z1 * transpose(x2), transpose(x1) * grad_z1
end
```

然后在 Julia 里我们用 `@benchmark` 宏来多次测量以获取运行时间

```julia
julia> @benchmark bench_tr_mul_autograd(autograd_x, autograd_y)
BenchmarkTools.Trial:
  memory estimate:  33.20 KiB
  allocs estimate:  82
  --------------
  minimum time:     50.218 μs (0.00% GC)
  median time:      62.364 μs (0.00% GC)
  mean time:        90.422 μs (9.86% GC)
  maximum time:     55.386 ms (99.86% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> @benchmark bench_tr_mul_yaad(yaad_x, yaad_y)
BenchmarkTools.Trial:
  memory estimate:  51.50 KiB
  allocs estimate:  16
  --------------
  minimum time:     10.387 μs (0.00% GC)
  median time:      13.429 μs (0.00% GC)
  mean time:        24.273 μs (45.13% GC)
  maximum time:     55.963 ms (99.96% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> @benchmark bench_tr_mul_zygote(xv, yv)
BenchmarkTools.Trial:
  memory estimate:  29.98 KiB
  allocs estimate:  10
  --------------
  minimum time:     42.527 μs (0.00% GC)
  median time:      46.640 μs (0.00% GC)
  mean time:        56.996 μs (15.31% GC)
  maximum time:     51.718 ms (99.90% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> @benchmark bench_tr_mul_base(xv, yv)
BenchmarkTools.Trial:
  memory estimate:  28.78 KiB
  allocs estimate:  5
  --------------
  minimum time:     6.413 μs (0.00% GC)
  median time:      8.201 μs (0.00% GC)
  mean time:        12.215 μs (31.57% GC)
  maximum time:     11.012 ms (99.87% GC)
  --------------
  samples:          10000
  evals/sample:     5

julia> @benchmark bench_tr_mul_flux(flux_x, flux_y)
BenchmarkTools.Trial:
  memory estimate:  30.25 KiB
  allocs estimate:  24
  --------------
  minimum time:     8.009 μs (0.00% GC)
  median time:      10.002 μs (0.00% GC)
  mean time:        14.412 μs (30.14% GC)
  maximum time:     16.286 ms (99.87% GC)
  --------------
  samples:          10000
  evals/sample:     3
```

然后 PyTorch (0.4.1) 上

```python
In [4]: x = torch.rand(30, 30, dtype=torch.float64, requires_grad=True)

In [5]: y = torch.rand(30, 30, dtype=torch.float64, requires_grad=True)

In [6]: %timeit bench_tr_mul_torch(x, y)
76.8 µs ± 1.68 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

所以我们花了小半天实现的这个自动微分还不赖嘛？只比基准性能慢了几个微秒，意外的是它竟然比 PyTorch 快了不少。然后 Flux 的 Tracker 性能竟然非常接近手动求导！
