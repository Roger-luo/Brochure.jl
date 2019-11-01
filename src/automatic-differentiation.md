# 自动微分

自动微分（automatic differentiation）技术在机器学习里也叫做后向传播，它的原理实际上是通过记录运算顺序，利用已经定义好的导数规则，生成一个正常计算程序对偶的程序。一般来说有两种自动微分方式，一种是前向自动微分（Forward Automatic Differentiation）另外一种是后向自动微分（Reverse Automatic Differentiation），后者更加适合多参数的情况（算法细节就不详述了，多参数的时候后向自动微分的时间复杂度更低，一次传播可以计算所有的参数）。
后向自动微分会讲所有的操作以一张图的方式存储下来，这张图称为计算图。这也是各大深度学习框架的核心所在——如何干净地产生一个计算图，然后高效计算它。
为了展示计算图是什么，我从Cornell，CS5740，2017sp这门课的课件里搬运了一些图，然后把他们做成了动画（动画使用 Luxor.jl 制作，你也许想看看我是怎么画出来的，绘图的脚本在这里：plot.jl）


我们以计算下面这个表达式为例：

```math
y = \mathbf{x}^T \mathbf{A} \mathbf{x} + \mathbf{b} \cdot \mathbf{x} + c
```

我们将会调用这样一些Julia函数来计算它：

1. ``z_1 = x^T``
2. ``z_2 = z_1 A``
3. ``y_1 = z_2 x``
4. ``y_2 = b \cdot x``
5. ``y_1 + y_2 + c``

而实际上我们可以把这个过程画成下面的这个图

![forward](http://blog.rogerluo.me/images/comput-graph-forward.gif)

![backward](http://blog.rogerluo.me/images/comput-graph-backward.gif)
