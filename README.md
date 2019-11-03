# Julia编程指南

[![Build Status](https://travis-ci.com/Roger-luo/Brochure.jl.svg?branch=master)](https://travis-ci.com/Roger-luo/Brochure.jl)
[在线阅读](http://rogerluo.me/Brochure.jl/dev/)

**一本我不知道什么时候才会写完的书，但是你可以先看着**

**注：原来的中文教程已经不再维护和更新，内容在 [`old` 这个分支](https://github.com/Roger-luo/Brochure.jl/tree/old)**

Julia语言在2018年正式发布了第一个长期支持版本（1.0版本），在这之后市面上出现了很多中文教程。但是，
为什么我还要编写本书呢？有如下的一些原因：

1. **市面上的中文书籍和教程主要都以入门Julia语言为主。**但是还没有一本**由有经验的Julia语言开发者**所编写的书籍。
2. 我们在[中文社区的论坛](https://discourse.juliacn.com)，QQ群等中文媒体上经常可以看到中文用户抱怨没有好的教材可以参考，缺乏可以学习的代码范例，并且中文网络上依然大量充斥着旧版Julia的代码（Julia 0.6甚至是更加古老的版本）。**这些旧的代码随着1.0版本的发布已经不再支持，对很多Julia学习者造成了困扰。**
3. 我发现由于Julia语言编程范式和其它流行语言（例如Python，C++，MATLAB）有很大不同，有很多Julia学习者在阅读过文档之后，即便已经学会了基本语法，**大部分Julia学习者入门之后依然无法很好的编写一个完整的，符合Julia范式的代码库。**
6. 我们有时候还是需要一些文档里不会讲的内容，比如一些经验性质的总结，一些更加和某个领域相关的完整工程展示。所以这也是这本书的目的之一：**提供编写Julia工程的实践范例**
7. 我在[知乎上写过很多零散的关于Julia语言的文章](https://zhuanlan.zhihu.com/halfinteger)，有很多人建议我将它们整理到一起，但是我不是很喜欢知乎的编辑器，它并不适合编写长篇的技术文章，对数学公式和代码的支持非常差。而微信公众号则更加糟糕，在经过调研后我觉得还是**需要使用开源工具自己来做这件事情。**
8. 大部分的书籍依然是传统的纸质媒体，或者以纸质媒体为前提进行编写。这在现在这个互联网和富文本时代是非常落后的。我希望以此为媒介，做一些现代书籍的实验。


本书使用纯Julia进行编写，除了构成静态网页的js/html/css 脚本以外，**这本书的Julia纯度为：100%**，当你下载这本书之后，可以用Julia的编译器运行下面这行命令

在命令行里

```sh
julia make.jl
```

或者打开Julia的REPL，然后运行

```jl
include("make.jl")
```

这本书就会以网页的形式挂在到 [localhost:8000](http://localhost:8000/index.html)，它使用了 [Documenter](https://github.com/JuliaDocs/Documenter.jl) 和 [LiveServer](https://github.com/asprionj/LiveServer.jl) 这两个Julia包进行编译和挂载。如果你喜欢黑夜模式（dark mode）
你还可以点击右上角的齿轮按钮选择黑夜模式。

## 赞赏

赞赏是让我完成它的最好鼓励

<img src="https://github.com/Roger-luo/Brochure.jl/raw/master/src/assets/buymecoffee.png" alt="buymecoffee" width="300"> </img>

## 协议

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。代码部分使用MIT协议。
