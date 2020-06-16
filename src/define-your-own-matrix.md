# 实现你自己的稀疏矩阵

Julia 语言有着世界上最好的矩阵和数组生态，这得益于 Julia 语言的类型系统（type system）和多重派发（multiple dispatch）。在这一部分，我们将通过自己实现一个稀疏矩阵类型来体验这一点。

## 一些关于稀疏矩阵的基础知识
我们这里所说的稀疏矩阵是指能够表示任意含有大量零元素的矩阵的数据结构。一般来说有这样几种数据结构

### COO 格式
这是最简单直接的格式，我们将每个非零元素的值，坐标存到一个表里。一般用于高效地构造矩阵。

### CSC 格式
CSC 是 *Compressed Sparse Column* 的缩写，一些实现里也使用 CSR 格式也就是*Compressed Sparse Row*。

## Julia 语言中的接口定义（Interface）

## Julia 语言中的广播机制
### Holy Trait
