# TutorialZH.jl

Julia语言的中文教程。

Chinese Tutorial for Julia Language.

## 使用方法

兼容 `v0.6` 和 `v0.7` (1.0的pre-release)，请通过Julia自带的包管理器进行安装。

在 v0.6 中，请使用 `Pkg` 模块进行安装

```julia
julia> Pkg.clone("https://github.com/Roger-luo/TutorialZH.jl.git")
```

在 v0.7 中，请使用REPL的 pkg mode 安装，按 `]` 键

```julia
(v0.7) pkg> dev https://github.com/Roger-luo/TutorialZH.jl.git#master
```

或者使用 `Pkg` 模块

```julia
julia> using Pkg; Pkg.develop("https://github.com/Roger-luo/TutorialZH.jl.git#master")
```

## 贡献和建议

欢迎通过 issue 和 PR 为这个中文教程提供贡献和建议。

## 内容

- [x] Julia 语言的简介
- [x] Julia 语法快速入门
- [ ] Julia 语言进阶
- [ ] Julia 语言的元编程和宏
- [ ] 编写高性能的Julia程序

## 致谢

感谢 **史雪松** 对C++部分的例子的建议。

## 开源协议

MIT协议
