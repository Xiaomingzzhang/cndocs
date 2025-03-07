# 非光滑一维流形
或许这个软件包最值得注意的地方就是其可以计算非光滑流形. 目前, 支持两类系统的非光滑流形计算:
- 时间周期的非光滑微分方程的一维流形计算
- 非光滑自治系统的二维流形计算

其中流形值得都是鞍点的不变流形, 前者需要取时间周期映射, 后者则需要取固定步长的时间-$T$-映射. 这两类系统中的非光滑因素可以是多种多样的, 包括分段光滑, 碰撞, 以及它们之间的组合. 这三类非光滑系统无需用户自己求解, 我们提供了三个封装的结构体:
- [`PiecewiseV`](@ref)
- [`BilliardV`](@ref)
- [`PiecewiseImpactV`](@ref)

并使用 [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) 中的 Callback 功能来进行时间映射的计算. 下面将以三个例子来介绍这三类非光滑系统不变流形的计算方法.

!!! warning 
    关于非光滑流形的计算严重依赖于求解 ODE 的算法与精度, 当求解失败或效果不佳时, 可以尝试更换算法, 提高求解 ODE 的精度, 或者降低 [`NSOneDManifoldProblem`](@ref) 中的参数 `ϵ`, `d`, `amax` 的值的大小.


## 分段光滑系统
考虑一个简单的分段光滑系统:

```math
\begin{aligned}
\dot{x}&=y,\\
\dot{y}&=f(x) + \epsilon \sin(2\pi t),
\end{aligned}
```
其中

```math
f(x) =
\begin{cases}
-k_1 x& \text{if } x < -d,\\
k_2 x & \text{if } -d<x<d,\\
-k_3 x& \text{if } x > d.
\end{cases}
```

$k_1,k_2,k_3,d>0$ 都为正的常数. 下面我们将计算时间周期映射的不变流形. 注意到当周期扰动很小时, 鞍点应当很接近原点.

首先加载用到的程序包
```@setup piecewise
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie
```
```@repl piecewise
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie
```
接着定义分段光滑的矢量场:
```@example piecewise
f1(x, p, t) = SA[x[2], p[1]*x[1]+p[4]*sin(2pi * t)]
f2(x, p, t) = SA[x[2], -p[2]*x[1]+p[4]*sin(2pi * t)]
f3(x, p, t) = SA[x[2], -p[3]*x[1]+p[4]*sin(2pi * t)]

hyper1(x, p, t) = x[1] - p[5]
hyper2(x, p, t) = x[1] + p[5]

dom1(x, p, t) = -p[5] < x[1] < p[5]
dom2(x, p, t) = x[1] > p[5]
dom3(x, p, t) = x[1] < -p[5]

vectorfield = PiecewiseV((f1, f2, f3), (dom1, dom2, dom3), (hyper1, hyper2))
```
传递给 `PiecewiseV` 这个结构体的参数分别为: 矢量场, 它们所在的区域, 以及分割这些区域的超平面. 更多细节可参考 [`PiecewiseV`](@ref).

接下来我们将求解时间周期映射的关键信息封装到另外一个结构体 [`NSSetUp`](@ref) 中:

```@repl piecewise
setup = setmap(vectorfield, (0.0, 1.0), Tsit5(), abstol=1e-8)
```
其中函数 [`setmap`](@ref) 用于封装时间映射的计算信息. 现在我们已经定义好求解时间周期映射的一切了.


接下来为了生成局部流形. 我们同样需要定位鞍点以及其不稳定特征向量. 取定参数:
```@repl piecewise
para = para = [2, 5, 5, 0.6, 2]
```
由于扰动是很小的, 鞍型的周期轨道应该仍然在 `dom1` 中. 因此我们可以使用 `findsaddle` 来计算鞍点的位置:
```@example piecewise
function df1(x, p, t)
    SA[0 1; p[1] 0]
end
initialguess = SA[0.0, 0.0]
saddle = findsaddle(f1, df1, (0.0,1.0), initialguess, para, abstol=1e-10)
```

接下来创建问题, 生成局部流形, 并进行延拓
```@repl piecewise
prob = NSOneDManifoldProblem(setup, para, ϵ = 1e-3)
segment = gen_segment(saddle)
manifold = growmanifold(prob, segment, 8)
```

注意到, `manifold.data` 的数据类型是 `Vector{Vector{S}}`, 其中 `S` 是插值函数. 所以绘制结果需要使用如下函数:
```@example piecewise
using CairoMakie
function manifold_plot(data)
    fig = Figure()
    axes = Axis(fig[1,1])
    for k in eachindex(data)
        for j in eachindex(data[k])
            points=data[k][j].u
            lines!(axes,first.(points),last.(points))
        end
    end
    fig
end
manifold_plot(manifold.data)
```

完整代码:
```julia
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie
f1(x, p, t) = SA[x[2], p[1]*x[1]+p[4]*sin(2pi * t)]
f2(x, p, t) = SA[x[2], -p[2]*x[1]+p[4]*sin(2pi * t)]
f3(x, p, t) = SA[x[2], -p[3]*x[1]+p[4]*sin(2pi * t)]
hyper1(x, p, t) = x[1] - p[5]
hyper2(x, p, t) = x[1] + p[5]
dom1(x, p, t) = -p[5] < x[1] < p[5]
dom2(x, p, t) = x[1] > p[5]
dom3(x, p, t) = x[1] < -p[5]
vectorfield = PiecewiseV((f1, f2, f3), (dom1, dom2, dom3), (hyper1, hyper2))
setup = setmap(vectorfield, (0.0, 1.0), Tsit5(), abstol=1e-8)
para = [2, 5, 5, 0.6, 2]
function df1(x, p, t)
    SA[0 1; p[1] 0]
end
initialguess = SA[0.0, 0.0]
saddle = findsaddle(f1, df1, (0.0,1.0), initialguess, para, abstol=1e-10)
prob = NSOneDManifoldProblem(setup, para, ϵ = 1e-3)
segment = gen_segment(saddle)
manifold = growmanifold(prob, segment, 8)
function manifold_plot(data)
    fig = Figure()
    axes = Axis(fig[1,1])
    for k in eachindex(data)
        for j in eachindex(data[k])
            points=data[k][j].u
            lines!(axes,first.(points),last.(points))
        end
    end
    fig
end
manifold_plot(manifold.data)
```

## 碰撞系统

考虑如下的受到激励的倒摆方程:

```math
\begin{aligned}
\dot{x}&= y,\\
\dot{y}&= \sin(x) - \epsilon \cos(2\pi t),
\end{aligned}
```
假设倒摆两边存在墙壁: 当 $x=\xi$ 或者 $x=-\xi$ 时, 有 $y\rightarrow - ry$. 同样地, 需要先构建非光滑矢量场
```@setup impact
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie
function manifold_plot(data)
    fig = Figure()
    axes = Axis(fig[1,1])
    for k in eachindex(data)
        for j in eachindex(data[k])
            points=data[k][j].u
            lines!(axes,first.(points),last.(points))
        end
    end
    fig
end
```
```@example impact
f(x, p, t) = SA[x[2], sin(x[1])-p[1]*cos(2 * pi * t)]

hyper1(x, p, t) = x[1] + p[2]
hyper2(x, p, t) = x[1] - p[2]

rule1(x, p, t) = SA[x[1], -p[3]*x[2]]
rule2(x, p, t) = SA[x[1], -p[3]*x[2]]

vectorfield = BilliardV(f, (hyper1, hyper2), (rule1, rule2))
```
接着封装求解时间周期映射的信息:
```@example impact
setup = setmap(vectorfield, (0.0, 1.0), Vern9(), abstol=1e-10)
```
寻找鞍点:
```@example impact
function df(x, p, t)
    SA[0 1; cos(x[1]) 0]
end
para = [0.2, pi / 4, 0.98]
initialguess = SA[0.0, 0.0]
saddle = findsaddle(f, df, (0.0,1.0), initialguess, para, abstol=1e-10)
```
接下来创建问题, 生成局部流形, 并进行延拓
```@repl impact
prob = NSOneDManifoldProblem(setup, para)
segment = gen_segment(saddle)
manifold = growmanifold(prob, segment, 11)
```

最后绘制结果: 
```@example impact
manifold_plot(manifold.data)
```

完整代码:
```julia
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie
f(x, p, t) = SA[x[2], sin(x[1])-p[1]*cos(2 * pi * t)]
hyper1(x, p, t) = x[1] + p[2]
hyper2(x, p, t) = x[1] - p[2]
rule1(x, p, t) = SA[x[1], -p[3]*x[2]]
rule2(x, p, t) = SA[x[1], -p[3]*x[2]]
vectorfield = BilliardV(f, (hyper1, hyper2), (rule1, rule2))
setup = setmap(vectorfield, (0.0, 1.0), Vern9(), abstol=1e-10)
function df(x, p, t)
    SA[0 1; cos(x[1]) 0]
end
para = [0.2, pi / 4, 0.98]
initialguess = SA[0.0, 0.0]
saddle = findsaddle(f, df, (0.0,1.0), initialguess, para, abstol=1e-10)
prob = NSOneDManifoldProblem(setup, para)
segment = gen_segment(saddle)
manifold = growmanifold(prob, segment, 11)
function manifold_plot(data)
    fig = Figure()
    axes = Axis(fig[1,1])
    for k in eachindex(data)
        for j in eachindex(data[k])
            points=data[k][j].u
            lines!(axes,first.(points),last.(points))
        end
    end
    fig
end
manifold_plot(manifold.data)
```

## 分段光滑与碰撞的组合的 ODE 系统
```@setup piecewiseimpact
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie
function manifold_plot(data)
    fig = Figure()
    axes = Axis(fig[1,1])
    for k in eachindex(data)
        for j in eachindex(data[k])
            points=data[k][j].u
            lines!(axes,first.(points),last.(points))
        end
    end
    fig
end
```
现在考虑同时存在分段光滑与碰撞的 ODE 系统:
```math
\begin{aligned}
\dot{x}&=y,\\
\dot{y}&=f(x) + \epsilon \sin(2\pi t),
\end{aligned}
```
其中

```math
f(x) =
\begin{cases}
-k_1 x& \text{if } x < -d,\\
k_2 x & \text{if } -d<x<d
\end{cases}
```

$k_1,k_2,d>0$ 都为正的常数. 当 $x=d$ 时, 有 $\dot{y}->-r\dot{y}$. 下面我们将计算时间周期映射的不变流形. 注意到当周期扰动很小时, 鞍点应当很接近原点. 首先加载用到的程序包
```@setup piecewiseimpact
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie
```
```@repl piecewiseimpact
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie
```
接着定义非光滑矢量场:
```@example piecewiseimpact
f1(x, p, t) = SA[x[2], p[1]*x[1]+p[3]*sin(2pi * t)]
f2(x, p, t) = SA[x[2], -p[2]*x[1]+p[3]*sin(2pi * t)]

hyper1(x, p, t) = x[1] - p[4]
hyper2(x, p, t) = x[1] + p[4]

dom1(x, p, t) = -p[4] < x[1]
dom2(x, p, t) = x[1] < -p[4]

impact_rule(x, p, t) = SA[x[1], -p[5]*x[2]]
id(x,p,t) = x

vectorfield = PiecewiseImpactV((f1, f2), (dom1, dom2), (hyper1, hyper2), (impact_rule, id), [1])
```
传递给 `PiecewiseImpactV` 这个结构体的参数分别为: 矢量场, 它们所在的区域, 以及分割这些区域的超平面, 作用在超平面上的规则, 以及具有碰撞效应的那些规则列表. 更多细节可参考 [`PiecewiseImpactV`](@ref).

接下来我们将求解时间周期映射的关键信息封装到另外一个结构体 [`NSSetUp`](@ref) 中:

```@repl piecewiseimpact
setup = setmap(vectorfield, (0.0, 1.0), Tsit5(), abstol=1e-8, reltol=1e-8)
```
其中函数 [`setmap`](@ref) 用于封装时间映射的计算信息. 现在我们已经定义好求解时间周期映射的一切了.


接下来为了生成局部流形. 我们同样需要定位鞍点以及其不稳定特征向量. 取定参数:
```@repl piecewiseimpact
para = [2, 5, 0.5, 2, 0.98]
```
由于扰动是很小的, 鞍型的周期轨道应该仍然在 `dom1` 中. 因此我们可以使用 `findsaddle` 来计算鞍点的位置:
```@example piecewiseimpact
function df1(x, p, t)
    SA[0 1; p[1] 0]
end
initialguess = SA[0.0, 0.0]
saddle = findsaddle(f1, df1, (0.0,1.0), initialguess, para, abstol=1e-10)
```

接下来创建问题, 生成局部流形, 并进行延拓
```@repl piecewiseimpact
prob = NSOneDManifoldProblem(setup, para)
segment = gen_segment(saddle)
manifold = growmanifold(prob, segment, 9)
```

注意到, `manifold.data` 的数据类型是 `Vector{Vector{S}}`, 其中 `S` 是插值函数. 所以绘制结果需要使用如下函数:
```@example piecewiseimpact
using CairoMakie
function manifold_plot(data)
    fig = Figure()
    axes = Axis(fig[1,1])
    for k in eachindex(data)
        for j in eachindex(data[k])
            points=data[k][j].u
            lines!(axes,first.(points),last.(points))
        end
    end
    fig
end
manifold_plot(manifold.data)
```

完整代码:
```julia
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie
f1(x, p, t) = SA[x[2], p[1]*x[1]+p[3]*sin(2pi * t)]
f2(x, p, t) = SA[x[2], -p[2]*x[1]+p[3]*sin(2pi * t)]
hyper1(x, p, t) = x[1] - p[4]
hyper2(x, p, t) = x[1] + p[4]
dom1(x, p, t) = -p[4] < x[1]
dom2(x, p, t) = x[1] < -p[4]
impact_rule(x, p, t) = SA[x[1], -p[5]*x[2]]
id(x,p,t) = x
vectorfield = PiecewiseImpactV((f1, f2), (dom1, dom2), (hyper1, hyper2), (impact_rule, id), [1])
setup = setmap(vectorfield, (0.0, 1.0), Tsit5(), abstol=1e-8, reltol=1e-8)
para = [2, 5, 0.5, 2, 0.98]
initialguess = SA[0.0, 0.0]
function df1(x, p, t)
    SA[0 1; p[1] 0]
end
saddle = findsaddle(f1, df1, (0.0,1.0), initialguess, para, abstol=1e-10)
segment = gen_segment(saddle)
prob = NSOneDManifoldProblem(setup, para)
manifold = growmanifold(prob, segment, 9)
function manifold_plot(data)
    fig = Figure()
    axes = Axis(fig[1,1])
    for k in eachindex(data)
        for j in eachindex(data[k])
            points=data[k][j].u
            lines!(axes,first.(points),last.(points))
        end
    end
    fig
end
manifold_plot(manifold.data)
```
