using InvariantManifolds, StaticArrays, OrdinaryDiffEq, CairoMakie
```
# 开始使用: 一维光滑流形

## 非线性映射
考虑如下的 Henon 映射:

```math
\begin{aligned}
x'&=1-\alpha x^2+y,\\
y'&=\beta x,
\end{aligned}
```

其中 $\alpha,\beta$ 为参数. 这个映射具有不动点:

```math
\begin{aligned}
(x_1,y_1)&=(\frac{-\sqrt{4 \alpha +\beta ^2-2 \beta +1}+\beta -1}{2 \alpha },\frac{1}{2} \left(\frac{\beta ^2}{\alpha }-\frac{\beta  \sqrt{4 \alpha +\beta ^2-2 \beta +1}}{\alpha }-\frac{\beta }{\alpha }\right)),\\
(x_2,y_2)&=(\frac{\sqrt{4 \alpha +\beta ^2-2 \beta +1}+\beta -1}{2 \alpha },\frac{1}{2} \left(\frac{\beta ^2}{\alpha }+\frac{\beta  \sqrt{4 \alpha +\beta ^2-2 \beta +1}}{\alpha }-\frac{\beta }{\alpha }\right)),
\end{aligned}
```

下面我们计算在经典参数 $\alpha=1.4,\beta=0.3$ 下这两个不动点的特征值:

```@example smooth_one
using StaticArrays, LinearAlgebra
function fixedpoints(p)
    a , b = p
    x1 = (-sqrt(4 * a + b^2 - 2 * b + 1) + b - 1) / (2 * a)
    y1 = (1 / 2) * (b^2 / a - b * sqrt(4 * a + b^2 - 2 * b + 1) / a - b / a)
    x2 = (sqrt(4 * a + b^2 - 2 * b + 1) + b - 1) / (2 * a)
    y2 = (1 / 2) * (b^2 / a + b * sqrt(4 * a + b^2 - 2 * b + 1) / a - b / a)
    return SA[x1, y1], SA[x2, y2]
end

function jacobian(x, p)
    a, b = p
    J = @SMatrix [-2 * a * x[1] 1.0; b 0.0]
    return J
end
```

```@repl smooth_one
eigen(jacobian(fixedpoints([1.4, 0.3])[1], [1.4, 0.3]))
```

```@repl smooth_one
eigen(jacobian(fixedpoints([1.4, 0.3])[2], [1.4, 0.3]))
```

可以看到, 在经典参数下, 这两个不动点均是不稳定的. 下面我们考虑使用 InvariantManifolds.jl 这个软件包来计算第二个不动点的不稳定流形分支的一支. 

InvariantManifolds.jl 这个软件包具有类似于很多 Julia 软件包的接口. 首先, 我们需要在 Julia 中加载这个包, 然后定义 Henon 映射:

```@repl smooth_one
using InvariantManifolds
function henonmap(x, p)
    y1 = 1 - p[1] * x[1]^2 + x[2]
    y2 = p[2] * x[1]
    SA[y1, y2]
end
```

由于这个映射在鞍点处的不稳定特征值为:
```@repl smooth_one
eigen(jacobian(fixedpoints([1.4, 0.3])[2], [1.4, 0.3])).values[1]
```
我们需要对这个映射进行两次迭代, 保证延拓时流形不会反向:

```@repl smooth_one
henonmap2(x, p)=henonmap(henonmap(x, p), p)
```

下面我们定义一个计算光滑映射的一维流形的问题:

```@repl smooth_one
para = [1.4, 0.3]
prob = OneDManifoldProblem(henonmap2, para)
```

为了计算流形, 我们需要一段起点在鞍点的一小段局部流形. 通常起点在鞍点, 长度非常小的不稳定特征矢量即可满足要求. InvariantManifolds.jl 提供了一个函数 [`gen_segment`](@ref) 来生成这样的局部流形:

```@example smooth_one
saddle = fixedpoints(para)[2]
unstable_direction = eigen(jacobian(fixedpoints([1.4, 0.3])[2], [1.4, 0.3])).vectors[:,1]
segment = gen_segment(saddle, unstable_direction)
```

在默认的关键字参数下, 这个函数会生成一个起点在鞍点, 长度为 $150$ 个单位, 步长为 $0.01$ 的局部流形. 下面我们使用这个局部流形来计算光滑流形:

```@repl smooth_one
manifold = growmanifold(prob, segment, 8)
```


这个软件包不提供绘图功能. 但是由于流形的计算结果保存在 `manifold.data` 中, 而 `manifold.data` 实际上是一个向量, 其元素是软件包 [DataInterpolations.jl](https://github.com/SciML/DataInterpolations.jl) 中的插值函数:
```@repl smooth_one
manifold.data
```

因此, 我们可以定义如下函数来绘制光滑流形:

```@example smooth_one
using CairoMakie
function manifold_plot(data)
    figure = Figure()
    axes = Axis(figure[1,1])
    for k in eachindex(data)
        points = data[k].u
        lines!(axes, first.(points), last.(points))
    end
    figure
end
manifold_plot(manifold.data)
```

完整代码:
```julia
using StaticArrays, LinearAlgebra, InvariantManifolds, CairoMakie
function fixedpoints(p)
    a , b = p
    x1 = (-sqrt(4 * a + b^2 - 2 * b + 1) + b - 1) / (2 * a)
    y1 = (1 / 2) * (b^2 / a - b * sqrt(4 * a + b^2 - 2 * b + 1) / a - b / a)
    x2 = (sqrt(4 * a + b^2 - 2 * b + 1) + b - 1) / (2 * a)
    y2 = (1 / 2) * (b^2 / a + b * sqrt(4 * a + b^2 - 2 * b + 1) / a - b / a)
    return SA[x1, y1], SA[x2, y2]
end
function jacobian(x, p)
    a, b = p
    J = @SMatrix [-2 * a * x[1] 1.0; b 0.0]
    return J
end
function henonmap(x, p)
    y1 = 1 - p[1] * x[1]^2 + x[2]
    y2 = p[2] * x[1]
    SA[y1, y2]
end
function henonmap2(x, p)
    henonmap(henonmap(x, p), p)
end
para = [1.4, 0.3]
prob = OneDManifoldProblem(henonmap2, para)
saddle = fixedpoints(para)[2]
unstable_direction = eigen(jacobian(fixedpoints([1.4, 0.3])[2], [1.4, 0.3])).vectors[:,1]
segment = gen_segment(saddle, unstable_direction)
manifold = growmanifold(prob, segment, 8)
function manifold_plot(data)
    figure = Figure()
    axes = Axis(figure[1,1])
    for k in eachindex(data)
        points = data[k].u
        lines!(axes, first.(points), last.(points))
    end
    figure
end
manifold_plot(manifold.data)
```

## 受到周期激励扰动的振子

下面我们考虑一个更高阶的例子. 考虑如下的受到周期激励扰动的振子:
```math
\begin{aligned}
\dot{x}&=y,\\
\dot{y}&=x-\delta x^3+\gamma \cos(\omega t).
\end{aligned}
```

当 $\gamma=0$ 时, 系统在 $(0,0)$ 处有一个鞍点. 在小的周期扰动之后, 这个鞍点变成了一个鞍周期轨道, 即映射 $T:X\mapsto \phi(X,2\pi/\omega,0)$ 的鞍点, 其中 $\phi(X,t,t_0)$ 是系统在初始条件 $X(t_0)=X\in\mathbb{R}^2$ 下的解. 幸运的是, 我们可以利用变分方程的解来获得映射 $T$ 的雅可比矩阵. 映射 $T$ 的鞍点位置和不稳定方向也可以通过数值方法获得.

InvariantManifolds.jl 提供了一个函数 [`findsaddle`](@ref) 来获得 $T$ 的鞍点位置和不稳定方向. 下面我们给出代码以示意如何使用这个函数:

```@example smooth_one
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie
f(x, p, t) = SA[x[2], x[1] - p[1]*(x[1]^3) + p[2]*cos(p[3]*t)]
df(x, p, t) = SA[0.0 1.0; 1-p[1]*3*(x[1]^2) 0.0]
initial_guess = SA[0.0, 0.0]
para = [1.0, 0.1, 2.2]
timespan = (0.0, 2pi/para[3])
saddle = findsaddle(f, df, timespan, initial_guess, para)
```

函数 `gen_segment` 可以直接作用于结构体 [`Saddle`](@ref). 因此我们可以使用如下代码来生成一个局部流形:
```@repl smooth_one
segment = gen_segment(saddle)
```

现在我们可以定义非线性映射:
```@repl smooth_one
function timeTmap(x, p)
    prob = ODEProblem{false}(f, x, (0.0, 2pi/p[3]), p)
    solve(prob, Vern9(), abstol=1e-10)[end]
end
```

接着创建问题, 求解:
```@repl smooth_one
prob = OneDManifoldProblem(timeTmap, para)
manifold = growmanifold(prob, segment, 7)
```
最后再利用上一小节定义的函数来绘制结果:
```@example smooth_one
manifold_plot(manifold.data)
```

完整代码:
```julia
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie
f(x, p, t) = SA[x[2], x[1] - p[1]*(x[1]^3) + p[2]*cos(p[3]*t)]
df(x, p, t) = SA[0.0 1.0; 1-p[1]*3*(x[1]^2) 0.0]
initial_guess = SA[0.0, 0.0]
para = [1.0, 0.1, 2.2]
timespan = (0.0, 2pi/para[3])
saddle = findsaddle(f, df, timespan, initial_guess, para)
segment = gen_segment(saddle)
function timeTmap(x, p)
    prob = ODEProblem{false}(f, x, (0.0, 2pi/p[3]), p)
    solve(prob, Vern9(), abstol=1e-10)[end]
end
prob = OneDManifoldProblem(timeTmap, para)
manifold = growmanifold(prob, segment, 7)
function manifold_plot(data)
    figure = Figure()
    axes = Axis(figure[1,1])
    for k in eachindex(data)
        points = data[k].u
        lines!(axes, first.(points), last.(points))
    end
    figure
end
manifold_plot(manifold.data)
```