# 非光滑两维流形

下面我们将继续以 Lorenz 系统为例, 不同的是我们将对这个系统引入一个人为的非光滑因素. 首先加载需要的包:

```@setup non_smooth_two
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie, DataInterpolations
```

```@repl non_smooth_two
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie, DataInterpolations
```

接着定义在远离原点时归一化的向量场:
```@repl non_smooth_two
function lorenz(x, p, t)
    σ, ρ, β = p
    v = SA[σ*(x[2]-x[1]),
        ρ*x[1]-x[2]-x[1]*x[3],
        x[1]*x[2]-β*x[3]
    ]
    v / sqrt(0.1 + norm(v)^2)
end
```
与先前的例子不同的是, 我们将引入如下非光滑因素: 当 $z=\xi$ 时, $(x,y,z)\rightarrow(x,-y,z)$. 
因此, 我们需要定义一个带有碰撞因素的向量场:
```@example non_smooth_two
hyper(x,p,t) = x[3]-p[4]
rule(x,p,t) = SA[x[1], -x[2], x[3]]
vectorfield =BilliardV(lorenz, (hyper,),(rule,))
```

接着如同前面的例子, 我们需要将求解这个微分方程的具体信息封装到 [`NSSetUp`](@ref) 中, 这需要使用函数 `setmap`:
```@repl non_smooth_two
setup = setmap(vectorfield, (0.0, -1.0), Tsit5(), abstol=1e-8)
```

接着同样需要生成一个局部流形:
```@example non_smooth_two
para = [10.0, 28.0, 8/3, 10.0]
function eigenv(p)
    σ, ρ, β = p
    [SA[0.0, 0.0, 1.0], SA[-(-1 + σ + sqrt(1 - 2 * σ + 4 * ρ * σ + σ^2))/(2*ρ), 1, 0]]
end
saddle = Saddle(SA[0, 0, 0.0], eigenv(para), [1.0, 1.0])
disk = gen_disk(saddle, r=1.0, d=0.1)
```
接着创建问题:
```@repl non_smooth_two
prob = NSVTwoDManifoldProblem(setup, para, amax=0.5, d=0.5, ϵ=0.2, dsmin=1e-3)
```
最后计算流形, 并绘制图像:
```@example non_smooth_two
manifold = growmanifold(prob, disk, 90, interp=LinearInterpolation)
function manifold_plot(annulus)
    fig = Figure()
    axes = LScene(fig[1, 1], show_axis=false, scenekw=(backgroundcolor=:white, clear=true))
    second(x) = x[2]
    for i in eachindex(annulus)
        for j in eachindex(annulus[i])
            points = annulus[i][j].u
            lines!(axes, first.(points), second.(points), last.(points), fxaa=true)
        end
    end
    fig
end
manifold_plot(manifold.data)
```

完整代码:
```julia
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie, DataInterpolations
function lorenz(x, p, t)
    σ, ρ, β = p
    v = SA[σ*(x[2]-x[1]),
        ρ*x[1]-x[2]-x[1]*x[3],
        x[1]*x[2]-β*x[3]
    ]
    v / sqrt(0.1 + norm(v)^2)
end
hyper(x,p,t) = x[3]-p[4]
rule(x,p,t) = SA[x[1], -x[2], x[3]]
vectorfield =BilliardV(lorenz, (hyper,),(rule,))
setup = setmap(vectorfield, (0.0, -1.0), Tsit5(), abstol=1e-8)
para = [10.0, 28.0, 8/3, 10.0]
function eigenv(p)
    σ, ρ, β = p
    [SA[0.0, 0.0, 1.0], SA[-(-1 + σ + sqrt(1 - 2 * σ + 4 * ρ * σ + σ^2))/(2*ρ), 1, 0]]
end
saddle = Saddle(SA[0, 0, 0.0], eigenv(para), [1.0, 1.0])
disk = gen_disk(saddle, r=1.0, d=0.1)
prob = NSVTwoDManifoldProblem(setup, para, amax=0.5, d=0.5, ϵ=0.2, dsmin=1e-3)
manifold = growmanifold(prob, disk, 90, interp=LinearInterpolation)
function manifold_plot(annulus)
    fig = Figure()
    axes = LScene(fig[1, 1], show_axis=false, scenekw=(backgroundcolor=:white, clear=true))
    second(x) = x[2]
    for i in eachindex(annulus)
        for j in eachindex(annulus[i])
            points = annulus[i][j].u
            lines!(axes, first.(points), second.(points), last.(points), fxaa=true)
        end
    end
    fig
end
manifold_plot(manifold.data)
```