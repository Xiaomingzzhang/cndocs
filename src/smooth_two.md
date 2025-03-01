# 光滑两维流形

在本质上, 我们并没有引入新的算法, 计算二维流形的核心函数跟一维流形的核心函数是一样的. 我们只是将二维流形表示为离得足够近的一维的圆圈.

```@setup lorenz
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie
```
## 自治向量场: Lorenz 流形
首先加载需要的包, 并定义 Lorenz 向量场:
```@repl lorenz
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie
function lorenz(x, p, t)
    σ, ρ, β = p
    v = SA[σ*(x[2]-x[1]),
        ρ*x[1]-x[2]-x[1]*x[3],
        x[1]*x[2]-β*x[3]
    ]
    v / sqrt(0.1 + norm(v)^2)
end
```

值得注意的是, 我们对向量场进行了一个近似的归一化, 使得向量场的模长保持在一个很小的范围内. 这样可以保证流形的扩张是均匀的. 在经典参数下:
```@repl lorenz
para = [10.0, 28.0, 8/3]
```
原点这个平衡点的 Jacobi 矩阵具有两个稳定方向, 这个稳定方向分别是:
```@example lorenz
function eigenv(p)
    σ, ρ, β = p
    [SA[0.0, 0.0, 1.0], SA[-(-1 + σ + sqrt(1 - 2 * σ + 4 * ρ * σ + σ^2))/(2*ρ), 1, 0]]
end
eigenv(para)
```
那么我们可以创建一个 `Saddle` 结构体来存储这个鞍点:
```@repl lorenz
saddle = Saddle(SA[0, 0, 0.0], eigenv(para), [1.0, 1.0])
```
这里特征值的大小可以随意指定, 不会影响计算结果. 由于计算的是稳定流形, 我们需要将流进行反向演化. 定义如下映射:
```@repl lorenz
function lorenz_map(x, p)
    prob = ODEProblem{false}(lorenz, x, (0.0, -1.0), p)
    sol = solve(prob, Vern9(), abstol = 1e-10)
    sol[end]
end
```

现在我们可以创建问题:
```@repl lorenz
prob = VTwoDManifoldProblem(lorenz_map, para, d=1.0, amax=1.0, dsmin=1e-3)
```
这些关键字参数的含义可参考 [`VTwoDManifoldProblem`](@ref).

与一维流形类似, 同样需要一个局部流形才能进行延拓. 相应的创建局部流形的函数为 [`gen_disk`](@ref):
```@repl lorenz
disk = gen_disk(saddle, r=1.0)
```
关于函数 `gen_disk` 的详细介绍请参考 [`gen_disk`](@ref). 现在我们可以进行延拓了:
```@repl lorenz
manifold = growmanifold(prob, disk, 200)
```

我们同样可以定义一个绘图函数来进行绘制结果:
```@example lorenz
using CairoMakie
function manifold_plot(annulus)
    fig = Figure()
    axes = LScene(fig[1, 1], show_axis=false, scenekw=(backgroundcolor=:white, clear=true))
    second(x) = x[2]
    for i in eachindex(annulus)
        points = annulus[i].u
        lines!(axes, first.(points), second.(points), last.(points), fxaa=true)
    end
    fig
end
manifold_plot(manifold.data)
```


## 非线性映射

考虑如下的非线性映射:

```math
f(X)=\varphi\circ\Lambda\circ\varphi^{-1}(X)
```
其中 $\varphi(x,y,z)=(x,y,z-\alpha x^2-\beta y^2)$ 是一个非线性映射, $\Lambda$ 是一个对角矩阵, 其对角元可用来控制映射 $f$ 在原点附近的 Jacobi 矩阵. 我们直接给出计算其不变流形的代码:
```@example nonlinearmap
using InvariantManifolds, LinearAlgebra, StaticArrays, OrdinaryDiffEq, CairoMakie
Λ = SDiagonal(SA[2.1, 6.3, 0.6])
φ(x, p)= SA[x[1],x[2],x[3]-p[1]*x[1]^2-p[2]*x[2]^2]
iφ(x, p)= SA[x[1],x[2],x[3]+p[1]*x[1]^2+p[2]*x[2]^2]
f(x,p) = φ(Λ*iφ(x, p),p)

para = [1.2,-1.2]
saddle = Saddle(SA[0.0, 0.0, 0.0], [SA[1.0, 0.0, 0.0], SA[0.0, 1.0, 0.0]], [2.1, 6.3])
prob = TwoDManifoldProblem(f, para, dcircle=0.05, d = 0.02, dsmin=1e-3)

disk = gen_disk(saddle, times=4, r= 0.05)
manifold = growmanifold(prob, disk, 3)
function manifold_plot(data)
    fig = Figure()
    axes = Axis3(fig[1,1])
    second(x) = x[2]
    for k in eachindex(data)
        for j in eachindex(data[k])
            points=data[k][j].u
            scatter!(axes,first.(points),second.(points),last.(points),fxaa=true)
        end
    end
    fig
end
manifold_plot(manifold.data)
```