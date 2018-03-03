#
# tests for online statistics
#

@testset "OnlineStatistics" begin

@testset "MeanVariance" begin
    @testset "1" begin
        x = rand(1, 1000)
        mv = MeanVariance(1)
        for i in 1:5:500
            update!(mv, x[:, i:(i+4)])
        end
        update!(mv, x[:,501:750])
        for i in 751:1000
            update!(mv, x[:, i])
        end

        @test isapprox([mean(mv), var(mv), std(mv)],
                first.([mean(x), var(x), std(x)]), 1e-14)
    end

    @testset "2" begin
        d = 5
        x = rand(d, 1000)
        mv = MeanVariance(d)

        @test all(isnan.(mean(mv)))
        @test all(isnan.(var(mv)))
        @test all(isnan.(cov(mv)))

        for i in 1:5:500
            update!(mv, x[:, i:(i+4)])
        end
        update!(mv, x[:, 501:750])
        for i in 751:1000
            update!(mv, x[:, i])
        end

        @test isapprox(mean(mv), mean(x, 2), 1e-14)
        @test isapprox(var(mv), var(x, 2), 1e-14)
        @test isapprox(cov(mv), cov(x, 2), 1e-14)
    end

    @testset "3" begin
        d = 5
        x = rand(d, 100)
        mv = MeanVariance(d)
        for i in 1:2:100
            update!(mv, x[:, i:(i+1)])
        end

        @test isapprox(mean(mv), mean(x, 2), 1e-14)
        @test isapprox(var(mv), var(x, 2), 1e-14)
        @test isapprox(cov(mv), cov(x, 2), 1e-14)
    end
end

@testset "Diagnostics" begin
    N = 1000
    ws = abs.(randn(N) ./ randn(N))
    d = Diagnostic()
    for i in 1:5:500
        update!(d, ws[i:(i+4)])
    end
    update!(d, ws[501:750])
    for i in 751:1000
        update!(d, ws[i])
    end
    @test isapprox([ne(d), neσ(d), neγ(d)],
        [sum(ws)^2/(sum(abs2, ws)),
        sum(abs2, ws)^2/(sum(ws.^4)),
        sum(abs2, ws)^3/(sum(ws.^3)^2)], 1e-15)
end

@testset "ControlVariates" begin
    @testset "1" begin
        Dg = 4
        Df = 2
        Q = rand(Dg, Dg)
        g(x::Vector) = Q*x - sin.(x)

        N = 1000
        xs = rand(Dg, N)*16
        gs = mapslices(g, xs, 1)
        β = rand(Dg, Df)
        fs = β'*gs

        C = cov(gs, fs, 2)

        cv = ControlVariate(Df, Dg)

        for i in 1:10:500
            update!(cv, fs[:, i:(i+9)], gs[:, i:(i+9)])
        end
        for i in 501:750
            update!(cv, fs[:, i], gs[:, i])
        end
        update!(cv, fs[:,751:1000], gs[:,751:1000])

        @test isapprox(coeffs(cv), β, 1e-14)
        @test isapprox(cov(cv), C, 1e-14)
    end

    @testset "2" begin
        f(x) = 5*sin(x)+sin(3*x)-2*sin(6*x)
        g(x) = [sin(x), sin(3*x), sin(6*x), sin(12*x)]
        xs = rand(1, 1000) * 16
        fs = f.(xs)
        gs = hcat(map(g, xs)...)

        C = cov(gs, fs, 2)
        β = cov(gs, 2)\C

        cv = ControlVariate(1, 4)

        for i in 1:10:500
            update!(cv, fs[:, i:(i+9)], gs[:, i:(i+9)])
        end
        for i in 501:750
            update!(cv, fs[:, i], gs[:, i])
        end
        update!(cv, fs[:,751:1000], gs[:,751:1000])

        @test isapprox(coeffs(cv), β, 1e-14)
        @test isapprox(cov(cv), C, 1e-14)
    end

    @testset "3" begin
        N = 25
        xs = rand(1, N)*10
        fs = 2*xs+16-xs.^2
        gs = vcat(xs, xs.^2)
        β = cov(gs, 2)\cov(gs, fs, 2)

        cv = ControlVariate(1, 2)
        j = 4
        for i in 1:(j+1):N
            update!(cv, fs[:, i:(i+j)], gs[:, i:(i+j)])
        end

        @test isapprox(coeffs(cv), β, 1e-14)
    end
end

end
