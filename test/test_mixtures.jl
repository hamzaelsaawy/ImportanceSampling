#
# tests for mixture distributions
#

@testset "Mixtures" begin
    using Distributions

    @testset "errors" begin
        @test_throws MethodError MixtureDistribution([Normal(), Poisson()])
        @test_throws MethodError MixtureDistribution([Normal(), MvNormal(1,1)])
        @test_throws MethodError MixtureDistribution([Wishart(3, eye(2)), MvNormal(1,1)])
    end

    @testset "1" begin
        # if it passes randomly, oh well :/
        wis = Wishart(20, eye(2))
        x = eye(2)*3 + randn(2,2)
        x = x*x'

        @test isapprox(pdf(MixtureDistribution([wis, wis], [2, 20]), x), pdf(wis, x))
        @test isapprox(logpdf(MixtureDistribution([wis, wis], [2, 20]), x), logpdf(wis, x))
    end

    @testset "2" begin
        mvn = MvNormal(eye(2))
        x = randn(2,6)

        @test isapprox(pdf(MixtureDistribution([mvn, mvn], [2, 20]), x), pdf(mvn, x))
        @test isapprox(logpdf(MixtureDistribution([mvn, mvn], [2, 20]), x), logpdf(mvn, x))
    end
end
