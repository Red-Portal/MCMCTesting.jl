
using Distributions
using Random
using Test

using MCMCTesting

include("models/normalnormalgibbs.jl")

@testset "normal-normal gibbs" begin
    model = Model(sqrt(10), sqrt(0.1))

    conf_level = 0.01
    n_samples  = 1000
    @testset for test in [TwoSampleTest(n_samples, n_samples),
                          TwoSampleGibbsTest(n_samples, n_samples),
                          ExactRankTest(n_samples, n_samples),]

        @testset "mcmctest" begin
            pvalue = mcmctest(test, TestSubject(model, GibbsRandScan()); show_progress=false)
            @test eltype(pvalue) <: Real
            @test all(@. 0 ≤ pvalue ≤ 1)
        end

        @testset "seqmcmctest correct" begin
            test_res = seqmcmctest(test, TestSubject(model, GibbsRandScan()), 0.001, 32; show_progress=false)
            @test test_res isa Bool
            @test test_res
        end

        @testset "seqmcmctest wrong mean" begin
            test_res = seqmcmctest(test, TestSubject(model, GibbsRandScanWrongMean()), 0.001, 32; show_progress=false)
            @test test_res isa Bool
            @test !test_res
        end

        @testset "seqmcmctest wrong var" begin
            test_res = seqmcmctest(test, TestSubject(model, GibbsRandScanWrongVar()), 0.001, 32; show_progress=false)
            @test test_res isa Bool
            @test !test_res
        end
    end
end
