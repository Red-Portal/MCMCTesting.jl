
using Distributions
using Random
using Test
using StableRNGs

using MCMCTesting

include("models/normalnormalgibbs.jl")

@testset "interfaces" begin
    model     = Model(sqrt(10), sqrt(0.1))
    n_samples = 100
    mcmc      = GibbsRandScan()
    subject   = TestSubject(model, mcmc)

    n_dim_param = 2
    n_dim_data  = 1

    @testset for test in [
        TwoSampleTest(n_samples, n_samples),
        TwoSampleGibbsTest(n_samples, n_samples),
        ExactRankTest(n_samples, n_samples)
    ]
        n_dim_default = if test isa TwoSampleTest || test isa TwoSampleGibbsTest
            (n_dim_param + n_dim_data)*2
        else
            n_dim_param*2
        end

        @testset "mcmctest" begin
            @testset "return type" begin
                pvalue = mcmctest(test, subject; show_progress=false)
                @test eltype(pvalue) <: Real
                @test length(pvalue) == n_dim_default
            end

            @testset "custom statistics" begin
                pvalue = mcmctest(test, subject; show_progress=false, statistics = θ -> θ)
                @test length(pvalue) == div(n_dim_default,2)
            end
        
            @testset "determinism" begin
                rng    = StableRNG(1)
                pvalue = mcmctest(rng, test, subject; show_progress=false)

                rng        = StableRNG(1)
                pvalue_rep = mcmctest(rng, test, subject; show_progress=false)
                @test pvalue == pvalue_rep
            end
        end

        @testset "seqmcmctest" begin
            @testset "return type" begin
                res = seqmcmctest(test, subject, 0.001, 32; show_progress=false)
                @test res isa Bool
            end
            
            @testset "custom statistics" begin
                seqmcmctest(test, subject, 0.001, 32; show_progress=false, statistics = θ -> θ)

                rng = StableRNG(1)
                seqmcmctest(rng, test, subject, 0.001, 32; show_progress=false, statistics = θ -> θ)
            end

            @testset "determinism" begin
                rng = StableRNG(1)
                res = seqmcmctest(rng, test, subject, 0.001, 32; show_progress=false)

                rng     = StableRNG(1)
                res_rep = seqmcmctest(rng, test, subject, 0.001, 32; show_progress=false)
                @test res == res_rep
            end
        end
    end

    @testset "rank simulation" begin
        test  = ExactRankTest(n_samples, n_samples)
        ranks = simulate_ranks(test, subject; show_progress=false)
        @test all(@. 1 ≤ ranks ≤ n_samples)
        @test eltype(ranks) <: Integer
        @test size(ranks) == (2*2, n_samples)
    end
end

@testset "normal-normal gibbs" begin
    model = Model(sqrt(10), sqrt(0.1))

    conf_level = 0.01
    n_samples  = 1000
    @testset for test in [TwoSampleTest(n_samples, n_samples),
                          TwoSampleGibbsTest(n_samples, n_samples),
                          ExactRankTest(n_samples, n_samples),]

        @testset "mcmctest" begin
            pvalue = mcmctest(test, TestSubject(model, GibbsRandScan()); show_progress=false)
            @test all(@. 0 ≤ pvalue ≤ 1)
        end

        @testset "seqmcmctest correct" begin
            test_res = seqmcmctest(test, TestSubject(model, GibbsRandScan()), 0.001, 32; show_progress=false)
            @test test_res
        end

        @testset "seqmcmctest wrong mean" begin
            test_res = seqmcmctest(test, TestSubject(model, GibbsRandScanWrongMean()), 0.001, 32; show_progress=false)
            @test !test_res
        end

        @testset "seqmcmctest wrong var" begin
            test_res = seqmcmctest(test, TestSubject(model, GibbsRandScanWrongVar()), 0.001, 32; show_progress=false)
            @test !test_res
        end
    end
end

