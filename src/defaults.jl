
function default_two_sample_test_pvalue(x, y)
    @error("The default two-sample test only supports real vectors. " *
           "Please supply your own two_sample_test_pvalue function.")
end

function default_two_sample_test_pvalue(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    HypothesisTests.pvalue(
        HypothesisTests.ApproximateTwoSampleKSTest(x, y)
    )
end

function default_uniformity_test_pvalue(x::AbstractVector{<:Integer})
    HypothesisTests.pvalue(
        HypothesisTests.ChisqTest(x)
    )
end

function default_statistics(x::AbstractVector{<:Real})
    vcat(x, x.^2)
end
