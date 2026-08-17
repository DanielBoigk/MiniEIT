using Statistics, Distributions
# This test checks that assembly and objective evaluation work, with σ, f, g
# all living in the same FE space as u.
@testset "No boundary assembly test" begin
    σ = rand(Uniform(1e-6, 1.0), n)

    # Try all this with Vectors
    g = randn(n)
    g .-= Statistics.mean(g)          # necessary - need zero mean RHS
    f = randn(n)
    f .-= Statistics.mean(f)          # not necessary


    # Try all this with Arrays:
    k = 10
    G = randn(n, k)
    G .-= Statistics.mean(G, dims=1)  # necessary `
    F = randn(n, k)
    F .-= Statistics.mean(F, dims=1)  # not necessary

    j, J, grad_j, grad_J = assemble_J(cellvalues, dh)

    # This is just to test whether the assembly works:
    obj_test = j(σ, f, g)
    @test obj_test >= 0
    obj_vec = J(σ, F, G)
    @test all(obj_vec .>= 0)
end

# --------------------------------------------------------------------------
# AD-backend differentiability MWE.
#
# This is kept as a minimal working example for Enzyme/Mooncake/Zygote
# maintainers to test against, but is NOT run as part of this package's own
# test suite: as of writing, Enzyme throws an EnzymeMutabilityException,
# Mooncake segfaults the whole process, and Zygote errors inside Ferrite's
# SIMD-based `reinit!` (all reproduce on the plain `j` closure above,
# independent of anything specific to this package). Whether/when these
# backends support this code path is up to their maintainers, not something
# this package's test suite should depend on. See test/02AdjointGradientTest.jl
# for an analytic (adjoint-method) gradient that doesn't depend on any AD
# backend.
#
# using DifferentiationInterface
# using ForwardDiff: ForwardDiff
# using Enzyme: Enzyme
# using Zygote: Zygote
# using Mooncake: Mooncake
#
# @testset "No boundary Differentiation test" begin
#     σ = rand(Uniform(1e-6, 1.0), n)
#     g = randn(n)
#     g .-= Statistics.mean(g)
#     f = randn(n)
#     f .-= Statistics.mean(f)
#
#     j, J, grad_j, grad_J = assemble_J(cellvalues, dh)
#
#     function make_grad_j(j_func, backend, σ_init, f_init, g_init)
#         prep = prepare_gradient(j_func, backend, σ_init, Constant(f_init), Constant(g_init))
#         return (σ, f, g) -> DifferentiationInterface.gradient(j_func, prep, backend, σ, Constant(f), Constant(g))
#     end
#
#     backends = (
#             "Enzyme"   => AutoEnzyme(),
#             "Mooncake" => AutoMooncake(),
#             "Zygote"   => AutoZygote(),
#         )
#
#     @testset "$name" for (name, backend) in backends
#         grad_j_ad = make_grad_j(j, backend, σ, f, g)
#         test_j = grad_j_ad(σ, f, g)
#         @test length(test_j) == n
#     end
# end
