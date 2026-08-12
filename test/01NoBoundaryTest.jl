using Statistics, Distributions
using DifferentiationInterface
using ForwardDiff: ForwardDiff
using Enzyme: Enzyme
using Zygote: Zygote
using Mooncake: Mooncake
# This is the actual test which tests autodifferentiation:
@testset "No boundary Differentiation test" begin
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

    j, J = assemble_J(cellvalues, dh)

    # This is just to test whether the assembly works:
    obj_test = j(σ, f, g)
    @test obj_test >= 0
    obj_vec = J(σ, F, G)
    @test all(obj_vec .>= 0)


    # Now the Differentiation tests:

    # I hope this is the correct way of doing this
    function make_grad_j(j_func, backend, σ_init, f_init, g_init)
        prep = prepare_gradient(j_func, backend, σ_init, Constant(f_init), Constant(g_init))
        return (σ, f, g) -> DifferentiationInterface.gradient(j_func, prep, backend, σ, Constant(f), Constant(g))
    end

    backends = (
            "Enzyme"   => AutoEnzyme(),
            "Mooncake" => AutoMooncake(),
            "Zygote"   => AutoZygote(),
        )

    # now the actual testset
    @testset "$name" for (name, backend) in backends
        grad_j = make_grad_j(j, backend, σ, f, g)

        test_j = grad_j(σ, f, g)
        @test length(test_j) == n

        # Evaluate with completely new inputs:
        σ_new = rand(Uniform(1e-6, 1.0), n)
        f_new = randn(n)
        g_new = randn(n) .- Statistics.mean(randn(n))

        test_j_new = grad_j(σ_new, f_new, g_new)
        @test length(test_j_new) == n
    end


    # If that works then I would do the same testset again for the Array input of F, G matrices

end
