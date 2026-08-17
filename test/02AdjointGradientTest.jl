using Statistics, Distributions, LinearAlgebra
# This test validates the analytic adjoint-method gradients (grad_j, grad_J)
# against a central finite difference of j/J along a random direction. This
# is a self-contained correctness check of the adjoint implementation itself
# and does not depend on any AD backend -- it's what AD-backend maintainers
# can compare their own gradient(j, ...) results against.

function directional_fd(f, x, d, args...; ε=1e-5)
    return (f(x .+ ε .* d, args...) - f(x .- ε .* d, args...)) / (2ε)
end

@testset "Adjoint gradient test" begin
    @testset "σ in the same space as u" begin
        σ = rand(Uniform(1e-2, 1.0), n)
        g = randn(n)
        g .-= Statistics.mean(g)
        f = randn(n)

        j, J, grad_j, grad_J = assemble_J(cellvalues, dh)

        gr = grad_j(σ, f, g)
        @test length(gr) == n

        d = randn(n)
        d ./= norm(d)
        fd = directional_fd(j, σ, d, f, g)
        @test isapprox(fd, dot(gr, d); rtol=1e-3, atol=1e-6)

        k = 5
        G = randn(n, k)
        G .-= Statistics.mean(G, dims=1)
        F = randn(n, k)

        grJ = grad_J(σ, F, G)
        @test length(grJ) == n

        dJ = randn(n)
        dJ ./= norm(dJ)
        fdJ = directional_fd(J, σ, dJ, F, G)
        @test isapprox(fdJ, dot(grJ, dJ); rtol=1e-3, atol=1e-6)
    end

    @testset "σ in its own P0 space" begin
        ip_σ = DiscontinuousLagrange{RefQuadrilateral,0}()
        qr_σ = QuadratureRule{RefQuadrilateral}(qr_order)
        cellvalues_σ = CellValues(qr_σ, ip_σ)
        dh_σ = DofHandler(grid)
        add!(dh_σ, :σ, ip_σ)
        close!(dh_σ)
        n_σ = ndofs(dh_σ)
        @test n_σ == ncells

        σ = rand(Uniform(1e-2, 1.0), n_σ)
        g = randn(n)
        g .-= Statistics.mean(g)
        f = randn(n)

        j, J, grad_j, grad_J = assemble_J(cellvalues, dh, cellvalues_σ, dh_σ)

        gr = grad_j(σ, f, g)
        @test length(gr) == n_σ

        d = randn(n_σ)
        d ./= norm(d)
        fd = directional_fd(j, σ, d, f, g)
        @test isapprox(fd, dot(gr, d); rtol=1e-3, atol=1e-6)
    end
end
