using Statistics, Distributions, LinearAlgebra
# This tests assemble_J_projected: instead of grounding an arbitrary pinned
# dof, the singular L is bordered with the "mean over electrodes" functional
# w. j/grad_j (single RHS) solve the resulting well-posed system directly;
# J/grad_J (matrix RHS) use Krylov's block_minres (there's no block-CG in
# Krylov.jl, and CG wouldn't apply anyway since L is only positive
# semi-definite -- block_minres is the symmetric-indefinite/singular
# counterpart). Validated the same way as 02/03: central finite difference
# along a random direction, no AD backend involved.

@testset "Projected boundary test" begin
    @testset "σ in the same space as u" begin
        σ = rand(Uniform(1e-2, 1.0), n)

        g = randn(m)
        g .-= Statistics.mean(g)  # charge conservation: sum(g)=0
        f = randn(m)

        k = 5
        G = randn(m, k)
        G .-= Statistics.mean(G, dims=1)
        F = randn(m, k)

        j, J, grad_j, grad_J = assemble_J_projected(cellvalues, dh, m, tb, ti)

        obj_test = j(σ, f, g)
        @test obj_test >= 0
        obj_vec = J(σ, F, G)
        @test all(obj_vec .>= 0)

        ε = 1e-5

        gr = grad_j(σ, f, g)
        @test length(gr) == n
        d = randn(n)
        d ./= norm(d)
        fd = (j(σ .+ ε .* d, f, g) - j(σ .- ε .* d, f, g)) / (2ε)
        @test isapprox(fd, dot(gr, d); rtol=1e-3, atol=1e-6)

        grJ = grad_J(σ, F, G)
        @test length(grJ) == n
        dJ = randn(n)
        dJ ./= norm(dJ)
        fdJ = (J(σ .+ ε .* dJ, F, G) - J(σ .- ε .* dJ, F, G)) / (2ε)
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

        σ = rand(Uniform(1e-2, 1.0), n_σ)

        g = randn(m)
        g .-= Statistics.mean(g)
        f = randn(m)

        k = 5
        G = randn(m, k)
        G .-= Statistics.mean(G, dims=1)
        F = randn(m, k)

        j, J, grad_j, grad_J = assemble_J_projected(cellvalues, dh, cellvalues_σ, dh_σ, m, tb, ti)

        ε = 1e-5

        gr = grad_j(σ, f, g)
        @test length(gr) == n_σ
        d = randn(n_σ)
        d ./= norm(d)
        fd = (j(σ .+ ε .* d, f, g) - j(σ .- ε .* d, f, g)) / (2ε)
        @test isapprox(fd, dot(gr, d); rtol=1e-3, atol=1e-6)

        grJ = grad_J(σ, F, G)
        @test length(grJ) == n_σ
        dJ = randn(n_σ)
        dJ ./= norm(dJ)
        fdJ = (J(σ .+ ε .* dJ, F, G) - J(σ .- ε .* dJ, F, G)) / (2ε)
        @test isapprox(fdJ, dot(grJ, dJ); rtol=1e-3, atol=1e-6)
    end
end
