using Statistics, Distributions, LinearAlgebra
# This tests assemble_J_projected: instead of grounding an arbitrary pinned
# dof, the singular L is bordered with the "mean over electrodes" functional
# w and solved via Krylov's block_minres (a block-CG doesn't exist in
# Krylov.jl, and wouldn't apply anyway since L is only positive
# semi-definite -- block_minres is the symmetric-indefinite/singular
# counterpart). Validated the same way as 02/03: central finite difference
# of J along a random direction, no AD backend involved.

@testset "Projected boundary test (block_minres)" begin
    @testset "σ in the same space as u" begin
        σ = rand(Uniform(1e-2, 1.0), n)

        k = 5
        G = randn(m, k)
        G .-= Statistics.mean(G, dims=1)  # charge conservation: sum(g)=0 per column
        F = randn(m, k)

        J, grad_J = assemble_J_projected(cellvalues, dh, m, tb, ti)

        obj_vec = J(σ, F, G)
        @test all(obj_vec .>= 0)

        grJ = grad_J(σ, F, G)
        @test length(grJ) == n

        d = randn(n)
        d ./= norm(d)
        ε = 1e-5
        fd = (J(σ .+ ε .* d, F, G) - J(σ .- ε .* d, F, G)) / (2ε)
        @test isapprox(fd, dot(grJ, d); rtol=1e-3, atol=1e-6)
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

        k = 5
        G = randn(m, k)
        G .-= Statistics.mean(G, dims=1)
        F = randn(m, k)

        J, grad_J = assemble_J_projected(cellvalues, dh, cellvalues_σ, dh_σ, m, tb, ti)

        grJ = grad_J(σ, F, G)
        @test length(grJ) == n_σ

        d = randn(n_σ)
        d ./= norm(d)
        ε = 1e-5
        fd = (J(σ .+ ε .* d, F, G) - J(σ .- ε .* d, F, G)) / (2ε)
        @test isapprox(fd, dot(grJ, d); rtol=1e-3, atol=1e-6)
    end
end
