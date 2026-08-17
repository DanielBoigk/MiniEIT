using Statistics, Distributions, LinearAlgebra
# This tests the boundary-restricted objective: current is injected and
# voltage measured only on a subset of electrode dofs (`electrode_dofs`,
# built from `runtests.jl` via boundary_dofs/select_electrodes/boundary_maps),
# rather than on every dof of `dh` like in 01/02. Same finite-difference
# validation approach as 02AdjointGradientTest.jl, no AD backend involved.

@testset "Boundary electrode test" begin
    @testset "σ in the same space as u" begin
        σ = rand(Uniform(1e-2, 1.0), n)
        g = randn(m)
        g .-= Statistics.mean(g)   # balanced injected current (no return path otherwise)
        f = randn(m)

        j, J, grad_j, grad_J = assemble_J(cellvalues, dh, m, tb, ti; pin=pin_dof)

        obj_test = j(σ, f, g)
        @test obj_test >= 0

        gr = grad_j(σ, f, g)
        @test length(gr) == n

        d = randn(n)
        d ./= norm(d)
        ε = 1e-5
        fd = (j(σ .+ ε .* d, f, g) - j(σ .- ε .* d, f, g)) / (2ε)
        @test isapprox(fd, dot(gr, d); rtol=1e-3, atol=1e-6)

        k = 5
        G = randn(m, k)
        G .-= Statistics.mean(G, dims=1)
        F = randn(m, k)

        obj_vec = J(σ, F, G)
        @test all(obj_vec .>= 0)

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

        j, J, grad_j, grad_J = assemble_J(cellvalues, dh, cellvalues_σ, dh_σ, m, tb, ti; pin=pin_dof)

        gr = grad_j(σ, f, g)
        @test length(gr) == n_σ

        d = randn(n_σ)
        d ./= norm(d)
        ε = 1e-5
        fd = (j(σ .+ ε .* d, f, g) - j(σ .- ε .* d, f, g)) / (2ε)
        @test isapprox(fd, dot(gr, d); rtol=1e-3, atol=1e-6)
    end
end
