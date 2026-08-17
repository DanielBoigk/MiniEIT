# This file takes the matrix assembler and returns the objective function j(σ,f,g) = ||f- L(σ)^(-1) g||^2,
# together with its adjoint-method gradient w.r.t. σ.
module MiniEIT
    using Ferrite, LinearAlgebra, SparseArrays
    using IterativeSolvers, Krylov

    include("MatrixAssembler.jl")

    export assemble_J

    # Convenience: σ lives in the same FE space as u.
    function assemble_J(cellvalues::CellValues, dh::DofHandler)
        return assemble_J(cellvalues, dh, cellvalues, dh)
    end

    # Assembles the objective j(σ,f,g) = ||f - L(σ)^{-1} g||^2 (and its
    # matrix-RHS counterpart J), together with their adjoint-method gradients
    # w.r.t. σ (grad_j, grad_J). σ lives in its own FE space (cellvalues_σ,
    # dh_σ), which may differ from u's space (cellvalues, dh).
    #
    # grad_j/grad_J are analytic reference gradients (state: Lu=g, adjoint:
    # Lλ=-2(u-f), dj/dσ_k = λ'(∂L/∂σ_k)u, see assemble_gradient!) meant for
    # AD-backend maintainers to validate gradient(j, ...)/gradient(J, ...)
    # against, independent of whether Enzyme/Zygote/Mooncake support this
    # code path.
    function assemble_J(cellvalues::CellValues, dh::DofHandler, cellvalues_σ::CellValues, dh_σ::DofHandler)
        ai = AssemblerInfo(cellvalues, dh, cellvalues_σ, dh_σ)
        n = ndofs(dh)
        n_σ = ndofs(dh_σ)
        u = zeros(n)
        λ = zeros(n)
        rhs = zeros(n)
        grad = zeros(n_σ)
        pin = ai.pin

        # L is grounded (dof `pin` pinned to zero, see assemble_L!/ground!),
        # so every RHS solved against it must have that entry zeroed too.
        j = (σ::AbstractVector, f::AbstractVector, g::AbstractVector) -> begin
            L = assemble_L!(ai, σ)
            rhs .= g
            rhs[pin] = 0
            u .= L \ rhs
            u .-= f
            dot(u, u)
        end

        grad_j = (σ::AbstractVector, f::AbstractVector, g::AbstractVector) -> begin
            L = assemble_L!(ai, σ)
            fact = factorize(L)
            rhs .= g
            rhs[pin] = 0
            u .= fact \ rhs
            r = u .- f
            rhs .= -2 .* r
            rhs[pin] = 0
            λ .= fact \ rhs
            assemble_gradient!(grad, ai, λ, u)
            return copy(grad)
        end

        J = (σ::AbstractVector, F::AbstractArray, G::AbstractArray) -> begin
            L = assemble_L!(ai, σ)
            RHS = copy(G)
            RHS[pin, :] .= 0
            U = L \ RHS
            U .-= F
            dot(U, U)
        end

        grad_J = (σ::AbstractVector, F::AbstractArray, G::AbstractArray) -> begin
            L = assemble_L!(ai, σ)
            fact = factorize(L)
            RHS = copy(G)
            RHS[pin, :] .= 0
            U = fact \ RHS
            R = U .- F
            RHS2 = -2 .* R
            RHS2[pin, :] .= 0
            Λ = fact \ RHS2
            total = zeros(n_σ)
            for col in 1:size(G, 2)
                assemble_gradient!(grad, ai, view(Λ, :, col), view(U, :, col))
                total .+= grad
            end
            return total
        end

        return j, J, grad_j, grad_J
    end

    # This is the same as the unrestricted assemble_J, but current is
    # injected and voltage measured only on a subset of dofs (e.g. discrete
    # electrodes on ∂Ω) rather than everywhere -- see boundary_dofs,
    # select_electrodes, boundary_maps for building `tb`/`ti`.
    # tb: R^n -> R^m, restricts the full state to the electrode dofs.
    # ti: R^m -> R^n, injects electrode data as a Neumann load vector, zero
    # elsewhere. Must be the transpose of `tb` for the adjoint gradient
    # below to be correct -- boundary_maps builds a matching tb/ti pair.
    # f, g have length m < n.
    #
    # `pin` (see AssemblerInfo/ground!) defaults to dof 1 of `dh`, same as
    # the unrestricted case, but here it matters which dof: if it coincides
    # with one of the dofs `tb`/`ti` treat as an electrode, that electrode's
    # injected current is silently grounded away. Pass a `pin` outside your
    # electrode set if that would be a problem for what you're testing.
    #
    # Convenience: σ lives in the same FE space as u.
    function assemble_J(cellvalues::CellValues, dh::DofHandler, m::Int, tb, ti; pin::Int=1)
        return assemble_J(cellvalues, dh, cellvalues, dh, m, tb, ti; pin=pin)
    end

    function assemble_J(cellvalues::CellValues, dh::DofHandler, cellvalues_σ::CellValues, dh_σ::DofHandler, m::Int, tb, ti; pin::Int=1)
        ai = AssemblerInfo(cellvalues, dh, cellvalues_σ, dh_σ; pin=pin)
        n = ndofs(dh)
        n_σ = ndofs(dh_σ)
        u = zeros(n)
        λ = zeros(n)
        rhs = zeros(n)
        u_b = zeros(m)
        grad = zeros(n_σ)
        pin = ai.pin

        j = (σ::AbstractVector, f::AbstractVector, g::AbstractVector) -> begin
            L = assemble_L!(ai, σ)
            rhs .= ti(g)
            rhs[pin] = 0
            u .= L \ rhs
            u_b .= tb(u)
            u_b .-= f
            dot(u_b, u_b)
        end

        # State: Lu = ti(g). Adjoint: Lλ = -2 ti(tb(u) - f) = -2 ti(r), since
        # tb is a 0/1 selection matrix and ti = tb'. Then, as in the
        # unrestricted case, dj/dσ_k = λ'(∂L/∂σ_k)u.
        grad_j = (σ::AbstractVector, f::AbstractVector, g::AbstractVector) -> begin
            L = assemble_L!(ai, σ)
            fact = factorize(L)
            rhs .= ti(g)
            rhs[pin] = 0
            u .= fact \ rhs
            u_b .= tb(u)
            u_b .-= f
            rhs .= ti(-2 .* u_b)
            rhs[pin] = 0
            λ .= fact \ rhs
            assemble_gradient!(grad, ai, λ, u)
            return copy(grad)
        end

        J = (σ::AbstractVector, F::AbstractArray, G::AbstractArray) -> begin
            L = assemble_L!(ai, σ)
            RHS = ti(G)
            RHS[pin, :] .= 0
            U = L \ RHS
            Ub = tb(U)
            Ub .-= F
            dot(Ub, Ub)
        end

        grad_J = (σ::AbstractVector, F::AbstractArray, G::AbstractArray) -> begin
            L = assemble_L!(ai, σ)
            fact = factorize(L)
            RHS = ti(G)
            RHS[pin, :] .= 0
            U = fact \ RHS
            Ub = tb(U)
            R = Ub .- F
            RHS2 = ti(-2 .* R)
            RHS2[pin, :] .= 0
            Λ = fact \ RHS2
            total = zeros(n_σ)
            for col in 1:size(G, 2)
                assemble_gradient!(grad, ai, view(Λ, :, col), view(U, :, col))
                total .+= grad
            end
            return total
        end

        return j, J, grad_j, grad_J
    end
end
