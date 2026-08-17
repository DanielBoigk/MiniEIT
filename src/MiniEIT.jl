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

    # Ignore this for now
    # This is the same but with a function that somehow restricts to the boundary
    # tb = from interior to boundary
    # ti = from boundary to interior
    function assemble_J(cellvalues::CellValues, dh::DofHandler, m::Int64 ,tb, ti)
        ai = AssemblerInfo(cellvalues, dh)
        n = size(ai.L)[1]
        u = zeros(n)
        rhs = zeros(m)
        u_b = zeros(m)
        # Assume f and g have length m < n
        j = (σ::AbstractVector, f::AbstractVector, g::AbstractVector) -> begin
            L = assemble_L!(ai,σ)
            rhs .= ti(g)
            u .= L \ rhs
            u_b .= tb(u)
            u_b .-= f
            dot(u_b,u_b)
        end
        J = (σ::AbstractVector, F::AbstractArray, G::AbstractArray) -> begin
            L = assemble_L!(ai,σ)
            RHS = ti(G)
            U = L \ RHS
            Ub = tb(U)
            Ub .-= F
            dot(Ub,Ub)
        end
        return j, J
    end
end
