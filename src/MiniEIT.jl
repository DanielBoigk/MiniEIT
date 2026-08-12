# This file takes the matrix assembler and returns the objective function j(σ,f,g) = ||f- L(σ)^(-1) g||^2
module MiniEIT
    using Ferrite, LinearAlgebra, SparseArrays
    using IterativeSolvers, Krylov

    include("MatrixAssembler.jl")

    export assemble_J


    # This assembles the function I want to differentiate:
    function assemble_J(cellvalues::CellValues, dh::DofHandler)
        ai = AssemblerInfo(cellvalues, dh)
        n = ndofs(dh)
        u = zeros(n)
        j = (σ::AbstractVector, f::AbstractVector, g::AbstractVector) -> begin
            L = assemble_L!(ai, σ)
            u .= L \ g
            u .-= f
            dot(u, u)
        end
        J = (σ::AbstractVector, F::AbstractArray, G::AbstractArray) -> begin
            L = assemble_L!(ai, σ)
            U = L \ G
            U .-= F
            dot(U, U)
        end

        return j, J
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
        j = (σ::AbstractVector, F::AbstractArray, G::AbstractArray) -> begin
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
