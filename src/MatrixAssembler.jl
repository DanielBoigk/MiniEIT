# This file contains the assembler for a weighted stiffness matrix,
# and the adjoint-method sensitivity assembler for its gradient.

export assemble_L, assemble_L!, assemble_gradient!, AssemblerInfo
export boundary_dofs, select_electrodes, boundary_maps
export bordered

struct AssemblerInfo
    L
    cellvalues
    dh
    cellvalues_σ
    dh_σ
    n_basefuncs
    Le
    assembler
    pin::Int
end

# σ lives in its own FE space (cellvalues_σ, dh_σ), which may differ from u's
# space (cellvalues, dh) -- e.g. σ piecewise constant (P0) via
# DiscontinuousLagrange{RefElem,0} while u is Lagrange order 2.
# cellvalues and cellvalues_σ must be built from the same QuadratureRule, so
# that quadrature point index q refers to the same physical point in both.
#
# `pin` is the dof of `dh` whose value is grounded to zero, see `ground!`.
function AssemblerInfo(cellvalues::CellValues, dh::DofHandler, cellvalues_σ::CellValues, dh_σ::DofHandler; pin::Int=1)
    L = allocate_matrix(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    Le = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(L)
    AssemblerInfo(L, cellvalues, dh, cellvalues_σ, dh_σ, n_basefuncs, Le, assembler, pin)
end

# Convenience constructor for σ living in the same space as u.
AssemblerInfo(cellvalues::CellValues, dh::DofHandler; pin::Int=1) = AssemblerInfo(cellvalues, dh, cellvalues, dh; pin=pin)


# This is supposed to assemble:
# L= ∫(σ * ∇(u)⋅∇(v))dΩ
#
# With pure Neumann data (no Dirichlet boundary anywhere) this bilinear form
# is only positive *semi*-definite: L has a 1-dimensional null space spanned
# by the constant vector (the potential is only defined up to an additive
# constant). That makes `L \ rhs` ill-posed/numerically unstable. We remove
# that null space by grounding a single reference dof (`ai.pin`) to zero, see
# `ground!` -- any dof works, since the constant null vector is nonzero
# everywhere. Callers must correspondingly zero out `rhs[ai.pin]` (or the
# matching row for a matrix RHS) before solving `L \ rhs`.
#
# Pass `ground=false` to get the raw, still-singular L back instead -- e.g.
# for solving via `bordered`, which removes the null space by a constraint
# instead of by eliminating a dof.
function assemble_L!(ai::AssemblerInfo, σ::AbstractVector; ground::Bool=true)
    L = ai.L
    cellvalues = ai.cellvalues
    cellvalues_σ = ai.cellvalues_σ
    dh = ai.dh
    dh_σ = ai.dh_σ
    n_basefuncs = ai.n_basefuncs
    Le = ai.Le
    assembler = ai.assembler

    fill!(L, 0.0)
    for (cell, cell_σ) in zip(CellIterator(dh), CellIterator(dh_σ))
        fill!(Le, 0)
        reinit!(cellvalues, cell)
        reinit!(cellvalues_σ, cell_σ)
        σ_local = σ[celldofs(cell_σ)]
        for q in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q)
            σ_f = function_value(cellvalues_σ, q, σ_local)
            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, q, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q, j)
                    Le[i, j] += σ_f * (∇v ⋅ ∇u) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Le)
    end
    ground && ground!(L, ai.pin)
    return L
end

# Symmetrically eliminates dof `pin` from the sparse system in place: zeroes
# its row and column and sets the diagonal to 1. Combined with zeroing
# `rhs[pin]`, this pins u[pin] = 0.
function ground!(L::SparseMatrixCSC, pin::Int)
    rows = rowvals(L)
    vals = nonzeros(L)
    for col in 1:size(L, 2)
        for idx in nzrange(L, col)
            row = rows[idx]
            if row == pin || col == pin
                vals[idx] = (row == pin && col == pin) ? 1.0 : 0.0
            end
        end
    end
    return L
end

# Adjoint-method sensitivity of the weighted stiffness form w.r.t. σ.
# For state Lu=g and adjoint Lλ=-2(u-f) (L symmetric), the objective
# j=||f-u||^2 has dj/dσ_k = λ'(∂L/∂σ_k)u = ∫ φ_k(x) ∇λ(x)⋅∇u(x) dΩ, where φ_k
# is the k-th σ basis function. Writes the result (length ndofs(dh_σ)) into
# `grad`.
function assemble_gradient!(grad::AbstractVector, ai::AssemblerInfo, λ::AbstractVector, u::AbstractVector)
    cellvalues = ai.cellvalues
    cellvalues_σ = ai.cellvalues_σ
    dh = ai.dh
    dh_σ = ai.dh_σ
    n_basefuncs_σ = getnbasefunctions(cellvalues_σ)

    fill!(grad, 0.0)
    for (cell, cell_σ) in zip(CellIterator(dh), CellIterator(dh_σ))
        reinit!(cellvalues, cell)
        reinit!(cellvalues_σ, cell_σ)
        λ_local = λ[celldofs(cell)]
        u_local = u[celldofs(cell)]
        dofs_σ = celldofs(cell_σ)
        for q in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q)
            s = (function_gradient(cellvalues, q, λ_local) ⋅ function_gradient(cellvalues, q, u_local)) * dΩ
            for a in 1:n_basefuncs_σ
                grad[dofs_σ[a]] += shape_value(cellvalues_σ, q, a) * s
            end
        end
    end
    return grad
end

# Borders the raw (ungrounded, singular) L with a constraint vector w into
# the symmetric (n+1)x(n+1) saddle-point matrix [L w; w' 0]. Solving
# [L w; w' 0] [u; ν] = [rhs; 0] gives the unique u with L u = rhs AND
# w'u = 0 -- i.e. it picks the one member of L's null-space-shifted solution
# family (u* + c*1, since null(L) = span(1)) satisfying a chosen linear
# constraint w'u=0, instead of picking it by pinning a single dof to zero.
# ν comes out ≈0 (so the top block rows really do solve L u = rhs
# unperturbed) provided rhs is compatible, i.e. sum(rhs) = 0 -- same
# zero-net-current condition pure-Neumann problems already need.
# w must have w'*1 ≠ 0 (e.g. w supported on a nonempty dof subset with
# positive entries) for the bordered matrix to be nonsingular.
function bordered(L::SparseMatrixCSC, w::AbstractVector)
    wcol = sparse(reshape(w, :, 1))
    return [L wcol; sparse(wcol') spzeros(eltype(L), 1, 1)]
end

# Sorted, unique global dof indices of `field` on `facetset` (e.g. the whole
# boundary ∂Ω, or a subset of it). Uses a throwaway zero-Dirichlet constraint
# purely to let Ferrite work out which dofs live on `facetset` -- this is not
# an actual boundary condition on the state.
function boundary_dofs(dh::DofHandler, field::Symbol, facetset)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(field, facetset, (x, t) -> 0))
    close!(ch)
    return ch.prescribed_dofs
end

# Picks `n_electrodes` dofs out of `dofs` to emulate a handful of discrete
# EIT electrodes on a continuous boundary dof set, rather than driving/
# measuring every boundary dof. Note: this spaces indices evenly through
# `dofs` as given (e.g. sorted by dof number), NOT evenly around the
# physical perimeter -- fine for exercising the machinery, but for a
# physically meaningful electrode layout you'd want to pick dofs by node
# coordinate instead.
function select_electrodes(dofs::AbstractVector{<:Integer}, n_electrodes::Int)
    1 <= n_electrodes <= length(dofs) || throw(ArgumentError("n_electrodes must be in 1:length(dofs)"))
    idx = unique(round.(Int, range(1, length(dofs), length=n_electrodes)))
    return dofs[idx]
end

# Builds the pair of linear maps assemble_J's boundary-restricted method
# needs: tb restricts a length-n state/RHS vector (or matrix) to the given
# electrode dofs, and ti is its transpose -- it embeds an electrode-length
# vector (or matrix) back into length n, zero everywhere else. assemble_J's
# adjoint gradient relies on ti = tb' holding exactly, so build both from the
# same `electrode_dofs` rather than writing them independently.
#
# Caveat: ti(g) becomes a Neumann load vector, i.e. injected current. If the
# entries of `g` don't sum to zero (charge conservation / no return path),
# the resulting `u` is still a well-defined solution (assemble_L! grounds a
# reference dof so the system can't be singular), but it no longer
# corresponds to a physically realizable closed-circuit measurement -- zero
# the mean of `g` yourself if that matters for what you're testing.
function boundary_maps(n::Int, electrode_dofs::AbstractVector{<:Integer})
    tb(u::AbstractVector) = u[electrode_dofs]
    tb(U::AbstractMatrix) = U[electrode_dofs, :]
    function ti(g::AbstractVector)
        out = zeros(eltype(g), n)
        out[electrode_dofs] .= g
        return out
    end
    function ti(G::AbstractMatrix)
        out = zeros(eltype(G), n, size(G, 2))
        out[electrode_dofs, :] .= G
        return out
    end
    return tb, ti
end
