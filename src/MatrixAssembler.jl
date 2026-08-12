# This file contains the assembler for a weighted stiffness matrix

export assemble_L, assemble_L!, AssemblerInfo

struct AssemblerInfo
    L
    cellvalues
    dh
    n_basefuncs
    Le
    assembler
end

function AssemblerInfo(cellvalues::CellValues, dh::DofHandler)
    L = allocate_matrix(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    Le = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(L)
    AssemblerInfo(L, cellvalues, dh, n_basefuncs, Le, assembler)
end


# This is supposed to assemble:
# L= ∫(σ * ∇(u)⋅∇(v))dΩ
function assemble_L!(ai::AssemblerInfo, σ::AbstractVector)
    L = ai.L
    cellvalues = ai.cellvalues
    dh = ai.dh
    n_basefuncs = ai.n_basefuncs
    Le = ai.Le
    assembler = ai.assembler

    #assembler = start_assemble(L)
    fill!(L, 0.0)
    for cell in CellIterator(dh)
        fill!(Le, 0)
        reinit!(cellvalues, cell)
        for q in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q)
            γe = σ[celldofs(cell)]
            γ = function_value(cellvalues, q, γe)
            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, q, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q, j)
                    Le[i, j] += γ * (∇v ⋅ ∇u) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Le)
    end
    return L
end
