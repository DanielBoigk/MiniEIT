using Test
using Ferrite, MiniEIT
using Statistics

# AD backend packages are only needed for the commented-out MWE at the
# bottom of test/01NoBoundaryTest.jl (kept there for Enzyme/Mooncake/Zygote
# maintainers to test against, not run as part of this suite). Uncomment if
# you want to exercise that block locally:
# using DifferentiationInterface
# using ForwardDiff: ForwardDiff
# using Enzyme: Enzyme
# using Zygote: Zygote
# using Mooncake: Mooncake

# this is the minimal Ferrite things needed to give an example:

N = 63
order = 2
qr_order = 3
grid = generate_grid(Quadrilateral, (N, N))

function return_space(::Type{RefElem}, grid, order::Int, qr_order::Int) where {RefElem}
    dim = Ferrite.getspatialdim(grid)
    ip = Lagrange{RefElem,order}()
    qr = QuadratureRule{RefElem}(qr_order)
    cellvalues = CellValues(qr, ip)
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    n = ndofs(dh)
    return cellvalues, dh, dim, n
end

cellvalues, dh, dim, n = return_space(RefQuadrilateral, grid, 2, 3)

# σ is piecewise constant per cell (P0), so it has one value per cell,
# not one per dof of `dh`:
ncells = getncells(grid)

# for later:
∂Ω = union(getfacetset.((grid,), ["left", "top", "right", "bottom"])...)

# Electrode dofs for the boundary-restricted objective: a subset of the
# boundary dofs of `dh`'s :u field, not every dof on ∂Ω (see
# select_electrodes for the caveat on what "evenly spaced" means here).
bdofs = boundary_dofs(dh, :u, ∂Ω)
n_electrodes = 16
electrode_dofs = select_electrodes(bdofs, n_electrodes)
tb, ti = boundary_maps(n, electrode_dofs)
m = length(electrode_dofs)

# The grounded reference dof must not be one of the electrodes (see
# assemble_J's boundary-restricted method), so pick one that isn't:
pin_dof = first(setdiff(1:n, electrode_dofs))

# This test is supposed to test whether the functional is differentiable:
include("01NoBoundaryTest.jl")
include("02AdjointGradientTest.jl")
include("03BoundaryElectrodeTest.jl")
