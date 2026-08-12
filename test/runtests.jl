using Test
using Ferrite, MiniEIT
using Statistics
using DifferentiationInterface
using ForwardDiff: ForwardDiff
using Enzyme: Enzyme
using Zygote: Zygote
using Mooncake: Mooncake

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


# for later:
∂Ω = union(getfacetset.((grid,), ["left", "top", "right", "bottom"])...)
m = length(∂Ω)

# This test is supposed to test whether the functional is differentiable:
include("01NoBoundaryTest.jl")
