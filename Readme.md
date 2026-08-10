# MiniEIT

Basically people from the Julia slack are of the opinion that autodifferentuation librares like 
- [ ] Enzyme.jl
- [ ] Mooncake.jl
- [ ] Zygote.jl?

are capable of differentiating through ferrite matrix assembly loops. 
However I was not capable of doing that.

Given that my other EIT library has grown sufficiently big I wanna provide a minimal example that provides such a matrix assembly + differentiation problem.

Imagine a function of the type:

$$ obj(\sigma, f, g) =  || f - L(\sigma)^(-1) g||^2 $$
with $\sigma\rightarrow L(\sigma)$ some matrix assembly,

Concretely I also need to plan for a function that restricts to  the boundary: $ \partial: \Omega\rightarrow\partial\Omega $
Then the full problem becomes something like:
$$ obj(\sigma, f, g) =  || \partial(f) - \partial(L(\sigma)^(-1) g) ||^2 $$


Additional problem would be: 
- a nonallocating version
- a BlockCG solved version for $F$,$G$ matrices
- a parallel assembly
- a version that can handle adaptive grid refining/coarsening


