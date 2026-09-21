#=
    kacs/condition.jl

    Interval-matching condition used by KACS classifiers, implemented
    with the Unordered Bound Representation (UBR) proposed by Stone &
    Bull (2003) for real-valued XCS conditions. Storing the two bounds
    (p, q) without fixing which one is the lower/upper bound removes
    the representational bias of the ordered min-max representation
    when applying crossover and mutation.
=#

"""
    UBR

Unordered Bound Representation for a one-dimensional interval
condition (Stone & Bull, 2003).

Bounds are stored as an unordered pair `(p, q)`; the actual lower and
upper bounds are recovered on demand via `get_lower_bound`,
`get_upper_bound`, or `get_lower_upper_bounds`. This avoids the bias
that ordered (lower, upper) representations introduce during
recombination.

# Fields
- `p::Float64`: First stored bound (order not significant).
- `q::Float64`: Second stored bound (order not significant).
"""
mutable struct UBR
    p::Float64
    q::Float64
end

"""
    get_lower_bound(ubr) -> Float64

Returns the interval's lower bound, i.e. `min(p, q)`.
"""
function get_lower_bound(ubr::UBR)::Float64
    return min(ubr.p, ubr.q)
end

"""
    get_upper_bound(ubr) -> Float64

Returns the interval's upper bound, i.e. `max(p, q)`.
"""
function get_upper_bound(ubr::UBR)::Float64
    return max(ubr.p, ubr.q)
end

"""
    get_lower_upper_bounds(ubr) -> (lower, upper)

Returns both bounds as an ordered tuple `(lower, upper)`, with
`lower <= upper` guaranteed regardless of storage order.
"""
function get_lower_upper_bounds(ubr::UBR)::Tuple{Float64,Float64}
    l = min(ubr.p, ubr.q)
    u = max(ubr.p, ubr.q)
    return (l, u)
end

"""
    is_equal(a, b) -> Bool

Exact structural equality of two `UBR` conditions. Note this compares
the stored `(p, q)` pair, not the resolved `(lower, upper)` bounds, so
two intervals with swapped `p`/`q` are considered different unless
their fields match exactly.
"""
function is_equal(a::UBR, b::UBR)::Bool
    return a.p == b.p && a.q == b.q
end
