#=
    parameters.jl

    Defines the `Parameters` struct that stores every hyperparameter of
    KACS / XCSF. Most of the notation follows the standard XCS
    hyperparameter naming convention (Wilson, 1995, 1998; Butz & Wilson,
    2002) so that readers already familiar with XCS-family classifier
    systems can map each field directly onto the original equations.

    For the exact meaning and default value of every parameter, run:
        julia main.jl --help
=#

"""
    Parameters

Hyperparameters shared by KACS and XCSF.

# Fields
- `N`: Maximum population size (macro-classifier count is unbounded, but
  total numerosity is capped at `N`).
- `beta`: Learning rate used to update the error, fitness, and action-set
  size estimates of a classifier.
- `e0`: Error threshold below which a classifier's accuracy is set to 1.
- `theta_EA`: Minimum time since a match set's last GA invocation before
  the genetic algorithm is triggered again.
- `chi`: Crossover probability.
- `mu`: Per-allele mutation probability.
- `theta_del`: Experience threshold above which a classifier's fitness is
  taken into account when computing its deletion vote.
- `delta`: Fraction of the population's mean fitness below which a
  classifier's fitness is considered for accelerated deletion.
- `F_I`: Initial fitness assigned to a newly generated classifier.
- `tau`: Tournament size (as a fraction of the action set) used for
  parent selection; if `0`, roulette-wheel selection is used instead.
- `m0`: Maximum magnitude of mutation applied to a condition bound.
- `r0`: Maximum spread used by the covering operator.
- `do_subsumption`: Whether GA offspring are tested for subsumption by
  their parents.
- `P_hash`: Probability of generating a "don't care" (wildcard) condition
  bound during covering.
- `theta_sub`: Experience threshold required before a classifier is
  eligible to subsume another.
"""
mutable struct Parameters
    N::Int64
    beta::Float64
    e0::Float64
    theta_EA::Int64
    chi::Float64
    mu::Float64
    theta_del::Int64
    delta::Float64
    F_I::Float64
    tau::Float64
    m0::Float64
    r0::Float64
    do_subsumption::Bool
    P_hash::Float64
    theta_sub::Int64
end

"""
    Parameters(args) -> Parameters

Builds a `Parameters` instance from the parsed command-line argument
dictionary produced by `parse_commandline()` in main.jl.
"""
function Parameters(args)
    return Parameters(
        args["N"],
        args["beta"],
        args["e0"],
        args["theta_EA"],
        args["chi"],
        args["mu"],
        args["theta_del"],
        args["delta"],
        args["F_I"],
        args["tau"],
        args["m0"],
        args["r0"],
        args["do_subsumption"],
        args["P_hash"],
        args["theta_sub"]
    )
end
