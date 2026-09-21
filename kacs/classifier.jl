#=
    kacs/classifier.jl

    Defines the KACS classifier (rule) and the genetic/reinforcement
    operators that act on it: covering, matching, crossover, mutation,
    subsumption, and deletion voting. Each classifier carries a linear
    local model (weights updated by Adam; Kingma & Ba, 2015) over a
    single scalar input, following the "computed prediction" idea
    introduced for XCSF (Wilson, 2002), but restricted to univariate
    inputs because KACS applies one classifier population per
    Kolmogorov-Arnold inner/outer function component (see
    kacs/xcs.jl).
=#

include("condition.jl")

using Random

"""
    Classifier

A single KACS rule. Each classifier matches a scalar input through a
`UBR` interval condition and predicts an output via a local linear
model `weights = [w0, w_coeff]`.

# Fields
- `id`: Unique classifier identifier.
- `condition`: `UBR` interval condition over the scalar input.
- `type`: `:psi` for an inner (univariate) component, or `:phi` for an
  outer (summed) component, following the two-stage Kolmogorov-Arnold
  decomposition used in kacs/xcs.jl.
- `q_idx`: Index of the outer summation term (`1` to `2n+1`).
- `p_idx`: Index of the input feature (`1` to `n`) for `:psi`
  classifiers; `nothing` for `:phi` classifiers.
- `weights`: Linear prediction weights `[w0, w_coeff]`.
- `moments`, `velocities`: First/second moment estimates used by the
  Adam optimizer (`adam!`) to update `weights`.
- `error`: Estimated absolute prediction error.
- `fitness`: Accuracy-based fitness (Wilson, 1995).
- `experience`: Number of times this classifier has been in a match
  set.
- `time_stamp`: Time step of the last GA invocation involving this
  classifier.
- `action_set_size`: Estimated average match-set size.
- `numerosity`: Number of "micro-classifiers" this classifier
  represents.
- `kappa`: Raw accuracy value from the most recent fitness update.
- `mu`: Self-adaptive mutation rate.
- `eta`: Self-adaptive gradient-descent learning rate (currently
  informational; weight updates use Adam's fixed step size).
"""
mutable struct Classifier
    id::Int64
    condition::UBR
    type::Symbol                       # :psi (inner) or :phi (outer)
    q_idx::Int64                       # outer summation index, 1..2n+1
    p_idx::Union{Int64, Nothing}       # input feature index, 1..n (psi only)
    weights::Vector{Float64}           # [w0, w_coeff]
    moments::Vector{Float64}           # Adam first-moment estimates
    velocities::Vector{Float64}        # Adam second-moment estimates
    error::Float64
    fitness::Float64
    experience::Int64
    time_stamp::Int64
    action_set_size::Float64
    numerosity::Int64
    kappa::Float64
end

"""
    Classifier(parameters, input_val, type, q_idx, p_idx=nothing) -> Classifier

Generates a new classifier via the covering operator. The condition is
built around `input_val` with a random spread up to `parameters.r0`
(clipped to `[0, 1]` for `:psi` classifiers, whose inputs are
normalized features). With probability `parameters.P_hash`, a
"don't care" condition spanning the full `[0, 1]` range is created
instead (only applicable to `:psi` classifiers, or whenever
`input_val == "?"`, i.e. a missing value).
"""
function Classifier(parameters, input_val::Union{Int64, Float64, String}, type::Symbol, q_idx::Int64, p_idx::Union{Int, Nothing}=nothing)
    weights::Vector{Float64} = fill(rand(Uniform(-1, 1)), 2)
    velocities::Vector{Float64} = zeros(2)
    moments::Vector{Float64} = zeros(2)

    if (rand() < parameters.P_hash || input_val == "?") && type == :psi
        # Wildcard ("don't care") condition covering the full normalized range.
        condition = rand() < 0.5 ? UBR(0.0, 1.0) : UBR(1.0, 0.0)
    else
        if type == :psi
            # Inner classifier: input is a normalized feature in [0, 1].
            if rand() < 0.5
                condition = UBR(min(max(input_val - rand() * parameters.r0, 0.0), 1.0),
                                 max(min(input_val + rand() * parameters.r0, 1.0), 0.0))
            else
                condition = UBR(max(min(input_val + rand() * parameters.r0, 1.0), 0.0),
                                 min(max(input_val - rand() * parameters.r0, 0.0), 1.0))
            end
        else
            # Outer classifier: input is an unbounded intermediate sum.
            if rand() < 0.5
                condition = UBR(input_val - rand() * parameters.r0, input_val + rand() * parameters.r0)
            else
                condition = UBR(input_val + rand() * parameters.r0, input_val - rand() * parameters.r0)
            end
        end
    end

    return Classifier(0, condition, type, q_idx, p_idx, weights, moments, velocities, 0, parameters.F_I, 0, 0, 1, 1, 1)
end

"""
    adam!(self, grad, i)

Applies one Adam optimization step (Kingma & Ba, 2015) to weight
component `i` of classifier `self`, given gradient `grad`. Moment
estimates are bias-corrected using the classifier's own `experience`
counter as the step count, so the correction reflects how many times
this specific classifier has actually been updated.
"""
function adam!(self::Classifier, grad, i)
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    eta = 0.001

    self.moments[i] = beta_1 * self.moments[i] + (1 - beta_1) * grad
    self.velocities[i] = beta_2 * self.velocities[i] + (1 - beta_2) * grad^2

    m_hat = self.moments[i] / (1 - beta_1^(self.experience + 1))
    v_hat = self.velocities[i] / (1 - beta_2^(self.experience + 1))

    self.weights[i] -= eta * m_hat / (sqrt(v_hat) + epsilon)
end

"""
    does_match(condition, input_val) -> Bool

Returns `true` if `input_val` falls within `condition`'s interval, or
if `input_val` is a missing value (`"?"`), which always matches.
"""
function does_match(condition::UBR, input_val::Union{Int64, Float64, String})::Bool
    if input_val == "?"
        return true
    end
    return get_lower_bound(condition) <= input_val <= get_upper_bound(condition)
end

"""
    apply_crossover!(child_1, child_2) -> Bool

Two-point crossover on the `(p, q)` bound pair: each bound is swapped
between the two children independently with probability 0.5. Returns
`true` if at least one swap occurred, so the caller can decide whether
to also average the children's inherited error/fitness.
"""
function apply_crossover!(child_1::Classifier, child_2::Classifier)::Bool
    is_changed::Bool = false

    if rand() < 0.5
        child_1.condition.p, child_2.condition.p = child_2.condition.p, child_1.condition.p
        is_changed = true
    end
    if rand() < 0.5
        child_1.condition.q, child_2.condition.q = child_2.condition.q, child_1.condition.q
        is_changed = true
    end

    return is_changed
end

"""
    apply_mutation!(self, m, mu, state)

With probability `mu`, perturbs both condition bounds by a uniform
random amount in `[-m, m]` (clipped to `[0, 1]` for `:psi`
classifiers)
"""
function apply_mutation!(self::Classifier, m::Float64, mu::Float64, state)
    if rand() < mu
        self.condition.p += 2.0 * m * rand() - m
        if self.type == :psi
            self.condition.p = min(max(0.0, self.condition.p), 1.0)
        end

        self.condition.q += 2.0 * m * rand() - m
        if self.type == :psi
            self.condition.q = min(max(0.0, self.condition.q), 1.0)
        end
    end
end

"""
    is_more_general(self, spec) -> Bool

Returns `true` if classifier `self` is strictly more general than
classifier `spec`: both classifiers must be of the same type (and
same `q_idx`/`p_idx` niche), and `self`'s interval must strictly
contain `spec`'s interval.
"""
function is_more_general(self::Classifier, spec::Classifier)::Bool
    if self.type != spec.type
        return false
    end

    if self.type == :psi
        if self.q_idx != spec.q_idx || self.p_idx != spec.p_idx
            return false
        end
    else
        if self.q_idx != spec.q_idx
            return false
        end
    end

    l_gen = get_lower_bound(self.condition)
    u_gen = get_upper_bound(self.condition)
    l_spec = get_lower_bound(spec.condition)
    u_spec = get_upper_bound(spec.condition)

    if !(l_gen <= l_spec && u_spec <= u_gen)
        return false
    end

    # Exclude the case of an identical interval (not strictly more general).
    if l_spec == l_gen && u_gen == u_spec
        return false
    end

    return true
end

"""
    is_equal_condition(self, other) -> Bool

Returns `true` if two classifiers have structurally identical
conditions (see `UBR.is_equal`). Used by `insert_in_population!` to
merge duplicate rules by incrementing numerosity instead of adding a
new macro-classifier.
"""
function is_equal_condition(self::Classifier, other::Classifier)::Bool
    return is_equal(self.condition, other.condition)
end

"""
    could_subsume(self, theta_sub, e0) -> Bool

Returns `true` if `self` is experienced (`experience > theta_sub`) and
accurate enough (`error < e0`) to act as a subsumer.
"""
could_subsume(self::Classifier, theta_sub::Int, e0::Float64)::Bool =
    self.experience > theta_sub && self.error < e0

"""
    does_subsume(self, tos, theta_sub, e0) -> Bool

Returns `true` if `self` both qualifies as a subsumer
(`could_subsume`) and is more general than `tos` (`is_more_general`).
"""
does_subsume(self::Classifier, tos::Classifier, theta_sub::Int, e0::Float64)::Bool =
    could_subsume(self, theta_sub, e0) && is_more_general(self, tos)

"""
    deletion_vote(self, average_fitness, theta_del, delta) -> Float64

Computes the classifier's vote weight in the deletion lottery
(Wilson, 1995): proportional to `action_set_size * numerosity`, scaled
up further if the classifier is experienced and under-performing
relative to `delta * average_fitness`.
"""
function deletion_vote(self::Classifier, average_fitness::Float64, theta_del::Int, delta::Float64)::Float64
    vote::Float64 = self.action_set_size * self.numerosity
    if self.experience > theta_del && self.fitness / self.numerosity < delta * average_fitness
        vote *= average_fitness / (self.fitness / self.numerosity)
    end
    return vote
end


"""
    is_equal_classifier(self, other) -> Bool

Identity comparison by classifier `id`.
"""
is_equal_classifier(self::Classifier, other::Classifier)::Bool = self.id == other.id
