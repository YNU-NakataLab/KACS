#=
    xcsf/classifier.jl

    Defines the XCSF classifier (rule) and the genetic/reinforcement
    operators that act on it: covering, matching, crossover, mutation,
    subsumption, and deletion voting. Unlike the single-scalar KACS
    classifier (kacs/classifier.jl), each XCSF classifier here matches
    the full `n`-dimensional feature vector through an `n`-dimensional
    hyperrectangle (`Vector{UBR}`), and predicts the target with a
    single linear "computed prediction" model over all `n` features,
    following Wilson's XCSF (Wilson, 2002).
=#

include("condition.jl")

using Random

"""
    Classifier

A single XCSF rule. Matches an `n`-dimensional feature vector through
`n` independent `UBR` interval bounds (one per feature), and predicts
the target via a linear model over the input features offset by the
condition's lower bounds (`weight[1]` is the bias/intercept term).

# Fields
- `id`: Unique classifier identifier.
- `condition`: `Vector{UBR}`, one interval bound per input feature.
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
- `weight`: Linear prediction weights, `[w0, w1, ..., wn]` (length
  `n + 1`, one bias term plus one weight per feature).
- `moments`, `velocities`: First/second moment estimates used by the
  Adam optimizer (`adam!`) to update `weight`.
"""
mutable struct Classifier
    id::Int64
    condition::Vector{UBR}
    error::Float64
    fitness::Float64
    experience::Int64
    time_stamp::Int64
    action_set_size::Float64
    numerosity::Int64
    kappa::Float64
    weight::Vector{Float64}
    moments::Vector{Float64}
    velocities::Vector{Float64}
end

"""
    Classifier(parameters, env, state) -> Classifier

Generates a new classifier via the covering operator. For every input
feature `i`, builds a `UBR` bound centered on `state[i]` with a random
spread up to `parameters.r0` (clipped to `[0, 1]`, since features are
normalized). With probability `parameters.P_hash`, or whenever
`state[i]` is missing (`"?"`), a "don't care" bound spanning the full
`[0, 1]` range is generated for that feature instead.
"""
function Classifier(parameters, env, state)
    condition = Vector{UBR}(undef, env.state_length)
    moments = zeros(env.state_length + 1)
    velocities = zeros(env.state_length + 1)
    weight::Vector{Float64} = fill(rand(Uniform(-1, 1)), env.state_length + 1)

    @inbounds @simd for i in 1:env.state_length
        if rand() < parameters.P_hash || state[i] == "?"
            # Wildcard ("don't care") bound covering the full normalized range.
            condition[i] = rand() < 0.5 ? UBR(0.0, 1.0) : UBR(1.0, 0.0)
        else
            if rand() < 0.5
                condition[i] = UBR(min(max(state[i] - rand() * parameters.r0, 0.0), 1.0),
                                    max(min(state[i] + rand() * parameters.r0, 1.0), 0.0))
            else
                condition[i] = UBR(max(min(state[i] + rand() * parameters.r0, 1.0), 0.0),
                                    min(max(state[i] - rand() * parameters.r0, 0.0), 1.0))
            end
        end
    end

    return Classifier(0, condition, 0, parameters.F_I, 0, 0, 1, 1, 1, weight, moments, velocities)
end

"""
    adam!(self, grad)

Applies one Adam optimization step (Kingma & Ba, 2015) to the entire
weight vector of classifier `self`, given the gradient vector `grad`.
"""
function adam!(self::Classifier, grad)
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    eta = 0.001

    self.moments = beta_1 .* self.moments .+ (1 - beta_1) .* grad
    self.velocities = beta_2 .* self.velocities .+ (1 - beta_2) .* grad .^ 2

    # self.experience was already incremented for this step in update_set!,
    # so it directly serves as the 1-based Adam step count t.
    m_hat = self.moments ./ (1 - beta_1^self.experience)
    v_hat = self.velocities ./ (1 - beta_2^self.experience)

    self.weight .-= eta * m_hat ./ (sqrt.(v_hat) .+ epsilon)
end

"""
    does_match(condition, state) -> Bool

Returns `true` if every feature of `state` falls within the
corresponding `UBR` bound in `condition`. A missing feature value
(`"?"`) always matches on that dimension.
"""
function does_match(condition::Vector{UBR}, state)::Bool
    @inbounds for i in 1:length(state)
        if state[i] == "?"
            continue
        end
        if !(get_lower_bound(condition[i]) <= state[i] <= get_upper_bound(condition[i]))
            return false
        end
    end
    return true
end

"""
    apply_crossover!(child_1, child_2) -> Bool

Uniform crossover applied independently to every feature dimension:
for each dimension, the `p` bound is swapped between the two children
with probability 0.5, and likewise for the `q` bound. Returns `true`
if at least one swap occurred anywhere, so the caller can decide
whether to also average the children's inherited error/fitness.
"""
function apply_crossover!(child_1::Classifier, child_2::Classifier)::Bool
    is_changed::Bool = false

    @inbounds @simd for i in 1:length(child_1.condition)
        if rand() < 0.5
            child_1.condition[i].p, child_2.condition[i].p = child_2.condition[i].p, child_1.condition[i].p
            is_changed = true
        end
        if rand() < 0.5
            child_1.condition[i].q, child_2.condition[i].q = child_2.condition[i].q, child_1.condition[i].q
            is_changed = true
        end
    end

    return is_changed
end

"""
    apply_mutation!(self, m, mu, state)

For every feature dimension, with probability `mu`, perturbs both
bounds of that dimension's `UBR` by a uniform random amount in
`[-m, m]`, clipped to `[0, 1]` (features are normalized).
"""
function apply_mutation!(self::Classifier, m::Float64, mu::Float64, state)
    @inbounds @simd for i in 1:length(self.condition)
        if rand() < mu
            self.condition[i].p += 2.0 * m * rand() - m
            self.condition[i].p = min(max(0.0, self.condition[i].p), 1.0)

            self.condition[i].q += 2.0 * m * rand() - m
            self.condition[i].q = min(max(0.0, self.condition[i].q), 1.0)
        end
    end
end

"""
    is_more_general(self, spec) -> Bool

Returns `true` if classifier `self` is strictly more general than
classifier `spec`: every one of `self`'s per-feature bounds must
contain the corresponding bound of `spec`, and at least one of those
bounds must be a strict superset (i.e., the two conditions must not be
identical across all dimensions).
"""
function is_more_general(self::Classifier, spec::Classifier)::Bool
    num_identical_dims::Int64 = 0

    @inbounds for i in 1:length(self.condition)
        l_gen = get_lower_bound(self.condition[i])
        u_gen = get_upper_bound(self.condition[i])
        l_spec = get_lower_bound(spec.condition[i])
        u_spec = get_upper_bound(spec.condition[i])

        if !(l_gen <= l_spec && u_spec <= u_gen)
            return false
        end

        if l_spec == l_gen && u_gen == u_spec
            num_identical_dims += 1
        end
    end

    # Reject the case where every dimension is identical (not strictly more general).
    if num_identical_dims == length(self.condition)
        return false
    end

    return true
end

"""
    is_equal_condition(self, other) -> Bool

Returns `true` if two classifiers have structurally identical
conditions in every dimension (see `UBR.is_equal`). Used by
`insert_in_population!` to merge duplicate rules by incrementing
numerosity instead of adding a new macro-classifier.
"""
function is_equal_condition(self::Classifier, other::Classifier)::Bool
    @inbounds for i in 1:length(self.condition)
        if !is_equal(self.condition[i], other.condition[i])
            return false
        end
    end
    return true
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
