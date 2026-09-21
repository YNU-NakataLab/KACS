#=
    xcsf/xcs.jl

    Reference XCSF implementation (Wilson, 2002) used as the baseline
    against KACS (kacs/xcs.jl) in this repository. Unlike KACS, XCSF
    keeps a single classifier population matching the full
    `n`-dimensional input at once, and each classifier predicts the
    target with one linear "computed prediction" model over all `n`
    features (Wilson, 2002), rather than decomposing the function via
    the Kolmogorov-Arnold representation. Population dynamics
    (covering, GA, subsumption, deletion) follow the standard XCS
    algorithmic template (Wilson, 1995, 1998; Butz & Wilson, 2002).
=#

include("classifier.jl")
using Distributions
using LinearAlgebra

"""
    XCS

Top-level XCSF learner holding a single classifier population that
matches the full input vector.

# Fields
- `env`: Data environment (see environment/csv.jl).
- `parameters`: Hyperparameters (see parameters.jl).
- `population`: All classifiers.
- `time_stamp`: Number of `run_experiment` calls so far.
- `covering_occur_num`, `subsumption_occur_num`, `deletion_occur_num`:
  Per-interval operator trigger counters (reset in main.jl's logging step).
- `global_id`: Monotonically increasing classifier ID counter.
"""
mutable struct XCS
    env::Environment
    parameters::Parameters
    population::Vector{Classifier}
    time_stamp::Int64
    covering_occur_num::Int64
    subsumption_occur_num::Int64
    deletion_occur_num::Int64
    global_id::Int64
end

"""
    XCS(env, parameters) -> XCS

Initializes an empty population for the given environment and
hyperparameters.
"""
function XCS(env, parameters)
    return XCS(env, parameters, [], 0, 0, 0, 0, 0)
end

"""
    run_experiment(self)

Runs one full online-learning step: draws a training sample, forms its
match set (covering if empty), updates classifier weights/fitness from
the resulting error (`update_set!`), then invokes the genetic
algorithm on the match set (`run_ga!`). Advances `self.time_stamp` by
one.
"""
function run_experiment(self::XCS)
    curr_state::Vector{Union{Float64, Int64, String}} = state(self.env)
    rho::Float64 = answer(self.env, curr_state)

    match_set::Vector{Classifier} = generate_match_set(self, curr_state)
    update_set!(self, match_set, rho, curr_state)
    run_ga!(self, match_set, curr_state)

    self.time_stamp += 1
end

"""
    generate_match_set(self, state, do_exploit=false) -> Vector{Classifier}

Returns every population classifier whose condition matches `state`.
While training (`do_exploit == false`), triggers covering if the
match set would otherwise be empty. While evaluating
(`do_exploit == true`), no covering is performed, so an empty match
set is possible.
"""
function generate_match_set(self::XCS, state::Vector{Union{Float64, Int64, String}}, do_exploit=false)::Vector{Classifier}
    match_set::Vector{Classifier} = []

    if !do_exploit
        match_set = filter(clas -> does_match(clas.condition, state), self.population)
        if isempty(match_set)
            clas::Classifier = generate_covering_classifier(self, match_set, state)
            push!(self.population, clas)
            delete_from_population!(self, true)
            push!(match_set, clas)
        end
    elseif do_exploit
        match_set = filter(clas -> does_match(clas.condition, state), self.population)
    else
        error("Invalid argument for do_exploit.")
    end

    return match_set
end

"""
    generate_covering_classifier(self, match_set, state) -> Classifier

Generates a new classifier covering `state` via the covering operator,
assigns it a fresh ID, stamps it with the current time step, and
increments the covering counter.
"""
function generate_covering_classifier(self::XCS, match_set::Vector{Classifier}, state::Vector{Union{Float64, Int64, String}})::Classifier
    clas::Classifier = Classifier(self.parameters, self.env, state)
    clas.time_stamp = self.time_stamp

    self.covering_occur_num += 1
    clas.id = self.global_id
    self.global_id += 1

    return clas
end

"""
    generate_prediction(self, match_set, state) -> Float64

Computes the system's fitness-weighted "computed prediction"
(Wilson, 2002) for `state`. Each classifier's local linear model is
evaluated relative to its own condition's lower bound, `l* = [0, l_1,
..., l_n]`, so its predicted output is
`dot(weight, [1, state] - l*)`; predictions are then combined across
the match set weighted by fitness. Returns `0.0` if the match set is
empty or has zero total fitness.
"""
function generate_prediction(self::XCS, match_set::Vector{Classifier}, state)::Float64
    prediction::Float64 = 0.0
    fitness_sum::Float64 = 0.0
    x_star = vcat(1.0, state)

    @inbounds @simd for clas in match_set
        l_star = vcat(0.0, [get_lower_bound(clas.condition[i]) for i in 1:length(state)])
        clas_prediction = dot(x_star - l_star, clas.weight)
        prediction += clas_prediction * clas.fitness
        fitness_sum += clas.fitness
    end

    return fitness_sum == 0.0 ? 0.0 : prediction / fitness_sum
end

"""
    update_set!(self, action_set, P, state)

Updates every classifier in `action_set` after observing the true
target `P`. For each classifier, computes the local prediction error
relative to its own condition-offset linear model
(`P - dot(x* - l*, weight)`), applies one Adam optimization step
(Kingma & Ba, 2015; see `adam!` in xcsf/classifier.jl) to its weight
vector, then updates its error and estimated action-set size before
refreshing accuracy-based fitness via `update_fitness!`.
"""
function update_set!(self::XCS, action_set::Vector{Classifier}, P::Float64, state)
    set_numerosity = sum(clas.numerosity for clas in action_set)
    x_star = vcat(1.0, state)
    beta = self.parameters.beta

    @inbounds @simd for clas in action_set
        clas.experience += 1

        l_star = vcat(0.0, [get_lower_bound(clas.condition[i]) for i in 1:length(state)])
        adam!(clas, -(P - dot(x_star - l_star, clas.weight)) .* (x_star - l_star))

        clas.error += beta * (abs(P - dot(x_star - l_star, clas.weight)) - clas.error)
        clas.action_set_size += beta * (set_numerosity - clas.action_set_size)
    end

    update_fitness!(self, action_set)
end

"""
    update_fitness!(self, action_set)

Standard XCS accuracy-based fitness update (Wilson, 1995):
classifier accuracy `kappa` is `1` if `error < e0`, and
`(error / e0)^-1` otherwise; fitness is then moved toward the
classifier's share of the match set's total (accuracy x numerosity).
"""
function update_fitness!(self::XCS, action_set::Vector{Classifier})
    kappa = Dict{Classifier, Float64}()
    beta = self.parameters.beta
    e0 = self.parameters.e0

    @simd for clas in action_set
        kappa[clas] = clas.error < e0 ? 1.0 : (clas.error / e0)^(-1)
        clas.kappa = kappa[clas]
    end

    accuracy_sum = sum(kappa[clas] * clas.numerosity for clas in action_set)

    @inbounds @simd for clas in action_set
        clas.fitness += beta * (kappa[clas] * clas.numerosity / accuracy_sum - clas.fitness)
        clas.kappa = kappa[clas]
    end
end

"""
    run_ga!(self, action_set, state)

Standard XCS niche genetic algorithm (Wilson, 1995): if the average
time since the action set's classifiers were last subject to the GA
exceeds `theta_EA`, select two parents (roulette-wheel or tournament,
see `select_offspring`), clone them into two children, apply crossover
(`apply_crossover!`) with probability `chi` and mutation
(`apply_mutation!`), then either let a sufficiently accurate and
general parent subsume the child or insert the child into the
population, followed by deletion to keep the population within `N`.
"""
function run_ga!(self::XCS, action_set::Vector{Classifier}, state)
    isempty(action_set) && return

    avg_time_stamp = mapreduce(clas -> clas.time_stamp * clas.numerosity, +, action_set) /
                     mapreduce(clas -> clas.numerosity, +, action_set)

    if self.time_stamp - avg_time_stamp > self.parameters.theta_EA
        @inbounds for clas in action_set
            clas.time_stamp = self.time_stamp
        end

        parent_1::Classifier = select_offspring(self, action_set)
        parent_2::Classifier = select_offspring(self, action_set)
        child_1::Classifier = deepcopy(parent_1)
        child_2::Classifier = deepcopy(parent_2)
        child_1.id = self.global_id
        child_2.id = self.global_id + 1
        self.global_id += 2

        child_1.fitness /= parent_1.numerosity
        child_2.fitness /= parent_2.numerosity
        child_1.numerosity = 1
        child_2.numerosity = 1
        child_1.experience = 0
        child_2.experience = 0

        if rand() < self.parameters.chi
            is_changed = apply_crossover!(child_1, child_2)
            if is_changed
                child_1.error = child_2.error = (parent_1.error + parent_2.error) / 2.0
                child_1.fitness = child_2.fitness = (parent_1.fitness + parent_2.fitness) / 2.0
            end
        end

        child_1.fitness *= 0.1
        child_2.fitness *= 0.1

        @inbounds @simd for child in (child_1, child_2)
            apply_mutation!(child, self.parameters.m0, self.parameters.mu, state)

            if self.parameters.do_subsumption
                if does_subsume(parent_1, child, self.parameters.theta_sub, self.parameters.e0)
                    self.subsumption_occur_num += 1
                    parent_1.numerosity += 1
                elseif does_subsume(parent_2, child, self.parameters.theta_sub, self.parameters.e0)
                    self.subsumption_occur_num += 1
                    parent_2.numerosity += 1
                else
                    insert_in_population!(self, child)
                end
            else
                insert_in_population!(self, child)
            end

            delete_from_population!(self, false)
        end
    end
end

"""
    select_offspring(self, action_set) -> Classifier

Selects a parent classifier from `action_set` for reproduction. Uses
fitness-proportionate (roulette-wheel) selection when
`parameters.tau == 0`; otherwise uses tournament selection with
relative tournament size `parameters.tau` over `numerosity`-weighted
copies of each classifier.
"""
function select_offspring(self::XCS, action_set::Vector{Classifier})::Classifier
    if self.parameters.tau == 0.0
        fitness_sum::Float64 = sum(clas.fitness for clas in action_set)
        choice_point::Float64 = rand() * fitness_sum

        fitness_sum = 0.0
        @inbounds for clas in action_set
            fitness_sum += clas.fitness
            if fitness_sum > choice_point
                return clas
            end
        end
    else
        parent::Any = nothing
        while parent === nothing
            @inbounds for clas in action_set
                if parent === nothing || parent.fitness / parent.numerosity < clas.fitness / clas.numerosity
                    for _ in 1:clas.numerosity
                        if rand() < self.parameters.tau
                            parent = clas
                            break
                        end
                    end
                end
            end
        end
        return parent
    end
end

"""
    insert_in_population!(self, clas)

Adds `clas` to the population, or, if a classifier with an identical
condition already exists, increments that classifier's numerosity
instead (macro-classifier merging).
"""
function insert_in_population!(self::XCS, clas::Classifier)
    @inbounds for c in self.population
        if is_equal_condition(c, clas)
            c.numerosity += 1
            return
        end
    end
    push!(self.population, clas)
end

"""
    delete_from_population!(self, is_cover)

Standard XCS deletion (Wilson, 1995): if total numerosity exceeds `N`,
runs a fitness-derating deletion lottery (`deletion_vote`) over the
whole population and decrements the numerosity of the selected
classifier, removing it from the population if numerosity reaches
zero. `is_cover` indicates whether this call was triggered by covering
(used only for bookkeeping counters).
"""
function delete_from_population!(self::XCS, is_cover::Bool)
    numerosity_sum::Float64 = mapreduce(clas -> clas.numerosity, +, self.population)
    numerosity_sum <= self.parameters.N && return

    average_fitness::Float64 = mapreduce(clas -> clas.fitness, +, self.population) / numerosity_sum
    vote_sum::Float64 = mapreduce(clas -> deletion_vote(clas, average_fitness, self.parameters.theta_del, self.parameters.delta), +, self.population)

    choice_point::Float64 = rand() * vote_sum
    vote_sum = 0.0

    @inbounds for clas in self.population
        vote_sum += deletion_vote(clas, average_fitness, self.parameters.theta_del, self.parameters.delta)
        if vote_sum > choice_point
            clas.numerosity -= 1
            if is_cover
                self.deletion_occur_num += 1
            end
            if clas.numerosity == 0
                @views filter!(e -> e != clas, self.population)
            end
            return
        end
    end
end

"""
    do_action_set_subsumption!(self, action_set)

Action-set subsumption (Wilson, 1998): finds the most general
classifier in `action_set` that is eligible to subsume
(`could_subsume`), and if found, merges every classifier it can
generalize (`is_more_general`) into it by accumulating numerosity and
removing the subsumed classifiers from both the action set and the
population.
"""
function do_action_set_subsumption!(self::XCS, action_set::Vector{Classifier})
    cl::Any = nothing
    @inbounds for c in action_set
        if could_subsume(c, self.parameters.theta_sub, self.parameters.e0)
            if cl === nothing || is_more_general(c, cl)
                cl = c
            end
        end
    end

    if cl !== nothing
        @inbounds for c in action_set
            if is_more_general(cl, c)
                self.subsumption_occur_num += 1
                cl.numerosity += c.numerosity
                @views filter!(e -> e != c, action_set)
                @views filter!(e -> e != c, self.population)
            end
        end
    end
end
