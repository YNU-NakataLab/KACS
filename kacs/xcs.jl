#=
    kacs/xcs.jl

    Core learning loop of KACS (Kolmogorov-Arnold Classifier System).

    KACS approximates a target function f(x_1, ..., x_n) using the
    Kolmogorov-Arnold representation theorem's two-stage decomposition:

        f(x) = sum_{q=1}^{2n+1} phi_q( sum_{p=1}^{n} psi_{q,p}(x_p) )

    Each inner function psi_{q,p} and each outer function phi_q is
    represented, in turn, by its own XCS-style rule population with
    univariate linear ("computed") predictions (Wilson, 2002) rather
    than a single fixed-form basis function. This file implements the
    match-set generation, prediction, parameter update, and genetic
    algorithm (covering / crossover / mutation / subsumption /
    deletion) that drive that population, following the standard XCS
    algorithmic template (Wilson, 1995, 1998; Butz & Wilson, 2002),
    adapted to this two-stage function-approximation setting.
=#

include("classifier.jl")
using Distributions
using LinearAlgebra

"""
    XCS

Top-level KACS learner: holds one shared classifier population that
covers both the `2n+1` outer (`:phi`) niches and the `n * (2n+1)` inner
(`:psi`) niches of the Kolmogorov-Arnold decomposition.

# Fields
- `env`: Data environment (see environment/csv.jl).
- `parameters`: Hyperparameters (see parameters.jl).
- `population`: All classifiers (both `:psi` and `:phi` types).
- `time_stamp`: Number of `run_experiment` calls so far.
- `covering_occur_num`, `subsumption_occur_num`, `deletion_occur_num`:
  Per-interval operator trigger counters (reset in main.jl's logging step).
- `covering_inner_num`, `covering_outer_num`, `deletion_inner_num`,
  `deletion_outer_num`: Same counters, split by `:psi` (inner) vs.
  `:phi` (outer) niche.
- `global_id`: Monotonically increasing classifier ID counter.
- `num_p`: Number of input features `n`.
- `num_q`: Number of outer summation terms, `2n + 1`.
"""
mutable struct XCS
    env::Environment
    parameters::Parameters
    population::Vector{Classifier}
    time_stamp::Int64
    covering_occur_num::Int64
    subsumption_occur_num::Int64
    deletion_occur_num::Int64
    covering_inner_num::Int64
    covering_outer_num::Int64
    deletion_inner_num::Int64
    deletion_outer_num::Int64
    global_id::Int64
    num_p::Int64
    num_q::Int64
end

"""
    XCS(env, parameters) -> XCS

Initializes an empty population sized for `n = env.state_length`
input features, with `num_q = 2n + 1` outer terms as prescribed by the
Kolmogorov-Arnold representation theorem.
"""
function XCS(env, parameters)
    n = env.state_length
    num_p = n
    num_q = 2 * n + 1
    return XCS(env, parameters, [], 0, 0, 0, 0, 0, 0, 0, 0, 0, num_p, num_q)
end

"""
    run_experiment(self)

Runs one full online-learning step: draws a training sample, predicts
its target, updates classifier weights/fitness from the resulting
error, then invokes the genetic algorithm in every inner and outer
match set. Advances `self.time_stamp` by one.
"""
function run_experiment(self::XCS)
    curr_state::Vector{Union{Float64, Int64, String}} = state(self.env)
    y::Float64 = answer(self.env, curr_state)

    y_hat, s_q_values, match_sets_psi, match_sets_phi = predict(self, curr_state)
    update_set!(self, curr_state, y, y_hat, s_q_values, match_sets_psi, match_sets_phi)

    for q in 1:self.num_q
        match_set_phi = get(match_sets_phi, q, Vector{Classifier}())
        run_ga!(self, match_set_phi, curr_state)
    end
    for q in 1:self.num_q
        for p in 1:self.num_p
            match_set_psi = get(match_sets_psi, (q, p), Vector{Classifier}())
            run_ga!(self, match_set_psi, curr_state)
        end
    end

    self.time_stamp += 1
end

"""
    predict(self, state, do_exploit=false)

Computes the system prediction `y_hat` for `state` using the two-stage
Kolmogorov-Arnold decomposition:

1. **Inner stage**: for every outer index `q` and feature index `p`,
   match the scalar input `x_p` against the `:psi_{q,p}` niche,
   triggering covering if the niche is empty and the system is
   training (`do_exploit == false`). Combine matching classifiers'
   linear predictions into `psi_hat_{q,p}`, then sum over `p` to get
   the intermediate value `s_q`.
2. **Outer stage**: for every outer index `q`, match `s_q` against the
   `:phi_q` niche (again covering if empty and training), and sum the
   resulting `phi_hat_q` values into the final prediction `y_hat`.

Returns `(y_hat, s_q_values, match_sets_psi, match_sets_phi)` so that
`update_set!` and `run_ga!` can reuse the match sets without
recomputing them.
"""
function predict(self::XCS, state::Vector{Union{Float64, Int64, String}}, do_exploit=false)::Tuple{Float64, Dict{Int64, Float64}, Dict{Tuple{Int64, Int64}, Vector{Classifier}}, Dict{Int64, Vector{Classifier}}}
    psi_hat_values = Dict{Tuple{Int64, Int64}, Float64}()
    s_q_values = Dict{Int64, Float64}()

    match_sets_psi = Dict{Tuple{Int64, Int64}, Vector{Classifier}}()
    match_sets_phi = Dict{Int64, Vector{Classifier}}()

    x0_constant = 1.0

    # --- Stage 1: inner functions psi_{q,p} and intermediate sums s_q ---
    for q in 1:self.num_q
        for p in 1:self.num_p
            x_p = state[p]
            match_set_psi = generate_match_set(self, x_p, :psi, q, p)

            if isempty(match_set_psi) && !do_exploit
                covering_classifier = generate_covering_classifier(self, x_p, :psi, q, p)
                push!(self.population, covering_classifier)
                delete_from_population!(self, true)
                push!(match_set_psi, covering_classifier)
            end

            match_sets_psi[(q, p)] = match_set_psi
            psi_hat_values[(q, p)] = weighted_average_prediction(match_set_psi, x_p, x0_constant)
        end
    end

    @simd for q in 1:self.num_q
        s_q_values[q] = sum(psi_hat_values[q, p] for p in 1:self.num_p)
    end

    # --- Stage 2: outer functions phi_q and final prediction y_hat ---
    y_hat::Float64 = 0.0

    for q in 1:self.num_q
        s_q = s_q_values[q]
        match_set_phi = generate_match_set(self, s_q, :phi, q)

        if isempty(match_set_phi) && !do_exploit
            covering_classifier = generate_covering_classifier(self, s_q, :phi, q)
            push!(self.population, covering_classifier)
            delete_from_population!(self, true)
            push!(match_set_phi, covering_classifier)
        end

        match_sets_phi[q] = match_set_phi
        y_hat += weighted_average_prediction(match_set_phi, s_q, x0_constant)
    end

    return y_hat, s_q_values, match_sets_psi, match_sets_phi
end

"""
    generate_match_set(self, input_val, type, q_idx, p_idx=nothing)

Returns every population classifier of the given `type` (`:psi` or
`:phi`) and niche index (`q_idx`, and `p_idx` for `:psi`) whose
condition matches `input_val`.
"""
function generate_match_set(self::XCS, input_val::Float64, type::Symbol, q_idx::Int, p_idx::Union{Int, Nothing}=nothing)::Vector{Classifier}
    match_set::Vector{Classifier} = []
    for cl in self.population
        if cl.type == type && cl.q_idx == q_idx
            if type == :psi && cl.p_idx != p_idx
                continue
            end
            if does_match(cl.condition, input_val)
                push!(match_set, cl)
            end
        end
    end
    return match_set
end

"""
    weighted_average_prediction(match_set, input_val, x0_constant) -> Float64

Fitness-weighted average of the linear "computed" predictions
(Wilson, 2002) of every classifier in `match_set`:

    sum_i F_i * (w0_i + w_coeff_i * input_val) / sum_i F_i

Returns `0.0` if the match set is empty or has zero total fitness.
"""
function weighted_average_prediction(match_set::Vector{Classifier}, input_val::Union{Float64, Int64, String}, x0_constant::Float64)::Float64
    if isempty(match_set)
        return 0.0
    end

    sum_weighted_predictions = 0.0
    sum_fitness = 0.0

    for cl in match_set
        sum_weighted_predictions += (cl.weights[1] * x0_constant + cl.weights[2] * input_val) * cl.fitness
        sum_fitness += cl.fitness
    end

    return sum_fitness == 0.0 ? 0.0 : sum_weighted_predictions / sum_fitness
end

"""
    generate_covering_classifier(self, input_val, type, q_idx, p_idx=nothing) -> Classifier

Generates a new classifier for the given niche via the covering
operator, assigns it a fresh ID, stamps it with the current time step,
and updates the appropriate covering counters.
"""
function generate_covering_classifier(self::XCS, input_val::Union{Float64, Int64, String}, type::Symbol, q_idx::Int64, p_idx::Union{Int, Nothing}=nothing)::Classifier
    clas::Classifier = Classifier(self.parameters, input_val, type, q_idx, p_idx)
    clas.time_stamp = self.time_stamp

    self.covering_occur_num += 1
    if type == :psi
        self.covering_inner_num += 1
    else
        self.covering_outer_num += 1
    end

    clas.id = self.global_id
    self.global_id += 1
    return clas
end

"""
    update_set!(self, x, y, y_hat, s_q_values, match_sets_psi, match_sets_phi)

Updates classifier weights and quality parameters after observing the
true target `y` and the system prediction `y_hat`.

Following the two-stage decomposition, the global error signal
`error_signal = y - y_hat` is propagated to:

- **Outer (`:phi`) classifiers**: weight gradients are proportional to
  the classifier's share of fitness in its match set, `F_i / sum F`.
- **Inner (`:psi`) classifiers**: the same error signal is additionally
  scaled by `bar_ws_phi_q`, the fitness-weighted average slope
  (`w_coeff`) of the outer classifiers in niche `q`, which approximates
  d(phi_q)/d(s_q) and thus back-propagates the error through the outer
  stage (a discrete analogue of the chain rule).

Both weight updates are performed via Adam (Kingma & Ba, 2015; see
`adam!` in kacs/classifier.jl). Error, fitness, and action-set-size
estimates are then updated for every classifier that participated in a
non-empty match set (see `update_parameters!` / `update_fitness!`).
"""
function update_set!(self::XCS, x::Vector{Union{Float64, Int64, String}}, y::Float64, y_hat::Float64,
    s_q_values::Dict{Int64, Float64}, match_sets_psi::Dict{Tuple{Int64, Int64}, Vector{Classifier}},
    match_sets_phi::Dict{Int64, Vector{Classifier}})

    x0_constant = 1.0

    error_signal = y - y_hat
    abs_error_signal = abs(error_signal)

    # --- Outer (:phi) classifiers ---
    for q in 1:self.num_q
        match_set_phi = get(match_sets_phi, q, Vector{Classifier}())
        isempty(match_set_phi) && continue

        sum_F_phi_q = sum(cl.fitness for cl in match_set_phi)
        s_q = s_q_values[q]

        if sum_F_phi_q > 0
            for cl_phi in match_set_phi
                adam!(cl_phi, (-error_signal * (cl_phi.fitness / sum_F_phi_q) * x0_constant), 1)
                adam!(cl_phi, (-error_signal * (cl_phi.fitness / sum_F_phi_q) * s_q), 2)
            end
            update_parameters!(self, match_set_phi, abs_error_signal, 1 / sum_F_phi_q)
        end
    end

    # --- Inner (:psi) classifiers ---
    for q in 1:self.num_q
        # Fitness-weighted average outer slope, used as the chain-rule factor
        # that back-propagates the error through phi_q into psi_{q,p}.
        match_set_phi_for_bar_ws = get(match_sets_phi, q, Vector{Classifier}())
        bar_ws_phi_q = 0.0
        if !isempty(match_set_phi_for_bar_ws)
            sum_F_phi_for_bar_ws = sum(cl.fitness for cl in match_set_phi_for_bar_ws)
            if sum_F_phi_for_bar_ws > 0
                bar_ws_phi_q = sum(cl.weights[2] * cl.fitness for cl in match_set_phi_for_bar_ws) / sum_F_phi_for_bar_ws
            end
        end

        for p in 1:self.num_p
            match_set_psi = get(match_sets_psi, (q, p), Vector{Classifier}())
            isempty(match_set_psi) && continue

            sum_F_psi_qp = sum(cl.fitness for cl in match_set_psi)
            x_p = x[p]

            if sum_F_psi_qp > 0
                for cl_psi in match_set_psi
                    adam!(cl_psi, (-error_signal * bar_ws_phi_q * (cl_psi.fitness / sum_F_psi_qp) * x0_constant), 1)
                    adam!(cl_psi, (-error_signal * bar_ws_phi_q * (cl_psi.fitness / sum_F_psi_qp) * x_p), 2)
                end
                update_parameters!(self, match_set_psi, abs_error_signal, bar_ws_phi_q / sum_F_psi_qp)
            end
        end
    end
end

"""
    update_parameters!(self, match_set, abs_error_signal, scaling_factor)

Updates experience, error (moving average toward `abs_error_signal`),
and estimated action-set size for every classifier in `match_set`,
then refreshes accuracy-based fitness via `update_fitness!`.
"""
function update_parameters!(self::XCS, match_set::Vector{Classifier}, abs_error_signal::Float64, scaling_factor::Float64)
    set_numerosity = sum(clas.numerosity for clas in match_set)

    for cl in match_set
        cl.experience += 1
        cl.error += self.parameters.beta * (abs_error_signal - cl.error)
        cl.action_set_size += self.parameters.beta * (set_numerosity - cl.action_set_size)
    end

    update_fitness!(self, match_set)
end

"""
    update_fitness!(self, match_set)

Standard XCS accuracy-based fitness update (Wilson, 1995):
classifier accuracy `kappa` is `1` if `error < e0`, and
`(error / e0)^-1` otherwise; fitness is then moved toward the
classifier's share of the match set's total (accuracy x numerosity).
"""
function update_fitness!(self::XCS, match_set::Vector{Classifier})
    kappa = Dict{Classifier, Float64}()
    beta = self.parameters.beta
    e0 = self.parameters.e0

    for clas in match_set
        kappa[clas] = clas.error < e0 ? 1.0 : (clas.error / e0)^(-1)
        clas.kappa = kappa[clas]
    end

    accuracy_sum = sum(kappa[clas] * clas.numerosity for clas in match_set)

    @inbounds for clas in match_set
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
condition already exists in the same niche, increments that
classifier's numerosity instead (macro-classifier merging).
"""
function insert_in_population!(self::XCS, clas::Classifier)
    @inbounds for c in self.population
        if is_equal_condition(c, clas) && c.type == clas.type && c.q_idx == clas.q_idx
            if clas.type == :psi && c.p_idx != clas.p_idx
                continue
            end
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
                if clas.type == :psi
                    self.deletion_inner_num += 1
                else
                    self.deletion_outer_num += 1
                end
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
