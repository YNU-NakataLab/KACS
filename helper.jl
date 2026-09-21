#=
    helper.jl

    Reporting and evaluation utilities shared by KACS and XCSF:
    console/CSV logging, classifier-population export, and the
    per-interval error and model-selection metrics (MAE, MSE, and AIC).
=#

using DataFrames, CSV
using Printf

"""
    Helper

Lightweight bundle of the environment and hyperparameters, used only
to dispatch the CSV-export helper functions below.
"""
mutable struct Helper
    env::Environment
    parameters::Parameters
end

function Helper(env, parameters)
    return Helper(env, parameters)
end

"""
    make_one_column_csv(filename, list)

Writes `list` as a single, headerless CSV column, e.g. for per-interval
metric time series.
"""
function make_one_column_csv(filename::String, list)
    dataframe = DataFrame(x=list)
    CSV.write(filename, dataframe, delim=',', writeheader=false)
end

"""
    make_matrix_csv(filename, list)

Writes a 2-D array (or table-like object) to `filename` as a
headerless CSV file.
"""
function make_matrix_csv(filename::String, list)
    tbl = Tables.table(list)
    CSV.write(filename, tbl, header=false)
end

"""
    get_number_of_parameters(xcs, system) -> Int64

Returns the total number of free (linear) model parameters in the
population, used as the model-complexity term `k` in the AIC
calculations below.

- `"kacs"`: every classifier contributes 2 weights (`w0`, `w_coeff`).
- otherwise (XCSF): every classifier contributes `dim + 1` weights,
  where `dim` is the input dimensionality of its condition.
"""
function get_number_of_parameters(xcs::XCS, system::String)::Int64
    if system == "kacs"
        return 2 * length(xcs.population)
    else
        dim = length(xcs.population[1].condition)
        return (dim + 1) * length(xcs.population)
    end
end

"""
    get_number_of_rules_per_p_idx(xcs, system, p) -> Int64

KACS-only: counts how many `:psi` classifiers are attached to input
feature index `p`, i.e., the local rule-population size for that
Kolmogorov-Arnold inner-function niche.
"""
function get_number_of_rules_per_p_idx(xcs::XCS, system::String, p::Int64)::Int64
    system != "kacs" && error("get_number_of_rules_per_p_idx can only be used by KACS")
    return length([clas for clas in xcs.population if clas.p_idx == p])
end

"""
    make_classifier_list(self, xcs, system) -> Array{Any,2}

Exports the full classifier population as a matrix (with a header
row) suitable for `make_matrix_csv`. The XCSF branch reports one
row per classifier with its multi-dimensional interval condition; the
KACS branch additionally reports each classifier's type (`:psi`/`:phi`)
and its `(q, p)` niche indices.
"""
function make_classifier_list(self::Helper, xcs::XCS, system::String)::Array{Any, 2}
    if system == "xcsf"
        classifier_list = Array{Any}(undef, length(xcs.population) + 1, 10)
        classifier_list[1, :] = ["ID", "Antecedent", "Consequent", "Fitness", "Error", "Accuracy",
                                  "Experience", "Time Stamp", "Match Set Size", "Numerosity"]

        i = 2
        for clf in xcs.population
            condition::String = "["
            for j in 1:length(clf.condition)
                l, u = get_lower_upper_bounds(clf.condition[j])
                condition *= string(round(l, digits=3)) * ":" * string(round(u, digits=3)) * ", "
            end
            condition = chop(condition, tail=2) * "]"

            classifier_list[i, :] = [clf.id, condition, clf.weight, clf.fitness, clf.error, clf.kappa,
                                       clf.experience, clf.time_stamp, clf.action_set_size, clf.numerosity]
            i += 1
        end
        return classifier_list
    else
        classifier_list = Array{Any}(undef, length(xcs.population) + 1, 13)
        classifier_list[1, :] = ["ID", "Antecedent", "Type", "q", "p", "Weight", "Fitness", "Error",
                                  "Accuracy", "Experience", "Time Stamp", "Match Set Size", "Numerosity"]

        i = 2
        for clf in xcs.population
            l, u = get_lower_upper_bounds(clf.condition)
            condition::String = "[" * string(round(l, digits=3)) * ", " * string(round(u, digits=3)) * "]"

            classifier_list[i, :] = [clf.id, condition, clf.type, clf.q_idx,
                                       (clf.p_idx === nothing ? "-" : clf.p_idx), clf.weights, clf.fitness,
                                       clf.error, clf.kappa, clf.experience, clf.time_stamp,
                                       clf.action_set_size, clf.numerosity]
            i += 1
        end
        return classifier_list
    end
end

"""
    make_parameter_list(args) -> Array{Any,2}

Exports the run's hyperparameters as a two-column (name, value) table
for archival alongside the results.
"""
function make_parameter_list(args)::Array{Any, 2}
    parameter_list = Array{Any}(undef, 15, 2)
    parameter_list[1, :]  = ["N", args["N"]]
    parameter_list[2, :]  = ["beta", args["beta"]]
    parameter_list[3, :]  = ["e0", args["e0"]]
    parameter_list[4, :]  = ["theta_EA", args["theta_EA"]]
    parameter_list[5, :]  = ["chi", args["chi"]]
    parameter_list[6, :]  = ["mu", args["mu"]]
    parameter_list[7, :]  = ["m0", args["m0"]]
    parameter_list[8, :]  = ["theta_del", args["theta_del"]]
    parameter_list[9, :]  = ["theta_sub", args["theta_sub"]]
    parameter_list[10, :] = ["delta", args["delta"]]
    parameter_list[11, :] = ["P_hash", args["P_hash"]]
    parameter_list[12, :] = ["r0", args["r0"]]
    parameter_list[13, :] = ["F_I", args["F_I"]]
    parameter_list[14, :] = ["tau", args["tau"]]
    parameter_list[15, :] = ["doSubsumption", args["do_subsumption"]]
    return parameter_list
end

"""
    val_with_spaces(val) -> String

Right-pads `val`'s string representation to a fixed 11-character-wide
column, for the aligned console log produced by `output_log_for_csv`.
"""
function val_with_spaces(val::Any)
    str_val = string(val)
    return " " ^ max(0, 11 - length(str_val)) * str_val * " "
end

"""
    output_log_for_csv(current_interval, num_iter_per_interval, num_interval, env, train_syserr,
                        test_syserr, popsize, covering_occur_num,
                        subsumption_occur_num, summary_list) -> Array{Any,2}

Prints one aligned console log row for the current interval and appends
the same statistics as a row of `summary_list`, which is later written
to `summary.csv`.
"""
function output_log_for_csv(current_interval::Int64, num_iter_per_interval::Int64, num_interval::Int64, env::Environment, train_syserr::Float64, test_syserr::Float64, popsize::Int64, covering_occur_num::Int64, subsumption_occur_num::Int64, summary_list::Array{Any, 2})::Array{Any, 2}
    if current_interval == 1
        println("   Interval   Iteration    TrainErr     TestErr     PopSize  CovOccRate   SubOccNum")
        println("=========== =========== =========== =========== =========== =========== ===========")
    end

    train_syserr_str = @sprintf("%.6f", train_syserr)
    test_syserr_str = @sprintf("%.6f", test_syserr)
    popsize_str = @sprintf("%.3f", popsize)

    println(val_with_spaces(current_interval), val_with_spaces(current_interval * num_iter_per_interval),
             val_with_spaces(train_syserr_str), val_with_spaces(test_syserr_str), val_with_spaces(popsize_str),
             val_with_spaces(round(covering_occur_num / num_iter_per_interval, digits=3)), val_with_spaces(subsumption_occur_num))

    summary_list[round(Int, current_interval), :] = [current_interval, current_interval * num_iter_per_interval, train_syserr,
                                                    test_syserr, popsize, covering_occur_num / num_iter_per_interval,
                                                    round(Int, subsumption_occur_num)]

    return summary_list
end

"""
    get_syserr_aic_per_interval(xcs, env, system)
        -> (train_mae, test_mae, train_mse, test_mse, aic)

Evaluates the current population on the full train and test splits and
returns:

- `train_mae`, `test_mae`: Mean absolute error.
- `train_mse`, `test_mse`: Mean squared error.
- `aic`: Akaike Information Criterion, `n_train * log(train_mse) + 2k`,
  treating the population's linear-model weights plus one shared
  residual-variance parameter as `k` free parameters
  (`get_number_of_parameters(xcs, system) + 1`).
"""
function get_syserr_aic_per_interval(
    xcs::XCS,
    env::Environment,
    system::String
)::Tuple{Float64, Float64, Float64, Float64, Float64}

    train_absolute_error = 0.0
    test_absolute_error = 0.0
    train_squared_error = 0.0
    test_squared_error = 0.0

    @inbounds for row in eachrow(env.train_data)
        state = row[1:end-1]
        target = Float64(row[end])
        pred = (system == "xcsf") ? prediction(xcs, state) : predict(xcs, state, true)[1]
        residual = pred - target
        train_absolute_error += abs(residual)
        train_squared_error += residual^2
    end

    @inbounds for row in eachrow(env.test_data)
        state = row[1:end-1]
        target = Float64(row[end])
        pred = (system == "xcsf") ? prediction(xcs, state) : predict(xcs, state, true)[1]
        residual = pred - target
        test_absolute_error += abs(residual)
        test_squared_error += residual^2
    end

    n_train = size(env.train_data, 1)
    n_test = size(env.test_data, 1)

    train_mae = n_train > 0 ? train_absolute_error / n_train : 0.0
    test_mae = n_test > 0 ? test_absolute_error / n_test : 0.0
    train_mse = n_train > 0 ? train_squared_error / n_train : 0.0
    test_mse = n_test > 0 ? test_squared_error / n_test : 0.0

    if n_train == 0
        return (train_mae, test_mae, train_mse, test_mse, Inf)
    end

    # Model complexity: classifier weights + one shared residual-variance parameter.
    k = get_number_of_parameters(xcs, system) + 1

    # Guard against log(0) when the training MSE is numerically zero.
    rss_per_sample = max(train_mse, eps(Float64))
    aic = n_train * log(rss_per_sample) + 2 * k

    return (train_mae, test_mae, train_mse, test_mse, aic)
end

"""
    prediction(xcs, state) -> Float64

XCSF-only exploitation-mode prediction helper: matches `state` against
the population (triggering no covering) and returns the fitness-weighted
combined prediction of the resulting match set.
"""
function prediction(xcs::XCS, state::Vector{Union{Float64, Int64, String}})::Float64
    match_set::Vector{Classifier} = @views generate_match_set(xcs, state, true)
    return generate_prediction(xcs, match_set, state)
end