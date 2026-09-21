#=
    main.jl

    Entry point for KACS / XCSF regression experiments on CSV
    datasets. Parses command-line hyperparameters, wires together the
    environment, parameters, and learner modules, runs one or more
    training/evaluation trials with independent train/test splits, and
    writes per-interval metrics (error, population size, AIC,
    operator counts) plus the final classifier population to disk.

    Usage examples:
        julia main.jl --csv dataset/example.csv -s kacs -i 100000 --num_trials 30
        julia main.jl --csv dataset/example.csv -s xcsf
        julia main.jl -s kacs --all true   # batch mode, see main_all_csv

    Run `julia main.jl --help` for the full list of hyperparameters.
=#

using ArgParse
using Random
using CSV
using DataFrames
using Dates
using Pkg

"""
    parse_commandline() -> Dict

Defines and parses every KACS/XCSF hyperparameter and run-control
option (dataset path, number of trials/iterations, population size, GA and
covering parameters, etc.). See parameters.jl for the algorithmic
meaning of each hyperparameter.
"""
function parse_commandline()
    s = ArgParseSettings(description="X-KAN classifier system")
    @add_arg_table s begin
        "--num_trials"
            help = "Number of independent trials (train/test splits) to run"
            arg_type = Int
            default = 30
        "--iteration", "-i"
            help = "Number of training iterations per trial. Must be an exact " *
                   "multiple of the fixed reporting-interval size " *
                   "(env.num_iter_per_interval = 2000, set in " *
                   "environment/csv.jl's Environment constructor): performance " *
                   "is evaluated and logged every num_iter_per_interval " *
                   "iterations, so a non-multiple value would silently run more " *
                   "iterations than requested (main_csv rounds the number of " *
                   "intervals up via ceil()). main_csv raises an error at " *
                   "startup if this is not satisfied."
            arg_type = Int
            default = 100000
        "--csv"
            help = "Path to the input CSV dataset"
            arg_type = String
            default = nothing
        "-a", "--all"
            help = "Run the batch mode over the dataset list in main_all_csv"
            arg_type = Bool
            default = nothing
        "-N"
            help = "Maximum population size (total numerosity)"
            arg_type = Int
            default = 6400
        "--beta"
            help = "Learning rate for updating fitness and match-set-size estimates"
            arg_type = Float64
            default = 0.2
        "--e0"
            help = "Error threshold under which a classifier's accuracy is set to one"
            arg_type = Float64
            default = 0.01
        "--theta_EA"
            help = "Minimum time between GA invocations in a match set"
            arg_type = Int
            default = 50
        "--chi"
            help = "Probability of applying crossover"
            arg_type = Float64
            default = 0.8
        "--mu"
            help = "Probability of mutating one allele"
            arg_type = Float64
            default = 0.04
        "--theta_del"
            help = "Experience threshold above which fitness affects deletion probability"
            arg_type = Int
            default = 50
        "--theta_sub"
            help = "Experience threshold required for subsumption eligibility"
            arg_type = Int
            default = 50
        "--delta"
            help = "Fraction of mean population fitness used as the deletion penalty threshold"
            arg_type = Float64
            default = 0.1
        "--r0"
            help = "Maximum spread used by the covering operator"
            arg_type = Float64
            default = 1.0
        "--m0"
            help = "Maximum magnitude of a condition mutation"
            arg_type = Float64
            default = 0.1
        "--F_I"
            help = "Initial fitness assigned to a newly generated classifier"
            arg_type = Float64
            default = 0.01
        "--tau"
            help = "Tournament size for parent selection (0 = roulette-wheel selection)"
            arg_type = Float64
            default = 0.4
        "--do_subsumption"
            help = "Whether GA offspring are tested for subsumption by their parents"
            arg_type = Bool
            default = true
        "--P_hash"
            help = "Probability of generating a wildcard ('don't care') condition during covering"
            arg_type = Float64
            default = 1.0
        "-s", "--system"
            help = "Which learner to run: 'kacs' or 'xcsf'"
            arg_type = String
            default = "kacs"
    end
    return parse_args(s)
end

"""
    main_csv(args; now_str=<timestamp>)

Runs `args["num_trials"]` independent trials on the dataset given by
`args["csv"]`. Each trial re-shuffles the train/test split, trains for
`args["iteration"]` iterations (online updates), evaluates on
both splits after every interval (`env.num_iter_per_interval` iterations),
logs progress to the console, and writes all per-interval metrics plus
the final population and hyperparameters to
`./result/<dataset>/<system>/<timestamp>/trial<n>/`.

Raises an error immediately if `args["iteration"]` is not an exact
multiple of `env.num_iter_per_interval`, since the training loop below
always executes whole intervals of `env.num_iter_per_interval`
iterations; a non-multiple `args["iteration"]` would otherwise be
silently rounded up (via `ceil`) and more iterations would run than
requested.
"""
function main_csv(args; now_str=Dates.format(Dates.now(), "Y-m-d-H-M-S"))
    env = Environment(args)

    if args["iteration"] % env.num_iter_per_interval != 0
        error("--iteration ($(args["iteration"])) must be an exact multiple of " *
              "the reporting-interval size (env.num_iter_per_interval = " *
              "$(env.num_iter_per_interval)). Choose a value evenly divisible " *
              "by $(env.num_iter_per_interval), e.g. " *
              "$(env.num_iter_per_interval * max(1, div(args["iteration"], env.num_iter_per_interval))) " *
              "or $(env.num_iter_per_interval * (div(args["iteration"], env.num_iter_per_interval) + 1)).")
    end

    param = Parameters(args)
    helper = Helper(env, param)

    println("[ Settings ]")
    println("    Environment = ", get_environment_name(env))
    println("          #Iter = ", args["iteration"])
    println("          #Inst = $(size(env.all_data, 1)) (Train:Test = $(size(env.train_data, 1)):$(size(env.test_data, 1)))")
    println("           #Fea = ", env.state_length)

    println("[ $(args["system"]) General Parameters ]")
    println("              N = ", param.N)
    println("           beta = ", param.beta)
    println("      epsilon_0 = ", param.e0)
    println("       theta_EA = ", param.theta_EA)
    println("            chi = ", param.chi)
    println("             mu = ", param.mu)
    println("      theta_del = ", param.theta_del)
    println("      theta_sub = ", param.theta_sub)
    println("          delta = ", param.delta)
    println("            m_0 = ", param.m0)
    println("            r_0 = ", param.r0)
    println("            F_I = ", param.F_I)
    println("  doSubsumption = ", Bool(param.do_subsumption))
    println("         P_hash = ", param.P_hash)
    println("            tau = ", param.tau)

    @time for n in 1:args["num_trials"]
        xcs::XCS = XCS(env, param)
        env.seed = n - 1
        Random.seed!(env.seed)
        shuffle_train_and_test_data!(env)
        accum_train_time = accum_test_time = 0

        println("\n[ Trial $(env.seed) / $(args["num_trials"]-1) ]\n")

        num_interval = div(args["iteration"], env.num_iter_per_interval)
        train_syserr_list = Vector{Float64}(undef, num_interval)
        test_syserr_list = Vector{Float64}(undef, num_interval)
        train_mse_list = Vector{Float64}(undef, num_interval)
        test_mse_list = Vector{Float64}(undef, num_interval)
        popsize_list = Vector{Int64}(undef, num_interval)
        micro_popsize_list = Vector{Int64}(undef, num_interval)
        number_of_parameters_list = Vector{Int64}(undef, num_interval)
        aic_list = Vector{Float64}(undef, num_interval)
        number_of_covering_list = Vector{Float64}(undef, num_interval)
        number_of_deletion_list = Vector{Float64}(undef, num_interval)
        train_time_list = Vector{Float64}(undef, 1)
        test_time_list = Vector{Float64}(undef, 1)
        summary_list = Array{Any}(undef, num_interval, 7)

        is_kacs = args["system"] == "kacs"
        if is_kacs
            number_of_covering_inner_list = Vector{Float64}(undef, num_interval)
            number_of_covering_outer_list = Vector{Float64}(undef, num_interval)
            number_of_deletion_inner_list = Vector{Float64}(undef, num_interval)
            number_of_deletion_outer_list = Vector{Float64}(undef, num_interval)
        end

        for e in 1:num_interval
            # --- Train ---
            env.is_exploit = false
            for _ in 1:env.num_iter_per_interval
                run_experiment(xcs)
            end

            # --- Evaluate ---
            env.is_exploit = true
            train_syserr, test_syserr, train_mse, test_mse, aic = get_syserr_aic_per_interval(xcs, env, args["system"])

            popsize::Int64 = length(xcs.population)
            nop::Int64 = get_number_of_parameters(xcs, args["system"])

            train_syserr_list[e] = train_syserr
            test_syserr_list[e] = test_syserr
            train_mse_list[e] = train_mse
            test_mse_list[e] = test_mse
            popsize_list[e] = popsize
            micro_popsize_list[e] = mapreduce(clas -> clas.numerosity, +, xcs.population)
            number_of_parameters_list[e] = nop
            aic_list[e] = aic
            number_of_covering_list[e] = xcs.covering_occur_num
            number_of_deletion_list[e] = xcs.deletion_occur_num
            if is_kacs
                number_of_covering_inner_list[e] = xcs.covering_inner_num
                number_of_covering_outer_list[e] = xcs.covering_outer_num
                number_of_deletion_inner_list[e] = xcs.deletion_inner_num
                number_of_deletion_outer_list[e] = xcs.deletion_outer_num
            end

            summary_list = output_log_for_csv(e, env.num_iter_per_interval, num_interval, env, train_syserr, test_syserr, popsize,
                                               xcs.covering_occur_num, xcs.subsumption_occur_num, summary_list)

            xcs.covering_occur_num = xcs.subsumption_occur_num = xcs.deletion_occur_num = 0
            if is_kacs
                xcs.covering_inner_num = xcs.covering_outer_num = xcs.deletion_inner_num = xcs.deletion_outer_num = 0
            end
        end

        dir_path = if args["all"] === nothing
            "./result/" * get_environment_name(env) * "/$(args["system"])/" * now_str * "/trial" * string(n - 1)
        else
            "./all" * now_str * "/$(args["system"])/" * basename(get_environment_name(env)) * "/trial" * string(n - 1)
        end

        mkpath(dir_path)
        make_one_column_csv(joinpath(dir_path, "train_syserr.csv"), train_syserr_list)
        make_one_column_csv(joinpath(dir_path, "test_syserr.csv"), test_syserr_list)
        make_one_column_csv(joinpath(dir_path, "train_mse.csv"), train_mse_list)
        make_one_column_csv(joinpath(dir_path, "test_mse.csv"), test_mse_list)
        make_one_column_csv(joinpath(dir_path, "popsize.csv"), popsize_list)
        make_one_column_csv(joinpath(dir_path, "micro_popsize.csv"), micro_popsize_list)
        make_one_column_csv(joinpath(dir_path, "number_of_parameters.csv"), number_of_parameters_list)
        make_one_column_csv(joinpath(dir_path, "aic.csv"), aic_list)
        make_one_column_csv(joinpath(dir_path, "number_of_covering.csv"), number_of_covering_list)
        make_one_column_csv(joinpath(dir_path, "number_of_deletion.csv"), number_of_deletion_list)
        make_matrix_csv(joinpath(dir_path, "classifier.csv"), make_classifier_list(helper, xcs, args["system"]))
        make_matrix_csv(joinpath(dir_path, "summary.csv"), summary_list)
        make_matrix_csv(joinpath(dir_path, "parameter.csv"), make_parameter_list(args))

        if is_kacs
            make_one_column_csv(joinpath(dir_path, "number_of_covering_inner.csv"), number_of_covering_inner_list)
            make_one_column_csv(joinpath(dir_path, "number_of_covering_outer.csv"), number_of_covering_outer_list)
            make_one_column_csv(joinpath(dir_path, "number_of_deletion_inner.csv"), number_of_deletion_inner_list)
            make_one_column_csv(joinpath(dir_path, "number_of_deletion_outer.csv"), number_of_deletion_outer_list)
        end

    end
end

"""
    main_all_csv(args)

Batch mode: runs `main_csv` sequentially over every dataset name listed
in `csv_list_array`, looking for `./dataset/<name>.csv`. `args["P_hash"]`
is set per dataset (0.8 for real-world benchmarks with wildcard
covering enabled, 0.0 for datasets where every feature should always be
specialized). Edit `csv_list_array` to select which datasets to run.
"""
function main_all_csv(args)
    now_str = Dates.format(Dates.now(), "Y-m-d")
    dir_path::String = "./dataset/"

    # Add the dataset base names (without the .csv extension) to run in batch.
    csv_list_array::Vector{String} = [
        "f1_rastrigin_10dim_1000", "f2_rosenbrock_10dim_1000", "f3_cross_10dim_1000", "f4_styblinski_10dim_1000",
        "asn", "ccpp", "cs", "eec",
    ]

    # Datasets that benefit from wildcard ("don't care") covering.
    wildcard_datasets = Set([
        "asn", "ccpp", "cs", "eec",
    ])

    for csv::String in csv_list_array
        args["csv"] = dir_path * csv * ".csv"
        args["P_hash"] = csv in wildcard_datasets ? 0.8 : 0.0
        main_csv(args; now_str)
    end
end

args = parse_commandline()
include("./environment/csv.jl")
include("parameters.jl")

# System setup: select the KACS or XCSF learner implementation.
if args["system"] == "xcsf"
    include("./xcsf/xcs.jl")
elseif args["system"] == "kacs"
    include("./kacs/xcs.jl")
end
include("helper.jl")

if args["csv"] === nothing && args["all"]
    main_all_csv(args)
elseif args["csv"] !== nothing && !args["all"]
    main_csv(args)
else
    error("Set either args[\"csv\"] or args[\"all\"]")
end