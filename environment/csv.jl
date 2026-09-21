#=
    environment/csv.jl

    CSV-file "environment" used by both KACS and XCSF. Loads a dataset
    where every row is one observation, the last column is the target
    value (regression response), and the remaining columns are the
    input features. Features are linearly rescaled to [0, 1] and the
    target is rescaled to [-1, 1] before training, matching the
    normalization convention typically used for XCS-family regression
    benchmarks.

    The train/test split follows a single Monte Carlo cross-validation
    split (90% train / 10% test), re-shuffled independently for every
    trial (see `shuffle_train_and_test_data!`).
=#

using CSV, DataFrames, Random

"""
    Environment

Holds the dataset and the bookkeeping state needed to stream training
samples (with per-epoch shuffling) and to evaluate a trained population
on held-out test data.

# Fields
- `num_iter_per_interval`: Number of online learning steps that make up
  one reporting/evaluation interval (currently fixed to 2000).
- `seed`: RNG seed for the current trial.
- `is_exploit`: `false` while training (explore/learn), `true` while
  evaluating on train/test data without further learning.
- `state_length`: Number of input features (columns minus the target).
- `row_index`: Pointer into `index_array`, advanced by `state()`.
- `train_data`, `test_data`: Normalized train/test matrices.
- `all_data`: Full dataset before the train/test split of the current
  trial (re-shuffled every trial).
- `file_path`: Path to the source CSV file.
- `index_array`: Shuffled row indices used to draw training samples
  without replacement within an epoch.
"""
mutable struct Environment
    num_iter_per_interval::Int64
    seed::Int64
    is_exploit::Bool
    state_length::Int64
    row_index::Int64
    train_data::Array{Union{Float64, Int64, String}, 2}
    test_data::Array{Union{Float64, Int64, String}, 2}
    all_data::Array{Union{Float64, Int64, String}, 2}
    file_path::String
    index_array::Vector{Int64}
end

"""
    Environment(args) -> Environment

Constructs an `Environment` by loading the CSV file specified in
`args["csv"]` and allocating (but not yet populating) the train/test
matrices. The actual split and normalization is performed later by
`shuffle_train_and_test_data!`, once per trial.
"""
function Environment(args)::Environment
    train_data, test_data, all_data = get_train_and_test_data_and_all_data(args["csv"], args)
    _num_actions, state_length = get_data_information(args["csv"])
    return Environment(2000, 0, false, state_length, 0, train_data, test_data, all_data, args["csv"], Vector(1:size(train_data, 1)))
end

"""
    get_train_and_test_data_and_all_data(filename, args)

Reads the CSV file and allocates empty train/test matrices sized
according to a 9:1 Monte Carlo cross-validation split. The returned
`all_data` matrix holds the raw (un-normalized) dataset; the returned
train/test matrices are placeholders filled in by
`shuffle_train_and_test_data!`.
"""
function get_train_and_test_data_and_all_data(filename::String, args)::Tuple{Array{Union{Float64, Int64, String}, 2}, Array{Union{Float64, Int64, String}, 2}, Array{Union{Float64, Int64, String}, 2}}
    all_data = CSV.File(filename; header=false) |> DataFrame
    all_data = Matrix(all_data)

    train_ratio = 0.9  # Monte Carlo CV split (train:test = 9:1)
    train_data_length::Int64 = Int(floor(size(all_data, 1) * train_ratio))
    test_data_length::Int64 = size(all_data, 1) - train_data_length

    train_data = Array{Any}(undef, train_data_length, size(all_data, 2))
    test_data = Array{Any}(undef, test_data_length, size(all_data, 2))

    return train_data, test_data, all_data
end

"""
    get_data_information(filename) -> (num_actions, state_length)

Returns the number of actions (always 1, since KACS/XCSF here solve a
regression task rather than a classification/RL task) and the number of
input features, inferred from the CSV column count minus the target
column.
"""
function get_data_information(filename::String)::Tuple{Int64, Int64}
    all_data = CSV.read(filename, DataFrame, header=false)
    num_actions::Int64 = 1
    state_length::Int64 = size(all_data, 2) - 1
    return num_actions, state_length
end

"""
    normalize_columns(data_train, data_test)

Min-max normalizes every input feature (columns `1:end-1`) to `[0, 1]`
using statistics computed from the training split only, and rescales
the target column (`end`) to `[-1, 1]`. Missing values, encoded as the
string `"?"`, are passed through unchanged so that classifier matching
can treat them as wildcards (see `does_match` in kacs/classifier.jl).
"""
function normalize_columns(data_train, data_test)::Tuple{Array{Union{Float64, Int64, String}, 2}, Array{Union{Float64, Int64, String}, 2}}
    # Normalize input features to [0, 1].
    @simd for j = 1:(size(data_train, 2) - 1)
        col = data_train[:, j]
        col_without_missing = col[col .!= "?"]
        col_without_missing = map(x -> parse(Float64, string(x)), col_without_missing)
        col_min = minimum(col_without_missing)
        col_max = maximum(col_without_missing)
        for data in (data_train, data_test)
            @simd for i = 1:size(data, 1)
                if data[i, j] != "?"
                    v = parse(Float64, string(data[i, j]))
                    if col_min == col_max
                        data[i, j] = 0.5
                    else
                        data[i, j] = max(0, min(1, (v - col_min) / (col_max - col_min)))
                    end
                else
                    data[i, j] = "?"
                end
            end
        end
    end

    # Normalize the target column to [-1, 1].
    j = size(data_train, 2)
    col = data_train[:, j]
    col_without_missing = col[col .!= "?"]
    col_without_missing = map(x -> parse(Float64, string(x)), col_without_missing)
    col_min = minimum(col_without_missing)
    col_max = maximum(col_without_missing)
    for data in (data_train, data_test)
        @simd for i = 1:size(data, 1)
            v = parse(Float64, string(data[i, j]))
            data[i, j] = max(-1, min(1, (2 * (v - col_min) / (col_max - col_min) - 1)))
        end
    end

    return data_train, data_test
end

"""
    shuffle_index_array_and_reset_row_index!(self)

Reshuffles `self.index_array` once every epoch (i.e., after all training
rows have been drawn once), so that samples are drawn without
replacement within an epoch and in a new random order across epochs.
"""
function shuffle_index_array_and_reset_row_index!(self::Environment)
    rng = MersenneTwister(self.seed)
    if self.row_index % size(self.train_data, 1) == 0
        self.index_array = Vector(1:size(self.train_data, 1))
        shuffle!(rng, self.index_array)
        self.row_index = 0
    end
end

"""
    shuffle_train_and_test_data!(self)

Draws a fresh 9:1 train/test split from `self.all_data` (re-permuted
using the trial's seed) and re-normalizes both splits. Called once at
the start of every trial in `main_csv`.
"""
function shuffle_train_and_test_data!(self::Environment)
    rng = MersenneTwister(self.seed)
    train_data_length::Int64 = size(self.train_data, 1)
    test_data_length::Int64 = size(self.test_data, 1)

    train_data = Array{Any}(undef, train_data_length, size(self.all_data, 2))
    test_data = Array{Any}(undef, test_data_length, size(self.all_data, 2))

    perm::Vector{Int64} = shuffle(rng, 1:train_data_length + test_data_length)
    self.all_data = self.all_data[perm, :]

    for i = 1:train_data_length
        train_data[i, :] = self.all_data[i, :]
    end
    for i = 1:test_data_length
        test_data[i, :] = self.all_data[train_data_length + i, :]
    end

    self.train_data, self.test_data = normalize_columns(train_data, test_data)
end

"""
    state(self) -> Vector

Draws the next training sample's feature vector, advancing the internal
row pointer (and reshuffling at epoch boundaries). Only valid while
`self.is_exploit == false`; use `train_data`/`test_data` directly for
evaluation.
"""
function state(self::Environment)::Vector{Union{Float64,Int64,String}}
    if self.is_exploit == false
        shuffle_index_array_and_reset_row_index!(self)

        self.row_index >= length(self.index_array) && error("Data pointer out of bounds")

        self.row_index += 1
        row_idx = self.index_array[self.row_index]
        return self.train_data[row_idx, 1:end-1]
    else
        throw(ErrorException("State sampling unavailable during exploitation phase. Use test data directly."))
    end
end

"""
    answer(self, state) -> Float64

Returns the ground-truth target value associated with the sample most
recently drawn by `state()`.
"""
function answer(self::Environment, state::Vector{Union{Float64, Int64, String}})::Float64
    if self.is_exploit == false
        return Float64(self.train_data[self.index_array[self.row_index], end])
    else
        error("answer() is only defined during the training (exploration) phase")
    end
end

"""
    get_environment_name(self) -> String

Returns the source CSV file path, used to name result output
directories.
"""
function get_environment_name(self::Environment)::String
    return "$(self.file_path)"
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
