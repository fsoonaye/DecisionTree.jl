# DecisionTree.jl Fork — Project Plan

## Fork overview

This is a fork of [JuliaAI/DecisionTree.jl](https://github.com/JuliaAI/DecisionTree.jl) for research simulations.
Upstream remote: `upstream` -> `https://github.com/JuliaAI/DecisionTree.jl.git`

## What we've done so far

### 1. Matrix-based feature importance (commits `260d78c`, `a5ea021`)

Refactored impurity importance from a 1D vector to a 2D matrix (features × depth).
This lets us train one forest to max depth and extract importance at any shallower depth
by truncating the matrix — no need to retrain separate forests per depth.

**Breaking change**: `featim` field in `Root` and `Ensemble` is now `Matrix{Float64}`.
`impurity_importance()` returns a `Matrix`.

Files modified:
- `src/DecisionTree.jl` — struct definitions
- `src/classification/main.jl` — classification tree/forest building
- `src/regression/main.jl` — regression tree/forest building
- Tests adapted in `test/classification/` and `test/miscellaneous/`

### 2. Upstream sync (2026-04-17)

Merged 3 upstream commits (CI pipeline fixes — Julia LTS version, `julia-actions/cache@v2`).
Only `.github/workflows/CI.yml` was touched. Clean merge, no conflicts with our changes.

---

## Next feature: custom feature subsampling distribution

### Motivation

In standard random forests, feature subsampling at each node is uniform: pick exactly `mtry`
features uniformly at random from the `d` available features. We want to replace this with
a user-supplied probability distribution over subsets of {1,...,d} of size ≤ mtry.

Two key differences from the standard approach:
1. **Non-uniform weights** — some subsets can be more likely than others
2. **Variable subset size** — subsets can have size < mtry, not just exactly mtry

### Design decision: callable interface

Representing the distribution explicitly (probability vector over all subsets) is
combinatorially infeasible for realistic `d`. Instead, we pass a **callable** that acts
as a subset sampler:

```julia
feature_sampler(rng::AbstractRNG, features::Vector{Int}) -> Vector{Int}
```

- `rng`: the random number generator (for reproducibility)
- `features`: the available (non-constant) feature indices at this node
- Returns: a vector of selected feature indices (the sampled subset)

**Why a callable:**
- No need to enumerate/store exponentially many subsets
- Supports any distribution the user wants (weighted sampling, precomputed tables, etc.)
- Naturally supports variable-size subsets
- Composable with `Distributions.jl` or any other package
- Idiomatic Julia — functions are first-class

**Default behavior**: a closure reproducing the current uniform sampling (uniform subset
of size exactly `mtry`), so existing API is unaffected.

### Where the change goes in the code

The feature subsampling logic lives in the `_split!` functions:
- `src/classification/tree.jl` (~line 113): hypergeometric draw + Fisher-Yates shuffle
- `src/regression/tree.jl` (~line 113): same pattern

Currently, `max_features` (i.e. mtry) is passed through the call chain:
```
build_forest -> build_tree -> _build_tree -> _split!
```

The plan:
1. Add `feature_sampler` as an optional keyword argument to `build_forest` / `build_tree`
2. Thread it through `_build_tree` down to `_split!`
3. In `_split!`, when `feature_sampler` is provided, call it instead of the
   hypergeometric + shuffle logic
4. Default value of `feature_sampler` is `nothing`, which triggers the original behavior
5. Expose through the scikit-learn API wrappers in `src/scikitlearnAPI.jl` as well

### Branch strategy

- `dev` branch: our stable fork baseline (synced with upstream)
- Create `feature/custom-subsampling` off `dev` for this work
- Merge back to `dev` when done
