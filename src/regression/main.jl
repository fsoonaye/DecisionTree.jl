include("tree.jl")

function _get_depth(node::treeregressor.NodeMeta)
    if node.is_leaf
        return 0
    else
        return 1 + max(_get_depth(node.l), _get_depth(node.r))
    end
end

function _convert(
    node::treeregressor.NodeMeta{S}, labels::Array{T}
) where {S,T<:AbstractFloat}
    if node.is_leaf
        return Leaf{T}(node.label, labels[node.region])
    elseif node.feature == 0
        # Pass-through node: data child goes on left so apply_tree (featid==0 → left) works.
        if length(node.l.region) > 0
            left = _convert(node.l, labels)
            right = _convert(node.r, labels)
        else
            left = _convert(node.r, labels)
            right = _convert(node.l, labels)
        end
        return Node{S,T}(0, node.threshold, left, right)
    else
        left = _convert(node.l, labels)
        right = _convert(node.r, labels)
        return Node{S,T}(node.feature, node.threshold, left, right)
    end
end

function update_using_impurity!(
    feature_importance::Matrix{Float64}, node::treeregressor.NodeMeta{S}, depth::Int
) where {S}
    if !node.is_leaf
        # Pass-through nodes (feature == 0) contribute zero impurity decrease;
        # skip to avoid out-of-bounds indexing.
        if node.feature != 0 && depth <= size(feature_importance, 2)
            feature_importance[node.feature, depth] +=
                node.node_impurity - node.l.node_impurity - node.r.node_impurity
        end
        update_using_impurity!(feature_importance, node.l, depth + 1)
        update_using_impurity!(feature_importance, node.r, depth + 1)
    end
    return nothing
end

function build_stump(
    labels::AbstractVector{T},
    features::AbstractMatrix{S};
    rng=Random.GLOBAL_RNG,
    impurity_importance::Bool=true,
) where {S,T<:AbstractFloat}
    return build_tree(labels, features, 0, 1; rng, impurity_importance)
end

function build_tree(
    labels::AbstractVector{T},
    features::AbstractMatrix{S},
    n_subfeatures=0,
    max_depth=-1,
    min_samples_leaf=5,
    min_samples_split=2,
    min_purity_increase=0.0;
    rng=Random.GLOBAL_RNG,
    impurity_importance::Bool=true,
    feature_sampler=nothing,
) where {S,T<:AbstractFloat}
    if max_depth == -1
        max_depth = typemax(Int)
    end
    if n_subfeatures == 0
        n_subfeatures = size(features, 2)
    end

    rng = mk_rng(rng)::Random.AbstractRNG
    t = treeregressor.fit(;
        X=features,
        Y=labels,
        W=nothing,
        max_features=Int(n_subfeatures),
        max_depth=Int(max_depth),
        min_samples_leaf=Int(min_samples_leaf),
        min_samples_split=Int(min_samples_split),
        min_purity_increase=Float64(min_purity_increase),
        rng,
        feature_sampler,
    )

    node = _convert(t.root, labels[t.labels])
    n_features = size(features, 2)
    if !impurity_importance
        return Root{S,T}(node, n_features, zeros(Float64, 0, 0))
    else
        fi_depth = max_depth
        if fi_depth == typemax(Int)
            fi_depth = _get_depth(t.root)
        end
        if fi_depth == 0
            fi_depth = 1
        end
        fi = zeros(Float64, n_features, fi_depth)
        update_using_impurity!(fi, t.root, 1)
        return Root{S,T}(node, n_features, fi ./ size(features, 1))
    end
end

function build_forest(
    labels::AbstractVector{T},
    features::AbstractMatrix{S},
    n_subfeatures=-1,
    n_trees=10,
    partial_sampling=0.7,
    max_depth=-1,
    min_samples_leaf=5,
    min_samples_split=2,
    min_purity_increase=0.0;
    rng::Union{Integer,AbstractRNG}=Random.GLOBAL_RNG,
    impurity_importance::Bool=true,
    feature_sampler=nothing,
) where {S,T<:AbstractFloat}
    if n_trees < 1
        throw("the number of trees must be >= 1")
    end
    if !(0.0 < partial_sampling <= 1.0)
        throw("partial_sampling must be in the range (0,1]")
    end

    if n_subfeatures == -1
        n_features = size(features, 2)
        n_subfeatures = round(Int, sqrt(n_features))
    end

    t_samples = length(labels)
    n_samples = floor(Int, partial_sampling * t_samples)

    forest = if impurity_importance
        Vector{Root{S,T}}(undef, n_trees)
    else
        Vector{LeafOrNode{S,T}}(undef, n_trees)
    end

    if rng isa Random.AbstractRNG
        shared_seed = rand(rng, UInt)
        Threads.@threads for i in 1:n_trees
            # The Mersenne Twister (Julia's default) is not thread-safe.
            _rng = Random.seed!(copy(rng), shared_seed + i)
            inds = rand(_rng, 1:t_samples, n_samples)
            forest[i] = build_tree(
                labels[inds],
                features[inds, :],
                n_subfeatures,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                min_purity_increase;
                rng=_rng,
                impurity_importance,
                feature_sampler,
            )
        end
    else # each thread gets its own seeded rng
        Threads.@threads for i in 1:n_trees
            Random.seed!(rng + i)
            inds = rand(1:t_samples, n_samples)
            forest[i] = build_tree(
                labels[inds],
                features[inds, :],
                n_subfeatures,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                min_purity_increase;
                impurity_importance,
                feature_sampler,
            )
        end
    end

    if impurity_importance
        max_ncols = 0
        for root in forest
            max_ncols = max(max_ncols, size(root.featim, 2))
        end

        for i in 1:length(forest)
            root = forest[i]
            current_ncols = size(root.featim, 2)
            if current_ncols < max_ncols
                padded_featim = hcat(
                    root.featim,
                    zeros(eltype(root.featim), root.n_feat, max_ncols - current_ncols),
                )
                forest[i] = Root(root.node, root.n_feat, padded_featim)
            end
        end
    end

    return _build_forest(forest, size(features, 2), n_trees, impurity_importance)
end
