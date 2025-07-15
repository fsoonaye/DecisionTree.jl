@testset "methods for `Ensemble` type" begin
    features, labels = load_data("iris")

    # combining identical ensembles:
    ensemble1 = build_forest(labels, features, 2, 7)
    @test DecisionTree.has_impurity_importance(ensemble1)
    @test ensemble1[1:2].trees == ensemble1.trees[1:2]
    @test length(ensemble1) == 7
    @test DecisionTree.n_features(ensemble1) == 4
    ensemble = vcat(ensemble1, ensemble1)
    @test ensemble.featim ≈ ensemble1.featim

    # combining heterogeneous ensembles:
    ensemble2 = build_forest(labels, features, 2, 10)
    ensemble = vcat(ensemble1, ensemble2)
    n1 = length(ensemble1.trees)
    n2 = length(ensemble2.trees)
    n = n1 + n2
    featim1 = ensemble1.featim
    featim2 = ensemble2.featim
    ncols1 = size(featim1, 2)
    ncols2 = size(featim2, 2)
    if ncols1 > ncols2
        featim2 = hcat(featim2, zeros(eltype(featim2), size(featim2, 1), ncols1 - ncols2))
    elseif ncols2 > ncols1
        featim1 = hcat(featim1, zeros(eltype(featim1), size(featim1, 1), ncols2 - ncols1))
    end
    @test n * ensemble.featim ≈ n1 * featim1 + n2 * featim2

    # including an ensemble without impurity importance should drop impurity importance from
    # the combination:
    ensemble3 = build_forest(labels, features, 2, 4; impurity_importance=false)
    @test !DecisionTree.has_impurity_importance(ensemble3)
    @test vcat(ensemble1, ensemble3).featim == zeros(Float64, 0, 0)
    @test vcat(ensemble3, ensemble1).featim == zeros(Float64, 0, 0)
    @test vcat(ensemble3, ensemble3).featim == zeros(Float64, 0, 0)

    # changing the number of features:
    ensemble4 = build_forest(labels, features[:, 1:3], 2, 4)
    @test_logs vcat(ensemble3, ensemble4) # ensemble 3 doesn't support importances
    @test_throws DecisionTree.ERR_ENSEMBLE_VCAT vcat(ensemble1, ensemble4)
end
