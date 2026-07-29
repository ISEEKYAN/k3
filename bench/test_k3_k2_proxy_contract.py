from k3_k2_proxy_contract import (
    FEATURE_MATRIX,
    PROXY_SPECS,
    k2_parameter_counts,
    k3_parameter_counts,
    relative_mismatch,
)


def test_two_bayan_proxy_axes_are_frozen():
    layer_proxy = PROXY_SPECS["reduced_layers_full_experts"]
    expert_proxy = PROXY_SPECS["full_layers_reduced_experts"]

    assert (layer_proxy.num_layers, layer_proxy.num_experts) == (2, 896)
    assert (layer_proxy.tp, layer_proxy.ep, layer_proxy.pp, layer_proxy.cp) == (
        1,
        8,
        1,
        1,
    )
    assert (expert_proxy.num_layers, expert_proxy.num_experts) == (93, 16)
    assert (expert_proxy.tp, expert_proxy.ep, expert_proxy.pp, expert_proxy.cp) == (
        4,
        1,
        2,
        1,
    )


def test_whole_model_and_activated_size_contracts_are_below_two_percent():
    for spec in PROXY_SPECS.values():
        k3 = k3_parameter_counts(spec)
        k2 = k2_parameter_counts(spec)
        assert relative_mismatch(k3.total, k2.total) < 0.02
        assert relative_mismatch(k3.activated, k2.activated) < 0.02


def test_counts_are_whole_model_not_parallel_shards():
    spec = PROXY_SPECS["reduced_layers_full_experts"]
    k3 = k3_parameter_counts(spec)
    k2 = k2_parameter_counts(spec)

    assert k3.total == 33_536_453_344
    assert k3.activated == 4_469_926_624
    assert k2.total == 33_507_793_920
    assert k2.activated == 4_486_683_648


def test_k3_features_are_present_or_explicitly_scoped_out():
    for arm in PROXY_SPECS:
        features = FEATURE_MATRIX[arm]
        assert features["attn_res"]["present"]
        assert features["kda"]["present"]
        assert features["mla"]["present"]
        assert features["noaux_tc_expert_bias"]["present"]
        assert features["shared_experts"]["present"]
        assert not features["mtp"]["present"]
        assert "production" in features["mtp"]["impact"]


if __name__ == "__main__":
    test_two_bayan_proxy_axes_are_frozen()
    test_whole_model_and_activated_size_contracts_are_below_two_percent()
    test_counts_are_whole_model_not_parallel_shards()
    test_k3_features_are_present_or_explicitly_scoped_out()
