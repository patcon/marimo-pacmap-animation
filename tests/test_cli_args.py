import json

import pytest

from _loader import cli


def test_parse_count_arg_absolute_integer():
    assert cli.parse_count_arg("500") == 500


def test_parse_count_arg_fraction_stays_float():
    assert cli.parse_count_arg("0.1") == 0.1


def test_parse_count_arg_1_is_treated_as_all_not_literal_count():
    assert cli.parse_count_arg("1") == 1
    assert isinstance(cli.parse_count_arg("1"), float)


def test_parse_iter_arg_single_value():
    assert cli.parse_iter_arg("150") == 150


def test_parse_iter_arg_range():
    assert cli.parse_iter_arg("50-300") == (50, 300)


def test_build_config_defaults_to_default_config():
    args = cli.parse_args([])
    cfg = cli.build_config(args)
    for key, value in cli.DEFAULT_CONFIG.items():
        if key == "num_iters":
            assert cfg[key] == tuple(value)
        else:
            assert cfg[key] == value


def test_build_config_cli_flag_overrides_default():
    args = cli.parse_args(["--n-neighbors", "5", "--mn-ratio", "0.8"])
    cfg = cli.build_config(args)
    assert cfg["n_neighbors"] == 5
    assert cfg["mn_ratio"] == 0.8


def test_build_config_n_components_defaults_to_2():
    args = cli.parse_args([])
    cfg = cli.build_config(args)
    assert cfg["n_components"] == 2


def test_build_config_n_components_cli_flag_overrides_default():
    args = cli.parse_args(["--n-components", "3"])
    cfg = cli.build_config(args)
    assert cfg["n_components"] == 3


def test_parse_args_n_components_rejects_invalid_choice():
    with pytest.raises(SystemExit):
        cli.parse_args(["--n-components", "4"])


def test_build_config_rotate_defaults_to_false():
    args = cli.parse_args([])
    cfg = cli.build_config(args)
    assert cfg["rotate"] is False


def test_build_config_rotate_flag_sets_true():
    args = cli.parse_args(["--rotate"])
    cfg = cli.build_config(args)
    assert cfg["rotate"] is True


def test_build_config_zoom_defaults_to_1():
    args = cli.parse_args([])
    cfg = cli.build_config(args)
    assert cfg["zoom"] == 1.0


def test_build_config_zoom_cli_flag_overrides_default():
    args = cli.parse_args(["--zoom", "2.5"])
    cfg = cli.build_config(args)
    assert cfg["zoom"] == 2.5


def test_build_config_n_all_resolves_to_none():
    args = cli.parse_args(["--n", "all"])
    cfg = cli.build_config(args)
    assert cfg["n"] is None


def test_build_config_n_fraction_stays_a_proportion_until_load_mnist():
    args = cli.parse_args(["--n", "0.1"])
    cfg = cli.build_config(args)
    assert cfg["n"] == 0.1


def test_build_config_num_iters_parsed_as_tuple():
    args = cli.parse_args(["--num-iters", "10,20,30"])
    cfg = cli.build_config(args)
    assert cfg["num_iters"] == (10, 20, 30)


def test_build_config_iter_range_parsed():
    args = cli.parse_args(["--iter", "50-300"])
    cfg = cli.build_config(args)
    assert cfg["iter"] == [(50, 300)]


def test_build_config_iter_single_value_parsed_as_single_item_list():
    args = cli.parse_args(["--iter", "150"])
    cfg = cli.build_config(args)
    assert cfg["iter"] == [150]


def test_build_config_iter_comma_separated_parsed_as_list_of_items():
    args = cli.parse_args(["--iter", "50,150,250-400"])
    cfg = cli.build_config(args)
    assert cfg["iter"] == [50, 150, (250, 400)]


def test_build_config_file_values_used_when_no_cli_override(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"n_neighbors": 7}))
    args = cli.parse_args(["--config", str(config_path)])
    cfg = cli.build_config(args)
    assert cfg["n_neighbors"] == 7


def test_build_config_cli_flag_wins_over_config_file(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"n_neighbors": 7}))
    args = cli.parse_args(["--config", str(config_path), "--n-neighbors", "9"])
    cfg = cli.build_config(args)
    assert cfg["n_neighbors"] == 9


def test_resolve_focus_label_passes_through_none():
    assert cli.resolve_focus_label(None, y=None) is None


def test_resolve_focus_label_parses_digit_string():
    assert cli.resolve_focus_label("3", y=None) == 3


def test_resolve_focus_label_prompts_and_validates_against_present_labels(monkeypatch):
    import numpy as np

    y = np.array([0, 1, 1, 2])
    replies = iter(["9", "not-a-number", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert cli.resolve_focus_label("__prompt__", y=y) == 1


def test_build_config_caching_is_on_by_default():
    cfg = cli.build_config(cli.parse_args([]))
    assert cfg["cache"] is True
    assert cfg["cache_dir"] == ".cache/fits"


def test_build_config_no_cache_flag_disables_caching():
    cfg = cli.build_config(cli.parse_args(["--no-cache"]))
    assert cfg["cache"] is False


def test_build_config_cache_dir_flag_overrides_default():
    cfg = cli.build_config(cli.parse_args(["--cache-dir", "/tmp/fits"]))
    assert cfg["cache_dir"] == "/tmp/fits"
    assert cfg["cache"] is True


def test_build_config_cache_can_be_disabled_from_a_config_file(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"cache": False}))
    cfg = cli.build_config(cli.parse_args(["--config", str(config_path)]))
    assert cfg["cache"] is False


def test_build_config_low_dist_thres_defaults_to_pacmaps_own_default():
    cfg = cli.build_config(cli.parse_args([]))
    assert cfg["low_dist_thres"] == 10.0


def test_build_config_low_dist_thres_flag_overrides_default():
    cfg = cli.build_config(cli.parse_args(["--low-dist-thres", "3.5"]))
    assert cfg["low_dist_thres"] == 3.5


def test_build_config_low_dist_thres_cli_flag_wins_over_config_file(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"low_dist_thres": 7.0}))
    cfg = cli.build_config(cli.parse_args(["--config", str(config_path), "--low-dist-thres", "2.0"]))
    assert cfg["low_dist_thres"] == 2.0


def test_build_config_schedule_preset_defaults_to_vanilla():
    """Vanilla is the default so that landing this feature changes nothing
    about an existing invocation - including its cached fits."""
    cfg = cli.build_config(cli.parse_args([]))
    assert cfg["schedule_preset"] == "vanilla"


def test_build_config_schedule_knobs_default_to_the_presets_own():
    """None means "whatever this preset defaults to" - a concrete default here
    would override every preset back into the same shape."""
    cfg = cli.build_config(cli.parse_args([]))
    assert cfg["schedule_period"] is None
    assert cfg["schedule_mn_min"] is None
    assert cfg["schedule_mn_max"] is None


def test_build_config_schedule_preset_flag_overrides_default():
    cfg = cli.build_config(cli.parse_args(["--schedule-preset", "cycle"]))
    assert cfg["schedule_preset"] == "cycle"


def test_build_config_schedule_knob_flags_override_defaults():
    cfg = cli.build_config(cli.parse_args([
        "--schedule-preset", "cycle",
        "--schedule-period", "250",
        "--schedule-mn-min", "0.5",
        "--schedule-mn-max", "50",
    ]))
    assert cfg["schedule_period"] == 250
    assert cfg["schedule_mn_min"] == 0.5
    assert cfg["schedule_mn_max"] == 50.0


def test_parse_args_schedule_preset_rejects_invalid_choice():
    with pytest.raises(SystemExit):
        cli.parse_args(["--schedule-preset", "bogus"])


def test_build_config_schedule_preset_settable_from_a_config_file(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"schedule_preset": "cycle", "schedule_period": 300}))
    cfg = cli.build_config(cli.parse_args(["--config", str(config_path)]))
    assert cfg["schedule_preset"] == "cycle"
    assert cfg["schedule_period"] == 300


def test_build_config_schedule_cli_flag_wins_over_config_file(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"schedule_preset": "vanilla", "schedule_period": 300}))
    cfg = cli.build_config(cli.parse_args([
        "--config", str(config_path), "--schedule-preset", "cycle", "--schedule-period", "50",
    ]))
    assert cfg["schedule_preset"] == "cycle"
    assert cfg["schedule_period"] == 50


def test_cycle_preset_holds_repulsion_unless_asked_otherwise():
    """cycle sweeps attraction alone by default: w_FP's bounds coincide at 1.0,
    which is the vanilla value."""
    d = cli.schedule.preset_defaults("cycle")
    assert d["fp_min"] == 1.0 and d["fp_max"] == 1.0
    assert d["fp_phase"] == 0.5  # antiphase, once the bounds are opened up


def test_build_config_fp_knob_flags_override_defaults():
    cfg = cli.build_config(cli.parse_args([
        "--schedule-preset", "cycle", "--schedule-fp-min", "0.2",
        "--schedule-fp-max", "5", "--schedule-fp-phase", "0",
    ]))
    assert cfg["schedule_fp_min"] == 0.2
    assert cfg["schedule_fp_max"] == 5.0
    assert cfg["schedule_fp_phase"] == 0.0


def test_parse_args_accepts_the_breathe_preset():
    cfg = cli.build_config(cli.parse_args(["--schedule-preset", "breathe"]))
    assert cfg["schedule_preset"] == "breathe"
