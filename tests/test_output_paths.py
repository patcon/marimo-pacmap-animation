from pathlib import Path

from _loader import cli


def test_resolve_output_dir_empty_defaults_to_outputs():
    assert cli.resolve_output_dir("") == Path("outputs")


def test_resolve_output_dir_relative_nests_under_outputs():
    assert cli.resolve_output_dir("myrun") == Path("outputs/myrun")


def test_resolve_output_dir_dot_slash_prefix_used_as_is():
    assert cli.resolve_output_dir("./myrun") == Path("./myrun")


def test_resolve_output_dir_dotdot_slash_prefix_used_as_is():
    assert cli.resolve_output_dir("../myrun") == Path("../myrun")


def test_resolve_output_dir_absolute_path_used_as_is():
    assert cli.resolve_output_dir("/tmp/myrun") == Path("/tmp/myrun")


def test_param_tag_is_default_when_nothing_differs():
    cfg = dict(cli.DEFAULT_CONFIG)
    assert cli.param_tag(cfg) == "default"


def test_param_tag_includes_only_changed_params():
    cfg = dict(cli.DEFAULT_CONFIG)
    cfg["n_neighbors"] = 5
    cfg["mn_ratio"] = 0.8
    assert cli.param_tag(cfg) == "nn5_mnr0.8"


def test_param_tag_excludes_algorithm_and_output_dir():
    cfg = dict(cli.DEFAULT_CONFIG)
    cfg["algorithm"] = "localmap"
    cfg["output_dir"] = "somewhere"
    assert cli.param_tag(cfg) == "default"


def test_param_tag_n_none_renders_as_all():
    cfg = dict(cli.DEFAULT_CONFIG)
    cfg["n"] = None
    assert cli.param_tag(cfg) == "nall"


def test_param_tag_includes_low_dist_thres():
    cfg = dict(cli.DEFAULT_CONFIG)
    cfg["low_dist_thres"] = 3.0
    assert cli.param_tag(cfg) == "ldt3.0"


def test_param_tag_excludes_cache_settings():
    cfg = dict(cli.DEFAULT_CONFIG)
    cfg["cache"] = False
    cfg["cache_dir"] = "somewhere"
    assert cli.param_tag(cfg) == "default"


def test_param_tag_num_iters_joined_with_dashes():
    cfg = dict(cli.DEFAULT_CONFIG)
    cfg["num_iters"] = [50, 50, 100]
    assert cli.param_tag(cfg) == "iters50-50-100"


def test_param_tag_excludes_n_components():
    # n_components is disambiguated via a filename marker (see
    # test_main_smoke.py), not a --tag-output slug entry - it's a pipeline
    # choice, not a "differing tunable param" like mn_ratio/n_neighbors.
    cfg = dict(cli.DEFAULT_CONFIG)
    cfg["n_components"] = 3
    assert cli.param_tag(cfg) == "default"


def test_unique_path_returns_path_unchanged_if_absent(tmp_path):
    p = tmp_path / "foo.mp4"
    assert cli.unique_path(p) == p


def test_unique_path_appends_suffix_on_decline(tmp_path, monkeypatch):
    p = tmp_path / "foo.mp4"
    p.write_text("existing")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    result = cli.unique_path(p)
    assert result == tmp_path / "foo_1.mp4"


def test_unique_path_skips_taken_incremented_names(tmp_path, monkeypatch):
    p = tmp_path / "foo.mp4"
    p.write_text("existing")
    (tmp_path / "foo_1.mp4").write_text("also existing")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    result = cli.unique_path(p)
    assert result == tmp_path / "foo_2.mp4"


def test_unique_path_overwrites_on_confirmation(tmp_path, monkeypatch):
    p = tmp_path / "foo.mp4"
    p.write_text("existing")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    result = cli.unique_path(p)
    assert result == p


def test_param_tag_includes_schedule_preset():
    cfg = dict(cli.DEFAULT_CONFIG)
    cfg["schedule_preset"] = "cycle"
    assert cli.param_tag(cfg) == "schedcycle"


def test_param_tag_includes_cycle_knobs_that_differ():
    cfg = dict(cli.DEFAULT_CONFIG)
    cfg["schedule_preset"] = "cycle"
    cfg["schedule_period"] = 200
    assert cli.param_tag(cfg) == "schedcycle_period200"


def test_param_tag_excludes_cycle_knobs_under_the_vanilla_preset():
    """Vanilla has no period, so a stray --schedule-period changes nothing
    about the render and must not split the comparison folder in two."""
    cfg = dict(cli.DEFAULT_CONFIG)
    cfg["schedule_period"] = 200
    cfg["schedule_mn_min"] = 0.5
    assert cli.param_tag(cfg) == "default"


def test_param_tag_includes_fp_knobs_under_a_cycling_preset():
    cfg = dict(cli.DEFAULT_CONFIG)
    cfg["schedule_preset"] = "breathe"
    cfg["schedule_fp_max"] = 5.0
    assert cli.param_tag(cfg) == "schedbreathe_fpmax5.0"


def test_param_tag_excludes_fp_knobs_under_the_vanilla_preset():
    cfg = dict(cli.DEFAULT_CONFIG)
    cfg["schedule_fp_max"] = 5.0
    cfg["schedule_fp_phase"] = 0.0
    assert cli.param_tag(cfg) == "default"
