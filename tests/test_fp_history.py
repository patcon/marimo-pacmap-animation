import numpy as np

from _loader import cli


def test_fp_resample_iterations_matches_localmap_condition():
    # boundary = num_iters[0] + num_iters[1] = 200, total = 500
    # itr > 200 and itr % 10 == 0 -> 210, 220, ..., 490
    result = cli.fp_resample_iterations((100, 100, 300))
    assert result == list(range(210, 500, 10))


def test_fp_resample_iterations_empty_when_phase_3_too_short():
    # boundary = 20, total = 25 -> no itr in range(25) is both > 20 and a
    # multiple of 10
    result = cli.fp_resample_iterations((10, 10, 5))
    assert result == []


def test_fp_resample_iterations_stays_within_total():
    result = cli.fp_resample_iterations((0, 0, 25))
    assert all(itr < 25 for itr in result)
    assert result == [10, 20]


def test_checkpoint_index_for_frame_before_first_checkpoint():
    checkpoint_frames = np.array([0, 211, 221, 231])
    assert cli.checkpoint_index_for_frame(5, checkpoint_frames) == 0


def test_checkpoint_index_for_frame_exact_and_between_checkpoints():
    checkpoint_frames = np.array([0, 211, 221, 231])
    assert cli.checkpoint_index_for_frame(211, checkpoint_frames) == 1
    assert cli.checkpoint_index_for_frame(215, checkpoint_frames) == 1
    assert cli.checkpoint_index_for_frame(230, checkpoint_frames) == 2


def test_checkpoint_index_for_frame_clamps_past_last_checkpoint():
    checkpoint_frames = np.array([0, 211, 221, 231])
    assert cli.checkpoint_index_for_frame(9999, checkpoint_frames) == 3


def test_checkpoint_index_for_frame_at_frame_zero():
    checkpoint_frames = np.array([0, 211, 221, 231])
    assert cli.checkpoint_index_for_frame(0, checkpoint_frames) == 0


def test_checkpoint_index_for_frame_simulated_render_selection():
    history = {0: "arr0", 5: "arr5", 12: "arr12"}
    checkpoint_frames = np.array(sorted(history))
    arrays = [history[f] for f in checkpoint_frames]
    expected = {0: "arr0", 3: "arr0", 5: "arr5", 8: "arr5", 12: "arr12", 20: "arr12"}
    for frame, want in expected.items():
        idx = cli.checkpoint_index_for_frame(frame, checkpoint_frames)
        assert arrays[idx] == want
