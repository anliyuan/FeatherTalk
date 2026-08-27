import numpy as np

from face_utils import (
    AUDIO_HALF_WINDOW,
    FACE_INNER_SIZE,
    gather_audio_window,
    mouth_soft_blend_mask,
    reshape_audio_feat,
)


def test_audio_window_contract_and_padding() -> None:
    features = np.ones((3, 2, 1024), dtype=np.float32)
    window = gather_audio_window(features, 0)
    assert window.shape == (AUDIO_HALF_WINDOW * 2, 2, 1024)
    tokens = reshape_audio_feat(window)
    assert tokens.shape == (40, 1024)
    assert tokens[: AUDIO_HALF_WINDOW * 2].count_nonzero() == 0


def test_mouth_blend_mask_contract() -> None:
    mask = mouth_soft_blend_mask()
    assert mask.shape == (1, FACE_INNER_SIZE, FACE_INNER_SIZE)
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0
    assert float(mask[:, 0].max()) == 0.0
