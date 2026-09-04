import numpy as np

from face_utils import (
    AUDIO_HALF_WINDOW,
    FACE_INNER_SIZE,
    gather_audio_window,
    reshape_audio_feat,
)
from inference import paste_prediction


def test_audio_window_contract_and_padding() -> None:
    features = np.ones((3, 2, 1024), dtype=np.float32)
    window = gather_audio_window(features, 0)
    assert window.shape == (AUDIO_HALF_WINDOW * 2, 2, 1024)
    tokens = reshape_audio_feat(window)
    assert tokens.shape == (40, 1024)
    assert tokens[: AUDIO_HALF_WINDOW * 2].count_nonzero() == 0


def test_audio_window_does_not_cross_clip_boundary() -> None:
    features = np.arange(8, dtype=np.float32).reshape(8, 1, 1)
    window = gather_audio_window(
        features,
        index=4,
        half_window=2,
        valid_start=4,
        valid_end=7,
    )
    assert window.shape == (4, 1, 1)
    assert window[:, 0, 0].tolist() == [0.0, 0.0, 4.0, 5.0]


def test_paste_prediction_directly_replaces_inner_face() -> None:
    crop_size = FACE_INNER_SIZE + 8
    image = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    face_crop = np.full_like(image, 17)
    prediction = np.full(
        (FACE_INNER_SIZE, FACE_INNER_SIZE, 3), 211, dtype=np.uint8
    )

    paste_prediction(
        image,
        prediction,
        face_crop,
        (0, 0, crop_size, crop_size),
        (crop_size, crop_size),
    )

    assert np.all(image[4:-4, 4:-4] == 211)
    assert np.all(image[:4] == 17)
