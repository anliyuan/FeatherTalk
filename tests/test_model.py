import torch

from model import AUDIO_FEATURE_DIM, AUDIO_TOKENS, IMAGE_SIZE, Model


def test_model_contract() -> None:
    model = Model().eval()
    image = torch.zeros(1, 6, IMAGE_SIZE, IMAGE_SIZE)
    audio = torch.zeros(1, AUDIO_TOKENS, AUDIO_FEATURE_DIM)
    with torch.no_grad():
        output = model(image, audio)
    assert output.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.all((0.0 <= output) & (output <= 1.0))


def test_model_rejects_wrong_audio_shape() -> None:
    model = Model().eval()
    image = torch.zeros(1, 6, IMAGE_SIZE, IMAGE_SIZE)
    audio = torch.zeros(1, AUDIO_TOKENS - 1, AUDIO_FEATURE_DIM)
    try:
        model(image, audio)
    except ValueError as error:
        assert "audio must have shape" in str(error)
    else:
        raise AssertionError("wrong audio shape was accepted")
