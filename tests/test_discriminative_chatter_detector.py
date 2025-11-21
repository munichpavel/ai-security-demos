import pytest

from ai_security.discriminative_chatter_detector import load_training_data


@pytest.mark.parametrize(
    'dataset_name',
    [
        'blackbriar',
        'extended',
        'baby-blackbriar',
    ]
)
def test_load_training_data(dataset_name):
    """Weak test"""
    df = load_training_data(dataset_name)
    assert df.shape[1] > 0
