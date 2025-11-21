from pathlib import Path
import os

import pytest

import ai_security.discriminative_chatter_detector
from ai_security.discriminative_chatter_detector import load_training_data


@pytest.mark.parametrize(
    'dataset_name',
    [
        'blackbriar',
        'extended',
        'baby-blackbriar',
        'toddler-blackbriar',
    ]
)
def test_load_training_data(monkeypatch, dataset_name):
    """Weak test"""
    def mock_get_data_root() -> Path:
        return Path(os.environ['DATA_ROOT'])

    data_root = mock_get_data_root()
    print('\n\nData root: ', data_root)
    print(f'Expected file: {data_root / "blackbriar-chatter-detection-dataset.csv"}')
    print(f'File exists: {(data_root / "blackbriar-chatter-detection-dataset.csv").exists()}')


    monkeypatch.setattr(
        'ai_security.discriminative_chatter_detector._get_data_root',
        mock_get_data_root
    )

    df = load_training_data(dataset_name)
    assert df.shape[1] > 0
