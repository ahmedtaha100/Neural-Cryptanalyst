from neural_cryptanalyst.preprocessing import PreprocessingConfig
import json

def test_config_roundtrip(tmp_path):
    config = PreprocessingConfig(sampling_rate=2e9, filter_order=3)
    file_path = tmp_path / 'conf.json'
    config.save(file_path)

    loaded = PreprocessingConfig.load(file_path)
    assert loaded.sampling_rate == 2e9
    assert loaded.filter_order == 3

def test_invalid_sampling_rate():
    try:
        PreprocessingConfig(sampling_rate=0)
    except ValueError:
        pass
    else:
        assert False, 'ValueError not raised'
