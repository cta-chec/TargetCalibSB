from TargetCalibSB.tf.tf_dc import TFDC
from TargetCalibSB.tf.base import make_monotonic
import numpy as np
import pytest
from copy import deepcopy
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from IPython import embed


def get_adc(voltage):
    return voltage * 0.5 + 4


@pytest.fixture(scope='module')
def tf():
    random = np.random.RandomState(1)
    n_events = 100
    n_pixels = 64
    n_samples = 64
    n_cells = 256
    voltages = np.linspace(-10, 10, 10)
    tf = TFDC(n_pixels, n_samples, n_cells, voltages)
    for voltage in voltages:
        amplitude_index = tf.get_input_amplitude_index(voltage)
        waveforms = np.full((n_pixels, n_samples), get_adc(voltage))
        for iev in range(n_events):
            fci = random.randint(0, n_cells, 1)
            tf.add_to_tf(waveforms, fci, amplitude_index)
    tf._apply_amplitudes = np.tile(tf._input_amplitudes, (*tf.shape[:-1], 1))
    return tf


def test_generation(tf):
    assert (tf.hits >= 10).all()
    assert (tf.tf[0, 0] == tf.tf).all()
    np.testing.assert_allclose(tf.tf[0, 0], get_adc(tf._input_amplitudes))


def test_apply(tf):
    random = np.random.RandomState(1)
    n_events = 10
    n_pixels = 1
    n_samples = 64
    n_cells = 256
    voltages = random.uniform(-3, 10, 100)
    for voltage in voltages:
        waveforms = np.full((n_pixels, n_samples), get_adc(voltage), dtype=np.float32)
        for iev in range(n_events):
            fci = random.randint(0, n_cells, 1)
            calibrated = tf.apply_tf(waveforms, fci)
            assert (calibrated != 0).any()
            np.testing.assert_allclose(calibrated, voltage, rtol=1e-4)

    waveforms = np.full((n_pixels, n_samples), tf.tf.min() - 1000, dtype=np.float32)
    calibrated = tf.apply_tf(waveforms, 0)
    assert (calibrated == tf._input_amplitudes.min()).all()

    waveforms = np.full((n_pixels, n_samples), tf.tf.max() + 1000, dtype=np.float32)
    calibrated = tf.apply_tf(waveforms, 0)
    assert (calibrated == tf._input_amplitudes.max()).all()


def test_make_tf_monotonic(tf):
    tf_test = deepcopy(tf)
    tf_test._tf[:, :, 0] = tf_test._tf.max() + 1
    tf_test._tf[:, :, 2] = tf_test._tf.min() - 1
    tf_test._tf[:, :, 5] = tf_test._tf.max() + 1
    tf_test._tf[:, :, -1] = tf_test._tf.min() - 1
    tf_test._tf[:, :, -3] = tf_test._tf.max() + 1
    tf_test._tf[:, :, -5] = tf_test._tf.min() - 1
    result_tf = make_monotonic(tf._tf, tf._apply_amplitudes)
    assert (np.diff(result_tf, axis=-1) >= 0).all()

    tf_test = deepcopy(tf)
    tf_test._tf[:, :, 1] = tf_test._tf.max() + 1
    tf_test._tf[:, :, -1] = tf_test._tf.min() - 1
    result_tf = make_monotonic(tf._tf, tf._apply_amplitudes)
    assert (np.diff(result_tf, axis=-1) >= 0).all()


def test_tcal(tf, tmp_path):
    original_tf = tf.tf.copy()
    original_hits = tf.hits.copy()
    original_amplitudes = tf._input_amplitudes.copy()

    path = tmp_path / "test.tcal"
    tf.save_tcal(path)
    assert path.exists()

    tf.load_tcal(path)
    np.testing.assert_allclose(original_tf, tf.tf)
    np.testing.assert_allclose(original_hits, tf.hits)
    np.testing.assert_allclose(original_amplitudes, tf._input_amplitudes)

    tf = TFDC.from_tcal(path)
    np.testing.assert_allclose(original_tf, tf.tf)
    np.testing.assert_allclose(original_hits, tf.hits)
    np.testing.assert_allclose(original_amplitudes, tf._input_amplitudes)

    from target_calib import TFInputArrayReader
    shape = (original_tf.shape[0] // 64, 64, *original_tf.shape[1:])
    tf = TFInputArrayReader(str(path))
    np.testing.assert_allclose(original_tf.reshape(shape), np.array(tf.GetTFInput()))
    np.testing.assert_allclose(original_hits.reshape(shape), np.array(tf.GetHits()))
    np.testing.assert_allclose(original_amplitudes, np.array(tf.GetAmplitude()))


if __name__ == '__main__':
    test_apply(tf())
    # test_make_tf_monotonic(tf())