import pytest
import numpy as np

from ..assessment import *
from ..config import *


@pytest.mark.passed
def test_set_bad_to_qso():
    SIZE = 10
    spectra = np.random.randn(SIZE, 2600)
    spectra[2] = 0
    prev_spectra = spectra.copy()
    # pred_cls_1d = np.random.randint(0, 3, size=SIZE)
    pred_cls_1d = np.zeros((SIZE, 2600))
    pred_cls_1d_backup = pred_cls_1d.copy()

    set_bad_to_qso(spectra, pred_cls_1d)
    assert np.all(prev_spectra == spectra)
    assert np.all(pred_cls_1d[2] == config.class_label['qso'])
    assert np.any(pred_cls_1d != pred_cls_1d_backup)
