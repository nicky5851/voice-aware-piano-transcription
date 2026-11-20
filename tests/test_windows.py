import numpy as np

from vapt.data.windows import voiced_windows


def test_voiced_windows_known_starts():
    # 5s at fps=10: frames 0-9 voiced, 10-29 unvoiced, 30-49 voiced.
    fps = 10
    voiced = np.zeros(50, dtype=bool)
    voiced[0:10] = True
    voiced[30:50] = True
    starts = voiced_windows(voiced, win_s=1.0, hop_s=1.0, min_frac=0.5, fps=fps)
    # windows: [0,10)->all voiced=1.0 pass; [10,20)..[20,30) unvoiced fail;
    # [30,40),[40,50) all voiced pass.
    assert starts == [0.0, 3.0, 4.0]


def test_voiced_windows_no_window_fits():
    voiced = np.ones(5, dtype=bool)
    starts = voiced_windows(voiced, win_s=10.0, hop_s=1.0, fps=10)
    assert starts == []
