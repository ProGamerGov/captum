import numpy as np


def rfft2d_freqs(height: int, width: int) -> np.ndarray:
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(height)[:, None]
    # on odd input dimensions we need to keep one additional frequency
    wadd = 2 if width % 2 == 1 else 1
    fx = np.fft.fftfreq(width)[: width // 2 + wadd]
    return np.sqrt((fx * fx) + (fy * fy))
