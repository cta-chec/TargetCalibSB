from numba import guvectorize, float32


@guvectorize([(float32[:], float32[:])], '(s)->(s)', nopython=True)
def correct_overflow(waveform, overflow_corrected):
    overflow_corrected[:] = waveform
    if waveform.max() > 2000:
        n_samples = waveform.size
        start = 0
        for isam in range(n_samples-1):
            this_sample = waveform[isam]
            next_sample = waveform[isam + 1]
            if (next_sample - this_sample < -500) and (next_sample < 600):
                start = isam + 1
            if (start > 0) and (next_sample - this_sample > 500):
                overflow_corrected[start:isam+1] += 4096
                start = 0
