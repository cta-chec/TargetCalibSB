import numpy as np
from numba import njit

N_BLOCKPHASE = 32
N_ROWS = 8


def calculate_row_column_blockphase(cell_id):
    blockphase = cell_id % N_BLOCKPHASE
    row = (cell_id // N_BLOCKPHASE) % N_ROWS
    column = (cell_id // N_BLOCKPHASE) // N_ROWS
    return row, column, blockphase


@njit
def get_cell_ids_for_waveform(first_cell_id, n_samples, n_cells):
    isample = np.arange(n_samples)
    block = first_cell_id // N_BLOCKPHASE
    blockphase = first_cell_id % N_BLOCKPHASE
    factor = 64 * (1 - 2 * (block % 2))
    shift = (((blockphase + isample) // N_BLOCKPHASE) % 2) * factor
    cell = (first_cell_id + isample + shift) % n_cells
    return cell
