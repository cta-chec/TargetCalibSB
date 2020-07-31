import fitsio
import numpy as np
import os


def load_tcal_pedestal(path):
    print(f"Loading pedestal tcal file: {path}")
    with fitsio.FITS(path) as file:
        header = file[0].read_header()
        n_pixels = int(header['TM'] * header['PIX'])
        n_blocks = int(header['BLOCKS'])
        n_bpisam = int(header['SAMPLESBP'])
        shape = (n_pixels, n_blocks, n_bpisam)

        try:
            pedestal = file["DATA"].read()["CELLS"].reshape(shape)
            if pedestal.dtype == np.dtype('>u2'):
                header = file[0].read_header()
                scale = header['SCALE']
                offset = header['OFFSET']
                pedestal = (pedestal.astype(np.float32) - offset) / scale
            if "HITS" and "STDDEV" in file:
                hits = file["HITS"].read()["CELLS"].reshape(shape)
                std = file["STDDEV"].read()["CELLS"].reshape(shape)
        except ValueError:
            raise ValueError("Incompatible pedestal class for file")

        return pedestal, hits, std


def save_tcal_pedestal(path, pedestal, hits, std, compress=False):
    print(f"Saving pedestal tcal file: {path}")
    if os.path.exists(path):
        os.remove(path)

    n_pixels, n_blocks, n_bpisam = pedestal.shape
    shape = (n_pixels, n_blocks * n_bpisam)

    header = dict(
        TYPE=1,
        TM=n_pixels // 64,
        PIX=64,
        BLOCKS=n_blocks,
        SAMPLESBP=n_bpisam,
    )

    if compress:
        min_ = pedestal.min()
        max_ = pedestal.max()
        range_ = max_ - min_
        scale = int(65535 / (range_ + 1))
        if scale < 1:
            scale = 1
        offset = int(-1 * min_ + 1) * scale
        header["SCALE"] = scale
        header["OFFSET"] = offset
        pedestal = np.round(pedestal * scale + offset).astype(np.uint16)

    with fitsio.FITS(path, 'rw') as file:
        file.create_image_hdu()
        file[0].write_keys(header)
        file.write(dict(CELLS=pedestal.reshape(shape)), extname="DATA")
        file.write(dict(CELLS=hits.reshape(shape)), extname="HITS")
        file.write(dict(CELLS=std.reshape(shape)), extname="STDDEV")


def load_tcal_tfinput(path):
    print(f"Loading TF tcal file: {path}")
    with fitsio.FITS(path) as file:
        header = file[0].read_header()
        n_pixels = int(header['TM'] * header['PIX'])
        n_cells = int(header['CELLS'])
        n_amplitudes = int(header['PNTS'])
        shape = (n_pixels, n_cells, n_amplitudes)

        try:
            tf = file["DATA"].read()["CELLS"].reshape(shape)
            hits = file["HITS"].read()["CELLS"].reshape(shape)
            input_amplitudes = file["AMPLITUDES"].read()["CELLS"]
        except ValueError:
            raise ValueError("Incompatible TF class for file")

    return tf, hits, input_amplitudes


def save_tcal_tfinput(path, tf, hits, amplitudes):
    print(f"Saving TF tcal file: {path}")
    if os.path.exists(path):
        os.remove(path)

    n_pixels, n_cells, n_amplitudes = tf.shape
    shape = (n_pixels, n_cells * n_amplitudes)

    header = dict(
        TYPE=4,
        TM=n_pixels // 64,
        PIX=64,
        CELLS=n_cells,
        PNTS=n_amplitudes,
    )
    with fitsio.FITS(path, 'rw') as file:
        file.create_image_hdu()
        file[0].write_keys(header)
        file.write(dict(CELLS=tf.reshape(shape)), extname="DATA")
        file.write(dict(CELLS=hits.reshape(shape)), extname="HITS")
        file.write(dict(CELLS=amplitudes), extname="AMPLITUDES")
