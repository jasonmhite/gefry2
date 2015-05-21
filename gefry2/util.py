import numpy as np
import os
import glob
import h5py
import re

def dream2HDF5(basename, outname, names, pth='./'):
    chainnum = re.compile(r'{}_chain(\d+).txt'.format(basename))

    nV = len(names)

    # These are apparently pointless
    varheader = zip(
        ['i', 'likelihood'] + names,
        [int] + (1 + nV) * [np.float64],
    )
    gelheader = zip(['i'] + names, [np.int16] + nV * [np.float64])

    with h5py.File(outname, 'w') as hf:
        chain_grp = hf.create_group("chains")
        chain_files = os.path.join(pth, basename + '_chain*.txt')

        for filename in glob.iglob(chain_files):
            f = os.path.basename(filename)
            i = chainnum.findall(f)[0]
            data = np.loadtxt(filename, skiprows=1, dtype=varheader)

            chain_grp.create_dataset(i, data=data)

        gelman_name = os.path.join(pth, basename + "_gr.txt")
        gelman_data = np.loadtxt(gelman_name, skiprows=1, dtype=gelheader)
        hf.create_dataset('gelman_rubin', data=gelman_data)
