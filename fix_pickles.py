#!/usr/bin/env python3
"""
One-time script to fix pickle files that were saved with a pandas version
that used StringDtype for string columns/indices, which causes a
NotImplementedError when loading with certain pandas builds.

Run this once: python3 fix_pickles.py
"""
import pickle
import numpy as np
import pandas as pd
from pandas._libs.arrays import NDArrayBacked
from pandas.core.arrays.string_ import StringDtype


class CompatUnpickler(pickle._Unpickler):
    pass


CompatUnpickler.dispatch = pickle._Unpickler.dispatch.copy()


def _load_build(self):
    stack = self.stack
    state = stack.pop()
    inst = stack[-1]
    if isinstance(inst, NDArrayBacked) and isinstance(state, tuple) and len(state) == 2:
        dtype, data = state
        if isinstance(dtype, StringDtype):
            arr = np.asarray(data, dtype=object)
            NDArrayBacked.__init__(inst, arr, np.dtype('O'))
            return
    setstate = getattr(inst, '__setstate__', None)
    if setstate is not None:
        setstate(state)
    else:
        slotstate = None
        if isinstance(state, tuple) and len(state) == 2:
            state, slotstate = state
        if state:
            inst.__dict__.update(state)
        if slotstate:
            for k, v in slotstate.items():
                setattr(inst, k, v)


CompatUnpickler.dispatch[ord(b'b')] = _load_build


def compat_read_pickle(path):
    with open(path, 'rb') as f:
        return CompatUnpickler(f).load()


if __name__ == '__main__':
    print("Fixing data/cluster.pkl...")
    cluster = compat_read_pickle('data/cluster.pkl')
    # Convert any StringDtype columns to object dtype
    for col in cluster.columns:
        if hasattr(cluster[col].dtype, 'storage'):
            cluster[col] = cluster[col].astype(object)
    cluster.to_pickle('data/cluster.pkl')
    print(f"  Saved: {cluster.shape}")

    print("Fixing data/embeddings.pkl...")
    embeddings = compat_read_pickle('data/embeddings.pkl')
    for col in embeddings.columns:
        if hasattr(embeddings[col].dtype, 'storage'):
            embeddings[col] = embeddings[col].astype(object)
    embeddings.to_pickle('data/embeddings.pkl')
    print(f"  Saved: {embeddings.shape}")

    print("Done. You can now run python3 main.py normally.")
