#!/usr/bin/env python3
"""Create a *-roi.h5 file from neuron_rois_cropped.h5 or neuron_rois.nrrd.

Usage:
    python create_roi_h5.py <source> <output>

Examples:
    python create_roi_h5.py neuron_rois_cropped.h5 2025-03-15-15-roi.h5
    python create_roi_h5.py neuron_rois.nrrd        2025-03-15-15-roi.h5
"""
import sys
import h5py
import numpy as np


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]

    if src.endswith(".h5"):
        with h5py.File(src, "r") as f:
            roi = f["roi"][:]
        print(f"Loaded .h5:   shape={roi.shape}")
    elif src.endswith(".nrrd"):
        import nrrd
        data, _ = nrrd.read(src)
        print(f"Loaded .nrrd: shape={data.shape}")
        # nrrd is (X, Y, Z) -> transpose to (Z, Y, X)
        roi = data.transpose(2, 1, 0)
        print(f"Transposed:   shape={roi.shape}")
    else:
        sys.exit(f"Unsupported format: {src}")

    roi = roi.astype(np.uint16)

    # Validate: output must be (Z, Y, X) where Z < Y < X
    if roi.ndim != 3 or not (roi.shape[0] <= roi.shape[1] <= roi.shape[2]):
        sys.exit(
            f"ERROR: unexpected output shape {roi.shape}. "
            f"Expected (Z, Y, X) where Z <= Y <= X."
        )

    with h5py.File(dst, "w") as f:
        f.create_dataset("roi", data=roi, compression="gzip")

    print(f"Created {dst}: roi shape={roi.shape}, dtype={roi.dtype}")


if __name__ == "__main__":
    main()
