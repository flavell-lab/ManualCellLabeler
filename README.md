# ManualCellLabeler

Interactive Streamlit app for manual neuron label confirmation using NeuroPAL and related HDF5/CSV data.

**Usage**
- **Activate project virtualenv** (from project root):

```bash
source .venv/bin/activate
```

- **Run the Streamlit app**:

```bash
source .venv/bin/activate && streamlit run app.py
```

- **Open the UI**: visit http://localhost:<port_number> (Firefox is the recommended browser for this app)

**Notes**
- Default data root used by the app: `/store1/shared/flv_utils_data/` (see `FLV_UTILS_DATA_DIR` in `app.py`).
- The app expects project subfolders with `autolabel_data`, `neuron_rois`, `processed_h5`, and `labels_csv`.

**Development**
- Python >= 3.10. Use the included `.venv` for reproducible environment.
- Dependencies are listed in `pyproject.toml`.


**Podman (containerized deployment)**

This project uses podman. Because the NFS-backed home directory does not support extended attributes, podman storage must be pointed to a local filesystem (e.g. `/tmp`).

Build:

```bash
# Set up local storage (required on NFS-backed machines; cleared on reboot)
export PODMAN_ROOT=/tmp/candy-podman-storage/root
export PODMAN_RUNROOT=/tmp/candy-podman-storage/runroot

podman --root $PODMAN_ROOT --runroot $PODMAN_RUNROOT build -t manual-cell-labeler .
```

Run:

```bash
podman --root $PODMAN_ROOT --runroot $PODMAN_RUNROOT run -p 8501:8501 \
  -v /store1/shared/flv_utils_data:/store1/shared/flv_utils_data:ro \
  -v /store1/shared/flv_utils_data/flagging/relabelled:/store1/shared/flv_utils_data/flagging/relabelled \
  manual-cell-labeler
```

Then open http://localhost:8501 (Firefox recommended).

Customize the `-v` mounts for your data and output directories. The cold build takes ~35s after a reboot (no cached layers).

Clean up storage:

```bash
podman --root $PODMAN_ROOT --runroot $PODMAN_RUNROOT system reset --force
```

**Integrate relabels to labels_csv**
```
uv run python update_labels_from_log.py /store1/shared/flv_utils_data/flagging/relabelled/AVA_candy_log.csv --apply 
```