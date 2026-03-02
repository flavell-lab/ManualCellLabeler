# ManualCellLabeler

Interactive Streamlit app for manual neuron label confirmation using NeuroPAL and related HDF5/CSV data.

**Usage**
- **Run the Streamlit app locally**:

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


**Containerized deployment**

Use `run.sh` to build (if needed) and launch the container:

```bash
./run.sh /store1/path/to/your/data_dir
```

Optional second argument for a custom output directory (defaults to `<data_dir>/relabelled`):

```bash
./run.sh /store1/path/to/your/data_dir /store1/path/to/output_dir
```

A free port is auto-selected from 3001-3030, so multiple users can run simultaneously. In case you would like to force a specific port (e.g. `3005`):

```bash
PORT=3005 ./run.sh /store1/path/to/your/data_dir
```

The assigned port is printed on startup (Firefox recommended).

The image is built automatically on first run or after a reboot (~35s cold build). Podman storage is kept under `/tmp/<user>-podman-storage/` to work around NFS xattr limitations.

**Integrate relabels to labels_csv on CLI**
```
uv run python update_labels_from_log.py /store1/path/to/output_dir/$(neuron_class)_$(user)_log.csv --apply 
```