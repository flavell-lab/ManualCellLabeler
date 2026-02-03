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

**Deprecated scripts**
- Some helper scripts are not required to launch the main app and have been moved to `deprecated/`.
- To restore a deprecated script, move it back to the project root.

**Development**
- Python >= 3.10. Use the included `.venv` for reproducible environment.
- Dependencies are listed in `pyproject.toml`.


**Integrate relabels to labels_csv**
```
from update_labels_from_log import update_labels_csv                                                                               
update_labels_csv("/store1/shared/flv_utils_data/flagging/relabelled/AVA_candy_log.csv", dry_run=False)    
```