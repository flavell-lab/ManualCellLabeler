import streamlit as st
import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from datetime import datetime

# --- 1. CONFIGURATION & STATE ---
st.set_page_config(layout="wide", page_title="ManualCellLabeler")

# Constants
FLV_UTILS_DATA_DIR = "/store1/shared/flv_utils_data/"
ARROW_STYLE = dict(facecolor='white', edgecolor='black', width=4, headwidth=12, shrink=0.05)
GAP, LENGTH = 2, 8
TAIL_DIST = GAP + LENGTH

# Initialize Session State
if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'decision' not in st.session_state:
    st.session_state.decision = "Not Sure"

def inject_ui_styling():
    st.markdown("""
        <style>
        div[data-testid="stSegmentedControl"] button { height: 4rem !important; min-height: 4rem !important; }
        div[data-testid="stSegmentedControl"] p { font-size: 22px !important; font-weight: bold !important; }
        .input-label { font-size: 22px !important; font-weight: bold !important; margin-bottom: -10px !important; margin-top: 15px !important; display: block; }
        .metadata-footer { color: #333; margin-top: 300px; font-size: 14px; border-top: 1px solid #222; padding-top: 10px; }
        </style>
        """, unsafe_allow_html=True)

# --- 2. UTILITIES ---

def get_valid_uids(prj_name):
    prj_root = os.path.join(FLV_UTILS_DATA_DIR, prj_name)
    label_dir = os.path.join(prj_root, "labels_csv")
    if not os.path.exists(label_dir): return []

    potential_uids = [f.replace(".csv", "") for f in os.listdir(label_dir) if f.endswith(".csv")]
    valid_uids = []
    for uid in potential_uids:
        paths = [
            os.path.join(prj_root, f"autolabel_data/{uid}-NeuroPAL.h5"),
            os.path.join(prj_root, f"neuron_rois/{uid}-roi.h5"),
            os.path.join(prj_root, f"processed_h5/{uid}-data.h5"),
            os.path.join(prj_root, f"labels_csv/{uid}.csv")
        ]
        if all(os.path.exists(p) for p in paths): valid_uids.append(uid)
    return valid_uids

@st.cache_data
def aggregate_matches(query, selected_configs):
    all_matches = []
    for prj, uid in selected_configs:
        path_csv = os.path.join(FLV_UTILS_DATA_DIR, prj, f"labels_csv/{uid}.csv")
        df = pd.read_csv(path_csv)
        subset = df[df['Neuron Class'].str.startswith(query, na=False)].copy()
        subset['prj_name'] = prj
        subset['data_uid'] = uid
        all_matches.append(subset)
    return pd.concat(all_matches).reset_index(drop=True) if all_matches else pd.DataFrame()

def load_roi_data(prj, uid):
    prj_root = os.path.join(FLV_UTILS_DATA_DIR, prj)
    try:
        with h5py.File(os.path.join(prj_root, f"neuron_rois/{uid}-roi.h5"), 'r') as f: rois = f['roi'][:]
        with h5py.File(os.path.join(prj_root, f"autolabel_data/{uid}-NeuroPAL.h5"), 'r') as f: raw = f['raw'][:]
        with h5py.File(os.path.join(prj_root, f"processed_h5/{uid}-data.h5"), 'r') as f:
            rev = f['behavior/reversal_vec'][:]
            traces = f['gcamp/trace_array_original'][:]
            match = f['neuropal_registration/roi_match'][:]
        return rois, raw, rev, traces, match
    except Exception as e:
        st.error(f"Failed to load {prj}/{uid}: {e}"); return None

def normalize(arr):
    p_low, p_high = np.percentile(arr, [5, 99.7])
    return (np.clip(arr, p_low, p_high) - p_low) / (p_high - p_low + 1e-5) if p_high != p_low else np.zeros_like(arr)

def create_composite_mip(neuropal_data, roi_volume, target_roi_id):
    mip = np.max(neuropal_data, axis=1)
    r, g, b, s = [normalize(mip[i]) for i in range(4)]
    roi_mip = np.max((roi_volume == target_roi_id).astype(float), axis=0)
    return np.stack([r, g, b], axis=-1), s, np.stack([roi_mip, roi_mip, np.zeros_like(roi_mip)], axis=-1)

def draw_arrows(ax, fx, fy):
    for dx, dy, ox, oy in [(0, -GAP, 0, -TAIL_DIST), (0, GAP, 0, TAIL_DIST), (-GAP, 0, -TAIL_DIST, 0), (GAP, 0, TAIL_DIST, 0)]:
        ax.annotate('', xy=(fx+dx, fy+dy), xytext=(fx+ox, fy+oy), arrowprops=ARROW_STYLE)

def save_current_entry(entry_data, path, q, user):
    os.makedirs(path, exist_ok=True)
    save_file = os.path.join(path, f"{q}_{user}_log.csv")
    pd.DataFrame([entry_data]).to_csv(save_file, mode='a', header=not os.path.exists(save_file), index=False)

# --- 3. PHASE 1: SETUP ---
if not st.session_state.setup_complete:
    st.title("🧫 ManualCellLabeler Setup")
    
    c1, c2 = st.columns(2)
    with c1:
        annotator = st.text_input("Enter flv-c username", value="candy")
        query = st.text_input("Neuron Class Query", value="NSM")
    with c2:
        dest_path = st.text_input("CSV Destination Path", value=f"/store1/{annotator}/")
        
    all_projects = [d for d in os.listdir(FLV_UTILS_DATA_DIR) if os.path.isdir(os.path.join(FLV_UTILS_DATA_DIR, d))]
    selected_projects = st.multiselect("Select Projects", options=all_projects)
    
    if selected_projects:
        final_configs = []
        for prj in selected_projects:
            valid_uids = get_valid_uids(prj)
            if valid_uids:
                chosen_uids = st.multiselect(f"UIDs for {prj}", options=valid_uids, default=valid_uids)
                for uid in chosen_uids: final_configs.append((prj, uid))
        
        if st.button("🚀 LAUNCH", type="primary", use_container_width=True):
            matches = aggregate_matches(query, final_configs)
            if not matches.empty:
                st.session_state.update({"matches": matches, "query": query, "annotator": annotator, "dest_path": dest_path, "setup_complete": True, "current_index": 0})
                st.rerun()

# --- 4. PHASE 2: ANNOTATION ---
else:
    inject_ui_styling()
    matches, query, annotator, dest_path = st.session_state.matches, st.session_state.query, st.session_state.annotator, st.session_state.dest_path

    if st.session_state.current_index < len(matches):
        row = matches.iloc[st.session_state.current_index]
        roi_id, curr_prj, curr_uid = int(row['ROI ID']), row['prj_name'], row['data_uid']
        
        # Decision logic based on confidence
        conf_val = int(row['Confidence']) if pd.notna(row['Confidence']) else 3
        if 'last_idx' not in st.session_state or st.session_state.last_idx != st.session_state.current_index:
            if conf_val >= 4: st.session_state.decision = "Yes"
            elif conf_val == 3: st.session_state.decision = "Not Sure"
            else: st.session_state.decision = "No"
            st.session_state.last_idx = st.session_state.current_index

        data = load_roi_data(curr_prj, curr_uid)
        if data:
            h5_rois, h5_raw, reversal_vec, traces_array, roi_match = data
            col_viz, col_ctrl = st.columns([6, 4], gap="large")

            with col_viz:
                coords = np.argwhere(h5_rois == roi_id)
                if coords.size > 0:
                    z_c, y_c, x_c = coords.mean(axis=0).astype(int)
                    rgb_f, s_f, _ = create_composite_mip(h5_raw, h5_rois, roi_id)
                    full_ctx = (rgb_f * 0.7) + (np.stack([s_f]*3, -1) * 0.3)
                    z_s, z_e = np.clip([z_c-4, z_c+4], 0, 63)
                    rgb_t, s_t, y_t = create_composite_mip(h5_raw[:, z_s:z_e], h5_rois[z_s:z_e], roi_id)
                    thin_ctx = (rgb_t * 0.7) + (np.stack([s_t]*3, -1) * 0.3)
                    thin_loc = np.where(y_t[..., 0:1] > 0, y_t, np.stack([s_t]*3, -1))

                    fig, axes = plt.subplots(4, 1, figsize=(7, 14), gridspec_kw={'hspace': 0.15})
                    fig.patch.set_alpha(0.0) 
                    imgs = [full_ctx, thin_ctx, thin_loc]
                    labels = ["FULL MIP", f"Z-SLICES\n{z_s}-{z_e}", f"ROI {roi_id}", "TRACE"]
                    for i in range(3):
                        axes[i].imshow(np.clip(imgs[i], 0, 1))
                        if i < 2: draw_arrows(axes[i], x_c, y_c)
                        axes[i].set_ylabel(labels[i], fontsize=12, fontweight='bold', color='white', rotation=0, labelpad=45, va='center')
                        axes[i].set_xticks([]); axes[i].set_yticks([]); axes[i].set_facecolor((0,0,0,0))
                    
                    trace_idx = roi_match[roi_id-1]
                    axes[3].plot(traces_array[..., trace_idx-1], color='lime', linewidth=1.5)
                    for idx in np.where(reversal_vec == 1)[0]: axes[3].axvspan(idx, idx+1, color='pink', alpha=0.3)
                    axes[3].set_ylabel(labels[3], fontsize=12, fontweight='bold', color='white', rotation=0, labelpad=45, va='center')
                    axes[3].tick_params(colors='white'); axes[3].set_facecolor((0,0,0,0))
                    st.pyplot(fig, use_container_width=True, clear_figure=True)

            with col_ctrl:
                st.markdown(f"<h1 style='text-align: center; margin-bottom: 20px;'>Is this {query}?</h1>", unsafe_allow_html=True)
                
                selection = st.segmented_control("Decision", options=["Yes", "Not Sure", "No"], selection_mode="single", default=st.session_state.decision, key=f"d_{roi_id}_{curr_uid}")
                if selection: st.session_state.decision = selection

                st.divider()
                st.markdown("<span class='input-label'>Confidence Level</span>", unsafe_allow_html=True)
                confidence = st.select_slider("C", options=["1", "2", "3", "4", "5"], value=str(conf_val), key=f"c_{roi_id}_{curr_uid}", label_visibility="collapsed")

                st.markdown("<span class='input-label'>Confirmed Neuron ID</span>", unsafe_allow_html=True)
                default_label = row['Neuron Class'] if pd.notna(row['Neuron Class']) else query
                final_label = st.text_input("L", value=default_label if st.session_state.decision != "No" else "", key=f"l_{roi_id}_{curr_uid}", label_visibility="collapsed")

                st.markdown("<span class='input-label'>Alternative ID</span>", unsafe_allow_html=True)
                alt_1 = st.text_input("A1", value="", key=f"a1_{roi_id}_{curr_uid}", label_visibility="collapsed")
                
                st.markdown("<span class='input-label'>Notes</span>", unsafe_allow_html=True)
                notes = st.text_input("N", value="", key=f"n_{roi_id}_{curr_uid}", label_visibility="collapsed")

                st.divider()
                
                b_quit, b_prev, b_next = st.columns([1, 1, 1.5])
                
                with b_quit:
                    if st.button("🚪 QUIT & SAVE", use_container_width=True):
                        log_entry = {"Project": curr_prj, "UID": curr_uid, "ROI_ID": roi_id, "Decision": st.session_state.decision, "Conf": confidence, "Label": final_label, "Alt_1": alt_1, "Notes": notes, "Timestamp": datetime.now().isoformat()}
                        save_current_entry(log_entry, dest_path, query, annotator)
                        st.toast(f"Your annotations are logged here: {dest_path}")
                        st.session_state.setup_complete = False
                        st.rerun()
                        
                with b_prev:
                    if st.button("⬅️ PREVIOUS", use_container_width=True, disabled=(st.session_state.current_index == 0)):
                        st.session_state.current_index -= 1; st.rerun()
                        
                with b_next:
                    if st.button("SAVE & NEXT ➡️", type="primary", use_container_width=True):
                        log_entry = {"Project": curr_prj, "UID": curr_uid, "ROI_ID": roi_id, "Decision": st.session_state.decision, "Conf": confidence, "Label": final_label, "Alt_1": alt_1, "Notes": notes, "Timestamp": datetime.now().isoformat()}
                        save_current_entry(log_entry, dest_path, query, annotator)
                        st.session_state.current_index += 1; st.rerun()

                # --- METADATA FOOTER ---
                st.markdown(f"""
                    <div class='metadata-footer'>
                        <b>Debug Metadata</b><br>
                        ROI: {roi_id}<br>
                        Project: {curr_prj}<br>
                        Data UID: {curr_uid}<br>
                        Saving to: {dest_path}
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.success("Session complete!")
        st.info(f"Your annotations are logged here: {dest_path}")
        st.balloons()
        if st.button("Return to Setup"):
            st.session_state.setup_complete = False
            st.rerun()