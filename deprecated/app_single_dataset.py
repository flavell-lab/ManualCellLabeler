import streamlit as st
import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="NeuroPAL Annotator")

# Initialize session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'decision' not in st.session_state:
    st.session_state.decision = "Not Sure"

def inject_ui_styling():
    st.markdown("""
        <style>
        /* Make Segmented Control (Yes/No/Not Sure) larger */
        div[data-testid="stSegmentedControl"] button {
            height: 4rem !important;
            min-height: 4rem !important;
        }
        div[data-testid="stSegmentedControl"] p {
            font-size: 22px !important;
            font-weight: bold !important;
        }
        
        /* Custom label styling for text boxes */
        .input-label {
            font-size: 22px !important;
            font-weight: bold !important;
            margin-bottom: -10px !important;
            margin-top: 15px !important;
            display: block;
        }
        </style>
        """, unsafe_allow_html=True)

# --- 2. DATA LOADING & PROCESSING ---
flv_utils_data_dir = "/store1/shared/flv_utils_data/"
prj_name = "prj_gfpneuropal"
query = 'NSM'
annotator = 'candy'
z_window = 8  
prj_root = f"{flv_utils_data_dir}/{prj_name}/"
data_uid = "2025-03-15-15"

path_neuropal_h5  = os.path.join(prj_root, f"autolabel_data/{data_uid}-NeuroPAL.h5")
path_roi_h5       = os.path.join(prj_root, f"neuron_rois/{data_uid}-roi.h5")
path_processed_h5 = os.path.join(prj_root, f"processed_h5/{data_uid}-data.h5")
path_labels_csv   = os.path.join(prj_root, f"labels_csv/{data_uid}.csv")

# ARROW SETTINGS (Gap=2, Length=8)
ARROW_STYLE = dict(facecolor='white', edgecolor='black', width=4, headwidth=12, shrink=0.05)
GAP = 2
LENGTH = 8
TAIL_DIST = GAP + LENGTH

@st.cache_resource
def load_all_data():
    try:
        with h5py.File(path_roi_h5, 'r') as f: rois = f['roi'][:]
        with h5py.File(path_neuropal_h5, 'r') as f: raw = f['raw'][:]
        with h5py.File(path_processed_h5, 'r') as f:
            rev = f['behavior/reversal_vec'][:]
            traces = f['gcamp/trace_array_original'][:]
            match = f['neuropal_registration/roi_match'][:]
            conf = f['neuropal_registration/roi_match_confidence'][:]
        df = pd.read_csv(path_labels_csv) if os.path.exists(path_labels_csv) else None
        return rois, raw, rev, traces, match, conf, df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None, None

def normalize(arr):
    p_low, p_high = np.percentile(arr, [5, 99.7])
    if p_high == p_low: return np.zeros_like(arr)
    return (np.clip(arr, p_low, p_high) - p_low) / (p_high - p_low + 1e-5)

def create_composite_mip(neuropal_data, roi_volume, target_roi_id):
    mip = np.max(neuropal_data, axis=1)
    r, g, b, s = [normalize(mip[i]) for i in range(4)]
    roi_mip = np.max((roi_volume == target_roi_id).astype(float), axis=0)
    return np.stack([r, g, b], axis=-1), s, np.stack([roi_mip, roi_mip, np.zeros_like(roi_mip)], axis=-1)

def draw_arrows(ax, fx, fy):
    for dx, dy, ox, oy in [(0, -GAP, 0, -TAIL_DIST), (0, GAP, 0, TAIL_DIST), (-GAP, 0, -TAIL_DIST, 0), (GAP, 0, TAIL_DIST, 0)]:
        ax.annotate('', xy=(fx+dx, fy+dy), xytext=(fx+ox, fy+oy), arrowprops=ARROW_STYLE)

h5_rois, h5_raw, reversal_vec, traces_array, roi_match, roi_match_conf, df_labels = load_all_data()

# --- 3. MAIN APP ---
inject_ui_styling()

if df_labels is not None:
    matches = df_labels[df_labels['Neuron Class'].str.startswith(query, na=False)].reset_index(drop=True)
    
    if st.session_state.current_index < len(matches):
        row = matches.iloc[st.session_state.current_index]
        roi_id = int(row['ROI ID'])

        col_viz, col_ctrl = st.columns([6, 4], gap="large")

        with col_viz:
            coords = np.argwhere(h5_rois == roi_id)
            if coords.size > 0:
                z_c, y_c, x_c = coords.mean(axis=0).astype(int)
                rgb_f, s_f, _ = create_composite_mip(h5_raw, h5_rois, roi_id)
                full_ctx = (rgb_f * 0.7) + (np.stack([s_f]*3, -1) * 0.3)
                
                half = z_window // 2
                z_s, z_e = np.clip([z_c-half, z_c+half], 0, 63)
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
                    axes[i].set_xticks([]); axes[i].set_yticks([])
                    axes[i].set_facecolor((0,0,0,0))

                # --- TRACE COLOR CHANGED TO LIME ---
                trace_idx = roi_match[roi_id-1]
                axes[3].plot(traces_array[..., trace_idx-1], color='lime', linewidth=1.5)
                for idx in np.where(reversal_vec == 1)[0]: axes[3].axvspan(idx, idx+1, color='pink', alpha=0.3)
                axes[3].set_ylabel(labels[3], fontsize=12, fontweight='bold', color='white', rotation=0, labelpad=45, va='center')
                axes[3].tick_params(colors='white')
                axes[3].set_facecolor((0,0,0,0))
                
                st.pyplot(fig, use_container_width=True, clear_figure=True)

        with col_ctrl:
            st.markdown(f"<h1 style='text-align: center; margin-bottom: 0px;'>Is this {query}?</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: gray; font-size: 20px;'>ROI: {roi_id}</p>", unsafe_allow_html=True)
            
            # --- LARGER SELECTION ---
            options = ["Yes", "Not Sure", "No"]
            icons = ["✅", "❓", "❌"]
            
            selection = st.segmented_control(
                "Decision",
                options=options,
                format_func=lambda x: f"{icons[options.index(x)]} {x}",
                selection_mode="single",
                default=st.session_state.decision,
                key="decision_input",
                label_visibility="collapsed"
            )
            
            if selection:
                st.session_state.decision = selection

            st.divider()
            
            # --- CONFIDENCE (From Matches) ---
            st.markdown("<span class='input-label'>Confidence Level</span>", unsafe_allow_html=True)
            default_conf = str(int(row['Confidence'])) if pd.notna(row['Confidence']) else "3"
            confidence = st.select_slider("C", options=["1", "2", "3", "4", "5"], value=default_conf, key=f"c_{roi_id}", label_visibility="collapsed")

            # --- LABELS (Larger text, Alternative blank) ---
            st.markdown("<span class='input-label'>Confirmed Neuron ID</span>", unsafe_allow_html=True)
            default_label = row['Neuron Class'] if pd.notna(row['Neuron Class']) else query
            final_label = st.text_input("L", value=default_label if st.session_state.decision != "No" else "", key=f"l_{roi_id}", label_visibility="collapsed")

            st.markdown("<span class='input-label'>Alternative ID</span>", unsafe_allow_html=True)
            alt_1 = st.text_input("A1", value="", key=f"a1_{roi_id}", label_visibility="collapsed")
            
            st.markdown("<span class='input-label'>Notes</span>", unsafe_allow_html=True)
            notes = st.text_input("N", value="", key=f"n_{roi_id}", label_visibility="collapsed")

            st.divider()
            
            # Navigation
            b1, b2 = st.columns(2)
            with b1:
                if st.button("⬅️ PREVIOUS", use_container_width=True, disabled=(st.session_state.current_index == 0)):
                    st.session_state.current_index -= 1
                    st.session_state.decision = "Not Sure"
                    st.rerun()
            with b2:
                if st.button("SAVE & NEXT ➡️", type="primary", use_container_width=True):
                    log_entry = {
                        "ROI_ID": roi_id, "Decision": st.session_state.decision, "Conf": confidence, 
                        "Label": final_label, "Alt_1": alt_1, "Notes": notes, 
                        "Timestamp": datetime.now().isoformat()
                    }
                    pd.DataFrame([log_entry]).to_csv(f"{query}_{annotator}_log.csv", mode='a', header=not os.path.exists(f"{query}_{annotator}_log.csv"), index=False)
                    
                    st.session_state.current_index += 1
                    st.session_state.decision = "Not Sure"
                    st.rerun()
else:
    st.error("Data not found.")
