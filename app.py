import streamlit as st
import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from datetime import datetime

# --- 1. CONFIGURATION & UI STYLING ---
st.set_page_config(layout="wide", page_title="NeuroPAL Annotator")

# Initialize session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'decision' not in st.session_state:
    st.session_state.decision = "Not Sure"

def inject_custom_css():
    selected = st.session_state.decision
    
    # Colors for each button (all yellow for now)
    green = "#ffc107"
    yellow = "#ffc107"
    red = "#ffc107"
    
    # Border and shadow logic: Thick white border + box shadow when selected
    y_border = "4px solid white" if selected == "Yes" else "2px solid rgba(255, 255, 255, 0.3)"
    y_shadow = "0 0 20px rgba(255, 255, 255, 0.8), 0 0 30px rgba(40, 167, 69, 0.6)" if selected == "Yes" else "none"
    
    s_border = "4px solid white" if selected == "Not Sure" else "2px solid rgba(255, 255, 255, 0.3)"
    s_shadow = "0 0 20px rgba(255, 255, 255, 0.8), 0 0 30px rgba(255, 193, 7, 0.6)" if selected == "Not Sure" else "none"
    
    n_border = "4px solid white" if selected == "No" else "2px solid rgba(255, 255, 255, 0.3)"
    n_shadow = "0 0 20px rgba(255, 255, 255, 0.8), 0 0 30px rgba(220, 53, 69, 0.6)" if selected == "No" else "none"

    st.markdown(f"""
        <style>
        .big-font {{ font-size:22px !important; font-weight: bold; margin-bottom: 8px; display: block; }}
        .conf-label {{ font-size: 24px !important; font-weight: bold !important; margin-bottom: 10px; }}
        
        /* YES Button - Yellow */
        div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {{ 
            background-color: {green} !important; 
            color: black !important; 
            border: {y_border} !important;
            box-shadow: {y_shadow} !important;
            transition: all 0.2s ease !important;
        }}
        
        /* NOT SURE Button - Yellow */
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {{ 
            background-color: {yellow} !important; 
            color: black !important; 
            border: {s_border} !important;
            box-shadow: {s_shadow} !important;
            transition: all 0.2s ease !important;
        }}
        
        /* NO Button - Yellow */
        div[data-testid="stHorizontalBlock"] > div:nth-child(3) button {{ 
            background-color: {red} !important; 
            color: black !important; 
            border: {n_border} !important;
            box-shadow: {n_shadow} !important;
            transition: all 0.2s ease !important;
        }}
        
        .stButton>button {{ height: 3.5em; font-size: 18px !important; font-weight: bold !important; }}
        
        /* Ensure background of app is consistent */
        .stApp {{ background-color: transparent; }}
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

ARROW_STYLE = dict(facecolor='white', edgecolor='black', width=4, headwidth=12, shrink=0.05)
GAP, LENGTH = 2.5, 7.5
TAIL_DIST = GAP + LENGTH

@st.cache_resource
def load_all_data():
    with h5py.File(path_roi_h5, 'r') as f: rois = f['roi'][:]
    with h5py.File(path_neuropal_h5, 'r') as f: raw = f['raw'][:]
    with h5py.File(path_processed_h5, 'r') as f:
        rev = f['behavior/reversal_vec'][:]
        traces = f['gcamp/trace_array_original'][:]
        match = f['neuropal_registration/roi_match'][:]
        conf = f['neuropal_registration/roi_match_confidence'][:]
    df = pd.read_csv(path_labels_csv) if os.path.exists(path_labels_csv) else None
    return rois, raw, rev, traces, match, conf, df

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
inject_custom_css()

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

                # SET FIGURE BACKGROUND TO TRANSPARENT
                fig, axes = plt.subplots(4, 1, figsize=(7, 14), gridspec_kw={'hspace': 0.15})
                fig.patch.set_alpha(0.0) 
                
                imgs = [full_ctx, thin_ctx, thin_loc]
                labels = ["FULL MIP", f"Z-SLICES\n{z_s}-{z_e}\n(out of 64)", f"ROI {roi_id}", "TRACE"]
                
                for i in range(3):
                    axes[i].imshow(np.clip(imgs[i], 0, 1))
                    if i < 2: draw_arrows(axes[i], x_c, y_c)
                    axes[i].set_ylabel(labels[i], fontsize=12, fontweight='bold', color='white', rotation=0, labelpad=45, va='center')
                    axes[i].set_xticks([]); axes[i].set_yticks([])
                    axes[i].set_facecolor((0,0,0,0)) # Transparent axes

                trace_idx = roi_match[roi_id-1]
                axes[3].plot(traces_array[..., trace_idx-1], color='white', linewidth=1.2) # White trace for dark theme
                for idx in np.where(reversal_vec == 1)[0]: axes[3].axvspan(idx, idx+1, color='pink', alpha=0.3)
                axes[3].set_ylabel(labels[3], fontsize=12, fontweight='bold', color='white', rotation=0, labelpad=45, va='center')
                axes[3].tick_params(colors='white')
                axes[3].set_facecolor((0,0,0,0))
                
                st.pyplot(fig, use_container_width=True, clear_figure=True)

        with col_ctrl:
            st.markdown(f"<h1 style='text-align: center; font-size: 42px;'>Is this {query}?</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; font-weight: bold; font-size: 18px;'>ROI: {roi_id} | Match {st.session_state.current_index + 1} of {len(matches)}</p>", unsafe_allow_html=True)
            st.divider()

            # Color-coded Decision Buttons with White Border and Glow for Selected
            btn_cols = st.columns(3)
            with btn_cols[0]:
                if st.button("YES", use_container_width=True): st.session_state.decision = "Yes"; st.rerun()
            with btn_cols[1]:
                if st.button("NOT SURE", use_container_width=True): st.session_state.decision = "Not Sure"; st.rerun()
            with btn_cols[2]:
                if st.button("NO", use_container_width=True): st.session_state.decision = "No"; st.rerun()

            st.markdown(f"<p class='big-font' style='text-align: center; color: #ffc107;'>Selected: {st.session_state.decision}</p>", unsafe_allow_html=True)
            
            st.markdown("<p class='conf-label'>Confidence</p>", unsafe_allow_html=True)
            confidence = st.select_slider("Conf", options=["1", "2", "3", "4", "5"], value="2", key=f"c_{roi_id}", label_visibility="collapsed")

            st.markdown("<p class='big-font'>Confirmed Neuron ID</p>", unsafe_allow_html=True)
            final_label = st.text_input("Label", value=query if st.session_state.decision != "No" else "", key=f"l_{roi_id}", label_visibility="collapsed")

            st.markdown("<p class='big-font'>Alternative ID #1 (Optional)</p>", unsafe_allow_html=True)
            alt_1 = st.text_input("A1", key=f"a1_{roi_id}", label_visibility="collapsed")
            
            st.markdown("<p class='big-font'>Alternative ID #2 (Optional)</p>", unsafe_allow_html=True)
            alt_2 = st.text_input("A2", key=f"a2_{roi_id}", label_visibility="collapsed")

            # Notes changed to single-row text_input
            st.markdown("<p class='big-font'>Notes (Optional)</p>", unsafe_allow_html=True)
            notes = st.text_input("N", key=f"n_{roi_id}", label_visibility="collapsed")

            st.divider()
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
                        "Label": final_label, "Alt_1": alt_1, "Alt_2": alt_2, "Notes": notes, 
                        "Timestamp": datetime.now().isoformat()
                    }
                    pd.DataFrame([log_entry]).to_csv(f"{query}_{annotator}_log.csv", mode='a', header=not os.path.exists(f"{query}_{annotator}_log.csv"), index=False)
                    st.session_state.decision = "Not Sure"
                    st.session_state.current_index += 1
                    st.rerun()
else:
    st.error("Data not found.")