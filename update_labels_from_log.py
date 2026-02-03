#!/usr/bin/env python3
"""
Update labels_csv files based on manual relabeling log.

Rules:
- Only update rows where Decision == "Yes"
- Always update Confidence
- Update Neuron Class only if Label is not empty
- Append Alt_1 to Alternatives if not empty
- Append Notes to Notes if not empty
"""

import pandas as pd
import os
from pathlib import Path

# Configuration
FLV_UTILS_DATA_DIR = "/store1/shared/flv_utils_data/"
RELABEL_LOG = "/store1/shared/flv_utils_data/flagging/relabelled/AVA_candy_log.csv"

def update_labels_csv(log_path, dry_run=True):
    """
    Update labels_csv files based on relabeling log.

    Args:
        log_path: Path to the relabeling log CSV
        dry_run: If True, only show what would be updated without making changes
    """
    # Read the relabeling log
    log_df = pd.read_csv(log_path)

    # Extract query from filename (e.g., RMG_candy_log.csv -> RMG)
    filename = os.path.basename(log_path)
    query = filename.split('_')[0]

    # Filter for rows to process:
    # 1. Decision='Yes' AND Label agrees with query (starts with query)
    # 2. Decision='No' AND Label='UNKNOWN' (any confidence)
    # 3. Decision='No' AND Label is a neuron class (not UNKNOWN) AND Conf >= 3
    def should_process(row):
        decision = row['Decision']
        label = str(row['Label']).strip() if pd.notna(row['Label']) else ''
        conf = row['Conf'] if pd.notna(row['Conf']) else 0

        if decision == 'Yes':
            # Only process if label agrees with query
            return label.startswith(query) or label == ''
        elif decision == 'No':
            if label == 'UNKNOWN' or label == '':
                return True  # Always process UNKNOWN/empty regardless of confidence
            else:
                return conf >= 3  # Neuron class labels need conf >= 3
        return False

    updates = log_df[log_df.apply(should_process, axis=1)].copy()

    yes_count = len(updates[updates['Decision'] == 'Yes'])
    no_unknown_count = len(updates[(updates['Decision'] == 'No') & ((updates['Label'] == 'UNKNOWN') | (updates['Label'].isna()) | (updates['Label'].str.strip() == ''))])
    no_neuron_count = len(updates[(updates['Decision'] == 'No') & (updates['Label'] != 'UNKNOWN') & (updates['Label'].notna()) & (updates['Label'].str.strip() != '')])

    print(f"Query: {query}")
    print(f"Found {yes_count} ROIs with Decision='Yes' (label agrees with query)")
    print(f"Found {no_unknown_count} ROIs with Decision='No' + UNKNOWN label")
    print(f"Found {no_neuron_count} ROIs with Decision='No' + neuron class (conf>=3)")
    print(f"Mode: {'DRY RUN (no files will be modified)' if dry_run else 'LIVE UPDATE'}")
    print("=" * 80)

    # Group by Project and UID for efficiency
    grouped = updates.groupby(['Project', 'UID'])

    summary = []

    for (project, uid), group in grouped:
        csv_path = os.path.join(FLV_UTILS_DATA_DIR, project, f"labels_csv/{uid}.csv")

        if not os.path.exists(csv_path):
            print(f"⚠️  WARNING: CSV not found: {csv_path}")
            continue

        # Read the original labels CSV
        labels_df = pd.read_csv(csv_path)

        print(f"\n📁 Processing: {project}/{uid}")
        print(f"   CSV: {csv_path}")

        # Track if any changes were made
        changes_made = False

        # Ensure ChangeLog column exists
        if 'ChangeLog' not in labels_df.columns:
            labels_df['ChangeLog'] = ""

        # Process each ROI in this group
        for idx, row in group.iterrows():
            roi_id = int(row['ROI_ID'])
            new_conf = int(row['Conf'])
            new_label = str(row['Label']).strip() if pd.notna(row['Label']) else ""
            alt_1 = str(row['Alt_1']).strip() if pd.notna(row['Alt_1']) else ""
            notes = str(row['Notes']).strip() if pd.notna(row['Notes']) else ""

            # Extract annotator from filename (e.g., AVA_candy_log.csv -> candy)
            filename = os.path.basename(log_path)
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 2:
                annotator = parts[1]  # Get 'candy' from 'AVA_candy_log'
            else:
                annotator = "unknown"

            # Extract date from timestamp (ignore time)
            timestamp_str = str(row['Timestamp']) if pd.notna(row['Timestamp']) else ""
            if timestamp_str:
                change_date = timestamp_str.split('T')[0]  # Get YYYY-MM-DD part
            else:
                from datetime import datetime
                change_date = datetime.now().strftime('%Y-%m-%d')

            # Find the matching row in labels_csv
            mask = labels_df['ROI ID'] == roi_id

            if not mask.any():
                print(f"   ⚠️  ROI {roi_id} not found in labels_csv")
                continue

            # Get the current values
            current_label = labels_df.loc[mask, 'Neuron Class'].iloc[0]
            current_conf = labels_df.loc[mask, 'Confidence'].iloc[0]

            # Check if Alternatives and Notes columns exist
            if 'Alternatives' in labels_df.columns:
                current_alts = labels_df.loc[mask, 'Alternatives'].iloc[0] if pd.notna(labels_df.loc[mask, 'Alternatives'].iloc[0]) else ""
            else:
                current_alts = ""
                labels_df['Alternatives'] = ""  # Create the column if it doesn't exist

            if 'Notes' in labels_df.columns:
                current_notes = labels_df.loc[mask, 'Notes'].iloc[0] if pd.notna(labels_df.loc[mask, 'Notes'].iloc[0]) else ""
            else:
                current_notes = ""
                labels_df['Notes'] = ""  # Create the column if it doesn't exist

            # Get current ChangeLog before any updates
            current_changelog = labels_df.loc[mask, 'ChangeLog'].iloc[0] if pd.notna(labels_df.loc[mask, 'ChangeLog'].iloc[0]) else ""

            # Prepare update message
            updates_desc = []
            decision = row['Decision']

            # NEW LOGIC based on Decision
            if decision == 'Yes':
                # Decision = Yes: Update confidence always
                # Update label only if it's different from original
                if current_conf != new_conf:
                    labels_df.loc[mask, 'Confidence'] = new_conf
                    updates_desc.append(f"Conf: {current_conf} → {new_conf}")
                    changes_made = True

                # If label is empty or same as current, keep current label
                # If label is different, update it
                if new_label and new_label != current_label:
                    labels_df.loc[mask, 'Neuron Class'] = new_label
                    updates_desc.append(f"Label: {current_label} → {new_label}")
                    changes_made = True

            elif decision == 'No':
                # Decision = No: Only update if new label provided with conf >= 3
                if new_label and new_conf >= 3:
                    # Update both label and confidence
                    if new_label != current_label:
                        labels_df.loc[mask, 'Neuron Class'] = new_label
                        updates_desc.append(f"Label: {current_label} → {new_label}")
                        changes_made = True

                    if current_conf != new_conf:
                        labels_df.loc[mask, 'Confidence'] = new_conf
                        updates_desc.append(f"Conf: {current_conf} → {new_conf}")
                        changes_made = True

            # Append Alt_1 to Alternatives if provided
            if alt_1:
                if current_alts:
                    new_alts = f"{current_alts} | Manual: {alt_1}"
                else:
                    new_alts = f"Manual: {alt_1}"
                labels_df.loc[mask, 'Alternatives'] = new_alts
                updates_desc.append(f"Added Alt: {alt_1}")
                changes_made = True

            # Append Notes if provided
            if notes:
                if current_notes:
                    new_notes = f"{current_notes} | Manual: {notes}"
                else:
                    new_notes = f"Manual: {notes}"
                labels_df.loc[mask, 'Notes'] = new_notes
                updates_desc.append(f"Added Note: {notes}")
                changes_made = True

            # Update ChangeLog if any changes were made
            if updates_desc:
                changelog_entry = f"{annotator} on {change_date}"

                if current_changelog:
                    new_changelog = f"{current_changelog} | {changelog_entry}"
                else:
                    new_changelog = changelog_entry

                labels_df.loc[mask, 'ChangeLog'] = new_changelog
                changes_made = True

            # Print update summary
            if updates_desc:
                changelog_entry = f"{annotator} on {change_date}"
                full_changelog = f"{current_changelog} | {changelog_entry}" if current_changelog else changelog_entry

                print(f"   ✓ ROI {roi_id}: {', '.join(updates_desc)}")
                print(f"      ChangeLog: {full_changelog}")
                summary.append({
                    'Project': project,
                    'UID': uid,
                    'ROI_ID': roi_id,
                    'Updates': ', '.join(updates_desc),
                    'ChangeLog': full_changelog
                })
            else:
                print(f"   - ROI {roi_id}: No changes needed")

        # Save the updated CSV if changes were made and not in dry run mode
        if changes_made and not dry_run:
            labels_df.to_csv(csv_path, index=False)
            print(f"   💾 Saved updated CSV")
        elif changes_made:
            print(f"   🔍 Would save updated CSV (dry run)")

    print("\n" + "=" * 80)
    print(f"Summary: {len(summary)} ROIs will be updated across {len(grouped)} files")

    return pd.DataFrame(summary)


if __name__ == "__main__":
    print("Manual Cell Labeler - Update Labels CSV from Log")
    print("=" * 80)
    print(f"Relabeling log: {RELABEL_LOG}")
    print()

    # Run in dry-run mode first
    summary_df = update_labels_csv(RELABEL_LOG, dry_run=True)

    print("\n" + "=" * 80)
    print("DRY RUN COMPLETE - No files were modified")
    print("=" * 80)
    print("\nTo apply these updates, run:")
    print("  python update_labels_from_log.py --apply")
    print("\nOr call update_labels_csv(RELABEL_LOG, dry_run=False)")
