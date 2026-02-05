# Label Update Logic

This document explains the logic flow for updating labels_csv files from manual review logs.

## Diagram

See `label_update_logic.png` for the complete flowchart.

## Processing Steps

### 1. Deduplication (Pre-processing)
- **Input**: Read all entries from log CSV
- **Action**: Remove duplicates based on (Project, UID, ROI_ID)
- **Rule**: Keep only the **latest entry** by timestamp
- **Output**: Warning shown if duplicates found

### 2. Decision Branch: Yes vs No

The logic splits into two main branches based on the `Decision` field:

---

## Branch A: Decision = "Yes"

**Goal**: Confirm the ROI belongs to the queried neuron class

### Rules
1. **Label Match Check**: Label must start with query string (e.g., "I2" for I2 query)
2. **Confidence Check**: Confidence must be >= 3
3. **ROI Existence**:
   - If ROI exists in labels_csv → **UPDATE** label and confidence
   - If ROI doesn't exist → **CREATE** new row

### Skip Conditions
- Label doesn't match query → Skip with warning
- Confidence < 3 → Skip with warning

---

## Branch B: Decision = "No"

**Goal**: Correct misidentified ROIs or mark them as special labels

### Sub-branch B1: Special Labels
**Special Labels**: UNKNOWN, granule, glia, GOOD

**Rules**:
- These labels are processed **regardless of confidence level**
- If ROI exists in labels_csv → **UPDATE** to special label
- If ROI doesn't exist → **CREATE** new row with special label

### Sub-branch B2: Other Neuron Classes
**Examples**: NSM, AVA, I1R, etc.

**Rules**:
- Confidence must be >= 3
- ROI must exist in labels_csv → **UPDATE** to new label
- If ROI doesn't exist → **SKIP** (only special labels can create new rows for Decision=No)

### Skip Conditions
- Non-special label with confidence < 3 → Skip with warning
- Non-special label, ROI doesn't exist → Skip (no warning, expected)

---

## Outcomes

### ✓ UPDATE
- Existing ROI gets new label and/or confidence
- ChangeLog updated with annotator and date
- Alternatives and Notes appended if provided

### ➕ CREATE
- New row added to labels_csv
- Fields populated: ROI ID, Neuron Class, Confidence, ChangeLog
- Only happens for:
  - Decision=Yes with conf>=3
  - Decision=No with special labels (any conf)

### ⚠️ SKIP (with warning)
- Entry doesn't meet processing criteria
- Warning printed at end with reason
- Examples:
  - Decision=Yes, conf < 3
  - Decision=No, non-special label, conf < 3

### SKIP (without warning)
- Entry filtered out but expected
- Example: Decision=No, non-special label, ROI doesn't exist

---

## Summary Statistics

After processing, the script reports:

1. **Duplicates found**: Number of duplicate entries removed
2. **Rules matched**:
   - Decision=Yes (label matches, conf>=3)
   - Decision=No + special labels (any conf)
   - Decision=No + other neuron classes (conf>=3)
3. **Label changes**: Full diff in format: `prj -- dataset -- roi -- label change`
4. **Warnings**: Entries that didn't match any processing rule

---

## Example Scenarios

### Scenario 1: High-confidence confirmation
```
Decision: Yes
Label: I2R
Confidence: 4
ROI exists: Yes
→ Result: UPDATE label to I2R, confidence to 4
```

### Scenario 2: New ROI identified
```
Decision: Yes
Label: I2L
Confidence: 3
ROI exists: No
→ Result: CREATE new row with I2L, confidence 3
```

### Scenario 3: Mark as UNKNOWN
```
Decision: No
Label: UNKNOWN
Confidence: 2
ROI exists: No
→ Result: CREATE new row with UNKNOWN, confidence 2
```

### Scenario 4: Misidentified neuron
```
Decision: No
Label: NSM
Confidence: 4
ROI exists: Yes
→ Result: UPDATE label to NSM, confidence to 4
```

### Scenario 5: Low confidence correction
```
Decision: No
Label: AVA
Confidence: 2
ROI exists: Yes
→ Result: SKIP with warning (conf < 3, not special label)
```

### Scenario 6: Granule cell (special)
```
Decision: No
Label: granule
Confidence: 2
ROI exists: No
→ Result: CREATE new row with granule, confidence 2
```

---

## Color Legend (in diagram)

- 🟢 **Green**: Start/Input, Success (Update/Create)
- 🔵 **Blue**: Decision Points
- 🟣 **Purple**: Special Rule (Special Labels)
- 🟠 **Orange**: Actions
- 🔴 **Red**: Skip/Warning

---

## Files

- `update_labels_from_log.py`: Main script
- `label_update_logic.png`: Visual flowchart
- `LABEL_UPDATE_LOGIC.md`: This document
