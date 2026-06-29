# RetinaGuard AI - Systematic Error Analysis Report

**Classification Threshold:** inf

**Total Samples Evaluated:** 10

**Total Errors:** 6 (Error Rate: 60.00%)

---

## 1. Error Categories Breakdown

| Category | Count | Percentage of Total |
|---|---|---|
| TP | 0 | 0.00% |
| TN | 4 | 40.00% |
| FP | 0 | 0.00% |
| FN | 6 | 60.00% |


## 2. High-Confidence Mistakes

> Defined as incorrect predictions where predicted probability is ≥ 0.80 or ≤ 0.20.

**Total Count:** 0

No high-confidence mistakes found 



## 3. Low-Confidence Correct Predictions

> Defined as correct predictions where predicted probability is between 0.40 and 0.60.

**Total Count:** 4

| Image ID | True Label | True DR | Prob | Type | Brightness | Sharpness |
|---|---|---|---|---|---|---|
| `IDRiD_041` | 0 | Grade 0 | 0.5046 | TN | 120.0 | 50.0 |
| `IDRiD_042` | 0 | Grade 1 | 0.5046 | TN | 120.0 | 50.0 |
| `IDRiD_046` | 0 | Grade 0 | 0.5046 | TN | 120.0 | 50.0 |
| `IDRiD_047` | 0 | Grade 1 | 0.5046 | TN | 120.0 | 50.0 |


## 4. Error Analysis by Disease Grade

| DR Grade | Count | Errors | Error Rate |
|---|---|---|---|
| Grade 0 | 2 | 0 | 0.00% |
| Grade 1 | 2 | 0 | 0.00% |
| Grade 2 | 2 | 2 | 100.00% |
| Grade 3 | 2 | 2 | 100.00% |
| Grade 4 | 2 | 2 | 100.00% |


## 5. Error Analysis by DME Grade

| DME Grade | Count | Errors | Error Rate |
|---|---|---|---|
| DME Risk 0 | 3 | 2 | 66.67% |
| DME Risk 1 | 4 | 2 | 50.00% |
| DME Risk 2 | 3 | 2 | 66.67% |


## 6. Error Analysis by Image Quality

### Errors by Brightness Quartile

| Brightness Quartile | Count | Errors | Error Rate |
|---|---|---|---|
| Q1 (Low) | 10 | 6 | 60.00% |


### Errors by Sharpness Quartile

| Sharpness Quartile | Count | Errors | Error Rate |
|---|---|---|---|
| Q1 (Low) | 10 | 6 | 60.00% |


## 7. Actionable Findings & Recommendations

- **False Negatives (FN):** These are cases of DR grade ≥ 2 that the model missed. Check if they have small localized lesions (like isolated microaneurysms) that might be lost during image resizing.
- **False Positives (FP):** Check if they have artifacts, vessel crossings, or bright spots that mimic exudates.
- **Image Quality:** If error rates are significantly higher in the Q1 brightness or sharpness quartiles, it suggests a need for stricter quality filtering or custom normalization.