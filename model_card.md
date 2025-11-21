# Model Card: Census Income Classification

## Model Details

- **Model type:** RandomForestClassifier (scikit-learn)
- **Task:** Binary classification – predict whether an individual's income is `<=50K` or `>50K`.
- **Framework:** Python 3.10, scikit-learn, FastAPI
- **Developer:** Student project for WGU / Udacity MLOps course
- **Version:** 1.0.0
- **Training script:** `train_model.py`
- **Serving script:** `main.py` (FastAPI app)

The model is trained on a cleaned version of the UCI Adult / Census Income dataset and wrapped in a FastAPI service for inference.

---

## Intended Use

- **Primary intended use:** Demonstration of an end-to-end ML pipeline (training, evaluation, deployment, CI) as part of a course project.
- **Input:** A single census-like record containing demographic and employment attributes (age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours per week, native country).
- **Output:** One of two labels:
  - `<=50K`
  - `>50K`

- **Intended users:** Instructors and reviewers evaluating the project, or developers studying the MLOps template.
- **Out of scope:** This model is **not** intended for real hiring, lending, or policy decisions. It is a teaching artifact.

---

## Data

- **Source:** Processed version of the UCI Adult / Census Income dataset.
- **Location in repo:** `data/census.csv`
- **Target / label:** `salary` (`<=50K` or `>50K`)
- **Feature types:**
  - Categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
  - Numerical: `age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`

- **Preprocessing:**
  - Categorical features one-hot encoded using `process_data` (in `ml/data.py`)
  - Labels binarized using `LabelBinarizer`
  - Train/test split: 80% train, 20% test, `random_state=42`

---

## Model Performance

### Overall test set performance

On the held-out test set (20% split):

- **Precision:** `PRECISION_HERE`
- **Recall:** `RECALL_HERE`
- **F1 (Fbeta, β=1):** `FBETA_HERE`

These values are printed by `train_model.py` and reflect the model’s ability to distinguish between `<=50K` and `>50K` income levels.

### Performance on data slices

To check for inconsistent performance across demographic groups, the model was evaluated on slices of the test data by each categorical feature. The results are written to `slice_output.txt`.

High-level observations:

- The model performs **reasonably well** on larger, common groups (for example, common education levels and the most frequent race and native-country values).
- Some smaller subgroups have **very low F1 scores** (including certain rare education levels or countries), indicating that the model is unreliable for those slices due to limited data.
- A few subgroups show near-perfect metrics (precision and recall close to 1.0). This is likely due to **very small sample sizes** rather than true generalization ability.

These slice metrics highlight that performance is **not uniform** across all demographic categories and should be monitored carefully in any real deployment.

---

## Ethical Considerations

- This model uses sensitive attributes such as **race**, **sex**, and **native-country** as input features.
- The training data reflects historical patterns and may encode existing societal biases.
- The slice-based analysis shows that some demographic groups get noticeably better performance than others, which would be problematic in real-world decision-making contexts (e.g., hiring, credit scoring).
- Because of these factors, this model **must not** be used in production settings where it can impact people’s lives.

---

## Caveats and Recommendations

1. **Data bias & representation**
   - Some demographic groups are underrepresented in the dataset, leading to unstable and unreliable slice metrics.
   - Additional, more balanced data would be needed for fair deployment.

2. **Model complexity**
   - The current RandomForest model uses default or simple hyperparameters and has not undergone extensive tuning.
   - Future work could include:
     - Hyperparameter tuning (e.g., GridSearchCV / RandomizedSearchCV)
     - Trying alternative models such as gradient boosting or logistic regression
     - Testing different decision thresholds per group.

3. **Monitoring**
   - In a real system, slice performance should be tracked over time to detect drift or widening gaps between demographic groups.
   - Additional fairness metrics (e.g., demographic parity, equalized odds) would be needed for a serious application.

4. **Intended use reminder**
   - This model is designed for an educational project to demonstrate MLOps tooling.
   - It should not be used directly for policy, employment, or financial decisions without substantial additional work and governance.

