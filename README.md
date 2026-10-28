# Explainable AI-Based Student Performance Prediction and Early Warning System

A student-support prototype built with Python, Pandas, XGBoost, SHAP, and Streamlit.

## What it does
- Analyzes early-term student indicators.
- Predicts whether a student may be at risk of receiving a final grade below the chosen threshold.
- Uses SHAP to show which inputs influenced a prediction.
- Provides supportive follow-up suggestions for faculty.

## Important limitations
- This prototype is trained on the public UCI Student Performance dataset, not your college's live records.
- The dataset's `absences` field is a count of recorded absences, not an attendance percentage.
- The target is defined as final grade `G3 < 10` on the dataset's 0–20 grading scale.
- This is a teaching/research prototype, not a validated student-risk assessment system. Do not use it as the sole basis for academic decisions. Validate with local data, check subgroup performance and calibration, protect student privacy, and keep a human in the loop.

## Files
- `student_early_warning_colab.ipynb`: Google Colab notebook for downloading data, analysis, training, evaluation, and saving model files.
- `app.py`: Streamlit web app.
- `requirements.txt`: Python packages.
- `model.joblib` and `model_info.json`: created by running the notebook; copy them beside `app.py`.

## Run in Google Colab
1. Upload/open `student_early_warning_colab.ipynb` in Google Colab.
2. Run all cells in order.
3. The notebook creates `model.joblib` and `model_info.json` and downloads them.
4. Put both files in the same folder as `app.py`.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publish to GitHub
1. Create a new GitHub repository, e.g. `explainable-ai-student-early-warning`.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the notebook.
3. Do not upload real student personal data.
4. Upload the generated `model.joblib` and `model_info.json` only if your repository size and project rules allow it. If GitHub rejects a large model file, use Git LFS or download the artifacts separately.
5. Add screenshots of the dashboard to the README if desired.

## Deploy with Streamlit Community Cloud
1. Push the repository to GitHub.
2. Sign in to Streamlit Community Cloud and connect GitHub.
3. Select the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy. Ensure `model.joblib` and `model_info.json` are available in the deployed repository.
