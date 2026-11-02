# Complete Execution Steps — ML Assignment 2

Follow these steps in order. Items marked **[BITS Lab]** must be done on the BITS Virtual Lab so you can capture the required screenshot.

---

## A. One-time setup on your PC (already prepared in this folder)

Project folder:

`C:\Users\mashok\Projects\dry-bean-classifier`

Contents already generated:

- `app.py` — Streamlit UI
- `train_models.py` — trains all 5 models
- `download_dataset.py` — downloads UCI Dry Bean data
- `Assignment2_Notebook.ipynb` — notebook for BITS Lab screenshot
- `requirements.txt`
- `README.md` (copy this into your submission PDF)
- `test_data.csv` — held-out test set
- `model/*.joblib` — five trained classifiers + label encoder + feature list

If you need to regenerate everything locally:

```powershell
cd C:\Users\mashok\Projects\dry-bean-classifier
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python download_dataset.py
python train_models.py
streamlit run app.py
```

---

## B. Create a public GitHub repository (required for Streamlit Cloud)

1. Create a GitHub account (if you do not have one): https://github.com/signup
2. Open https://github.com/new
3. Repository name: `dry-bean-classifier` (or any unique name)
4. Visibility: **Public** (Streamlit Community Cloud needs a public repo on the free tier)
5. Do **not** add a README / .gitignore on GitHub (this project already has them)
6. Click **Create repository**
7. On your PC, run (replace `YOUR_GITHUB_USERNAME`):

```powershell
cd C:\Users\mashok\Projects\dry-bean-classifier
git init -b main
git add app.py train_models.py download_dataset.py Assignment2_Notebook.ipynb requirements.txt README.md test_data.csv model .gitignore EXECUTION_STEPS.md artifacts/metrics.json artifacts/classification_reports.json
git commit -m "Add Dry Bean classification models and Streamlit app for ML Assignment 2"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/dry-bean-classifier.git
git push -u origin main
```

8. Copy your repo URL, for example:

`https://github.com/YOUR_GITHUB_USERNAME/dry-bean-classifier`

9. Paste that URL into **README.md section c** and into your submission PDF.

---

## C. Deploy on Streamlit Community Cloud

1. Go to https://streamlit.io/cloud
2. Sign in with the **same GitHub account**
3. Click **New app**
4. Select repository: `dry-bean-classifier`
5. Branch: `main`
6. Main file path: `app.py`
7. Click **Deploy**
8. Wait until the app is live (usually 1–3 minutes)
9. Copy the live URL, for example:

`https://dry-bean-classifier-xxxxx.streamlit.app`

10. Paste it into README section c and into the submission PDF.

### How to use the live app (for your own check + screenshots if needed)

1. Click **Browse files** / upload
2. Upload `test_data.csv` from the repo
3. Choose each model from the dropdown
4. Confirm metrics + confusion matrix / classification report appear

Required app features covered:

- Dataset upload (CSV test data)
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix + classification report
- No retraining on upload (loads saved `.joblib` models)

---

## D. BITS Virtual Lab execution + screenshot **[1 mark]**

1. Log in to **BITS Virtual Lab**
2. Upload / clone this project folder (or at least the notebook + requirements)
3. Open a terminal in the project folder and install deps:

```bash
pip install -r requirements.txt
```

4. Open `Assignment2_Notebook.ipynb` in Jupyter / VS Code on the lab
5. Run all cells (**Kernel → Restart & Run All**)
6. When the **comparison table** cell finishes, take **ONE** clear screenshot showing:
   - BITS Virtual Lab environment visible (browser/desktop chrome)
   - Notebook with metrics / comparison table output
7. Keep that screenshot for the submission PDF (section order item 3)

If the Virtual Lab has issues, email:

- `csislabsupport@wilp.bits-pilani.ac.in`
- `raj.rohan@wilp.bits-pilani.ac.in`

Subject: `NSP1 ML Assignment 2: BITS Lab issue`

---

## E. Build the single submission PDF (you create this)

Maintain this **exact order**:

1. **GitHub Repository Link** (clickable)
2. **Live Streamlit App Link** (clickable; must open interactive UI)
3. **Screenshot** of assignment execution on BITS Virtual Lab
4. **Full README.md content** (sections a–d, including both tables)

Suggested PDF title page line:

`M.Tech (AIML) — Machine Learning — Assignment 2`

Deadline: **08-Nov-2026, 23:59** — submit (not draft).

---

## F. Final checklist before SUBMIT

- [ ] GitHub repo link works and is public
- [ ] Repo contains: `app.py`, `requirements.txt`, `README.md`, `test_data.csv`, `model/` with 5 classifiers
- [ ] Streamlit app link opens without errors
- [ ] Upload test CSV works; model dropdown works; metrics + CM/report visible
- [ ] BITS Lab screenshot included
- [ ] README content copied into the PDF
- [ ] Click **SUBMIT** (drafts are not accepted)
