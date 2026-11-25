# ISL Recognizer (Indian Sign Language) – Frontend + Backend

Webcam ya photo se ISL alphabets/numbers pehchaanne wala web app.

- **Frontend:** HTML/CSS/JS (webcam, hand skeleton overlay, confidence bar, text builder, speak button)
- **Backend:** Flask + MediaPipe (hand landmarks) + scikit-learn (KNN / Random Forest / SVM)

## Folder structure
```
isl_app/
  app.py            Flask backend (/, /predict, /health)
  train.py          dataset -> landmarks -> models -> reports
  features.py       MediaPipe landmark extraction (shared)
  templates/        index.html
  static/           style.css, app.js
  models/           isl_model.joblib (train ke baad), hand_landmarker.task (auto-download)
  reports/          charts + tables (train ke baad)
  data/             landmarks.csv (train ke baad)
```

## Setup (Python 3.9 – 3.12 recommended)
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Step 1: Dataset
Dataset folder aisa hona chahiye (har sign ka ek folder):
```
dataset/
  A/  img1.jpg img2.jpg ...
  B/  ...
  1/  ...
```
Karthikeyu repo ka Google Drive dataset extract karke yahi structure check kar lo
(agar extra folder level hai to andar wale folder ka path do).

## Step 2: Train
```bash
python train.py --data path/to/dataset --flip
```
- Pehli baar MediaPipe model (~8 MB) internet se auto-download hota hai. Agar fail ho to
  `features.py` mein diye link se `hand_landmarker.task` download karke `models/` mein rakho.
- 42,000 images mein extraction ~15–40 minute le sakta hai. **Pehle quick test:**
  `python train.py --data path/to/dataset --max-per-class 100`
- Landmarks `data/landmarks.csv` mein save ho jaate hain, dobara train karna fast hai.
  Naya dataset ho to `--rebuild` lagao.
- `--tune` se SVM ka GridSearch (improvement step ke liye).

Output: `models/isl_model.joblib` + `reports/` (class distribution, sample images,
detection rate, accuracy comparison, confusion matrix, classification report, results table).

## Step 3: App chalao
```bash
python app.py
```
Browser mein **http://127.0.0.1:5000** kholo (localhost pe hi camera allow hota hai).

## Case-study 4 steps ke saath mapping
| Step | Kahan |
|---|---|
| Problem statement | Communication gap kam karna: ISL fingerspelling ko text mein badalna |
| Data pre-processing | `train.py`: landmarks extract, undetected images hatana, normalize, stratified split |
| Model building | KNN, Random Forest, SVM compare (`reports/results_table.csv`) |
| Evaluation + improvement | confusion matrix, most-confused signs, `--flip`, `--tune`, naye data pe test |

## Accuracy ke tips
1. Report mein **do numbers** do: dataset ka test accuracy aur apne webcam se li hui nayi images ka accuracy.
2. Agar app mein predictions ulte/galat aayein, "Mirror image before predicting" tick karke dekho
   (dataset mirrored ho sakta hai). Train mein `--flip` lagane se dono chal jaate hain.
3. Confusion matrix mein jo signs aapas mein confuse hon (jaise M/N), unki zyada images lo.
4. Do haath wale signs ke liye dono haath frame mein poore dikhne chahiye.
