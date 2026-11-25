"""Train the ISL recognition model.

Usage:
    python train.py --data path/to/dataset            # folder with one sub-folder per sign (A, B, ... 1, 2 ...)
    python train.py --data path/to/dataset --flip     # also learn from mirrored images (more robust)
    python train.py --data path/to/dataset --max-per-class 300   # quick test run
    python train.py --csv data/landmarks.csv          # reuse already extracted landmarks

Steps (same as the case-study format):
    1. Problem + data understanding (EDA charts)
    2. Pre-processing (MediaPipe landmarks -> Pandas DataFrame -> stratified split)
    3. Model building (KNN, Random Forest, SVM)
    4. Evaluation + improvement (accuracy table, confusion matrix, optional tuning)
Everything is saved in reports/ and models/.
"""
import argparse
import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import GridSearchCV, train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from features import FEATURE_NAMES  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(BASE, "reports")
MODELS = os.path.join(BASE, "models")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(data_dir, max_per_class=None):
    rows = []
    classes = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    if not classes:
        raise SystemExit(f"No class folders found inside {data_dir}. Expected data/A, data/B, ...")
    for c in classes:
        files = sorted(
            f for f in os.listdir(os.path.join(data_dir, c)) if f.lower().endswith(IMG_EXT)
        )
        if max_per_class:
            files = files[:max_per_class]
        rows += [(os.path.join(data_dir, c, f), c) for f in files]
    return pd.DataFrame(rows, columns=["path", "label"])


def eda_charts(files_df):
    counts = files_df["label"].value_counts().sort_index()
    plt.figure(figsize=(11, 4))
    counts.plot(kind="bar", color="#2a7f8f")
    plt.title("Images per sign (class distribution)")
    plt.xlabel("Sign")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, "class_distribution.png"), dpi=150)
    plt.close()

    import cv2

    firsts = files_df.groupby("label").first().reset_index().head(36)
    cols = 9
    rows = int(np.ceil(len(firsts) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.7))
    for ax in np.array(axes).reshape(-1):
        ax.axis("off")
    for ax, (_, r) in zip(np.array(axes).reshape(-1), firsts.iterrows()):
        img = cv2.imread(r["path"])
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(r["label"], fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, "sample_images.png"), dpi=130)
    plt.close()


def extract_dataset(files_df, flip):
    import cv2

    from features import HandExtractor

    ex = HandExtractor()
    feats, meta = [], []
    total = len(files_df)
    t0 = time.time()
    for i, (path, label) in enumerate(zip(files_df["path"], files_df["label"]), 1):
        img = cv2.imread(path)
        if img is None:
            continue
        variants = [(img, 0)] + ([(cv2.flip(img, 1), 1)] if flip else [])
        for im, aug in variants:
            f, _ = ex.features(im)
            if f is not None:
                feats.append(f)
                meta.append((label, path, aug))
            elif aug == 0:
                meta.append((label, path, -1))  # hand not detected (kept only for stats)
                feats.append(np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32))
        if i % 500 == 0:
            print(f"  {i}/{total} images  ({time.time() - t0:.0f}s)")
    ex.close()
    df = pd.DataFrame(feats, columns=FEATURE_NAMES)
    df["label"] = [m[0] for m in meta]
    df["path"] = [m[1] for m in meta]
    df["aug"] = [m[2] for m in meta]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="dataset folder (one sub-folder per sign)")
    ap.add_argument("--csv", help="use an existing landmarks CSV instead of reading images")
    ap.add_argument("--flip", action="store_true", help="add mirrored copies to the training set")
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--tune", action="store_true", help="GridSearchCV on SVM (slower)")
    ap.add_argument("--rebuild", action="store_true", help="ignore cached CSV")
    args = ap.parse_args()

    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    cache = args.csv or os.path.join(BASE, "data", "landmarks.csv")

    # ---------- Step 1 and 2: data + pre-processing ----------
    if os.path.exists(cache) and not args.rebuild:
        print(f"Loading cached landmarks: {cache}  (use --rebuild to extract again)")
        df = pd.read_csv(cache)
    else:
        if not args.data:
            raise SystemExit("Give --data <dataset folder> (or --csv).")
        files_df = list_images(args.data, args.max_per_class)
        print(f"Found {len(files_df)} images in {files_df['label'].nunique()} classes")
        eda_charts(files_df)
        print("Extracting hand landmarks with MediaPipe (this takes a while)...")
        df = extract_dataset(files_df, args.flip)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_csv(cache, index=False)
        print(f"Saved landmarks table: {cache}")

    orig = df[df["aug"] != 1]
    det = orig.assign(detected=orig["aug"] != -1).groupby("label")["detected"].mean().sort_index()
    print(f"\nHand detection rate: {det.mean() * 100:.1f}% overall")
    plt.figure(figsize=(11, 4))
    (det * 100).plot(kind="bar", color="#c46b2f")
    plt.title("Hand detection rate per sign (%)")
    plt.ylabel("%")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, "detection_rate.png"), dpi=150)
    plt.close()

    df = df[df["aug"] != -1].dropna(subset=FEATURE_NAMES)
    base = df[df["aug"] == 0]
    aug = df[df["aug"] == 1]
    train_base, test = train_test_split(
        base, test_size=args.test_size, stratify=base["label"], random_state=42
    )
    train = pd.concat([train_base, aug[aug["path"].isin(set(train_base["path"]))]])
    X_train, y_train = train[FEATURE_NAMES].values, train["label"].values
    X_test, y_test = test[FEATURE_NAMES].values, test["label"].values
    print(f"Train: {len(train)}   Test: {len(test)}   Classes: {base['label'].nunique()}")

    # ---------- Step 3: model building ----------
    models = {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
        "SVM": make_pipeline(
            StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=42)
        ),
    }
    if args.tune:
        grid = GridSearchCV(
            make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
            {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.01, 0.001]},
            cv=3, n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        print("Best SVM params:", grid.best_params_)
        models["SVM (tuned)"] = grid.best_estimator_

    results, fitted = [], {}
    for name, m in models.items():
        t0 = time.time()
        m.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        results.append({"model": name, "test_accuracy": round(acc * 100, 2),
                        "train_time_s": round(time.time() - t0, 1)})
        fitted[name] = m
        print(f"  {name:14s} accuracy = {acc * 100:.2f}%")

    # ---------- Step 4: evaluation ----------
    res = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
    res.to_csv(os.path.join(REPORTS, "results_table.csv"), index=False)
    plt.figure(figsize=(6, 3.5))
    plt.bar(res["model"], res["test_accuracy"], color="#2a7f8f")
    plt.ylim(0, 100)
    plt.ylabel("Test accuracy (%)")
    plt.title("Model comparison")
    for i, v in enumerate(res["test_accuracy"]):
        plt.text(i, v + 1, f"{v:.1f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, "accuracy_comparison.png"), dpi=150)
    plt.close()

    best_name = res.iloc[0]["model"]
    best = fitted[best_name]
    pred = best.predict(X_test)
    labels = sorted(np.unique(y_test))
    cm = confusion_matrix(y_test, pred, labels=labels)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=len(labels) <= 20, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, "confusion_matrix.png"), dpi=150)
    plt.close()

    report = classification_report(y_test, pred, zero_division=0)
    with open(os.path.join(REPORTS, "classification_report.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n\n{report}")

    # most confused pairs (useful for the 'improvement' discussion)
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    pairs = sorted(
        ((cm_off[i, j], labels[i], labels[j]) for i in range(len(labels))
         for j in range(len(labels)) if cm_off[i, j] > 0), reverse=True)[:5]
    if pairs:
        print("\nMost confused signs (actual -> predicted):")
        for n, a, p in pairs:
            print(f"  {a} -> {p}: {n} times")

    joblib.dump({"model": best, "classes": list(best.classes_), "name": best_name,
                 "accuracy": float(res.iloc[0]["test_accuracy"])},
                os.path.join(MODELS, "isl_model.joblib"))
    print(f"\nBest: {best_name} ({res.iloc[0]['test_accuracy']}%). Saved models/isl_model.joblib")
    print("Charts and reports are in the reports/ folder.")


if __name__ == "__main__":
    main()
