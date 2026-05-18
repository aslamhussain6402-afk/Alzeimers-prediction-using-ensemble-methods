"""
Alzheimer Detection Web App — Flask Backend
Run: python app.py  →  http://localhost:5000
"""

import os, pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score,
                              classification_report, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

app = Flask(__name__)

MODEL_DIR  = "model_artifacts"
CSV_PATH   = "alzheimers_disease_data.csv"
DROP_COLS  = ["PatientID", "DoctorInCharge"]
TARGET_COL = "Diagnosis"

os.makedirs(MODEL_DIR, exist_ok=True)

pipeline     = {}
metrics      = {}
feature_info = {}

# ── Human-readable field labels ───────────────────────────
FIELD_LABELS = {
    "Age":"Age","Gender":"Gender","Ethnicity":"Ethnicity",
    "EducationLevel":"Education Level","BMI":"BMI","Smoking":"Smoking",
    "AlcoholConsumption":"Alcohol Consumption (units/week)",
    "PhysicalActivity":"Physical Activity (hrs/week)",
    "DietQuality":"Diet Quality Score","SleepQuality":"Sleep Quality Score",
    "FamilyHistoryAlzheimers":"Family History of Alzheimer's",
    "CardiovascularDisease":"Cardiovascular Disease","Diabetes":"Diabetes",
    "Depression":"Depression","HeadInjury":"Head Injury","Hypertension":"Hypertension",
    "SystolicBP":"Systolic BP (mmHg)","DiastolicBP":"Diastolic BP (mmHg)",
    "CholesterolTotal":"Total Cholesterol (mg/dL)","CholesterolLDL":"LDL Cholesterol (mg/dL)",
    "CholesterolHDL":"HDL Cholesterol (mg/dL)",
    "CholesterolTriglycerides":"Triglycerides (mg/dL)",
    "MMSE":"MMSE Score (0-30)","FunctionalAssessment":"Functional Assessment (0-10)",
    "MemoryComplaints":"Memory Complaints","BehavioralProblems":"Behavioral Problems",
    "ADL":"ADL Score (0-10)","Confusion":"Confusion","Disorientation":"Disorientation",
    "PersonalityChanges":"Personality Changes",
    "DifficultyCompletingTasks":"Difficulty Completing Tasks","Forgetfulness":"Forgetfulness",
}

BINARY_OPTIONS = {
    "Gender":                   [{"value":0,"label":"Male"},{"value":1,"label":"Female"}],
    "Smoking":                  [{"value":0,"label":"No"},{"value":1,"label":"Yes"}],
    "FamilyHistoryAlzheimers":  [{"value":0,"label":"No"},{"value":1,"label":"Yes"}],
    "CardiovascularDisease":    [{"value":0,"label":"No"},{"value":1,"label":"Yes"}],
    "Diabetes":                 [{"value":0,"label":"No"},{"value":1,"label":"Yes"}],
    "Depression":               [{"value":0,"label":"No"},{"value":1,"label":"Yes"}],
    "HeadInjury":               [{"value":0,"label":"No"},{"value":1,"label":"Yes"}],
    "Hypertension":             [{"value":0,"label":"No"},{"value":1,"label":"Yes"}],
    "MemoryComplaints":         [{"value":0,"label":"No"},{"value":1,"label":"Yes"}],
    "BehavioralProblems":       [{"value":0,"label":"No"},{"value":1,"label":"Yes"}],
    "Confusion":                [{"value":0,"label":"No"},{"value":1,"label":"Yes"}],
    "Disorientation":           [{"value":0,"label":"No"},{"value":1,"label":"Yes"}],
    "PersonalityChanges":       [{"value":0,"label":"No"},{"value":1,"label":"Yes"}],
    "DifficultyCompletingTasks":[{"value":0,"label":"No"},{"value":1,"label":"Yes"}],
    "Forgetfulness":            [{"value":0,"label":"No"},{"value":1,"label":"Yes"}],
}

CAT_OPTIONS = {
    "Ethnicity":     [{"value":0,"label":"Caucasian"},{"value":1,"label":"African American"},
                      {"value":2,"label":"Asian"},{"value":3,"label":"Other"}],
    "EducationLevel":[{"value":0,"label":"None"},{"value":1,"label":"High School"},
                      {"value":2,"label":"Bachelor's"},{"value":3,"label":"Higher"}],
}


def build_feature_info(df):
    X = df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    info = {}
    for col in X.columns:
        mn  = float(X[col].min())
        mx  = float(X[col].max())
        uni = sorted(X[col].dropna().unique().tolist())
        is_bin = set(uni) <= {0, 1}
        is_cat = len(uni) <= 8 and all(float(v).is_integer() for v in uni)
        entry = {
            "name":           col,
            "label":          FIELD_LABELS.get(col, col),
            "min":            round(mn, 2),
            "max":            round(mx, 2),
            "is_binary":      is_bin,
            "is_categorical": is_cat and not is_bin,
            "options":        None,
            "hint":           f"{round(mn,2)} – {round(mx,2)}",
            "step":           1 if str(X[col].dtype) == "int64" else "any",
        }
        if is_bin:
            entry["options"] = BINARY_OPTIONS.get(col, [{"value":0,"label":"No"},{"value":1,"label":"Yes"}])
        elif is_cat:
            entry["options"] = CAT_OPTIONS.get(col, [{"value":int(v),"label":str(int(v))} for v in uni])
        info[col] = entry
    return info


# ── Terminal metrics printer ───────────────────────────────
def print_metrics(m, title="Model Performance"):
    W = 56
    print(f"\n{'═'*W}")
    print(f"  {title}")
    print(f"{'═'*W}")
    rows = [("Accuracy",m["accuracy"]),("Precision",m["precision"]),
            ("Recall",m["recall"]),("F1 Score",m["f1"]),("AUC-ROC",m["auc"])]
    print(f"  {'Metric':<18} {'Score':>8}   {'Progress Bar'}")
    print("  " + "─"*52)
    for name, val in rows:
        bar = "█"*int(val*32) + "░"*(32-int(val*32))
        print(f"  {name:<18} {val*100:>7.2f}%   {bar}")
    print(f"{'═'*W}")

    if "base_accuracy" in m:
        print(f"\n  Base Model Comparison:")
        print("  " + "─"*52)
        for name, acc in m["base_accuracy"].items():
            bar = "█"*int(acc*32) + "░"*(32-int(acc*32))
            print(f"  {name:<18} {acc*100:>7.2f}%   {bar}")
        bar = "█"*int(m["accuracy"]*32) + "░"*(32-int(m["accuracy"]*32))
        print(f"  {'Stacked Ensemble':<18} {m['accuracy']*100:>7.2f}%   {bar}  ← FINAL")

    if "report" in m:
        print(f"\n  Classification Report:\n  {'─'*52}")
        for line in m["report"].splitlines():
            print("  " + line)

    if "confusion" in m:
        classes = m.get("classes", [])
        print(f"\n  Confusion Matrix (rows=actual, cols=predicted):\n  {'─'*52}")
        hdr = "         " + "  ".join(f"{c:>6}" for c in classes)
        print("  " + hdr)
        for i, row in enumerate(m["confusion"]):
            lbl = f"{classes[i]:>6}" if i < len(classes) else f"{i:>6}"
            print("  " + lbl + "   " + "  ".join(f"{v:>6}" for v in row))
    print()


# ── Training pipeline ──────────────────────────────────────
def train_and_save(csv_path=CSV_PATH):
    global pipeline, metrics, feature_info

    print(f"\n{'═'*56}")
    print("  NeuroScan — Hybrid Ensemble · Alzheimer Detection")
    print(f"{'═'*56}")
    print(f"  Loading dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    feature_info = build_feature_info(df)
    print(f"  Dataset  : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Target   : {df[TARGET_COL].value_counts().to_dict()}")

    X = df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    y = df[TARGET_COL]

    le_target   = LabelEncoder()
    y_enc       = le_target.fit_transform(y)
    num_classes = len(le_target.classes_)
    print(f"  Classes  : {list(le_target.classes_)}")

    feature_names_raw = list(X.columns)
    numeric_cols      = X.select_dtypes(include=["int64","float64"]).columns
    categorical_cols  = X.select_dtypes(include=["object"]).columns

    num_imputer = SimpleImputer(strategy="median")
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
    cat_imputer = SimpleImputer(strategy="most_frequent")
    if len(categorical_cols):
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder(); X[col] = le.fit_transform(X[col].astype(str)); le_dict[col] = le

    vt   = VarianceThreshold(threshold=0.01); X_vt = vt.fit_transform(X)
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_vt)

    print("\n  [1/5] RFE Feature Selection…")
    rf_tmp   = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    n_sel    = min(20, X_scaled.shape[1])
    rfe      = RFE(rf_tmp, n_features_to_select=n_sel)
    X_rfe    = rfe.fit_transform(X_scaled, y_enc)
    vt_mask  = vt.get_support(); rfe_mask = rfe.get_support()
    after_vt = [feature_names_raw[i] for i,m in enumerate(vt_mask) if m]
    sel_names= [after_vt[i] for i,m in enumerate(rfe_mask) if m]
    print(f"  Selected {len(sel_names)} features: {sel_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_rfe, y_enc, test_size=0.30, random_state=42, stratify=y_enc)
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
    print(f"  Train (post-SMOTE): {X_train.shape}  |  Test: {X_test.shape}")

    is_bin   = (num_classes == 2)
    xgb_obj  = "binary:logistic" if is_bin else "multi:softprob"
    lgb_obj  = "binary"           if is_bin else "multiclass"
    dnn_out  = 1                   if is_bin else num_classes
    dnn_act  = "sigmoid"           if is_bin else "softmax"
    dnn_loss = "binary_crossentropy" if is_bin else "sparse_categorical_crossentropy"

    print("\n  [2/5] Training XGBoost…")
    xgb_m = XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=6,
        subsample=0.8,colsample_bytree=0.8,objective=xgb_obj,
        num_class=num_classes if not is_bin else None,
        eval_metric="mlogloss" if not is_bin else "logloss",
        verbosity=0,use_label_encoder=False)
    xgb_m.fit(X_train,y_train)
    xgb_acc = accuracy_score(y_test, xgb_m.predict(X_test))
    print(f"  XGBoost accuracy      : {xgb_acc*100:.2f}%")

    print("\n  [3/5] Training Random Forest…")
    rf_m = RandomForestClassifier(n_estimators=400,random_state=42,n_jobs=-1)
    rf_m.fit(X_train,y_train)
    rf_acc = accuracy_score(y_test, rf_m.predict(X_test))
    print(f"  Random Forest accuracy: {rf_acc*100:.2f}%")

    print("\n  [4/5] Training LightGBM…")
    lgb_m = lgb.LGBMClassifier(n_estimators=500,learning_rate=0.05,max_depth=7,
        objective=lgb_obj,num_class=num_classes if not is_bin else None,verbose=-1)
    lgb_m.fit(X_train,y_train)
    lgb_acc = accuracy_score(y_test, lgb_m.predict(X_test))
    print(f"  LightGBM accuracy     : {lgb_acc*100:.2f}%")

    print("\n  [5/5] Training Deep Neural Network…")
    dnn_m = Sequential([
        Dense(256,activation="relu",input_shape=(X_train.shape[1],)),Dropout(0.4),
        Dense(128,activation="relu"),Dropout(0.3),
        Dense(64,activation="relu"),Dropout(0.2),
        Dense(dnn_out,activation=dnn_act)])
    dnn_m.compile(optimizer=tf.keras.optimizers.Adam(0.0005),loss=dnn_loss,metrics=["accuracy"])
    dnn_m.fit(X_train,y_train,epochs=100,batch_size=16,validation_split=0.2,
        callbacks=[EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)],verbose=0)
    dnn_pred = (dnn_m.predict(X_test,verbose=0)>0.5).astype(int).flatten() if is_bin \
               else np.argmax(dnn_m.predict(X_test,verbose=0),axis=1)
    dnn_acc  = accuracy_score(y_test, dnn_pred)
    print(f"  DNN accuracy          : {dnn_acc*100:.2f}%")

    print("\n  Training Stacking Meta-Model…")
    def gp(mdl,X,dnn=False):
        if dnn:
            p=mdl.predict(X,verbose=0); return p.reshape(-1,1) if is_bin else p
        return mdl.predict_proba(X)
    mX_tr = np.hstack([gp(xgb_m,X_train),gp(rf_m,X_train),gp(lgb_m,X_train),gp(dnn_m,X_train,dnn=True)])
    meta_m = LogisticRegression(max_iter=1000); meta_m.fit(mX_tr,y_train)

    mX_te = np.hstack([gp(xgb_m,X_test),gp(rf_m,X_test),gp(lgb_m,X_test),gp(dnn_m,X_test,dnn=True)])
    fp    = meta_m.predict_proba(mX_te)
    fpred = np.argmax(fp,axis=1)
    avg   = "binary" if is_bin else "macro"
    tnames= [str(c) for c in le_target.classes_]

    metrics.update({
        "accuracy" : round(float(accuracy_score(y_test,fpred)),4),
        "precision": round(float(precision_score(y_test,fpred,average=avg,zero_division=0)),4),
        "recall"   : round(float(recall_score(y_test,fpred,average=avg,zero_division=0)),4),
        "f1"       : round(float(f1_score(y_test,fpred,average=avg,zero_division=0)),4),
        "auc"      : round(float(roc_auc_score(y_test,
                        fp if not is_bin else fp[:,1],
                        multi_class="ovr" if not is_bin else "raise",
                        average="macro" if not is_bin else None)),4),
        "classes"  : tnames,
        "report"   : classification_report(y_test,fpred,target_names=tnames),
        "confusion" : confusion_matrix(y_test,fpred).tolist(),
        "base_accuracy": {"XGBoost":round(xgb_acc,4),"RandomForest":round(rf_acc,4),
                          "LightGBM":round(lgb_acc,4),"DNN":round(dnn_acc,4)},
    })

    print_metrics(metrics, "Hybrid Ensemble — Final Test Performance")

    pipeline.update({
        "num_imputer":num_imputer,"cat_imputer":cat_imputer,"le_dict":le_dict,
        "vt":vt,"scaler":scaler,"rfe":rfe,"xgb":xgb_m,"rf":rf_m,"lgb":lgb_m,
        "meta":meta_m,"le_target":le_target,
        "numeric_cols":list(numeric_cols),"categorical_cols":list(categorical_cols),
        "feature_names_raw":feature_names_raw,"selected_features":sel_names,
        "is_binary":is_bin,"num_classes":num_classes,
    })
    with open(f"{MODEL_DIR}/pipeline.pkl","wb")     as f: pickle.dump(pipeline,f)
    with open(f"{MODEL_DIR}/metrics.pkl","wb")      as f: pickle.dump(metrics,f)
    with open(f"{MODEL_DIR}/feature_info.pkl","wb") as f: pickle.dump(feature_info,f)
    dnn_m.save(f"{MODEL_DIR}/dnn_model.keras")
    print("  Artifacts saved to ./model_artifacts/\n")


def load_artifacts():
    global pipeline, metrics, feature_info
    with open(f"{MODEL_DIR}/pipeline.pkl","rb")     as f: pipeline     = pickle.load(f)
    with open(f"{MODEL_DIR}/metrics.pkl","rb")      as f: metrics      = pickle.load(f)
    with open(f"{MODEL_DIR}/feature_info.pkl","rb") as f: feature_info = pickle.load(f)
    pipeline["dnn"] = tf.keras.models.load_model(f"{MODEL_DIR}/dnn_model.keras")
    print("  Artifacts loaded.\n")
    print_metrics(metrics, "Loaded Model — Performance Summary")


# ── Prediction ─────────────────────────────────────────────
def predict_single(input_dict):
    p  = pipeline
    df = pd.DataFrame([input_dict])
    for col in p["feature_names_raw"]:
        if col not in df.columns: df[col] = 0
    df = df[p["feature_names_raw"]]
    df[p["numeric_cols"]]     = p["num_imputer"].transform(df[p["numeric_cols"]])
    if p["categorical_cols"]:
        df[p["categorical_cols"]] = p["cat_imputer"].transform(df[p["categorical_cols"]])
    for col, le in p["le_dict"].items():
        df[col] = le.transform(df[col].astype(str))
    X = p["vt"].transform(df)
    X = p["scaler"].transform(X)
    X = p["rfe"].transform(X)

    def gp(mdl, is_dnn=False):
        if is_dnn:
            r = mdl.predict(X,verbose=0)
            return r.reshape(1,-1) if p["is_binary"] else r
        return mdl.predict_proba(X)
    mX   = np.hstack([gp(p["xgb"]),gp(p["rf"]),gp(p["lgb"]),gp(p["dnn"],is_dnn=True)])
    prob = p["meta"].predict_proba(mX)[0]
    idx  = int(np.argmax(prob))
    lbl  = p["le_target"].classes_[idx]
    probs= {str(p["le_target"].classes_[i]): round(float(prob[i])*100,2) for i in range(len(prob))}
    return str(lbl), probs


# ── Flask routes ───────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/metrics")
def api_metrics():
    safe = {k:v for k,v in metrics.items() if k not in ("report","confusion")}
    return jsonify(safe)

@app.route("/api/features")
def api_features():
    ordered  = [feature_info[k] for k in pipeline.get("feature_names_raw",[]) if k in feature_info]
    selected = pipeline.get("selected_features",[])
    return jsonify({"features": ordered, "selected": selected})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    try:
        lbl, probs = predict_single(data)
        return jsonify({"prediction": lbl, "probabilities": probs, "status":"ok"})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 400


if __name__ == "__main__":
    ok = all(os.path.exists(f"{MODEL_DIR}/{f}")
             for f in ["pipeline.pkl","metrics.pkl","feature_info.pkl","dnn_model.keras"])
    if ok:
        print("\n[INFO] Loading pre-trained model…")
        load_artifacts()
    else:
        print("\n[INFO] No saved model — training now…")
        train_and_save(CSV_PATH)
        load_artifacts()
    print("  Open browser → http://localhost:5000\n")
    app.run(debug=True, port=5000)