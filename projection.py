import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import json
import plotly.graph_objs as go
import streamlit as st  # En üste ekleyin

BASE_DIR = "data/records"
MODEL_PATH = "random_forest_risk_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model dosyası bulunamadı. Önce modeli eğitmelisiniz!")
        return None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def load_patient_grade(pid):
    grade_path = os.path.join(BASE_DIR, pid, "grade.json")
    if os.path.exists(grade_path):
        try:
            with open(grade_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("grade"), data.get("danger_score")
        except:
            return None, None
    return None, None

def load_patient_features(pid):
    base = os.path.join(BASE_DIR, pid)
    try:
        mood_files = sorted(os.listdir(os.path.join(base, "mood_tracking")))
        mood_df = pd.read_csv(os.path.join(base, "mood_tracking", mood_files[-1]))
        mood_avg = mood_df["average"].values[0]
    except:
        mood_avg = 3.0

    try:
        func_files = sorted(os.listdir(os.path.join(base, "functioning_score")))
        func_df = pd.read_csv(os.path.join(base, "functioning_score", func_files[-1]))
        func_score = func_df["score"].values[0]
    except:
        func_score = 50.0

    try:
        health_path = os.path.join(base, "healthkit")
        health_files = sorted([f for f in os.listdir(health_path) if f.endswith(".csv")])
        dfs = [pd.read_csv(os.path.join(health_path, f)) for f in health_files]
        health_df = pd.concat(dfs, ignore_index=True)
        steps = int(health_df["steps"].mean())
        sleep = round(health_df["hours"].mean(), 2)
    except:
        steps, sleep = 5000, 6.0

    form_scores = {}
    for form in ["PHQ9", "GAD7", "PSS10", "PSQI", "IESR"]:
        try:
            form_path = os.path.join(base, "forms", form)
            form_files = sorted([f for f in os.listdir(form_path) if f.endswith(".json")])
            with open(os.path.join(form_path, form_files[-1]), encoding="utf-8") as f:
                data = json.load(f)
                form_scores[form] = data["score"]
        except:
            form_scores[form] = 0

    features = [
        mood_avg, steps, sleep,
        form_scores.get("PHQ9", 0),
        form_scores.get("GAD7", 0),
        form_scores.get("PSS10", 0),
        form_scores.get("PSQI", 0),
        form_scores.get("IESR", 0),
        func_score
    ]
    return features

def predict_risk_score(model, features):
    X = np.array(features).reshape(1, -1)
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    try:
        class_nums = np.array([int(str(c).replace("I", "1").replace("II", "2").replace("III", "3").replace("IV", "4").replace("V", "5")) for c in classes])
    except:
        class_nums = np.arange(1, len(classes)+1)
    risk_score = np.dot(proba, class_nums) * 20
    # Maksimum 100 ile sınırla🐇🐇🐇
    risk_score = min(max(risk_score, 0), 100)
    print("Başlangıç risk skoru:", risk_score)
    return risk_score

def simulate_intervention_curves(x0, day_count, grade=None):
    t = np.linspace(0, day_count - 1, day_count)
    if 90 <= x0 <= 100:
        base = 90
        no_intervention = base + 5 * np.sin(0.09 * t)
        no_intervention = np.clip(no_intervention, 90, 100)
    else:
        no_intervention = x0 + 0.04 * t + 1.0 * np.sin(0.09 * t)
        no_intervention = np.clip(no_intervention, 0, 100)

    # Kademeli iyileşme için katsayılar ve merkez noktaları 🐇🐇🐇🐇
    if grade == "III":
        k_therapy = 0.18
        k_pharma = 0.20
        k_combine = 0.25
        center = 0.35
    elif grade == "IV":
        k_therapy = 0.10
        k_pharma = 0.12
        k_combine = 0.16
        center = 0.55
    elif grade == "V":
        k_therapy = 0.06
        k_pharma = 0.08
        k_combine = 0.11
        center = 0.75
    else:
        k_therapy = 0.10
        k_pharma = 0.12
        k_combine = 0.16
        center = 0.55

    therapy = x0 - (x0 - 10) / (1 + np.exp(-k_therapy * (t - center * day_count))) + 2.0 * np.cos(0.05 * t)
    pharma  = x0 - (x0 - 10) / (1 + np.exp(-k_pharma  * (t - (center+0.05) * day_count))) + 1.5 * np.sin(0.05 * t)
    combine = x0 - (x0 - 10) / (1 + np.exp(-k_combine * (t - (center-0.1) * day_count))) + 1.0 * np.cos(0.04 * t)

    therapy = np.clip(therapy, 0, 100)
    pharma = np.clip(pharma, 0, 100)
    combine = np.clip(combine, 0, 100)
    return t, no_intervention, therapy, pharma, combine


def run_simulation_for_patient(pid, day_count=120):
    model = load_model()
    if model is None:
        return "Model bulunamadı."

    grade, _ = load_patient_grade(pid)
    if grade is None:
        return f"[{pid}] için Grade bilgisi bulunamadı."

    if grade in ["I", "II"]:
        msg = f"[{pid}] Grade {grade} olduğundan, risk projeksiyonu yapılmaz."
        return msg

    # Grade'a göre başlangıç risk skoru sabitleniyor🐇🐇🐇🐇
    grade_map = {"III": 50, "IV": 70, "V": 90}
    x0 = grade_map.get(grade, 50)

    t, f1, f2, f3, f4 = simulate_intervention_curves(x0, day_count, grade=grade)
    f1[0] = f2[0] = f3[0] = f4[0] = x0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=f1, mode='lines+markers', name="Müdahale Yok", line=dict(dash='solid')))
    fig.add_trace(go.Scatter(x=t, y=f2, mode='lines+markers', name="Terapötik Müdahale (CBT)", line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=t, y=f3, mode='lines+markers', name="Farmakolojik Müdahale (SSRI)", line=dict(dash='dashdot')))
    fig.add_trace(go.Scatter(x=t, y=f4, mode='lines+markers', name="Kombine Müdahale", line=dict(dash='dot')))

    for y, dash, label in [(20, 'dot', "Grade I/II Sınırı (20)"),
                           (40, 'dash', "Grade II/III Sınırı (40)"),
                           (60, 'dashdot', "Grade III/IV Sınırı (60)"),
                           (80, 'dot', "Grade IV/V Sınırı (80)")]:
        fig.add_hline(y=y, line_dash=dash, line_color="gray", annotation_text=label, annotation_position="top left")

    fig.add_annotation(x=t[0], y=f1[0], text=f"{f1[0]:.1f}", showarrow=True, arrowhead=1, yshift=-20, font=dict(color="blue"))
    fig.add_annotation(x=t[0], y=f2[0], text=f"{f2[0]:.1f}", showarrow=True, arrowhead=1, yshift=-35, font=dict(color="orange"))
    fig.add_annotation(x=t[0], y=f3[0], text=f"{f3[0]:.1f}", showarrow=True, arrowhead=1, yshift=-50, font=dict(color="green"))
    fig.add_annotation(x=t[0], y=f4[0], text=f"{f4[0]:.1f}", showarrow=True, arrowhead=1, yshift=-65, font=dict(color="red"))

    fig.update_layout(
        title=f"{pid} için Risk ve Müdahale Eğrileri Simülasyonu",
        xaxis_title="Gün",
        yaxis_title="Risk Skoru",
        yaxis=dict(range=[min(0, x0-10), 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )

    return fig  # Sadece figürü döndür 🐇🐇