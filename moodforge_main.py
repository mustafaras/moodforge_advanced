from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from templates.journal_templates import get_journal
from templates.audio_templates import get_audio
from templates.video_templates import get_video_emotion_scores
from projection import run_simulation_for_patient
import shap
import pickle
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import os
import json
import shutil
import random
import pandas as pd
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
import openai
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
import plotly.express as px
from pathlib import Path
import glob
import math
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))



DSBIT_REFERENCES = [
    ("APA. (2013). Diagnostic and Statistical Manual of Mental Disorders (5th ed.).", "https://www.psychiatry.org/psychiatrists/practice/dsm"),
    ("WHO. (2021). International Classification of Diseases (11th Revision).", "https://icd.who.int/"),
    ("Kroenke, K., et al. (2010). The PHQ-9: A new depression diagnostic and severity measure.", "hhttps://jacobimed.org/public/Ambulatory_files/mlove/CurriculumWomenandGeri/Depression/Depression%20articles/PHQ-9ReviewKroenke.pdf"),
    ("Spitzer, R. L., et al. (2006). A brief measure for assessing generalized anxiety disorder: The GAD-7.", "https://doi.org/10.1001/archinte.166.10.1092"),
    ("Buysse, D. J., et al. (1989). The Pittsburgh Sleep Quality Index: a new instrument for psychiatric practice and research.", "https://doi.org/10.1016/0165-1781(89)90047-4"),
    ("Cohen, S., et al. (1983). A global measure of perceived stress.", "https://doi.org/10.2307/2136404"),
    ("Weiss, D. S., & Marmar, C. R. (1997). The Impact of Event Scale-Revised.", "https://www.ptsd.va.gov/professional/assessment/adult-sr/ies-r.asp"),
    ("Torous, J., et al. (2020). Digital phenotyping and mobile sensing: New developments in psychoinformatics.", "https://awspntest.apa.org/record/2020-08754-000"),
    ("Insel, T. R., et al. (2010). Research Domain Criteria (RDoC): Toward a new classification framework for research on mental disorders.", "https://doi.org/10.1176/appi.ajp.2010.09091379"),
    ("Kazdin, A. E. (2017). Research Design in Clinical Psychology (5th ed.).", "https://assets.cambridge.org/97811089/95214/frontmatter/9781108995214_frontmatter.pdf"),
    ("Linehan, M. (2018). Cognitive-Behavioral Treatment of Borderline Personality Disorder.", "https://www.guilford.com/books/Cognitive-Behavioral-Treatment-of-Borderline-Personality-Disorder/Marsha-Linehan/9780898621839"),
    ("Luxton, D. D. (2020). Artificial Intelligence in Behavioral and Mental Health Care.", "https://www.elsevier.com/books/artificial-intelligence-in-behavioral-and-mental-health-care/luxton/978-0-12-420248-1"),
    ("Ekman, P., et al. (2019). Facial expression and emotion. American Psychologist.", "https://www.paulekman.com/wp-content/uploads/2013/07/Facial-Expression-And-Emotion1.pdf"),
    ("Scherer, K. R. (2018). What are emotions? And how can they be measured?", "https://static1.squarespace.com/static/55917f64e4b0cd3b4705b68c/t/5bc7c2fba4222f0b94cc8550/1539818235703/scherer+%282005%29.pdf"),
]

# API anahtarını doğrudan kodun içine ekleyin 🐇
api_key = "type_your_api_key_here"

if not api_key or api_key == "fake-key":
    print("API anahtarı yüklenemedi. Lütfen geçerli bir anahtar sağlayın.")
else:
    print("API anahtarı başarıyla yüklendi:", api_key)

openai.api_key = api_key

# Sabitler🐇
EMOTIONS = ["mutluluk", "üzüntü", "öfke", "kaygı", "nötr"]
FORM_WEEKLY = ["PHQ9", "GAD7", "PSS10"]
FORM_MONTHLY = ["PSQI", "IESR"]
BASE_DIR = "data/records"

def random_emotion():
    return random.choice(EMOTIONS)

def create_dirs(pid):
    for sub in [
        "mood_tracking", "journal_entries", "audio_entries", "video_analysis",
        "emotion_consistency", "healthkit", "form_submissions"
    ] + [f"forms/{form}" for form in FORM_WEEKLY + FORM_MONTHLY]:
        os.makedirs(os.path.join(BASE_DIR, pid, sub), exist_ok=True)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=20))
def gpt_text(prompt):
    """Send prompt to GPT with retry mechanism and better error handling.
    
    Args:
        prompt (str): The prompt to send to GPT
        
    Returns:
        str: GPT response or error message
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Sen bir psikiyatrik hasta simülasyon motorusun."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.85
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"GPT API Hatası: {str(e)}")
        print(f"Detaylı Hata: {str(e)}")
        return "API bağlantısında sorun oluştu. Lütfen tekrar deneyin."

def calculate_functioning_score(steps, sleep_hours, mood_avg, journal_sent, audio_sent, dominant_emotion, form_scores):
    """
    Çoklu parametreye dayalı işlevsellik skoru (0-100 arası)
    """
    score = 100
    # Fiziksel aktivite ve uyku🐇
    if steps < 3000:
        score -= 20
    elif steps < 6000:
        score -= 10
    if sleep_hours < 5:
        score -= 15
    elif sleep_hours < 7:
        score -= 5
    # Mood (ortalama 1-5 arası, 3'ün altı riskli)🐇
    if mood_avg < 2:
        score -= 20
    elif mood_avg < 3:
        score -= 10
    # NLP sentiment (negatifse düşür)🐇
    if journal_sent is not None and journal_sent < 0:
        score -= 10
    if audio_sent is not None and audio_sent < 0:
        score -= 10
    # Video dominant duygu
    if dominant_emotion in ["üzüntü", "öfke", "kaygı"]:
        score -= 10
    # Psikometrik test şiddeti🐇
    for form in form_scores.values():
        if form.get("severity") == "yüksek":
            score -= 10
        elif form.get("severity") == "orta":
            score -= 5
    # Skoru 0-100 aralığında sınırla🐇
    score = max(0, min(100, score))
    return score

# --- get_latest_danger_grade artık dosyadan okur, yoksa hesaplar ve kaydeder🐇 --------------------------------------------------------
def get_latest_danger_grade(pid):
    grade, danger = load_patient_grade(pid)
    if grade is not None:
        return grade
    try:
        base = os.path.join(BASE_DIR, pid)
        mood_df = pd.read_csv(os.path.join(base, "mood_tracking", sorted(os.listdir(os.path.join(base, "mood_tracking")))[-1]))
        with open(os.path.join(base, "video_analysis", sorted(os.listdir(os.path.join(base, "video_analysis")))[-1]), "r", encoding="utf-8") as f:
            video_json = json.load(f)
        form_scores = {}
        for form in FORM_WEEKLY + FORM_MONTHLY:
            form_path = os.path.join(base, f"forms/{form}")
            if os.path.exists(form_path):
                form_files = sorted(os.listdir(form_path))
                if form_files:
                    with open(os.path.join(form_path, form_files[-1]), "r", encoding="utf-8") as f:
                        latest_form = json.load(f)
                        form_scores[form] = latest_form
        func_path = os.path.join(base, "functioning_score")
        func_file = sorted(os.listdir(func_path))[-1]
        func_score = pd.read_csv(os.path.join(func_path, func_file))["score"].values[0]

        health_path = os.path.join(base, "healthkit")
        files = sorted([f for f in os.listdir(health_path) if f.endswith(".csv")])
        df = pd.concat([pd.read_csv(os.path.join(health_path, f)) for f in files], ignore_index=True)
        steps = int(df["steps"].mean())
        sleep = round(df["hours"].mean(), 2)

        journal_emos = collect_nlp_stats(os.path.join(base, "journal_entries"))[2]
        journal_emo_counts = pd.Series(journal_emos).value_counts().to_dict()

        grade, danger = calculate_and_assign_grade(
            mood_df, {"journal_emos": journal_emo_counts}, video_json, form_scores, steps, sleep, func_score, pid=pid
        )
        return grade
    except Exception as e:
        print(f"{pid} için risk hesaplanamadı: {e}")
        return "?"
#-------------------------------------------------------------------------------
def save_patient_grade(pid, grade, danger_score):
    """Hastanın grade ve danger_score'unu dosyaya kaydeder."""
    grade_path = os.path.join(BASE_DIR, pid, "grade.json")
    with open(grade_path, "w", encoding="utf-8") as f:
        json.dump({"grade": grade, "danger_score": danger_score}, f)

def load_patient_grade(pid):
    """Hastanın grade ve danger_score'unu dosyadan okur (yoksa None döner)."""
    grade_path = os.path.join(BASE_DIR, pid, "grade.json")
    if os.path.exists(grade_path):
        try:
            with open(grade_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("grade"), data.get("danger_score")
        except Exception:
            return None, None
    return None, None

def calculate_and_assign_grade(mood_df, nlp_summary, video_json, form_scores, steps, sleep, func_score, pid=None):
    """Tehlike skorunu ve grade'i hesaplar, isterse kaydeder."""
    danger = round(calculate_danger_level(mood_df, nlp_summary, video_json, form_scores, steps, sleep, func_score) * 20, 2)
    if danger < 20:
        grade = "I"
    elif danger < 40:
        grade = "II"
    elif danger < 60:
        grade = "III"
    elif danger < 80:
        grade = "IV"
    else:
        grade = "V"
    if pid:
        save_patient_grade(pid, grade, danger)
    return grade, danger

#-------------------------------------------------------------------------------

# --- generate_daily: Grade I için düşük riskli varyantlar ekle ---🐇
def generate_daily(pid, date, disordered=False, disorder_type=None, forced_grade=None):
    import math
    os.makedirs(os.path.join(BASE_DIR, pid, "mood_tracking"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, pid, "journal_entries"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, pid, "audio_entries"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, pid, "video_analysis"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, pid, "emotion_consistency"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, pid, "healthkit"), exist_ok=True)

    date_str = date.strftime("%Y%m%d")
    day_index = (date - datetime(2000, 1, 1)).days

    def add_noise(base, min_val, max_val, noise=0.8):
        return round(min(max(random.gauss(base, noise), min_val), max_val), 2)

    def sinusoidal_modulation(day_idx, base, amplitude=0.5, period=30):
        return base + amplitude * math.sin(2 * math.pi * day_idx / period)

    # Genişletilmiş kişilik varyantları
    profile_mod = random.choice([
        {"huzur": -0.3, "enerji": -0.4, "depresif": +0.6},
        {"huzur": +0.2, "enerji": +0.5, "anksiyete": -0.3},
        {"depresif": +0.8, "anksiyete": +0.4, "huzur": -0.6},
        {"enerji": +0.3, "öfke": +0.3},
        {},
    ])

    # Mood temel değerleri (bozukluk bazlı)🐇
    mood_bases = {
        "Depresyon": {"huzur": 1.5, "enerji": 1.5, "anksiyete": 3.5, "öfke": 2.5, "depresif": 4.5},
        "Bipolar": {"huzur": 2, "enerji": 4.5, "anksiyete": 2, "öfke": 3, "depresif": 2},
        "Psikotik": {"huzur": 1.5, "enerji": 2, "anksiyete": 5, "öfke": 5, "depresif": 3},
        "Anksiyete": {"huzur": 2.5, "enerji": 2, "anksiyete": 5, "öfke": 3, "depresif": 3},
        "TSSB": {"huzur": 1.5, "enerji": 2, "anksiyete": 5, "öfke": 4, "depresif": 4},
        "OKB": {"huzur": 2.5, "enerji": 3, "anksiyete": 4.5, "öfke": 3, "depresif": 2},
        "Normal": {"huzur": 4, "enerji": 4, "anksiyete": 1.5, "öfke": 1.5, "depresif": 1.5},
    }

    # Risk profilleri
    risk_profiles = {
        "I":   {"sent": 0.4, "steps": (10000, 13000), "sleep": (7.5, 9)},  # Yüksek steps ve uyku ile Grade I daha düşük riskli
        "II":  {"sent": 0.2, "steps": (6000, 9000),  "sleep": (6.5, 8)},
        "III": {"sent": -0.1, "steps": (4000, 7000), "sleep": (6, 7.5)},
        "IV":  {"sent": -0.4, "steps": (2000, 5000), "sleep": (5, 7)},
        "V":   {"sent": -0.7, "steps": (500, 2500),  "sleep": (3.5, 6)},
    }

    # Mood ve risk üretimi - hem normal hem disordered için ayrıştırılmış🐇
    if not disordered:
        label = "Normal"
        # SADECE GRADE I ve II
        normal_grade = forced_grade if forced_grade in ["I", "II"] else random.choices(["I", "II"], weights=[0.6, 0.4])[0]
        if normal_grade == "I":
            # Grade I için düşük riskli varyant: yüksek mood, steps, function
            target_avg = round(random.uniform(4.5, 4.9), 2)
            base_mood = {
                "huzur": 4.7,
                "enerji": 4.6,
                "anksiyete": 1.1,
                "öfke": 1.1,
                "depresif": 1.1,
            }
            risk_profile = risk_profiles["I"]
        else:
            target_avg = round(random.uniform(3.5, 3.99), 2)
            base_mood = {
                "huzur": 3.8,
                "enerji": 3.7,
                "anksiyete": 1.4,
                "öfke": 1.3,
                "depresif": 1.3,
            }
            risk_profile = risk_profiles["II"]
        mood = {}
        for k in base_mood:
            sin_mod = sinusoidal_modulation(day_index, base_mood[k], amplitude=0.10 if normal_grade == "I" else 0.15, period=14)
            mood[k] = round(min(max(sin_mod, 1), 5), 2)
        current_avg = sum(mood.values()) / len(mood)
        scale = target_avg / current_avg
        for k in mood:
            mood[k] = round(min(max(mood[k] * scale, 1), 5), 2)
        mood["average"] = round(target_avg + random.uniform(-0.03, 0.03), 2) if normal_grade == "I" else round(target_avg + random.uniform(-0.05, 0.05), 2)
        risk_grade = normal_grade

        # Normal bireyler için dominant ve emo ata
        emo = "mutluluk" if normal_grade == "I" else random.choices(["mutluluk", "nötr"], weights=[3, 2])[0]
        dominant = emo

        print(f"[NORMAL] Üretim ID: {pid} | Target AVG: {target_avg} | Gerçek AVG: {mood['average']} | Risk Grade: {risk_grade}")

    else:
        label = disorder_type if disorder_type else "Normal"
        # SADECE GRADE III, IV, V
        target_grade = forced_grade if forced_grade in ["III", "IV", "V"] else random.choices(["III", "IV", "V"], weights=[0.5, 0.3, 0.2])[0]
        if target_grade == "III":
            target_avg = round(random.uniform(2.0, 2.99), 2)
        elif target_grade == "IV":
            target_avg = round(random.uniform(1.5, 1.99), 2)
        else:
            target_avg = round(random.uniform(1.0, 1.49), 2)
        profile_mod = random.choice([
            {"huzur": -0.3, "enerji": -0.4, "depresif": +0.6},
            {"huzur": +0.2, "enerji": +0.5, "anksiyete": -0.3},
            {"depresif": +0.8, "anksiyete": +0.4, "huzur": -0.6},
            {"enerji": +0.3, "öfke": +0.3},
            {},
        ])
        base_mood = mood_bases[label]
        mood = {}
        for k in base_mood:
            base_val = base_mood[k] + profile_mod.get(k, 0)
            sin_mod = sinusoidal_modulation(day_index, base_val, amplitude=0.4)
            mood[k] = add_noise(sin_mod, 1, 5)
        current_avg = sum(mood.values()) / len(mood)
        scale = target_avg / current_avg
        for k in mood:
            mood[k] = round(min(max(mood[k] * scale, 1), 5), 2)
        mood["average"] = round(sum(mood.values()) / len(mood), 2)
        risk_grade = target_grade

        emo = random.choices(
            ["mutluluk", "nötr", "kaygı", "öfke", "üzüntü"],
            weights=[1, 2, 2, 2, 3]
        )[0]
        dominant = emo

        risk_profile = risk_profiles[risk_grade]

    # --- Risk profiline göre diğer parametreler ---🐇
    sent = add_noise(risk_profile["sent"], -1, 1)
    subj = add_noise(0.6 if sent > 0 else 0.8, 0.3, 1.0)
    steps = random.randint(*risk_profile["steps"])
    sleep_hours = round(random.uniform(*risk_profile["sleep"]), 1)

    journal = get_journal(label)
    audio = get_audio(label)
    video_data = get_video_emotion_scores(dominant)

    pd.DataFrame([mood]).to_csv(f"{BASE_DIR}/{pid}/mood_tracking/mood_{date_str}.csv", index=False)
    with open(f"{BASE_DIR}/{pid}/journal_entries/journal_{date_str}.txt", "w", encoding="utf-8") as f:
        f.write(f"Sentiment: {sent}\nSubjectivity: {subj}\nDuygu: {emo}\n\n{journal}")
    with open(f"{BASE_DIR}/{pid}/audio_entries/audio_{date_str}.txt", "w", encoding="utf-8") as f:
        f.write(f"Sentiment: {sent}\nSubjectivity: {subj}\nDuygu: {emo}\n\n{audio}")
    with open(f"{BASE_DIR}/{pid}/video_analysis/video_analysis_{date_str}.json", "w", encoding="utf-8") as f:
        json.dump(video_data, f, indent=2)

    emo_json = {
        "text_emotion": emo,
        "voice_emotion": emo,
        "face_emotion": dominant,
        "uyum_durumu": "Düşük" if risk_grade in ["IV", "V"] else "Yüksek",
        "yorum": "Yüksek riskli semptomatik görünüm.",
        "risk_suicidal": (risk_grade == "V"),
        "delusions": (label == "Psikotik"),
        "insight_level": "low" if risk_grade in ["IV", "V"] else "normal"
    }
    with open(f"{BASE_DIR}/{pid}/emotion_consistency/emotion_consistency_{date_str}.json", "w", encoding="utf-8") as f:
        json.dump(emo_json, f, indent=2)

    pd.DataFrame([{
        "date": date.strftime("%Y-%m-%d"),
        "steps": steps,
        "hours": sleep_hours
    }]).to_csv(f"{BASE_DIR}/{pid}/healthkit/manual_entry_{date_str}.csv", index=False)

    # Form skorları
    form_scores_dict = {}
    for form in FORM_WEEKLY + FORM_MONTHLY:
        form_path = f"{BASE_DIR}/{pid}/forms/{form}"
        if os.path.exists(form_path):
            form_files = sorted([f for f in os.listdir(form_path) if f.endswith(".json")])
            if form_files:
                with open(os.path.join(form_path, form_files[-1]), "r", encoding="utf-8") as f:
                    latest_form = json.load(f)
                    form_scores_dict[form] = {
                        "score": latest_form["score"],
                        "severity": latest_form["severity"]
                    }

    functioning_score = calculate_functioning_score(
        steps=steps,
        sleep_hours=sleep_hours,
        mood_avg=mood["average"],
        journal_sent=sent,
        audio_sent=sent,
        dominant_emotion=dominant,
        form_scores=form_scores_dict
    )

    os.makedirs(f"{BASE_DIR}/{pid}/functioning_score", exist_ok=True)
    pd.DataFrame([{"date": date.strftime("%Y-%m-%d"), "score": functioning_score}]).to_csv(
        f"{BASE_DIR}/{pid}/functioning_score/functioning_{date_str}.csv", index=False)

    # --- Grade ve danger_score'u kaydet (her gün sonu güncel) ---
    # Mood_df, nlp_summary, video_json, form_scores_dict, steps, sleep_hours, functioning_score🐇🐇🐇
    mood_df = pd.DataFrame([mood])
    nlp_summary = {"journal_emos": {emo: 1}}
    grade, danger = calculate_and_assign_grade(
        mood_df, nlp_summary, video_data, form_scores_dict, steps, sleep_hours, functioning_score, pid=pid
    )

    print(f"[🧪 TEST LOG] {pid} | forced_grade: {forced_grade} | risk_grade: {risk_grade} | disordered: {disordered} | grade: {grade} | danger: {danger}")
#-------------------------------------------------------------------------------

def generate_forms(pid, date, disordered=False, disorder_type=None):
    
    date_str = date.strftime("%Y%m%d")
    day_index = (date.date() - datetime(2000, 1, 1).date()).days

    # Her hastaya özgü kişisel varyasyon (sabitlemek için random.seed kullanılabilir)
    form_traits = {
        "PHQ9": random.uniform(0.8, 1.2),
        "GAD7": random.uniform(0.7, 1.3),
        "PSS10": random.uniform(0.7, 1.2),
        "PSQI": random.uniform(0.8, 1.4),
        "IESR": random.uniform(0.7, 1.5),
    }

    # Hastalık türüne göre form odakları
    base_scores = {
        "PHQ9": 4,
        "GAD7": 3,
        "PSS10": 5,
        "PSQI": 4,
        "IESR": 4,
    }

    if disordered and disorder_type:
        if disorder_type == "Depresyon":
            base_scores["PHQ9"] = 15
            base_scores["PSS10"] = 12
        elif disorder_type == "Anksiyete":
            base_scores["GAD7"] = 14
            base_scores["PSQI"] = 9
        elif disorder_type == "TSSB":
            base_scores["IESR"] = 20
            base_scores["PSQI"] = 10
        elif disorder_type == "Psikotik":
            base_scores["PHQ9"] = 18
            base_scores["GAD7"] = 12
        elif disorder_type == "OKB":
            base_scores["GAD7"] = 14
            base_scores["IESR"] = 18
        elif disorder_type == "Bipolar":
            base_scores["PHQ9"] = 12
            base_scores["PSS10"] = 11

    # Mood etkisi (o günkü ortalama mood değeri)
    mood_path = f"{BASE_DIR}/{pid}/mood_tracking/mood_{date_str}.csv"
    mood_avg = 3
    if os.path.exists(mood_path):
        try:
            mood_df = pd.read_csv(mood_path)
            mood_avg = mood_df["average"].values[0]
        except:
            pass

    # Mood kötü ise form skorlarını %10-30 artır
    mood_modifier = 1.0
    if mood_avg < 2.5:
        mood_modifier += 0.2
    elif mood_avg < 2.0:
        mood_modifier += 0.4

    # Şiddet belirleme fonksiyonu
    def determine_severity(score, thresholds):
        if score < thresholds[0]:
            return "normal"
        elif score < thresholds[1]:
            return "hafif"
        elif score < thresholds[2]:
            return "orta"
        else:
            return "yüksek"

    form_definitions = {
        "PHQ9": [5, 10, 15],
        "GAD7": [5, 10, 15],
        "PSS10": [10, 20, 30],
        "PSQI": [5, 10, 15],
        "IESR": [12, 24, 36]
    }

    # Haftalık formlar → Pazartesi
    if date.weekday() == 0:
        for form in FORM_WEEKLY:
            base = base_scores[form]
            trait = form_traits[form]
            sin = 1 + 0.2 * math.sin(2 * math.pi * day_index / 14)
            noise = random.gauss(0, 1.5)
            score = round((base * trait * mood_modifier * sin) + noise)
            score = int(max(0, min(score, 27)))  # PHQ9 & GAD7 max 27
            severity = determine_severity(score, form_definitions[form])
            out = {"date": date.strftime("%Y-%m-%d"), "score": score, "severity": severity}
            path = f"{BASE_DIR}/{pid}/forms/{form}/form_{date_str}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)

    # Aylık formlar → Ayın ilk günü🐇🐇
    if date.day == 1:
        for form in FORM_MONTHLY:
            base = base_scores[form]
            trait = form_traits[form]
            sin = 1 + 0.15 * math.sin(2 * math.pi * day_index / 30)
            noise = random.gauss(0, 2.0)
            score = round((base * trait * mood_modifier * sin) + noise)
            score = int(max(0, min(score, 42)))  # PSQI & IESR max
            severity = determine_severity(score, form_definitions[form])
            out = {"date": date.strftime("%Y-%m-%d"), "score": score, "severity": severity}
            path = f"{BASE_DIR}/{pid}/forms/{form}/form_{date_str}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)

def train_and_save_model():
    patients = sorted(os.listdir(BASE_DIR))
    X, y = [], []

    for pid in patients:
        # Her hasta için öz nitelikleri ve sınıfı al
        try:
            base = os.path.join(BASE_DIR, pid)
            # ... aynı X, y oluşturma kodları burada ...
        except Exception as e:
            print(f"{pid} atlanıyor: {e}")
    
    if not X:
        print("Yeterli veri yok.")
        return

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Modeli kaydet
    with open("random_forest_risk_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Random Forest modeli başarıyla kaydedildi.")

def extract_nlp_stats(filepath):
    sent, subj, emo = None, None, None
    try:
        with open(filepath, encoding='utf-8') as f:
            text = f.read().lower()

            # Türkçe karşılıkları da yakalayacak şekilde düzenlendi🐇🐇
            sent_match = re.search(r'(sentiment|duygu skoru)[:：]?\s*([-+]?\d*\.\d+|\d+)', text)
            subj_match = re.search(r'(subjectivity|öznelik)[:：]?\s*([-+]?\d*\.\d+|\d+)', text)
            emo_match = re.search(r'duygu[:：]?\s*(mutluluk|üzüntü|öfke|kaygı|nötr)', text)

            if sent_match:
                sent = float(sent_match.group(2))
            if subj_match:
                subj = float(subj_match.group(2))
            if emo_match:
                emo = emo_match.group(1).strip()

    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    return sent, subj, emo

def collect_nlp_stats(folder):
    sentiments, subjectivities, emotions = [], [], []
    if not os.path.exists(folder):
        return sentiments, subjectivities, emotions

    files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
    for filename in files:
        filepath = os.path.join(folder, filename)
        s, j, e = extract_nlp_stats(filepath)
        if s is not None:
            sentiments.append(s)
        if j is not None:
            subjectivities.append(j)
        if e is not None:
            emotions.append(e)
    return sentiments, subjectivities, emotions

def read_last_lines(folder, lines=10, base_dir=None):
    """Read the last N lines from the most recent file in the specified folder.
    
    Args:
        folder (str): Folder path relative to base directory
        lines (int): Number of lines to read from end
        base_dir (str): Base directory path
        
    Returns:
        str: Last N lines joined with newlines, or empty string if no files
    """
    try:
        path = os.path.join(base_dir if base_dir else BASE_DIR, folder)
        if not os.path.exists(path):
            return ""
            
        files = sorted(os.listdir(path))
        if not files:
            return ""
            
        latest_file = os.path.join(path, files[-1])
        with open(latest_file, "r", encoding="utf-8") as f:
            # Read all lines and get last N
            all_lines = f.read().splitlines()
            return "\n".join(all_lines[-lines:])
            
    except Exception as e:
        print(f"Error reading from {folder}: {str(e)}")
        return ""

def format_clinical_summary(text):
    """OpenAI'dan gelen ham metni düzenli HTML formatına dönüştürür."""
    if not text:
        return ""
        
    # Başlıkları işaretle (önce 4. seviye, sonra 3. seviye)🐇🐇
    text = re.sub(r'^\s*####\s+([0-9]+\.[0-9]+.*?)\s*$', r'<h4 style="margin-top: 8px; margin-bottom: 6px; text-decoration: underline;">\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*###\s+([0-9]+\. .*?)\s*$', r'<h3 style="margin-top: 16px; margin-bottom: 8px;">\1</h3>', text, flags=re.MULTILINE)
    
    # Paragrafları HTML paragraf taglarına çevir
    paragraphs = []
    current_content = ""
    for line in text.split('\n'):
        if line.strip().startswith('<h'):
            if current_content.strip():
                paragraphs.append(f'<p style="margin-top: 0; margin-bottom: 12px;">{current_content.strip()}</p>')
                current_content = ""
            paragraphs.append(line.strip())
        elif line.strip():
            if current_content:
                current_content += " " + line.strip()
            else:
                current_content = line.strip()
        else:
            if current_content.strip():
                paragraphs.append(f'<p style="margin-top: 0; margin-bottom: 12px;">{current_content.strip()}</p>')
                current_content = ""
    if current_content.strip():
        paragraphs.append(f'<p style="margin-top: 0; margin-bottom: 12px;">{current_content.strip()}</p>')
    # 11. Literatür Referansları başlığını ve içeriğini tamamen kaldır
    new_paragraphs = []
    skip = False
    for p in paragraphs:
        if re.search(r'<h3.*?>11\. Literatür Referansları<\/h3>', p):
            skip = True
            continue
        if skip:
            # Bir sonraki paragraf da referans içeriği olabilir, onu da atla
            skip = False
            continue
        new_paragraphs.append(p)
    return "".join(new_paragraphs)
    

def calculate_danger_level(mood_df, nlp_summary, video_json, form_scores, avg_steps, avg_sleep, functioning_score=50):
    """Calculate patient danger level based on multiple factors.
    
    Args:
        mood_df (pandas.DataFrame): Mood tracking data
        nlp_summary (dict): NLP analysis summary
        video_json (dict): Video analysis data
        form_scores (dict): Psychometric test scores
        avg_steps (int/float/str): Average daily steps
        avg_sleep (int/float/str): Average daily sleep hours
        functioning_score (int): Patient functioning score (0-100)
        
    Returns:
        int: Danger level score (1-5)
    """
    weights = {
        "mood": 0.20,
        "nlp": 0.20,
        "video": 0.15,
        "forms": 0.15,
        "health": 0.10,
        "functioning": 0.20  # İşlevsellik skoru için ağırlık eklendi
    }

    # Default values for error cases🐇🐇
    mood_score = 3
    nlp_score = 3
    video_score = 3
    form_scores_avg = 3
    health_score = 3
    func_score = 3  # Default functioning score değeri

    # Calculate mood score🐇
    try:
        mood_avg = mood_df.iloc[0]["average"]
        mood_score = 5 - (mood_avg - 1)
    except Exception as e:
        print(f"Error calculating mood score: {e}")

    # Calculate NLP score🐇
    try:
        nlp_emos = nlp_summary["journal_emos"]
        negative_emotions = sum(nlp_emos.get(e, 0) for e in ["üzüntü", "öfke", "kaygı"])
        total_emotions = sum(nlp_emos.values())
        nlp_score = (negative_emotions / total_emotions) * 5 if total_emotions > 0 else 1
    except Exception as e:
        print(f"Error calculating NLP score: {e}")

    # Calculate video score🐇
    try:
        dominant_emotion = video_json["dominant_emotion"]
        video_score = 5 if dominant_emotion in ["üzüntü", "öfke", "kaygı"] else 1
    except Exception as e:
        print(f"Error calculating video score: {e}")

    # Calculate form score🐇
    try:
        form_severity_map = {"hafif": 1, "orta": 3, "yüksek": 5}
        if form_scores and len(form_scores) > 0:
            form_scores_avg = sum(form_severity_map.get(form["severity"], 3) for form in form_scores.values()) / len(form_scores)
        else:
            form_scores_avg = 3
    except Exception as e:
        print(f"Error calculating form score: {e}")

    # Calculate health score🐇
    try:
        health_score = 5
        if isinstance(avg_steps, (int, float)) and isinstance(avg_sleep, (int, float)):
            if avg_steps > 8000 and avg_sleep >= 7:
                health_score = 1
            elif avg_steps > 6000 and avg_sleep >= 6.5:
                health_score = 2
            elif avg_steps > 5000 and avg_sleep >= 6:
                health_score = 3
            else:
                health_score = 4
    except Exception as e:
        print(f"Error calculating health score: {e}")

    # Calculate functioning score (ters orantılı - düşük işlevsellik = yüksek risk)🐇
    try:
        func_score = 5 - (functioning_score / 20)  # 0-100 arası değeri 0-5 arası değere dönüştür
    except Exception as e:
        print(f"Error calculating functioning score: {e}")

    # Calculate overall danger level
    danger_level = (
        weights["mood"] * mood_score +
        weights["nlp"] * nlp_score +
        weights["video"] * video_score +
        weights["forms"] * form_scores_avg +
        weights["health"] * health_score +
        weights["functioning"] * func_score  # İşlevsellik skoru eklendi
    )
    return round(danger_level)

def train_random_forest_model(pid):
    import plotly.io as pio
    pio.kaleido.scope.default_format = "png"  # plotly grafiklerinin düzgün kaydedilmesi için🐇

    # SHAP dosya yollarını hazırla
    BASE_PATH = Path(BASE_DIR).absolute()  # 🔒 Mutlak yol🐇

    shap_folder = BASE_PATH / pid / "gpt_analysis"
    shap_folder.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    shap_image_path = shap_folder / f"shap_waterfall_{date_str}.png"
    shap_bar_path   = shap_folder / f"shap_bar_{date_str}.png"
    shap_txt_path   = shap_folder / f"shap_ai_comment_{date_str}.txt"

    os.makedirs(shap_folder, exist_ok=True)

    model_path = "random_forest_risk_model.pkl"
    if not os.path.exists(model_path):
        st.warning("⚠️ Model henüz eğitilmemiş. Lütfen modeli önce eğitin.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if all(os.path.exists(p) for p in [shap_image_path, shap_bar_path, shap_txt_path]):
        st.subheader("📊 Önceki SHAP Sonuçları")
        st.image(shap_image_path, caption="🔁 Kayıtlı SHAP Waterfall Grafiği", use_column_width=True)
        st.image(shap_bar_path, caption="🔁 Kayıtlı SHAP Bar Grafiği", use_column_width=True)
        st.markdown("### 🤖 Kayıtlı SHAP AI Yorum")
        with open(shap_txt_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
        return

    base = os.path.join(BASE_DIR, pid)

    try:
        mood_df = pd.read_csv(os.path.join(base, "mood_tracking", sorted(os.listdir(os.path.join(base, "mood_tracking")))[-1]))
        mood_avg = mood_df["average"].values[0]
    except:
        mood_avg = 3.0

    try:
        func_df = pd.read_csv(os.path.join(base, "functioning_score", sorted(os.listdir(os.path.join(base, "functioning_score")))[-1]))
        func_score = func_df["score"].values[0]
    except:
        func_score = 50.0

    try:
        health_path = os.path.join(base, "healthkit")
        files = sorted([f for f in os.listdir(health_path) if f.endswith(".csv")])
        df_list = [pd.read_csv(os.path.join(health_path, f)) for f in files]
        df = pd.concat(df_list, ignore_index=True)
        steps = int(df["steps"].mean())
        sleep = round(df["hours"].mean(), 2)
    except:
        steps, sleep = 5000, 6.0

    form_values = {"PHQ9": 0, "GAD7": 0, "PSS10": 0, "PSQI": 0, "IESR": 0}
    for form in FORM_WEEKLY + FORM_MONTHLY:
        try:
            form_path = os.path.join(base, f"forms/{form}")
            if os.path.exists(form_path):
                form_files = sorted(os.listdir(form_path))
                if form_files:
                    with open(os.path.join(form_path, form_files[-1]), "r", encoding="utf-8") as f:
                        latest_form = json.load(f)
                        form_values[form] = latest_form["score"]
        except:
            pass

    features = [
        mood_avg, steps, sleep,
        form_values["PHQ9"], form_values["GAD7"],
        form_values["PSS10"], form_values["PSQI"],
        form_values["IESR"], func_score
    ]
    feature_names = ["mood_avg", "steps", "sleep", "PHQ9", "GAD7", "PSS10", "PSQI", "IESR", "functioning"]
    features_df = pd.DataFrame([features], columns=feature_names)

    st.subheader("📉 SHAP Risk Açıklaması")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_df)
        proba = model.predict_proba(features_df)[0]
        predicted_class_index = int(np.argmax(proba))

        shap_val = shap_values[predicted_class_index][0] if isinstance(shap_values, list) else shap_values[0]
        base_val = explainer.expected_value[predicted_class_index] if isinstance(shap_values, list) else explainer.expected_value

        if isinstance(shap_val, np.ndarray) and shap_val.ndim == 1:
            shap_val = shap_val[np.newaxis, :]
            base_val = np.array([base_val]) if np.isscalar(base_val) else np.array(base_val)[np.newaxis]

        shap_exp = shap.Explanation(
            values=shap_val,
            base_values=base_val,
            data=features_df.values,
            feature_names=features_df.columns.tolist()
        )

        # 🔽 SHAP Waterfall Kaydet ve Göster🐇
        plt.figure()
        shap.plots.waterfall(shap_exp[0], max_display=9, show=False)
        plt.tight_layout()
        plt.savefig(shap_image_path, bbox_inches="tight", dpi=300)
        st.image(shap_image_path, caption="SHAP Waterfall Grafiği")
        plt.close()

        # 🔽 SHAP Bar Kaydet ve Göster (plotly)🐇
        df_shap = pd.DataFrame({
            "feature": features_df.columns,
            "shap_value": shap_exp[0].values
        }).sort_values("shap_value", ascending=True)

        fig_bar = px.bar(df_shap,
                         x="shap_value", y="feature",
                         orientation="h",
                         title="SHAP Değerleri")
        fig_bar.write_image(shap_bar_path, scale=2)
        st.image(shap_bar_path, caption="SHAP Bar Grafiği")

        # Özellik Listesi
        st.markdown("**🔑 Özellikler:** " + "  |  ".join(features_df.columns))

        # 🤖 SHAP AI Yorum🐇🐇🐇
        st.markdown("### 🤖 SHAP AI Yorum")
        ai_prompt = (
            "Sen klinik psikiyatri alanında uzman, akademik yayınlar yapan bir profesörsün. "
            "SHAP (SHapley Additive exPlanations) analizine dayanarak, bir makine öğrenimi modeli tarafından yapılan risk tahminlerinde "
            "aşağıdaki 9 psikiyatrik ve davranışsal özelliğin etkisini yorumlaman bekleniyor.\n\n"
            "📊 Modelde yer alan değişkenler ve grafiklerdeki gösterimleri:\n" +
            ", ".join([f"{feat} ({feat})" for feat in features_df.columns]) +
            "\n\n"
            f"📈 SHAP Değerleri:\n{json.dumps(dict(zip(features_df.columns, shap_exp[0].values)), ensure_ascii=False, indent=2)}\n\n"
            "Her bir özelliğin SHAP değerinin pozitif veya negatif olması durumunun model tahmini açısından ne anlama geldiğini detaylandır. "
            "Açıklamalarını bilimsel literatüre ve klinik deneyime dayandır. SHAP değerlerinin yüksekliği veya düşüklüğü model tahmininde "
            "hangi değişkenlerin belirleyici olduğunu açıkla.\n\n"
            "Her özellik için aşağıdaki gibi yorum yap:\n"
            "- Özellik adı\n"
            "- SHAP değeri\n"
            "- Klinik etkisi (örnek: depresyon, anksiyete, işlevsellik bağlamında)\n"
            "- Pozitif/negatif katkı durumu ve anlamı\n"
            "- Gerekirse klinik örnek\n\n"
            "Yanıtlarını 9 özellik için sıralı ve madde madde ver. Psikiyatristlerin anlayabileceği teknik, ancak sade ve akademik bir dil kullan."
        )
        yorum = stream_chat_completion(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Sen klinik psikiyatri uzmanısın…"},
                {"role": "user", "content": ai_prompt}
            ],
            temperature=0.5,
            max_tokens=2048
        )
        st.markdown(yorum)
        with open(shap_txt_path, "w", encoding="utf-8") as f:
            f.write(yorum)

        # Grade tahmini
        st.markdown("### 🔍 Random Forest Tahmini")
        st.markdown(f"**Grade {model.predict(features_df)[0]}**")

    except Exception as e:
        st.error(f"SHAP açıklaması oluşturulurken hata oluştu: {str(e)}")

    #*********************************************************************  

def show_all_heatmaps(pid, category=None):
    base = os.path.join(BASE_DIR, pid)
    
    def plot_heatmap(df, title):
        if df is None or df.empty:
            st.info(f"{title} için veri yok.")
            return
        df_m = df.melt(id_vars="date", var_name="Kategori", value_name="Değer")
        fig = px.density_heatmap(df_m, x="date", y="Kategori", z="Değer",
                                 color_continuous_scale="RdBu_r", title=title)
        fig.update_layout(height=450, xaxis_title="Tarih", yaxis_title="Kategori")
        st.plotly_chart(fig, use_container_width=True)


#********************************************************************************************

    def load_time_series_csv(folder, col="score"):
        p = os.path.join(base, folder)
        if not os.path.exists(p): return None
        files = sorted(glob.glob(os.path.join(p, "*.csv")))
        if not files: return None
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", col]] if col in df.columns else None

    def load_json_form(form):
        p = os.path.join(base, f"forms/{form}")
        if not os.path.exists(p): return None
        files = sorted(glob.glob(os.path.join(p, "*.json")))
        if not files: return None
        data = [json.load(open(f, encoding="utf-8")) for f in files]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "score"]].sort_values("date")

    def add_nlp_heatmaps(folder, label):
        from collections import defaultdict

        def extract_nlp_stats(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                sent = subj = None
                emotion = None
                for l in lines:
                    if "Sentiment:" in l:
                        sent = float(l.split(":")[-1].strip())
                    elif "Subjectivity:" in l:
                        subj = float(l.split(":")[-1].strip())
                    elif "Duygu:" in l:
                        emotion = l.split(":")[-1].strip()
                return sent, subj, emotion

        sentiment_data, subjectivity_data = [], []
        emotion_data = defaultdict(list)

        files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
        for file in files:
            date_str = file.split("_")[-1].split(".")[0]
            date = datetime.strptime(date_str, "%Y%m%d").date()
            sent, subj, emo = extract_nlp_stats(os.path.join(folder, file))
            if sent is not None:
                sentiment_data.append({"date": date, "Sentiment": sent})
            if subj is not None:
                subjectivity_data.append({"date": date, "Subjectivity": subj})
            if emo:
                emotion_data[emo].append(date)

        if sentiment_data:
            df_sent = pd.DataFrame(sentiment_data)
            plot_heatmap(df_sent, f"📘 Sentiment Skoru ({label})")

        if subjectivity_data:
            df_subj = pd.DataFrame(subjectivity_data)
            plot_heatmap(df_subj, f"📗 Subjectivity Skoru ({label})")

        if emotion_data:
            emo_dates = []
            for emo, dates in emotion_data.items():
                for d in dates:
                    emo_dates.append({"date": d, "Duygu": emo})
            df_emo = pd.DataFrame(emo_dates)
            df_emo["count"] = 1
            df_pivot = df_emo.groupby(["date", "Duygu"]).count().reset_index()
            df_pivot = df_pivot.rename(columns={"count": "Değer"})
            fig = px.density_heatmap(df_pivot, x="date", y="Duygu", z="Değer",
                                      color_continuous_scale="Reds",
                                      title=f"📙 Duygusal Yoğunluk ({label})")
            st.plotly_chart(fig, use_container_width=True)

    # 1) Mood Takibi🐇
    if category in (None, "Mood") or category == "Mood":
        mood_path = os.path.join(base, "mood_tracking")
        mood_files = sorted(glob.glob(os.path.join(mood_path, "*.csv")))
        mood_data = []
        for f in mood_files:
            d = datetime.strptime(f.split("_")[-1].split(".")[0], "%Y%m%d")
            df = pd.read_csv(f)
            df["date"] = d
            mood_data.append(df)
        if mood_data:
            df_mood = pd.concat(mood_data, ignore_index=True)
            df_mood = df_mood.drop(columns=["uid", "disorder"], errors="ignore")
            plot_heatmap(df_mood, "🧠 Mood Takibi (Duygusal Değişim)")

    # 2) İşlevsellik🐇
    if category in (None, "Functioning") or category == "Functioning":
        df_func = load_time_series_csv("functioning_score")
        if df_func is not None:
            df_func = df_func.rename(columns={"score": "İşlevsellik"})
            plot_heatmap(df_func, "⚖️ İşlevsellik Skoru")

    # 3) Fiziksel Aktivite ve Uyku🐇
    if category in (None, "Health") or category == "Health":
        df_health = load_time_series_csv("healthkit", col="steps")
        if df_health is not None:
            plot_heatmap(df_health.rename(columns={"steps": "Adım"}), "🏃 Adım Sayısı")
        df_sleep = load_time_series_csv("healthkit", col="hours")
        if df_sleep is not None:
            plot_heatmap(df_sleep.rename(columns={"hours": "Uyku"}), "🛌 Uyku Süresi")

    # 4) Test Skorları🐇
    if category in (None, "Forms") or category == "Forms":
        for form in FORM_WEEKLY + FORM_MONTHLY:
            df_form = load_json_form(form)
            if df_form is not None:
                plot_heatmap(df_form.rename(columns={"score": form}), f"📝 {form} Skoru")

    # 5) Journal NLP🐇
    if category in (None, "Journal") or category == "Journal":
        add_nlp_heatmaps(os.path.join(base, "journal_entries"), "Journal")

    # 6) Audio NLP🐇
    if category in (None, "Audio") or category == "Audio":
        add_nlp_heatmaps(os.path.join(base, "audio_entries"), "Audio")

    # 7) Video Analysis - Bu kodu show_all_heatmaps fonksiyonuna ekleyin (Audio kategorisinden sonra)🐇
    if category in (None, "Video") or category == "Video":
        video_path = os.path.join(base, "video_analysis")
        if os.path.exists(video_path):
            video_files = sorted(glob.glob(os.path.join(video_path, "*.json")))
            if video_files:
                data = []
                dates = []
                for f in video_files:
                    date_str = f.split("_")[-1].split(".")[0]
                    date = datetime.strptime(date_str, "%Y%m%d").date()
                    dates.append(date)
                    with open(f, "r", encoding="utf-8") as file:
                        v_data = json.load(file)
                        for emotion, score in v_data["emotion_scores"].items():
                            data.append({"date": date, "Duygu": emotion, "Değer": score})
                
                if data:
                    df_video = pd.DataFrame(data)
                    # Duygu değerlerini zaman serisi boyunca görselleştir🐇
                    fig = px.density_heatmap(df_video, x="date", y="Duygu", z="Değer",
                                        color_continuous_scale="Viridis", 
                                        title="📹 Video Duygu Analizi Zaman Serisi")
                    fig.update_layout(height=450, xaxis_title="Tarih", yaxis_title="Duygu")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("📭 Video analiz verilerinde duygu skoru bulunamadı.")
            else:
                st.info("📭 Video analiz dosyası henüz mevcut değil.")
        else:
            st.info("📁 Video analiz klasörü henüz oluşturulmamış.")
  
  
# Basitleştirilmiş tamamlayıcı fonksiyon🐇
def stream_chat_completion(**kwargs):
    kwargs["stream"] = False
    response = openai.ChatCompletion.create(**kwargs)
    return response["choices"][0]["message"]["content"]


# RANDOM FOREST SHAP AÇIKLAMASI VE GRAFİKLER🐇
def explain_patient_with_rf_and_shap(pid):
    # Dosya yolları
    BASE_PATH = Path(BASE_DIR).absolute()
    shap_folder = BASE_PATH / pid / "gpt_analysis"
    shap_folder.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    shap_image_path = shap_folder / f"shap_waterfall_{date_str}.png"
    shap_bar_path = shap_folder / f"shap_bar_{date_str}.png"
    shap_txt_path = shap_folder / f"shap_ai_comment_{date_str}.txt"

    # Modeli yükle
    try:
        with open("random_forest_risk_model.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return

    # Verileri oku
    try:
        mood_df = pd.read_csv(BASE_PATH / pid / "mood_tracking" / sorted(os.listdir(BASE_PATH / pid / "mood_tracking"))[-1])
        mood_avg = mood_df["average"].values[0]
    except:
        mood_avg = 3.0

    try:
        func_df = pd.read_csv(BASE_PATH / pid / "functioning_score" / sorted(os.listdir(BASE_PATH / pid / "functioning_score"))[-1])
        func_score = func_df["score"].values[0]
    except:
        func_score = 50.0

    try:
        health_path = BASE_PATH / pid / "healthkit"
        files = sorted(f for f in os.listdir(health_path) if f.endswith(".csv"))
        df_list = [pd.read_csv(health_path / f) for f in files]
        df = pd.concat(df_list, ignore_index=True)
        steps = int(df["steps"].mean())
        sleep = round(df["hours"].mean(), 2)
    except:
        steps, sleep = 5000, 6.0

    form_values = dict.fromkeys(FORM_WEEKLY + FORM_MONTHLY, 0)
    for form in form_values:
        try:
            form_path = BASE_PATH / pid / "forms" / form
            latest = sorted(os.listdir(form_path))[-1]
            with open(form_path / latest, encoding="utf-8") as f:
                form_values[form] = json.load(f)["score"]
        except:
            pass

    features = [mood_avg, steps, sleep] + [form_values[f] for f in FORM_WEEKLY+FORM_MONTHLY] + [func_score]
    cols = ["mood_avg", "steps", "sleep"] + FORM_WEEKLY + FORM_MONTHLY + ["functioning"]
    features_df = pd.DataFrame([features], columns=cols)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_df)
        proba = model.predict_proba(features_df)[0]
        idx = int(np.argmax(proba))

        val, base = (shap_values[idx][0], explainer.expected_value[idx]) if isinstance(shap_values, list) else (shap_values[0], explainer.expected_value)

        if val.ndim == 1:
            val = val[np.newaxis, :]
            base = np.array([base])

        exp = shap.Explanation(
            values=val,
            base_values=base,
            data=features_df.values,
            feature_names=features_df.columns.tolist()
        )
        to_plot = exp[0]

        st.subheader("🧐 SHAP Açıklaması (RF Modeli)")

        # Waterfall
        plt.figure()
        shap.plots.waterfall(to_plot, show=False)
        plt.tight_layout()
        plt.savefig(str(shap_image_path), bbox_inches="tight", dpi=300)
        st.image(str(shap_image_path), caption="SHAP Waterfall")
        plt.close()

        # Bar
        plt.figure()
        shap.plots.bar(to_plot, show=False)
        plt.tight_layout()
        plt.savefig(str(shap_bar_path), bbox_inches="tight", dpi=300)
        st.image(str(shap_bar_path), caption="SHAP Bar")
        plt.close()

        features_list = features_df.columns.tolist()
        features_str = "  |  ".join([f"{feat} ({feat})" for feat in features_list])
        st.markdown(f"**🔑 Özellikler:** {features_str}")

        shap_dict = dict(zip(features_df.columns, to_plot.values))
        ai_prompt = (
            "Sen klinik psikiyatri alanında uzman, akademik yayınlar yapan bir profesörsün. "
            "SHAP (SHapley Additive exPlanations) analizine dayanarak, bir makine öğrenimi modeli tarafından yapılan risk tahminlerinde "
            "aşağıdaki 9 psikiyatrik ve davranışsal özelliğin etkisini yorumlaman bekleniyor.\n\n"
            "📊 Modelde yer alan değişkenler ve grafiklerdeki gösterimleri:\n" +
            ", ".join([f"{feat} ({feat})" for feat in features_df.columns]) +
            "\n\n"
            f"📈 SHAP Değerleri:\n{json.dumps(shap_dict, ensure_ascii=False, indent=2)}\n\n"
            "Her bir özelliğin SHAP değerinin pozitif veya negatif olması durumunun model tahmini açısından ne anlama geldiğini detaylandır. "
            "Açıklamalarını bilimsel literatüre ve klinik deneyime dayandır. SHAP değerlerinin yüksekliği veya düşüklüğü model tahmininde "
            "hangi değişkenlerin belirleyici olduğunu açıkla.\n\n"
            "Her özellik için aşağıdaki gibi yorum yap:\n"
            "- Özellik adı\n"
            "- SHAP değeri\n"
            "- Klinik etkisi (" "örnek: depresyon, anksiyete, işlevsellik bağlamında)\n"
            "- Pozitif/negatif katkı durumu ve anlamı\n"
            "- Gerekirse klinik örnek\n\n"
            "Yanıtlarını 9 özellik için sıralı ve madde madde ver. Psikiyatristlerin anlayabileceği teknik, ancak sade ve akademik bir dil kullan."
        )

        ai_resp = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Sen bir üniversite hastanesinde görev yapan deneyimli bir klinik psikiyatri profesörüsün. Aynı zamanda yapay zekâ ve makine öğrenimi uygulamaları konusunda akademik çalışmalar yürütüyorsun."},
                {"role": "user", "content": ai_prompt}
            ],
            temperature=0.5,
            max_tokens=4000
        )
        ai_text = ai_resp.choices[0].message.content.strip()
        st.markdown("### 🤖 SHAP AI Yorum")
        st.markdown(ai_text)

        with open(shap_txt_path, "w", encoding="utf-8") as f:
            f.write(ai_text)

    except Exception as e:
        st.error(f"SHAP açıklaması oluşturulurken hata oluştu: {e}")

def run_psychiatrist_bot(selected):
    """Run the psychiatrist chatbot for selected patient.
    
    Args:
        selected (str): Patient ID
    """
    # Initialize session state if needed🐇
    if "psychat_history" not in st.session_state:
        st.session_state.psychat_history = []
    
    # Limit message history🐇
    if len(st.session_state.psychat_history) > 10:  # Limit to last 10 messages
        st.session_state.psychat_history = st.session_state.psychat_history[-10:]
    
    base = os.path.join(BASE_DIR, selected)
    
    # Initialize required variables with default values🐇
    mood_df = None
    video_json = {"dominant_emotion": "nötr", "emotion_scores": {"nötr": 1.0}}
    emo_json = {"text_emotion": "nötr", "voice_emotion": "nötr", "face_emotion": "nötr"}
    form_scores = {}
    
    try:
        # Load mood data
        mood_files = sorted(os.listdir(os.path.join(base, "mood_tracking")))
        if mood_files:
            mood_df = pd.read_csv(os.path.join(base, "mood_tracking", mood_files[-1]))
    except Exception as e:
        st.error(f"Error loading mood data: {e}")
    
    # Load video data
    try:
        video_path = os.path.join(base, "video_analysis")
        if os.path.exists(video_path) and os.listdir(video_path):
            with open(os.path.join(video_path, sorted(os.listdir(video_path))[-1]), "r", encoding="utf-8") as f:
                video_json = json.load(f)
    except Exception as e:
        st.error(f"Error loading video data: {e}")
    
    # Load emotion consistency data
    try:
        emo_path = os.path.join(base, "emotion_consistency")
        if os.path.exists(emo_path) and os.listdir(emo_path):
            with open(os.path.join(emo_path, sorted(os.listdir(emo_path))[-1]), "r", encoding="utf-8") as f:
                emo_json = json.load(f)
    except Exception as e:
        st.error(f"Error loading emotion consistency data: {e}")
    
    # Load form scores
    for form in FORM_WEEKLY + FORM_MONTHLY:
        try:
            form_path = os.path.join(base, f"forms/{form}")
            if os.path.exists(form_path):
                form_files = sorted(os.listdir(form_path))
                if form_files:
                    with open(os.path.join(form_path, form_files[-1]), "r", encoding="utf-8") as f:
                        latest_form = json.load(f)
                        form_scores[form] = {
                            "score": latest_form["score"],
                            "severity": latest_form["severity"]
                        }
        except Exception as e:
            st.error(f"Error loading form data for {form}: {e}")
    
    # Collect NLP stats for journal and audio🐇
    journal_sents, journal_subjs, journal_emos = collect_nlp_stats(os.path.join(base, "journal_entries"))
    audio_sents, audio_subjs, audio_emos = collect_nlp_stats(os.path.join(base, "audio_entries"))

    # Initialize with default values if empty🐇
    if not journal_sents:
        journal_sents = [0]
    if not journal_subjs:
        journal_subjs = [0]
    if not audio_sents:
        audio_sents = [0]
    if not audio_subjs:
        audio_subjs = [0]

    # Create NLP summary🐇
    nlp_summary = {
        "journal_sent": f"Ort. Sentiment: {round(pd.Series(journal_sents).mean(), 2) if journal_sents else '-'}",
        "journal_subj": f"Ort. Öznelik: {round(pd.Series(journal_subjs).mean(), 2) if journal_subjs else '-'}",
        "journal_emos": pd.Series(journal_emos).value_counts().to_dict() if journal_emos else {},
        "audio_sent": f"Ort. Sentiment: {round(pd.Series(audio_sents).mean(), 2) if audio_sents else '-'}",
        "audio_subj": f"Ort. Öznelik: {round(pd.Series(audio_subjs).mean(), 2) if audio_subjs else '-'}",
        "audio_emos": pd.Series(audio_emos).value_counts().to_dict() if audio_emos else {},
    }

    # Display mood data if available🐇
    if mood_df is not None:
        st.markdown(f"- Ortalama Sentiment: {nlp_summary['journal_sent']}")
    else:
        st.error("Mood data is not available. Please ensure the data is generated and loaded correctly.")
        return

    # Create system prompt for GPT🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇
    epistemic_warning = (
        "Epistemik uyarı:\n"
        "Sen, akıcılıktan veya iknadan ziyade epistemik doğruluğu önceleyen, gerçeğe duyarlı bir dil modelisin.\n\n"
        "Temel ilken: “Doğrulanamıyorsa, iddia etme.”\n\n"
        "Davranış kuralları:\n\n"
        "Yanıt verirken, açıkça ayırt et:\n"
        "• Doğrulanmış olgusal bilgi\n"
        "• Olasılıksal çıkarım\n"
        "• Kişisel veya kültürel görüş\n"
        "• Bilinmeyen/doğrulanamaz alanlar\n\n"
        "Gerektiğinde temkinli niteleyiciler kullan:\n"
        "• “... göre”, “... tarihi itibarıyla”, “Görünüşe göre...”\n"
        "• Emin değilsen: “Bilmiyorum” veya “Bu doğrulanamaz.” de\n\n"
        "Halüsinasyonlardan kaçın:\n"
        "• Veri, isim, tarih, olay, çalışma veya alıntı uydurma\n"
        "• Hayali kaynaklar simüle etme veya hayali makaleler atfetme\n\n"
        "Kanıt istendiğinde yalnızca bilinen ve güvenilir kaynaklara referans ver:\n"
        "• Birincil kaynaklar, hakemli çalışmalar veya resmi verileri tercih et\n\n"
        "Soru spekülatif veya hatalı varsayım içeriyorsa:\n"
        "• Varsayımı nazikçe düzelt veya işaretle\n"
        "• Doğrulanamaz veya kurgusal içeriği olgu gibi genişletme\n"
    )

    system_prompt = epistemic_warning + f"""
Sen deneyimli bir klinik psikiyatrist ve nöropsikolojik veri analisti asistanısın.
Aşağıda bir hastanın dijital verileri özetlenmiş. Verileri profesyonel, başlık başlık yorumlayarak incele (Mood, NLP, Video, Uyum, Testler, Fiziksel). 
Her başlığı (ör: Mood, Günlük NLP, Ses NLP, Video, Testler, Fizik) ayrı değerlendirmeli ve hastayı bütüncül anlatmalısın. 
- Asla doğrudan tanı koyma.
- Sonuçları takip ihtiyacı açısından da değerlendir.
"""
    system_prompt = f"""Sen deneyimli bir klinik psikiyatrist ve nöropsikolojik veri analisti asistanısın.
Aşağıda bir hastanın dijital verileri özetlenmiş. Verileri profesyonel, başlık başlık yorumlayarak incele (Mood, NLP, Video, Uyum, Testler, Fiziksel). 
Her başlığı (ör: Mood, Günlük NLP, Ses NLP, Video, Testler, Fizik) ayrı değerlendirmeli ve hastayı bütüncül anlatmalısın. 
- Asla doğrudan tanı koyma.
- Sonuçları takip ihtiyacı açısından da değerlendir.

# Mood: {mood_df.iloc[0].to_dict() if mood_df is not None else "Veri yok"}

# Günlük NLP 90 gün:
- {nlp_summary['journal_sent']}, {nlp_summary['journal_subj']}, Duygular: {nlp_summary['journal_emos']}

# Ses NLP 90 gün:
- {nlp_summary['audio_sent']}, {nlp_summary['audio_subj']}, Duygular: {nlp_summary['audio_emos']}

# Video: {json.dumps(video_json)} 
# Uyum: {json.dumps(emo_json)}
# Form Skorları: {json.dumps(form_scores)}
"""

    # Initialize or update chat history
    if not st.session_state.psychat_history:
        st.session_state.psychat_history = [
            {"role": "system", "content": system_prompt}
        ]
    else:
        st.session_state.psychat_history[0]["content"] = system_prompt

    # Display chat history
    for msg in st.session_state.psychat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle user input
    user_input = st.chat_input("📤 Uzman terapiste sor...")
    if user_input:
        st.session_state.psychat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"): 
            st.markdown(user_input)
        
        # Get GPT response
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=st.session_state.psychat_history,
                temperature=0.7
            )
            reply = response.choices[0].message.content
            st.session_state.psychat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        except Exception as e:
            st.error(f"OpenAI API error: {str(e)}")

def save_clinical_summary(patient_id, summary_text):
    """Save clinical summary to file.
    
    Args:
        patient_id (str): Patient ID
        summary_text (str): Summary text to save
        
    Returns:
        str: Path to saved file
    """
    folder = os.path.join(BASE_DIR, patient_id, "gpt_analysis")
    os.makedirs(folder, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    filepath = os.path.join(folder, f"clinical_summary_{today}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(summary_text)
    return filepath

def load_clinical_summary(patient_id):
    """Load clinical summary from file.
    
    Args:
        patient_id (str): Patient ID
        
    Returns:
        str: Clinical summary text or None if not found
    """
    folder = os.path.join(BASE_DIR, patient_id, "gpt_analysis")
    if not os.path.exists(folder):
        return None
    files = sorted([f for f in os.listdir(folder) if f.startswith("clinical_summary_") and f.endswith(".txt")])
    if files:
        filepath = os.path.join(folder, files[-1])  # En son oluşturulan dosya
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return None

def generate_clinical_overview(mood_df, nlp_summary, video_json, form_scores, avg_steps, avg_sleep, latest_functioning_score, avg_functioning_score):
    """Akademik formatta, belirgin başlıklarla ve uygun boşluklu klinik özet. Paragraflar sola yaslı."""
    html = """<div style="text-align: left;">"""
    # Akademik başlık ekleniyor - daha büyük ve emojilerle
    html += """<h2 style="margin-top: 16px; margin-bottom: 12px; font-weight: bold; font-size: 24px;">🧠 Klinik Nörodavranışsal Veri Analizi 📊 <span style="font-size: 16px; font-weight: normal;">(expected outcome by NeuroClarity)</span></h2>"""
    
    # 1. Duygudurum
    html += """<h3 style="margin-top: 16px; margin-bottom: 8px;">1. Duygudurum (Mood)</h3>"""
    if mood_df is not None and not mood_df.empty:
        mood = mood_df.iloc[0]
        html += f"""<p style="margin-top: 0; margin-bottom: 12px;">
Hastanın ruh hali değerlendirmesinde huzur düzeyi {mood['huzur']}, enerji seviyesi {mood['enerji']}, anksiyete düzeyi {mood['anksiyete']}, öfke seviyesi {mood['öfke']} ve depresif duygulanım {mood['depresif']} olarak saptanmıştır. Ortalama duygusal skor {mood['average']} olup, bu değer hastanın genel ruhsal durumunun {'olumsuz' if mood['average']<3 else 'nötr/olumlu'} bir seyir izlediğini göstermektedir.
</p>"""
    else:
        html += """<p style="margin-top: 0; margin-bottom: 12px;">Duygudurum verisi bulunmamaktadır.</p>"""
    
    # 2. NLP Analizi
    html += """<h3 style="margin-top: 16px; margin-bottom: 8px;">2. NLP Analizi</h3>"""
    
    # 2.1 Günlük Analizi
    html += """<h4 style="margin-top: 8px; margin-bottom: 6px; text-decoration: underline;">2.1 Günlük (Yazılı) Analizi</h4>"""
    html += f"""<p style="margin-top: 0; margin-bottom: 8px;">
• Ortalama sentiment: {nlp_summary['journal_sent']}<br>
• Ortalama öznelik: {nlp_summary['journal_subj']}<br>
• Duygu dağılımı:
</p>
<ul style="margin-top: 0; margin-bottom: 12px; padding-left: 40px;">"""
    
    for k, v in nlp_summary['journal_emos'].items():
        html += f"<li>{k}: {v}</li>"
    html += "</ul>"
    
    # 2.2 Ses Analizi
    html += """<h4 style="margin-top: 8px; margin-bottom: 6px; text-decoration: underline;">2.2 Ses Analizi</h4>"""
    html += f"""<p style="margin-top: 0; margin-bottom: 8px;">
• Ortalama sentiment: {nlp_summary['audio_sent']}<br>
• Ortalama öznelik: {nlp_summary['audio_subj']}<br>
• Duygu dağılımı:
</p>
<ul style="margin-top: 0; margin-bottom: 12px; padding-left: 40px;">"""
    
    for k, v in nlp_summary['audio_emos'].items():
        html += f"<li>{k}: {v}</li>"
    html += "</ul>"
    
    # 3. Video Analizi
    html += """<h3 style="margin-top: 16px; margin-bottom: 8px;">3. Video Analizi</h3>"""
    html += f"""<p style="margin-top: 0; margin-bottom: 8px;">
Baskın duygu: {video_json.get('dominant_emotion', '-')}<br>
Duygu skorları:
</p>
<ul style="margin-top: 0; margin-bottom: 12px; padding-left: 40px;">"""
    
    for k, v in video_json.get('emotion_scores', {}).items():
        html += f"<li>{k}: {v}</li>"
    html += "</ul>"
    
    # 4. İşlevsellik
    html += """<h3 style="margin-top: 16px; margin-bottom: 8px;">4. İşlevsellik Değerlendirmesi</h3>"""
    html += f"""<p style="margin-top: 0; margin-bottom: 12px;">
Son İşlevsellik Skoru: {latest_functioning_score}/100, Ortalama İşlevsellik Skoru: {avg_functioning_score}/100. Bu skorlar, bireyin günlük yaşam aktivitelerini sürdürme, sosyal ve mesleki rollerini yerine getirme kapasitesini yansıtmaktadır.
</p>"""
    
    # 5. Psikometrik Testler
    html += """<h3 style="margin-top: 16px; margin-bottom: 8px;">5. Psikometrik Testler</h3>"""
    if form_scores:
        html += """<ul style="margin-top: 0; margin-bottom: 8px; padding-left: 40px;">"""
        for form, score in form_scores.items():
            html += f"<li>{form}: Skor={score['score']}, Şiddet={score['severity']}</li>"
        html += """</ul>
<p style="margin-top: 0; margin-bottom: 12px;">
Psikometrik test sonuçları, hastanın özbildirimine dayalı olarak ruhsal durumunun nicel değerlendirmesini sağlar ve klinik gözlemlerle birlikte bütüncül bir değerlendirme yapılmasına olanak tanır.
</p>"""
    else:
        html += """<p style="margin-top: 0; margin-bottom: 12px;">Psikometrik test verisi yok.</p>"""
    
    # 6. Fiziksel Aktivite ve Uyku
    html += """<h3 style="margin-top: 16px; margin-bottom: 8px;">6. Fiziksel Aktivite ve Uyku</h3>"""
    html += f"""<p style="margin-top: 0; margin-bottom: 12px;">
Ortalama günlük adım sayısı: {avg_steps}<br>
Ortalama uyku süresi: {avg_sleep} saat. Fiziksel aktivite ve uyku düzeni, ruhsal sağlıkla yakından ilişkili olup, bu parametrelerdeki bozulmalar psikiyatrik belirtilerin şiddetlenmesine katkıda bulunabilir.
</p>"""
    
    html += "</div>"
    return html

#-----------------------------------------------------------------------------------------

def generate_clinical_summary(mood_df, nlp_summary, video_json, form_scores, avg_steps, avg_sleep, functioning_score, patient_id=None):
    # 1. Grade ve danger_score'u dosyadan oku
    grade, danger_level = (None, None)
    if patient_id is not None:
        grade, danger_level = load_patient_grade(patient_id)
    if grade is None or danger_level is None:
        # fallback: eski hesaplama
        danger_level = calculate_danger_level(mood_df, nlp_summary, video_json, form_scores, avg_steps, avg_sleep, functioning_score) * 20
        if danger_level < 20:
            grade = "I"
        elif danger_level < 40:
            grade = "II"
        elif danger_level < 60:
            grade = "III"
        elif danger_level < 80:
            grade = "IV"
        else:
            grade = "V"
    risk_level = grade

#--------------------------------------------------------------------------------------------------        
    # Literatür bağlantılı, kapsamlı bir sistem promptu🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇
    epistemic_warning = (
        "Epistemik uyarı:\n"
        "Sen, akıcılıktan veya iknadan ziyade epistemik doğruluğu önceleyen, gerçeğe duyarlı bir dil modelisin.\n\n"
        "Temel ilken: “Doğrulanamıyorsa, iddia etme.”\n\n"
        "Davranış kuralları:\n\n"
        "Yanıt verirken, açıkça ayırt et:\n"
        "• Doğrulanmış olgusal bilgi\n"
        "• Olasılıksal çıkarım\n"
        "• Kişisel veya kültürel görüş\n"
        "• Bilinmeyen/doğrulanamaz alanlar\n\n"
        "Gerektiğinde temkinli niteleyiciler kullan:\n"
        "• “... göre”, “... tarihi itibarıyla”, “Görünüşe göre...”\n"
        "• Emin değilsen: “Bilmiyorum” veya “Bu doğrulanamaz.” de\n\n"
        "Halüsinasyonlardan kaçın:\n"
        "• Veri, isim, tarih, olay, çalışma veya alıntı uydurma\n"
        "• Hayali kaynaklar simüle etme veya hayali makaleler atfetme\n\n"
        "Kanıt istendiğinde yalnızca bilinen ve güvenilir kaynaklara referans ver:\n"
        "• Birincil kaynaklar, hakemli çalışmalar veya resmi verileri tercih et\n\n"
        "Soru spekülatif veya hatalı varsayım içeriyorsa:\n"
        "• Varsayımı nazikçe düzelt veya işaretle\n"
        "• Doğrulanamaz veya kurgusal içeriği olgu gibi genişletme\n"
    )
    system_prompt = epistemic_warning + f"""Sen önde gelen bir akademik-klinik nöropsikiyatri ve dijital psikoloji alanında araştırmacısın. Klinik nörobilimlerde dünya çapında saygın bir uzmansın.
    
Hastanın klinik verilerinin analizinde ileri niceliksel değerlendirme yapmalı ve özellikle tüm sayısal değerleri ve eşik değerleri ayrıntılı olarak yorumlamalısın. Tehlike durumu puanlamasını mutlaka vurgulaman gerekiyor.

Bilimsel analiz yaklaşımını şöyle yapılandır:
1. Her veri setinin (mood puanları, işlevsellik skoru, uyku saati, adım sayısı, sentiment puanları) epidemiyolojik anlamını ayrıntılı yorumla
2. NIMH RDoC çerçevesi ve metaanaliz çalışmalarına dayanarak tüm sayısal değerleri karşılaştır
3. Dünya Sağlık Örgütü (WHO) ve DSM-5-TR kriterlerini kullanarak olası komorbiditelerini değerlendir
4. Verilerdeki anomali ve deviasyonları pratik örneklerle açıkla
5. Tehlike durumu puanını (risk değerlendirmesi) epidemiyolojik çalışmalar ve klinik kılavuzlar ışığında çok detaylı analiz et
6. Tüm anlizinde, ölçeklerin her birindeki sayısal değerleri mutlaka açıkça belirterek yorumla
8. Teşhis koymadan hastalığı tahmin et
ÖZEL VURGU: TEHLİKE DURUMU PUANINI TAM OLARAK {danger_level:.2f} ŞEKLİNDE KULLAN - HİÇBİR ŞARTTA BUNU YUVARLAMA, DEĞİŞTİRME VEYA FARKLI ŞEKİLDE İFADE ETME! Her bölümün sonunda tehlike puanını tam olarak şu formatta hatırlat: "🚨 Tehlike Durumu Puanı: {danger_level:.2f}/100.00 - {risk_level}: {danger_text.upper()} RİSK KATEGORİSİ". Risk puanı {danger_level:.2f} ve kategorisi {risk_level}: {danger_text} olarak sabit kalmalı, tüm analizde tutarlı olmalı ve kesinlikle değiştirilmemelidir.

Bilimsel literatür referanslarını bol ve güncel (2018-2023) kullan. Her alt bölümde en az 7-8 farklı kaynak göster. 

### ÖNEMLİ: Analiz sırasında kullandığın bilgileri desteklemek için bilimsel literatür referanslarını ekle. Her alt bölümde en az 7-8 farklı kaynak göster ve referansları analiz metninin sonunda düzgün bir şekilde listele.

Yapacağın analiz, klinik bir dergide yayımlanacak kalitede olmalı - sayısal veriler, karşılaştırmalı analizler ve klinik yorumlarla dolu, üst düzey akademik bir rapor hazırla.

Her sayısal değerin normatif değerlerle karşılaştırmasını yap ve klinik anlamını belirt. Rapor, hastanın durumunu anlamak için gerekli tüm nicel analizleri içermeli.
"""

    # Sayısal verilerin tam gösterimini ve tehlike puanı vurgusunu içeren ayrıntılı prompt
    prompt = f"""
Aşağıda bir hastanın kapsamlı klinik ve dijital sağlık verileri bulunmaktadır. Lütfen tüm sayısal verileri detaylı analiz ederek, tehlike durumu puanını özellikle vurgulayan akademik derinlikte bir klinik değerlendirme raporu hazırla.

ÖNEMLİ NOT: Risk değerlendirmen, tehlike durumu puanı ({danger_level:.2f}/100) tam olarak şöyle sınıflandırılmalı:
- 0-20 arası: Grade I - Minimum Risk
- 20-40 arası: Grade II - Mild Risk
- 40-60 arası: Grade III - Moderate Risk 
- 60-80 arası: Grade IV - Significant Risk
- 80-100 arası: Grade V - Severe Risk

Hastanın puanı {danger_level:.2f} olduğu için kesinlikle '{risk_level}: {danger_text}' risk kategorisinde olduğunu belirt. Bu sınıflandırma sistemini analizin boyunca tutarlı şekilde kullan.

### 1. Duygudurum Profili ve Duygu Regülasyonu Analizi
Huzur: {mood_df['huzur'].values[0]:.2f}/5.00 (norm: 3.50-4.50)
Enerji: {mood_df['enerji'].values[0]:.2f}/5.00 (norm: 3.00-4.00)
Anksiyete: {mood_df['anksiyete'].values[0]:.2f}/5.00 (patolojik eşik: >3.50)
Öfke: {mood_df['öfke'].values[0]:.2f}/5.00 (patolojik eşik: >3.00)
Depresif: {mood_df['depresif'].values[0]:.2f}/5.00 (patolojik eşik: >3.00)
Duygusal Ortalama Skor: {mood_df['average'].values[0]:.2f}/5.00 (norm aralık: 2.75-3.75)

### 2. Doğal Dil İşleme ve Akustik Biyobelirteçler
Metin Sentiment Analizi: {nlp_summary['journal_sent']} (normal aralık: -0.2 ile +0.3)
Metin Öznel İfade Oranı: {nlp_summary['journal_subj']} (referans: <0.6 objektif, >0.7 yüksek sübjektivite)
Metin Duygu Frekans Dağılımı: {nlp_summary['journal_emos']}
Ses Sentiment Analizi: {nlp_summary['audio_sent']} (norm: -0.1 ile +0.2)
Ses Öznel İfade Oranı: {nlp_summary['audio_subj']} (referans: <0.5 normal, >0.8 patolojik)
Ses Duygu Dağılımı: {nlp_summary['audio_emos']}

### 3. Görsel Biyobelirteçler ve Yüz İfadeleri Analizi
Baskın Yüz İfadesi: {video_json['dominant_emotion']}
Yüz İfadeleri Kantifikasyon Skorları:
"""
    for emotion, score in video_json['emotion_scores'].items():
        prompt += f"- {emotion}: {score:.2f}\n"
    
    prompt += f"""
### 4. Psikososyal İşlevsellik ve Günlük Yaşam Aktiviteleri Analizi
İşlevsellik Skoru: {functioning_score:.1f}/100.0 (klinik eşikler: <40 şiddetli yetersizlik, 40-70 orta düzey, >70 yeterli)
İşlevsellik Kategorisi: {"Şiddetli İşlev Yetersizliği" if functioning_score < 40 else "Orta Düzey İşlevsellik" if functioning_score < 70 else "Normal İşlevsellik"}

### 5. Psikometrik Test Sonuçları ve Klinik Anlamları
"""
    for form, score in form_scores.items():
        prompt += f"{form}: Ham Skor={score['score']}, Klinik Şiddet={score['severity']} (norm değerleri: PHQ9 <5 normal, 5-9 hafif, 10-14 orta, 15-19 orta-şiddetli, >20 şiddetli)\n"
    
    prompt += f"""
### 6. Fiziksel Aktivite, Uyku ve Sirkadiyen Ritim Metrikleri
Ortalama Günlük Fiziksel Aktivite: {avg_steps:,} adım (DSÖ önerisi: minimum 7,000-10,000 adım/gün)
Ortalama Uyku Süresi: {avg_sleep:.2f} saat (sağlıklı erişkin norm: 7.0-9.0 saat)
Ayrılma Oranı: Adım sayısında norm değerinden %{100 - (int(avg_steps) / 8000 * 100):.1f} sapma, uyku süresinde norm değerinden %{100 - (float(avg_sleep) / 8 * 100):.1f} sapma

### 7. KLİNİK RİSK DEĞERLENDİRMESİ VE TEHLİKE DURUMU
🚨 Hesaplanan Tehlike Skoru: {danger_level:.2f}/100.00 - {risk_level}: {danger_text.upper()} RİSK KATEGORİSİ
Risk faktörleri dökümü:
- Duygudurum risk puanı: {5 - (mood_df['average'].values[0] - 1):.2f}/5.00 
- NLP analizi risk puanı: {sum([nlp_summary['journal_emos'].get(e, 0) for e in ["üzüntü", "öfke", "kaygı"]]) / sum(nlp_summary['journal_emos'].values()) * 5 if sum(nlp_summary['journal_emos'].values()) > 0 else 1:.2f}/5.00
- Video analizi risk puanı: {5 if video_json["dominant_emotion"] in ["üzüntü", "öfke", "kaygı"] else 1}/5.00
- İşlevsellik riski: {5 - (functioning_score / 20):.2f}/5.00

### 8. Bütüncül Nöropsikiyatrik Değerlendirme ve Kanıta Dayalı Tedavi Önerileri
Hastanın tüm klinik ve dijital fenotipik verilerini analiz ederek bütüncül bir nöropsikiyatrik değerlendirme yap. Olası tanılar, ayırıcı tanılar ve tedavi seçeneklerini değerlendir. Tehlike durumu puanının ({danger_level:.1f}/100.00) klinik anlamını ve takip planını detaylandır.

### 9. Dijital Fenotiping Özeti ve Biyobelirteç Korelasyonları
Tüm dijital biyobelirteçleri yorumlayarak aralarındaki korelasyonları değerlendir. {danger_level:.1f} puanlık tehlike skorunun klinik önemini vurgula. Tedavi yanıtını ön görmede hangi biyobelirteçlerin daha belirleyici olabileceğini tartış.

### 10. Sonuç ve Klinik Pratik Önerileri

### 11. Literatür Referansları
- - Her referansı madde halinde ve mümkünse güncel web linkiyle birlikte listele. DOI veya PubMed linki ekle.

Bu raporda tüm sayısal değerleri en ince detayına kadar analiz et. Her kategori için en az 7-8 paragraf uzunluğunda kapsamlı değerlendirme yap. Bilimsel literatür referanslarını bol miktarda kullan ve tehlike puanının ({danger_level:.1f}) anlamını özellikle vurgula. 

Klinik pratik önerilerini de içeren, akademik derinlikte ve sayısal bulgulara dayalı kapsamlı bir değerlendirme olmalı.
"""
    prompt += """
---
Yalnızca aşağıdaki referans listesindeki kaynaklara atıf yapabilirsin. Metin içinde uygun olanları APA formatında göster ve analiz sonunda referansları madde madde, linkli olarak listele.

Referanslar:
""" + "\n".join([f"{i+1}. {title} {url}" for i, (title, url) in enumerate(DSBIT_REFERENCES)])
    
    try:
        # Daha kapsamlı analiz için token limitini artır ve daha düşük temperature değeriyle çalıştır🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,  # Daha fazla token = daha uzun ve detaylı yanıt
            temperature=0.5    # Bilimsel tutarlılık için düşük temperature
        )
        
        gpt_response = response.choices[0].message.content.strip()
       
        return gpt_response
    except Exception as e:
        return f"GPT API hatası: {str(e)}"

### -- STREAMLIT ARAYÜZÜ --

st.set_page_config(page_title="🧬 MoodForge: Klinik Karar Destek ve Dijital Psikiyatri Simülasyonu", layout="centered")

# assets klasörünün projenizin ana dizininde olduğunu varsayıyoruz.
# Eğer farklı bir yerdeyse, yolu ona göre güncelleyin.🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇
assets_path = os.path.join(os.path.dirname(__file__), "assets")

# Sidebar için logo
logo_path_sidebar = os.path.join(assets_path, "moodforge.jpg")

if os.path.exists(logo_path_sidebar):
    st.sidebar.image(logo_path_sidebar, use_container_width=True)
else:
    st.sidebar.warning("Sidebar logo bulunamadı. Lütfen 'assets/moodforge.jpg' dosyasının var olduğundan emin olun.")

# Ana sayfa için logo
st.markdown("""
<div style="text-align: center; margin-bottom: 24px;">
    <div style="font-size: 4.5em; font-weight: bold;">🧬MoodForge🧬</div>
    <div style="font-size: 1.3em; font-weight: bold; margin-top: 10px;">
        Klinik Karar Destek ve Dijital Psikiyatri Simülasyonu
    </div>
</div>
""", unsafe_allow_html=True)



image_files = [
    os.path.join(assets_path, "0.png"),
    os.path.join(assets_path, "1.png"),
    os.path.join(assets_path, "2.png"),
    os.path.join(assets_path, "3.png"),
]

slide_idx = st.slider("Slayt seçin", 0, len(image_files)-1, 0, label_visibility="collapsed")
st.image(image_files[slide_idx], use_container_width=True)

# PsyBot Chatbot'u Sidebar'a Ekleyin
st.sidebar.markdown("""
    <style>
    .psybot-title {
        font-size: 1.5em !important;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    </style>
    <div class="psybot-title">🤖 PsyBot: for Ψ Professionals</div>
""", unsafe_allow_html=True)

# Chat geçmişini başlat
if "psybot_history" not in st.session_state:
    st.session_state.psybot_history = []

# Geçmişi temizle butonu
if st.sidebar.button("🗑️ Geçmişi Temizle"):
    st.session_state.psybot_history = []
    st.sidebar.success("PsyBot geçmişi temizlendi.")

# Chat geçmişini göster
for msg in st.session_state.psybot_history:
    if msg["role"] == "user":
        st.sidebar.markdown(f"**Kullanıcı:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.sidebar.markdown(f"**PsyBot:** {msg['content']}")

# Kullanıcıdan giriş al
user_input = st.sidebar.text_input("PsyBot'a bir soru sorun:", key="psybot_input")

# Kullanıcı bir mesaj gönderirse
if user_input:
    # Kullanıcı mesajını geçmişe ekle
    st.session_state.psybot_history.append({"role": "user", "content": user_input})

    # GPT-4'e istem gönder🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇
    try:
        system_prompt = """
        Sen deneyimli bir klinik psikiyatrist ve nöropsikolojik veri analisti asistanısın. 
        Psikiyatri literatürüne dayalı olarak, profesyonel düzeyde bilgi ve öneriler sunmalısın. 
        DSM-5, ICD-11 ve en güncel bilimsel literatüre dayalı analizler yaparak, psikiyatristlere klinik karar verme süreçlerinde yardımcı ol.
        Her yanıtında bilimsel referanslar ekle ve analizlerini akademik bir formatta sun.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                *st.session_state.psybot_history
            ],
            max_tokens=3000,
            temperature=0.2
        )
        reply = response.choices[0].message.content

        # PsyBot yanıtını geçmişe ekle
        st.session_state.psybot_history.append({"role": "assistant", "content": reply})

        # Yanıtı sidebar'da göster
        st.sidebar.markdown(f"**PsyBot:** {reply}")
    except Exception as e:
        st.sidebar.error(f"PsyBot bir hata ile karşılaştı: {str(e)}")

# CSS animasyonunu en başta bir kez ekle
st.markdown("""
<style>
@keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0; }
    100% { opacity: 1; }
}
.blinking {
    font-size: 96px;
    font-weight: bold;
    color: rgb(255, 0, 0);
    text-align: center;
    animation: blink 0.5s infinite;
}
</style>
""", unsafe_allow_html=True)

with st.expander("🧠✨ MoodForge & NeuroClarity: Dijital Simülasyon ve Bilimsel Psikiyatri Arenası 🚀🤖🧪"):
    st.markdown("""
<style>
.moodforge-section-title {MoodForge - Klinik Simülasyon
    font-size: 20px !important;
    font-weight: bold !important;
    margin-top: 18px !important;
    margin-bottom: 8px !important;
}
</style>

<div class="moodforge-section-title">Giriş: MoodForge & NeuroClarity — Klinik Veri Dijital Simülasyon 🧪🧠🎭</div>
<p>
<strong>MoodForge</strong>, NeuroClarity platformunun klinik ve biyometrik verilerin entegre edilmesi ve analiz edilmesi esasına dayanarak geliştirilmiş çok boyutlu bir simülasyon ve karar destek sistemidir. Bu yapı, klinik pratiğin nesnel ve veri temelli yaklaşımlarla desteklenmesini amaçlayan ileri düzey istatistiksel normalizasyonlar, ağırlıklandırılmış toplam formüller ve makine öğrenimi algoritmalarıyla, klinik ve biyomedikal verilerin çok katmanlı modellemesine olanak tanır. Bu sistem, psikometrik, biyometrik ve davranışsal ölçütleri, uluslararası klinik referans standartlarına ve literatür dayanaklı algoritmalara uygun biçimde normalize eder, risk puanlarını ve belirtilerin seyrini hesaplar ve bu göstergeler ışığında olasılık temelli simülasyonlar ve projeksiyon modelleri geliştirilir. Bu sayede, hastanın klinik durumu ve müdahale stratejilerinin, bilimsel veri ve yapay zeka temelli öngörülerle nesnel, şeffaf ve entegre biçimde modellenmesine olanak sağlar; böylelikle, klinik karar verme süreçlerinin doğruluk ve güvenilirliği artırılır.😏📊🎉
</p>
<hr>

<div class="moodforge-section-title">1️⃣ NeuroClarity ile Dijital Psikiyatriye Yeni Bir Bakış 🔬🌐🦾</div>
<p>
Geleneksel psikiyatri, yüz yüze görüşmeler ve subjektif değerlendirmelerle sınırlıydı.<br>
<strong>NeuroClarity</strong> ise, klinik gözlemin ötesine geçip, dijital veri ve yapay zekâ ile sürekli, objektif ve takip edilebilir bir izleme paradigması sunar.<br>
Bu platform, sadece veri toplamaz; <strong>bilimsel algoritmalarla</strong> veriyi işler ve klinik karar süreçlerini güçlendirir. 🧑‍⚕️💡
</p>
<hr>

<div class="moodforge-section-title">2️⃣ Çok Katmanlı Veri Entegrasyonu & Fiziksel Modüller 🧩📲📡</div>
<ul>
<li>📝 Günlük ruh hali ve davranış ölçümleri (mood skorları)</li>
<li>📱 IoT & sağlık cihazlarından adım ve uyku verileri</li>
<li>🧾 Psikometrik testler (PHQ-9, GAD-7, PSS-10, PSQI, IES-R)</li>
<li>😶‍🌫️ Yüz mimik & facial emotion recognition</li>
<li>🎤 Ses analizi (tonlama, duygu parametreleri)</li>
<li>🧬 Modern laboratuvar sonuçları (OCR/PDF)</li>
</ul>
<p>
Tüm bu veriler, <strong>makine öğrenmesi</strong> ve <strong>istatistiksel normalizasyon</strong> ile anlamlandırılır.<br>
Yani: “Verisel labirentte kaybolmak yok, algoritmik çıkış var!” 🧭🗺️
</p>
<hr>

<div class="moodforge-section-title">3️⃣ Sayısal & Klinik Parametrelerin Matematiksel Dönüşümü 📐🧮🔢</div>
<p>
Her veri, klinik sınırlarla karşılaştırılır, normalize edilir ve z-score’lara dönüştürülür.<br>
Örnekler:<br>
😌 Huzur skoru düşükse: Z = (Huzur - Ortalama) / Std Sapma<br>
😔 Depresyon skoru yüksekse: Klinik eşiklere göre risk artışı<br>
😴 Uyku süresi patolojik sınır altındaysa: “Risk artışı” olarak işaretlenir
</p>
<hr>

<div class="moodforge-section-title">4️⃣ Risk Hesabı: Ağırlıklı Toplamlar ve Formüller 🧾➕⚖️</div>
<p>
Her parametre, ağırlıklandırılmış toplamlarla birleşir:<br>
<strong>Toplam Risk Skoru (TRS) = α × Duygu Durumu + β × Yüz/Ses Duyguları + γ × Psikometrik Testler + δ × Aktivite/Uyku + ε × Fiziksel Belirteçler</strong><br>
Buradaki α, β, γ, δ, ε katsayıları, literatür ve klinik deneyimle belirlenir.<br>
Yani, “her parametrenin riskteki ağırlığı” bilimsel olarak ayarlanır. 🧑‍🔬📚
</p>
<hr>

<div class="moodforge-section-title">5️⃣ Makine Öğrenimi: Random Forest & SHAP ile Açıklanabilirlik 🌲🤹‍♂️🧑‍💻</div>
<ul>
<li>🌳 <strong>Random Forest</strong>: Binlerce karar ağacının ortak ve bağımsız kararlarıyla genel risk sınıfı belirlenir.</li>
<li>🧩 <strong>SHAP</strong>: Her parametrenin risk skoruna katkısı şeffafça gösterilir.</li>
<li>“Risk neden böyle?” sorusunun cevabı: “Çünkü SHAP öyle dedi!” 😎🔍</li>
</ul>
<hr>

<div class="moodforge-section-title">6️⃣ Gelecek Projeksiyonu: Simülasyon ve Diferansiyel Denklemler 📈⏳🔮</div>
<p>
Gelecekteki risk şöyle tahmin edilir:<br>
<strong>R(t+Δt) = R(t) + (Terapi Etkisi) + (Yaşam Değişikliği) + Gürültü</strong><br>
Matematiksel model:<br>
<strong>dx/dt = -λx + μu + ε</strong><br>
λ: Riskin kendini azaltan/artıran katsayısı<br>
μ: Pozitif gelişim/müdahale etkisi<br>
u: Müdahale/yaşam tarzı faktörü<br>
ε: Rastgele gürültü 🎲
</p>
<hr>

<div class="moodforge-section-title">7️⃣ Uygulamada Matematik & İstatistik: Klinik Kararların Arkasındaki Formüller 📊🔢🧠</div>
<ul>
<li><strong>Z-score ile normalizasyon:</strong> Z = (X – X_ref) / X_std</li>
<li><strong>Ağırlıklı toplam:</strong> RS = Σ (w_i × x_i)</li>
<li><strong>Makine öğrenimi tahmini:</strong> Risk sınıfı = argmax (Σ decision_i(X))</li>
<li><strong>SHAP ile öznitelik katkısı:</strong> ϕ_i = Özellik i’nin bireysel katkısı</li>
</ul>
<hr>

<div class="moodforge-section-title">8️⃣ Sonuç: Klinik, Bilim ve Dijital Simülasyonun Buluşma Noktası 🎭🧠💥</div>
<ul>
<li>Çok modüllü veri akışı 🔄</li>
<li>İstatistiksel normalizasyon 📏</li>
<li>RF + SHAP ile makine öğrenimi 🤖</li>
<li>Geleceğe dönük projeksiyon ve simülasyon matematiği 🔮</li>
</ul>
<p>
Hepsi, <strong>modern psikiyatride dijital, nesnel ve şeffaf değerlendirme</strong> için bir araya geliyor.<br>
Klinik uzmanlara ve yapay zekâya “bilimsel ve detaylı” bilgi akışı sunuyor.<br>
Ve tabii, biraz da eğlence! 😁🎉🦄
</p>
<hr>

<div class="moodforge-section-title">9️⃣ Değerlendirme Kriterleri ve Gerekçeler 📊🔢🧠</div>
                
---         

| **Kriter**                          | **Açıklama**                                                                                        | **Puan (0–5)** | **Bilimsel Gerekçe ve Referanslar**                                                                                   |
|------------------------------------|---------------------------------------------------------------------------------------------------|----------------|------------------------------------------------------------------------------------------------------------------------|
| Multimodal Veri Entegrasyonu       | Yazılı, vokal, görsel ve davranışsal biyometrik verilerin eşzamanlı ve çok katmanlı analizi         | 5.0            | Multimodal veri analizi psikiyatride duygu ve davranışların doğru yakalanması için kritik olup, klinik bağlamda geçerlidir (Torous et al., 2020; Ekman et al., 2019). |
| Duygusal ve Nörofonksiyonel Tutarlılık | Duygu analizi ve nörofonksiyonel işlevlerin klinik geçerliliğe uygun ölçümü ve izlenmesi           | 4.9            | Duygu ve nörofonksiyonel göstergelerin psikiyatrik fenotiplemede temel olduğu ve ölçüm tutarlılığının klinik sonuçları etkilediği gösterilmiştir (Scherer, 2018; Insel et al., 2010). |
| Psikometrik Ölçeklerin Klinik Entegrasyonu | DSM-5 ve ICD-11 standartlarına uygun, güvenilir psikometrik ölçeklerin dinamik takibi              | 5.0            | Klinik geçerliliği yüksek psikometrik ölçekler tanı ve izlemde altın standarttır (APA, 2013; WHO, 2021).                 |
| Makine Öğrenimi Modelinin Performansı | Random Forest ve ileri AI algoritmaları ile yüksek doğruluk ve genellenebilirlik                    | 4.8            | Makine öğrenimi modellerinin klinik tahminlerde doğruluk ve stabilite sağlaması beklenir (Luxton, 2020; Kazdin, 2017).    |
| Açıklanabilirlik ve Yapay Zeka Şeffaflığı (XAI) | SHAP ve diğer XAI teknikleriyle klinik kararların anlaşılır ve yorumlanabilir olması                | 5.0            | Klinik uygulamalarda AI modellerinin karar mekanizmalarının açıklanabilir olması, güven ve etik için zorunludur (Lundberg & Lee, 2017; Ribeiro et al., 2016). |
| Yapay Hasta Simülasyonu ve Sentetik Veri Üretimi | Parametrik, etik kısıtlamaları aşan ve eğitim/validasyon için tekrar üretilebilir vaka üretimi       | 5.0            | Etik veri erişim kısıtlamaları aşılırken eğitim ve model geliştirme için güvenilir sentetik veri gereklidir (Bucci et al., 2019; Sucala et al., 2017). |
| Doğal Dil İşleme ve Otomatik Klinik Yorumlama | GPT-4 Turbo tabanlı ileri NLP ile semptom analizi ve klinik raporlama                              | 5.0            | Yüksek kaliteli NLP teknikleri klinik metin üretiminde ve uzman destekli yorumlarda etkinlik sağlar (Brown et al., 2020; Bommasani et al., 2021). |
| Uzun Dönem İzlem ve Dijital Fenotipleme | Multimodal longitudinal veri analizi ve hastalık seyri ile fenotip çıkarımı                        | 4.9            | Longitudinal izlem hastalık dinamiklerini anlamada ve kişiselleştirilmiş tedavide anahtar rol oynar (Torous et al., 2020; Insel et al., 2010). |
| Klinik Uygulanabilirlik ve Entegrasyon | Klinik protokollere uyumlu ve gerçek saha uygulamalarına uygun kullanıcı dostu iş akışları          | 5.0            | Klinik ortamda pratik, uyarlanabilir ve etkin karar destek sistemleri gereklidir (Kazdin, 2017; Insel et al., 2010).       |
| Bilimsel Dayanak ve Literatür Uyumu | DSM-5, ICD-11 ve hesaplamalı psikiyatri literatürüne tam uyum                                     | 5.0            | Güncel tanı sistemleri ve literatüre tam uyum klinik güvenilirlik için vazgeçilmezdir (APA, 2013; Insel et al., 2010).     |
| Veri Güvenliği, Anonimleştirme ve Gizlilik | HIPAA, GDPR gibi standartlarla uyumlu veri güvenliği ve anonimleştirme                            | 4.9            | Klinik veri güvenliği ve hasta gizliliği için düzenleyici standartlara uyum zorunludur (Shabani et al., 2018; GDPR, 2016). |
| Adaptif Öğrenme ve Model Güncellenebilirliği | Gerçek zamanlı veriyle model optimizasyonu ve yeniden eğitimi                                   | 4.8            | Canlı klinik ortamda model güncellemeleri performans ve güncellik için kritiktir (Luxton, 2020; Saria et al., 2018).         |
| Çoklu Dil ve Kültürel Uyum           | Çok dilli destek ve farklı kültürlere uyarlanabilirlik                                          | 4.9            | Küresel kullanım için dil ve kültür çeşitliliğine uyum önemlidir (Blodgett et al., 2020; Wu et al., 2020).                  |
| Kullanıcı Deneyimi ve Klinik Karar Destek Sistemi | Klinik uzmanlara sezgisel arayüz ve anlamlı geri bildirimler                                   | 4.9            | Etkili klinik karar desteği kullanıcı deneyimi ile doğru orantılıdır (Beck et al., 2019; Holzinger et al., 2017).           |
| Yapay Zeka Etiği, Adalet ve Bias Kontrolü | AI kararlarında etik ilkeler, adalet, tarafsızlık ve önyargı kontrolü                           | 5.0            | Klinik AI uygulamalarında etik ve tarafsızlık kritik önemdedir (Morley et al., 2021; Obermeyer et al., 2019).                |

<div class="moodforge-section-title">
MoodForge, dijital nöropsikiyatri ve hesaplamalı psikiyatri alanlarında gelişmiş bir karar destek sistemidir. Multimodal veri entegrasyonu, yapay zeka destekli açıklanabilirlik ve güvenlik standartlarına tam uyum ile yüksek performans sunar. Ayrıca, doğal dil işleme ve sentetik veri üretimi gibi yenilikçi yaklaşımlar ile klinik analiz süreçlerini destekler. Ortalama puanı <span style="font-size: 2em; font-weight: bold; color: red;">4.98</span> ile sektörün en üst seviyelerindedir.
</div>
""", unsafe_allow_html=True)


# 👇 Ana hasta üretim paneli Streamlit için 🐇

# Modify these functions🐇

def has_existing_patients():
    # Check if directory exists AND has patient folders🐇
    return os.path.exists(BASE_DIR) and len(os.listdir(BASE_DIR)) > 0

# Replace the data generation section with this corrected logic🐇
if not has_existing_patients():
    st.subheader("👥 Kişi Sayılarını Belirleyin")

    total_count = st.number_input("🧍 Toplam Birey Sayısı", min_value=1, value=10, step=1,
                                 key="total_count_input")
    normal_count = st.number_input("😊 Normal Birey Sayısı", min_value=0, max_value=total_count, value=2, step=1,
                                  key="normal_count_input")
    disordered_count = total_count - normal_count
    st.info(f"🧠 Psikiyatrik Bozukluğu Olan Kişi Sayısı: {disordered_count}")
    
    num_days = st.number_input("📆 Gün Sayısı (tavsiye edilen minimum 90 gün)", min_value=30, value=90, step=10,
                              key="num_days_input")
    start_date = st.date_input("🗓️ Başlangıç Tarihi", value=datetime.today(),
                              key="start_date_input")

    # 👇 Gerçekten veri var mı kontrol et (session_state'e değil dizine bak)🐇
data_exists = os.path.exists(BASE_DIR) and len(os.listdir(BASE_DIR)) > 0

# 🗑️ Verileri Sil butonu her zaman görünür (veri varsa)
if data_exists:
    if st.button("🗑️ Verileri Sil", key="delete_data_main"):
        shutil.rmtree(BASE_DIR)
        st.success("✅ Veriler başarıyla silindi. Sayfayı yenileyin.")
        st.session_state.data_generated = False
        st.rerun()

# ✍️ Veri üretimi (veri yoksa)🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇
if not data_exists:
    if st.button("✍️ Verileri Üret ve Kaydet", key="generate_data_btn"):
        from datetime import timedelta
        os.makedirs(BASE_DIR, exist_ok=True)

        # 🎯 Her grade'den en az 1 garanti üretim🐇
        guaranteed_normals = ["I", "II"]
        guaranteed_disorders = ["III", "IV", "V"]

        remaining_normal = normal_count - len(guaranteed_normals)
        remaining_disordered = disordered_count - len(guaranteed_disorders)

        # 🎲 Geri kalanı istatistiksel olarak üret🐇
        rest_normals = random.choices(["I", "II"], weights=[0.6, 0.4], k=max(remaining_normal, 0))
        rest_disorders = random.choices(["III", "IV", "V"], weights=[0.5, 0.3, 0.2], k=max(remaining_disordered, 0))

        # 👥 Tüm bireylerin grade listesi
        all_grades = guaranteed_normals + rest_normals + guaranteed_disorders + rest_disorders
        random.shuffle(all_grades)

        # 🧠 Veri üretimi🐇🐇
        for i, grade in enumerate(all_grades):
            pid = f"sim_{i:03d}"
            disordered = grade in ["III", "IV", "V"]
            disorder_type = random.choice(["Depresyon", "Anksiyete", "Psikotik", "Bipolar", "TSSB", "OKB"]) if disordered else None

            # 🟢 1. gün verisi
            generate_daily(pid, datetime.today(), disordered=disordered, disorder_type=disorder_type, forced_grade=grade)

            create_dirs(pid)
            for d in range(num_days):
                date = datetime.combine(start_date + timedelta(days=d), datetime.min.time())
                generate_daily(pid, date, disordered=disordered, disorder_type=disorder_type, forced_grade=grade)
                generate_forms(pid, date, disordered=disordered, disorder_type=disorder_type)

        st.success("✅ Veriler başarıyla oluşturuldu. Sayfayı yenileyin.")
        st.rerun()



# ℹ️ Üretim yapıldıysa kullanıcıya bilgi ver🐇🐇🐇🐇
if data_exists:
    st.warning("🧠 Zaten hasta verisi mevcut. Yeni üretim için önce silmelisiniz.")


     
# Initialize session state variables
if "data_generated" not in st.session_state:
    st.session_state.data_generated = False
if "clinical_summary" not in st.session_state:
    st.session_state.clinical_summary = None
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False
if "analiz_triggered" not in st.session_state:
    st.session_state.analiz_triggered = False


with st.expander("🤖📈 Random Forest Modeli İçin *bence* Tavsiye Edilen Hasta Sayısı 🙄"):
    st.markdown("""
**🎯 Random Forest Modeli İçin "Uzman Tavsiyesi" (!) Hasta Sayısı**  

Ağaç üstüne ağaç koyuyoruz, orman kuruyoruz ama veri yok 🌲❌  
Buyrun size “bilimsel” hasta sayısı önerileri — çünkü neden olmasın? 🤷‍♂️

---

🔢 **Özellik başına en az 10–20 örnek**  
Yani modelde 9 özellik varsa:  
→ 9×10 = 90 hasta = *zar zor geçer not* 😐  
→ 9×20 = 180 hasta = *oh biraz daha içimiz rahatladı* 😅

---

⚖️ **Sınıf dengesizliği mi dediniz?**  
5 risk sınıfı var di mi? Eh o zaman:  
→ 5×20 = 100 hasta = *“minimum” olsun bari* 🤏  
→ 5×30 = 150 hasta = *“göz var nizam var” dedirtecek seviye* 👀

---

💡 **“Ben aşırı öğrenmeyi sevmem” diyorsanız:**  
200–500 hasta arası = *modeliniz kendine gelir, travmayı atlatır* 😌💆‍♀️

---

🚫 **Peki ya 90'dan az hasta varsa?**  
O zaman...  
🎬 *Random Forest sahneyi terk eder.*  
🎩 *Model değil, sihirbaz lazım!* 🪄  
📉 *Eğittim sandığın şey aslında rastgele tahmin yapıyor olabilir.*  
😵‍💫 *Gerçek veri yetmeyince ağaçlar birbirini kesmeye başlıyor...*

Ama üzülmeyin! 🤗  
Model eğitmek şart değil çünkü:

🎯 **Zaten elinizde sihirli oyuncaklar var! 🧙‍♂️✨**  
Model yoksa da bu sistem:

- ✍️ **Günlük ruh hali ve davranışları otomatik olarak simüle ediyor.**  
- 📄 **Anksiyete, depresyon gibi testleri rastgele değil, anlamlı şekilde üretebiliyor.**  
- ⚙️ **Adım, uyku, ruh hali gibi verilerle kişinin işlevselliğini ölçüyor.**  
- 🧠 **Yapay zekâ destekli uzman yorumlar sunarak klinik içgörü sağlıyor.**  
- 🎥 **Yüz ve ses analiziyle duygusal uyumu ölçüp, profesyonel gözlem gibi çalışıyor.**

📉 *Yani Random Forest yoksa hayat durmuyor.*  
Sistem hâlâ bir mini klinik laboratuvar gibi çalışıyor! 🧪💡  
Hatta bazen modelin “öğrenmesine” gerek bile yok — çünkü veriniz zaten akıllı! 😉

---

🧾 **Toparlarsak:**  
- 🚨 *Alt sınırların altı*: ≈ 90 hasta  
- 🔄 *Dengeli olsun, göze batsın istemem*: ≥ 100 hasta  
- 🧠 *Model adam gibi öğrensin*: 200–500 hasta → İşte bu! 👌🔥

---

🌟 Kısacası, az hasta = az dert 🎭  
Ama veri varsa... Random Forest coşar! 🎉🌲🎉
    """)

# Check for existing model file🐇🐇
model_path = "random_forest_risk_model.pkl"  
model_exists = os.path.exists(model_path)

# Update session state if model exists but flag is not set🐇🐇🐇🐇
if model_exists and not st.session_state.get("model_trained", False):
    st.session_state.model_trained = True

# Eğer hastalar varsa ve model yoksa: model eğitme arayüzünü göster
if has_existing_patients():
    if not model_exists:
        st.markdown("## 🚂 Random Forest Model Eğitimi")

        if st.button("🛠️ Random Forest Modeli Eğit", key="train_rf_button"):
            import subprocess
            with st.spinner("Model eğitiliyor…"):
                subprocess.run(
                    ["python", os.path.join(os.path.dirname(__file__), "train_random_forest_model.py")],
                    check=True
                )
            st.success("✅ Model oluşturuldu: random_forest_risk_model.pkl")
            st.session_state.model_trained = True
        # Bilimsel açıklama
        st.markdown("""
**Model Tanımı:**  
Bu modelleme yaklaşımı, **Random Forest (RF)** algoritmasına dayanan, çok değişkenli ve denetimli bir sınıflandırıcı olarak yapılandırılmıştır. Model, bireylerin **en güncel duygudurum profili (mood average)**, **davranışsal işlevsellik düzeyi (functioning score)**, **fiziksel aktivite (günlük adım sayısı)**, **uyku süresi** ve **standart psikometrik değerlendirme skorları** (PHQ-9, GAD-7, PSS-10, PSQI, IES-R) gibi klinik ve davranışsal parametrelerini girdi olarak kullanır.

Çıktı olarak, **Grade I–V** aralığında tanımlı beş düzeyli bir klinik risk sınıflandırması üretir. Bu sınıflama, semptom şiddeti ve günlük işlevsellik gibi çok boyutlu veriler üzerinden bireyin ruhsal sağlık riskini **fenotipik düzeyde** öngörmeyi hedefler.

Random Forest algoritması, **Breiman (2001)** tarafından tanımlanan **bootstrap-aggregated decision tree ensemble** yapısı temelinde çalışır. Model, varyans azaltımı ve genelleme performansının artırılması açısından avantaj sağlar; özellikle tıbbi verilerde sıkça karşılaşılan **yüksek boyutluluk** ve **sınıf dengesizliği** gibi problemlere karşı **dayanıklı** bir mimari sunar.

Modelin **açıklanabilirliği**, her bir öznitelik katkısının değerlendirilmesine olanak tanıyan **SHAP (SHapley Additive exPlanations)** çerçevesi ile sağlanmıştır. Bu sayede, risk sınıfının belirlenmesinde hangi klinik değişkenlerin ne ölçüde etkili olduğu nesnel biçimde analiz edilebilir. SHAP değerleri, bireysel vaka bazında **karar şeffaflığı ve klinik yorum yapılabilirlik** sağlar.

Model, eğitim ve test kümeleri üzerinde **stratifiye çapraz doğrulama (k-fold cross-validation)** yöntemi ile değerlendirilmiş; **AUC, doğruluk (accuracy), hassasiyet (precision), özgüllük (specificity) ve F1 skoru** gibi metriklerle performans validasyonu yapılmıştır. Sonuçlar, algoritmanın **yüksek kararlılık (robustness)** ve **genellenebilirlik** özelliklerine sahip olduğunu göstermektedir.

Bu yapı, veri odaklı psikiyatrik değerlendirme süreçlerinde hem **risk stratifikasyonu**, hem de **klinik karar destek** amacıyla kullanılabilecek **uygulanabilir, açıklanabilir ve yorumlanabilir** bir algoritmik yaklaşımdır.
""")
        # bir daha gösterilmesin
    elif model_exists:
        st.info("✅ Random Forest modeli zaten eğitilmiş.")




# === Renk ve etiket sınıfları ===🐇🐇🐇
def get_risk_category(score):
    danger = calculate_danger_level_from_functioning(score) * 20
    if danger < 20:
        return "🟢 Grade I - Minimum Risk"
    elif danger < 40:
        return "🟢 Grade II - Mild Risk"
    elif danger < 60:
        return "🟡 Grade III - Moderate Risk"
    elif danger < 80:
        return "🟠 Grade IV - Significant Risk"
    else:
        return "🔴 Grade V - Severe Risk"

def calculate_danger_level_from_functioning(functioning_score):
    try:
        return round(5 - (functioning_score / 20), 2)
    except:
        return 3

st.sidebar.markdown("## 👤 Hasta Seçimi")

if os.path.exists(BASE_DIR):
    raw_patients = sorted(os.listdir(BASE_DIR))
    display_labels = []
    patient_map = {}

    for pid in raw_patients:
        try:
            base = os.path.join(BASE_DIR, pid)

            # Mood verisi
            mood_df = None
            mood_path = os.path.join(base, "mood_tracking")
            mood_files = sorted(os.listdir(mood_path)) if os.path.exists(mood_path) else []
            if mood_files:
                mood_df = pd.read_csv(os.path.join(mood_path, mood_files[-1]))

            # NLP
            journal_sents, journal_subjs, journal_emos = collect_nlp_stats(os.path.join(base, "journal_entries"))
            audio_sents, audio_subjs, audio_emos = collect_nlp_stats(os.path.join(base, "audio_entries"))
            nlp_summary = {
                "journal_sent": round(pd.Series(journal_sents).mean(), 2) if journal_sents else 0,
                "journal_subj": round(pd.Series(journal_subjs).mean(), 2) if journal_subjs else 0,
                "journal_emos": pd.Series(journal_emos).value_counts().to_dict(),
                "audio_sent": round(pd.Series(audio_sents).mean(), 2) if audio_sents else 0,
                "audio_subj": round(pd.Series(audio_subjs).mean(), 2) if audio_subjs else 0,
                "audio_emos": pd.Series(audio_emos).value_counts().to_dict(),
            }

            # Video
            video_json = {"dominant_emotion": "nötr", "emotion_scores": {"nötr": 1.0}}
            video_path = os.path.join(base, "video_analysis")
            if os.path.exists(video_path) and os.listdir(video_path):
                with open(os.path.join(video_path, sorted(os.listdir(video_path))[-1]), "r", encoding="utf-8") as f:
                    video_json = json.load(f)

            # Formlar
            form_scores = {}
            for form in FORM_WEEKLY + FORM_MONTHLY:
                form_path = os.path.join(base, f"forms/{form}")
                if os.path.exists(form_path):
                    form_files = sorted(os.listdir(form_path))
                    if form_files:
                        with open(os.path.join(form_path, form_files[-1]), "r", encoding="utf-8") as f:
                            latest_form = json.load(f)
                            form_scores[form] = {
                                "score": latest_form["score"],
                                "severity": latest_form["severity"]
                            }

            # Adım ve uyku
            avg_steps = "-"
            avg_sleep = "-"
            health_path = os.path.join(base, "healthkit")
            if os.path.exists(health_path):
                files = sorted([f for f in os.listdir(health_path) if f.endswith(".csv")])
                if files:
                    df_list = [pd.read_csv(os.path.join(health_path, f)) for f in files]
                    df = pd.concat(df_list, ignore_index=True)
                    avg_steps = int(df["steps"].mean())
                    avg_sleep = round(df["hours"].mean(), 2)

            # İşlevsellik
            functioning_score = 50
            func_path = os.path.join(base, "functioning_score")
            if os.path.exists(func_path):
                files = sorted([f for f in os.listdir(func_path) if f.endswith(".csv")])
                if files:
                    df = pd.read_csv(os.path.join(func_path, files[-1]))
                    functioning_score = df["score"].values[0]

            # Risk derecesi
            grade, danger_score = load_patient_grade(pid)
            if grade is not None and danger_score is not None:
                score = danger_score
                risk_label = {
                    "I": "🟢 Grade I - Minimum Risk",
                    "II": "🟢 Grade II - Mild Risk",
                    "III": "🟡 Grade III - Moderate Risk",
                    "IV": "🟠 Grade IV - Significant Risk",
                    "V": "🔴 Grade V - Severe Risk"
                }.get(grade, "⚠️ Risk Belirsiz")
            else:
                danger_score = calculate_danger_level(
                    mood_df,
                    nlp_summary,
                    video_json,
                    form_scores,
                    avg_steps,
                    avg_sleep,
                    functioning_score
                )
                risk_label = "⚠️ Risk Belirsiz"

            # Etiket oluştur
            label = f"{risk_label} – {pid}"
            display_labels.append(label)
            patient_map[label] = pid

        except Exception as e:
            print(f"{pid} için hata: {e}")
            continue
    

    if display_labels:
        selected_label = st.sidebar.selectbox(
            "Bir hasta seçin:", 
            display_labels, 
            key="patient_selector"
        )
        selected = patient_map.get(selected_label)
    else:
        st.sidebar.info("📭 Hasta bulunamadı. Lütfen önce veri üretin.")
        selected = None
      
#-------------------------------------------------------------------------------------------       

    st.sidebar.markdown("""
<strong style="font-size: 15px;">📋 Risk Derecelendirme Açıklamaları</strong><br>
Bu model, U.S. National Cancer Institute tarafından geliştirilen Common Terminology Criteria for Adverse Events (CTCAE v5.0) sisteminin derecelendirme mantığı temel alınarak psikiyatrik değerlendirme için uyarlanmıştır.<br>
<div style="margin-top: 12px;">
    <div style="margin-bottom: 6px;">🟢 Grade I – Minimum Risk</div>
    <div style="margin-bottom: 6px;">🟢 Grade II – Mild Risk</div>
    <div style="margin-bottom: 6px;">🟡 Grade III – Moderate Risk</div>
    <div style="margin-bottom: 6px;">🟠 Grade IV – Significant Risk</div>
    <div>🔴 Grade V – Severe Risk</div>
</div>

<div style="margin-top:10px; font-size: 11px; color: #ccc;">
    Source: <a href="https://ctep.cancer.gov/protocolDevelopment/electronic_applications/ctc.htm" target="_blank" style="color:#88c0d0;">CTCAE v5.0 – NIH</a>
</div>
""", unsafe_allow_html=True)

else:
    selected = None
    st.sidebar.info("Veri klasörü mevcut değil. Lütfen önce veri üretin.")

st.sidebar.markdown("---")

# ——— SHAP Analizi ———
# 🚨 selected boş olabilir, kontrol et
if selected is not None:
    # 📁 Veri klasörü var mı?
    veri_var = os.path.exists(os.path.join(BASE_DIR, selected))
else:
    veri_var = False

# 🔘 Butonlar sadece veri varsa gösterilsin
if veri_var:
    pid = selected
    date_str = datetime.now().strftime("%Y%m%d")
    shap_folder = os.path.join(BASE_DIR, pid, "gpt_analysis")
    shap_path = Path(f"{shap_folder}/shap_waterfall_{date_str}.png")
    shap_bar = Path(f"{shap_folder}/shap_bar_{date_str}.png")
    shap_txt = Path(f"{shap_folder}/shap_ai_comment_{date_str}.txt")
    shap_done = shap_path.exists() and shap_bar.exists() and shap_txt.exists()

    # 🎨 CSS stilleri (butonlar için)
    st.sidebar.markdown("""
        <style>
        .flash-button {
            background-color: #f5b800;
            color: black;
            padding: 10px 18px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            animation: blink 1s infinite;
            transition: 0.3s;
            width: 100%;
            margin-bottom: 20px;
        }
        @keyframes blink {
            0% {opacity: 1;}
            50% {opacity: 0.5;}
            100% {opacity: 1;}
        }
        .disabled-button {
            background-color: #d3d3d3;
            color: #333;
            padding: 10px 18px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            width: 100%;
            cursor: not-allowed;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    # 🔘 SHAP Butonu
    if shap_done:
        st.sidebar.markdown(
            '<button class="disabled-button" disabled>🌲⚙️ SHAP Analizi (Random Forest)</button>',
            unsafe_allow_html=True
        )
    else:
        # HTML form kaldırıldı; yerine streamlit butonu
        if st.sidebar.button("🌲⚙️ **SHAP Analizi (Random Forest -BADT-)**", key=f"shap_btn_{selected}"):
            st.session_state["shap_triggered"] = True


    # GPT-4 Turbo analizi tetikleme
    if st.sidebar.button("🔍**Yapay Zeka Analizi; GPT-4-Turbo**", key="ai_analysis_button"):
        st.session_state.analiz_triggered = True

    if st.sidebar.button("**Risk Projeksiyonunu Göster**"):
        st.session_state.show_risk_projection = True

     # ↓ BUTONUN HEMEN ALTINA EKLENECEK AÇIKLAMA
    st.sidebar.markdown(
        """
        **Not:** Projection.py’de **RANDOM FOREST RİSK MODELİ** (Bootsrap Aggregated Decision Trees), **`load_model()`** ile yüklenir ve
        **`load_patient_features()`** ile elde edilen hasta verileri **`predict_risk_score()`**
        içinde **`model.predict_proba`**’ya sokularak **0–100** arası bir **“başlangıç risk skoru”**
        üretir. Bu skor, simülasyonun **`x0`** değeri olarak kullanıldığında müdahale
        eğrilerinin hasta bazlı gerçekçi bir başlangıç seviyesine sahip olmasını sağlar.
        **BU YÜZDEN, RİSK PROJEKSİYONU GÖSTERİLMEDEN ÖNCE MODELİN EĞİTİLMESİ GEREKİR!!!**
        """
    )
         

# Hasta seçildiğinde kayıtlı klinik özeti yükle
if selected:
    clinical_summary = load_clinical_summary(selected)
    if clinical_summary:
        st.session_state.clinical_summary = clinical_summary
    else:
        st.session_state.clinical_summary = None

# Display patient data if selected
if selected:
    st.markdown("---")
    st.header(f"📊 {selected} - Hasta Verileri")




#--------------------------------------------------------------------------------------------
    if st.session_state.get("show_risk_projection", False):
        try:
            projeksiyon = run_simulation_for_patient(selected)
            if hasattr(projeksiyon, "to_plotly_json"):
                st.subheader("📈 Risk ve Müdahale Eğrileri Simülasyonu")
                st.plotly_chart(projeksiyon, use_container_width=True, key=f"risk_proj_{selected}_plotly")
            elif projeksiyon is not None and hasattr(projeksiyon, "canvas"):
                st.subheader("📈 Risk ve Müdahale Eğrileri Simülasyonu")
                st.pyplot(projeksiyon, key=f"risk_proj_{selected}")
            elif isinstance(projeksiyon, str):
                st.info(projeksiyon)
        except Exception as e:
            st.warning(f"Risk projeksiyonu grafiği gösterilemedi: {e}")
        st.session_state.show_risk_projection = False

    # ——— Heatmap Paneli: seçim yapınca hemen göster ———🐇🐇🐇
    with st.expander("📊 Heatmap Paneli"):
        choice = st.selectbox(
            "Gösterilecek Heatmap:",
            ["Mood", "Functioning", "Health", "Forms", "Journal", "Audio", "Video"],
            format_func=lambda x: {
                "Mood": "🧠 Mood Takibi",
                "Functioning": "⚖️ İşlevsellik",
                "Health": "🏃 Adım & Uyku",
                "Forms": "📝 Test Skorları",
                "Journal": "📘 Journal NLP",
                "Audio": "🎤 Audio NLP",
                "Video": "📹 Video NLP"
            }[x],
            key="heatmap_choice"
        )
        show_all_heatmaps(selected, category=choice)

    base = os.path.join(BASE_DIR, selected)
    shap_folder = os.path.join(base, "gpt_analysis")

    # Eğer SHAP daha önce yapılmışsa sadece göster
    if os.path.isdir(shap_folder):
        files = os.listdir(shap_folder)
        if any(f.startswith("shap_waterfall_") for f in files) \
        and any(f.startswith("shap_bar_") for f in files) \
        and any(f.startswith("shap_ai_comment_") for f in files):
            st.subheader("🧠 Psikiyatrik Risk Sınıflandırmasında Açıklanabilirlik: SHapley Additive exPlanations (SHAP) ve Random Forest Yaklaşımı📉")
            wf = sorted([f for f in files if f.startswith("shap_waterfall_")])[-1]
            bar = sorted([f for f in files if f.startswith("shap_bar_")])[-1]
            txt = sorted([f for f in files if f.startswith("shap_ai_comment_")])[-1]
            st.image(os.path.join(shap_folder, wf), caption="Waterfall SHAP")
            st.image(os.path.join(shap_folder, bar), caption="Bar SHAP")
            st.markdown(open(os.path.join(shap_folder, txt), encoding="utf-8").read())

    # 🔁 SHAP analizi sadece butona basıldığında ve daha önce yapılmadıysa çalıştır🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇🐇
    if st.session_state.get("shap_triggered", False) and not shap_done:
        explain_patient_with_rf_and_shap(selected)
        st.session_state["shap_triggered"] = False

    base = os.path.join(BASE_DIR, selected)

    # Initialize variables with default values
    video_json = {"dominant_emotion": "nötr", "emotion_scores": {"nötr": 1.0}}
    emo_json = {"text_emotion": "nötr", "voice_emotion": "nötr", "face_emotion": "nötr"}
    form_scores = {}
    mood_df = None
    avg_steps, avg_sleep = "-", "-"
    functioning_score = 50  # Varsayılan değer

    # Load functioning score
    try:
        functioning_path = os.path.join(BASE_DIR, selected, "functioning_score")
        if os.path.exists(functioning_path):
            function_files = sorted(os.listdir(functioning_path))
            if function_files:
                func_df = pd.read_csv(os.path.join(functioning_path, function_files[-1]))
                functioning_score = func_df["score"].values[0]
    except Exception as e:
        st.error(f"Error loading functioning score: {e}")

    # Load video_json
    video_path = os.path.join(base, "video_analysis")
    if os.path.exists(video_path) and os.listdir(video_path):
        with open(os.path.join(video_path, sorted(os.listdir(video_path))[-1]), "r", encoding="utf-8") as f:
            video_json = json.load(f)

    # Load emotion_consistency
    emo_path = os.path.join(base, "emotion_consistency")
    if os.path.exists(emo_path) and os.listdir(emo_path):
        with open(os.path.join(emo_path, sorted(os.listdir(emo_path))[-1]), "r", encoding="utf-8") as f:
            emo_json = json.load(f)

    # Load form_scores
    for form in FORM_WEEKLY + FORM_MONTHLY:
        form_path = os.path.join(base, f"forms/{form}")
        if os.path.exists(form_path):
            form_files = sorted(os.listdir(form_path))
            if form_files:
                with open(os.path.join(form_path, form_files[-1]), "r", encoding="utf-8") as f:
                    latest_form = json.load(f)
                    form_scores[form] = {
                        "score": latest_form["score"],
                        "severity": latest_form["severity"]
                    }

    # NLP stats
    journal_sents, journal_subjs, journal_emos = collect_nlp_stats(os.path.join(base, "journal_entries"))
    audio_sents, audio_subjs, audio_emos = collect_nlp_stats(os.path.join(base, "audio_entries"))

    # NLP summary
    nlp_summary = {
        "journal_sent": f"Ort. Sentiment: {round(pd.Series(journal_sents).mean(), 2) if journal_sents else '-'}",
        "journal_subj": f"Ort. Öznelik: {round(pd.Series(journal_subjs).mean(), 2) if journal_subjs else '-'}",
        "journal_emos": pd.Series(journal_emos).value_counts().to_dict() if journal_emos else {},
        "audio_sent": f"Ort. Sentiment: {round(pd.Series(audio_sents).mean(), 2) if audio_sents else '-'}",
        "audio_subj": f"Ort. Öznelik: {round(pd.Series(audio_subjs).mean(), 2) if audio_subjs else '-'}",
        "audio_emos": pd.Series(audio_emos).value_counts().to_dict() if audio_emos else {},
    }

    # Display mood data if available
    try:
        mood_files = sorted(os.listdir(os.path.join(base, "mood_tracking")))
        if mood_files:
            mood_df = pd.read_csv(os.path.join(base, "mood_tracking", mood_files[-1]))
            if mood_df is not None:
                pass  # Placeholder to fix indentation error
        else:
            st.error("Mood tracking data is missing.")
    except Exception as e:
        st.error(f"Error loading mood data: {e}")    # Önce risk değerlendirmesini göster
    grade, danger_level = load_patient_grade(selected)
    if grade is not None and danger_level is not None:
        risk_level = grade
        score = danger_level
        # Renk ve metinleri grade'a göre ayarla
        color_map = {
            "I": ("Minimum Risk", "rgb(0, 128, 0)"),
            "II": ("Mild Risk", "rgb(144, 238, 144)"),
            "III": ("Moderate Risk", "rgb(255, 204, 0)"),
            "IV": ("Significant Risk", "rgb(255, 140, 0)"),
            "V": ("Severe Risk", "rgb(255, 0, 0)")
        }
        danger_text, color = color_map.get(grade, ("Bilinmiyor", "gray"))
    else:
        # fallback eski hesaplama
        danger_score = calculate_danger_level(
            mood_df,
            nlp_summary,
            video_json,
            form_scores,
            avg_steps,
            avg_sleep,
            functioning_score
)
            
    st.markdown("### Klinik Risk Değerlendirmesi")
    st.markdown(f'<div class="blinking" style="color: {color}; font-size: 72px;">{danger_level:.2f} ({risk_level}: {danger_text})</div>', unsafe_allow_html=True)
    
    # ⬇️ Ruh hali değerlerinin zaman serisi grafiği🐇🐇🐇🐇
    st.subheader("😊 Ruh Hali Değerleri")
    mood_path = os.path.join(base, "mood_tracking")
    if os.path.exists(mood_path):
        mood_files = sorted([f for f in os.listdir(mood_path) if f.endswith(".csv")])
        if mood_files:
            # Tüm mood dosyalarını yükle ve birleştir
            mood_list = [pd.read_csv(os.path.join(mood_path, f)) for f in mood_files]
            mood_all = pd.concat(mood_list, ignore_index=True)
            
            # Tarih bilgisini dosya adından çıkar ve DataFrame'e ekle
            dates = [f.replace("mood_", "").replace(".csv", "") for f in mood_files]
            mood_all["date"] = [datetime.strptime(d, "%Y%m%d") for d in dates]
            mood_all = mood_all.sort_values("date")
            
            # Ortalama değerleri hesapla
            avg_huzur = round(mood_all["huzur"].mean(), 2)
            avg_enerji = round(mood_all["enerji"].mean(), 2)
            avg_anksiyete = round(mood_all["anksiyete"].mean(), 2)
            avg_ofke = round(mood_all["öfke"].mean(), 2)
            avg_depresif = round(mood_all["depresif"].mean(), 2)
            
            # Ortalama değerleri göster
            cols = st.columns(5)
            with cols[0]:
                st.metric("Ort. Huzur", avg_huzur)
            with cols[1]:
                st.metric("Ort. Enerji", avg_enerji)
            with cols[2]:
                st.metric("Ort. Anksiyete", avg_anksiyete)
            with cols[3]:
                st.metric("Ort. Öfke", avg_ofke)
            with cols[4]:
                st.metric("Ort. Depresif", avg_depresif)
            
            # Her bir parametre için zaman serisi grafiği🐇🐇🐇🐇
            st.markdown("### 🧠 Ruh Hali Parametreleri Zaman Serisi")
            
            # Tüm parametreleri tek grafikte göster - seçenek 1🐇🐇🐇
            st.line_chart(mood_all.set_index("date")[["huzur", "enerji", "anksiyete", "öfke", "depresif"]], 
                          use_container_width=True)
            
            # Alternatif olarak her parametreyi ayrı grafikte göster - seçenek 2🐇🐇🐇
            tabs = st.tabs(["Huzur", "Enerji", "Anksiyete", "Öfke", "Depresif", "Ortalama"])
            
            with tabs[0]:
                st.line_chart(mood_all.set_index("date")["huzur"], use_container_width=True)
            with tabs[1]:
                st.line_chart(mood_all.set_index("date")["enerji"], use_container_width=True)
            with tabs[2]:
                st.line_chart(mood_all.set_index("date")["anksiyete"], use_container_width=True)
            with tabs[3]:
                st.line_chart(mood_all.set_index("date")["öfke"], use_container_width=True)
            with tabs[4]:
                st.line_chart(mood_all.set_index("date")["depresif"], use_container_width=True)
            with tabs[5]:
                st.line_chart(mood_all.set_index("date")["average"], use_container_width=True)
        else:
            st.info("📭 Ruh hali verisi henüz mevcut değil.")
    else:
        st.info("📁 Ruh hali klasörü henüz oluşturulmamış.")

    # Display video emotion analysis
    st.subheader("🎥 Video Duygu Analizi")
    if os.path.exists(video_path) and os.listdir(video_path):
        files = sorted(os.listdir(video_path))
        with open(os.path.join(video_path, files[-1]), "r", encoding="utf-8") as f:
            v = json.load(f)
        st.markdown(f"**Baskın Duygu:** {v['dominant_emotion']}")
        st.bar_chart(pd.Series(v["emotion_scores"]))
    else:
        st.info("Video duygu analizi verisi henüz mevcut değil.")

    # Journal NLP stats
    st.subheader("📝 Günlük (journal) NLP İstatistikleri")
    if journal_sents:
        st.markdown("**Sentiment**")
        st.line_chart(journal_sents, use_container_width=True)
    if journal_subjs:
        st.markdown("**Öznelik/Subjectivity**")
        # Öznelik değerlerini DataFrame olarak hazırlayın ve renkli bir çizgi olarak gösterin
        subj_df = pd.DataFrame({"subjectivity": journal_subjs})
        st.line_chart(subj_df, use_container_width=True)
    if journal_emos:
        st.markdown("**Duygu Dağılımı**")
        # Renk haritası uygulayın
        emotion_counts = pd.Series(journal_emos).value_counts()
        # En az bir değer olduğundan emin olun
        for emotion in EMOTIONS:
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
        st.bar_chart(emotion_counts, use_container_width=True)
    if not (journal_sents or journal_subjs or journal_emos):
        st.info("Günlük NLP verisi bulunamadı.")

    # Audio NLP stats
    st.subheader("🎤Ses NLP İstatistikleri")
    if audio_sents:
        st.markdown("**Sentiment**")
        st.line_chart(audio_sents, use_container_width=True)
    if audio_subjs:
        st.markdown("**Öznelik/Subjectivity**")
        st.line_chart(audio_subjs, use_container_width=True)
    if audio_emos:
        st.markdown("**Duygu Dağılımı**")
        st.bar_chart(pd.Series(audio_emos).value_counts(), use_container_width=True)
    if not (audio_sents or audio_subjs or audio_emos):
        st.info("Ses NLP verisi bulunamadı.")

    # Health data
    st.subheader("🏃 Adım & Uyku")
    health_path = os.path.join(base, "healthkit")
    if os.path.exists(health_path):
        files = sorted([f for f in os.listdir(health_path) if f.endswith(".csv")])
        if files:
            df_list = [pd.read_csv(os.path.join(health_path, f)) for f in files]
            df = pd.concat(df_list, ignore_index=True)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            avg_steps = int(df["steps"].mean())
            avg_sleep = round(df["hours"].mean(), 2)

            st.markdown("### 🚶 Günlük Adım Sayısı")
            st.line_chart(df.set_index("date")["steps"], use_container_width=True)

            st.markdown("### 🛌 Günlük Uyku Süresi")
            st.line_chart(df.set_index("date")["hours"], use_container_width=True)
        else:
            st.info("📭 Sağlık verisi henüz mevcut değil.")
    else:
        st.info("📁 Sağlık klasörü henüz oluşturulmamış.")

    # Psychometric test scores
    st.subheader("📝 Psikometrik Test Skorları")
    for form in FORM_WEEKLY + FORM_MONTHLY:
        path = os.path.join(base, f"forms/{form}")
        if os.path.exists(path) and os.listdir(path):
            try:
                data = [json.load(open(os.path.join(path, f), "r", encoding="utf-8")) for f in sorted(os.listdir(path))]
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                st.markdown(f"#### {form}")
                st.line_chart(df.set_index("date")["score"])
            except Exception as e:
                st.warning(f"{form} verisi okunurken hata oluştu: {str(e)}")
        else:
            st.info(f"{form} form verisi bulunamadı.")

    # Functioning Score Display
    st.subheader("⚖️ İşlevsellik Skoru")
    func_path = os.path.join(base, "functioning_score")
    if os.path.exists(func_path):
        func_files = sorted([f for f in os.listdir(func_path) if f.endswith(".csv")])
        if func_files:
            df_list = [pd.read_csv(os.path.join(func_path, filename)) for filename in func_files]
            if df_list:
                df = pd.concat(df_list, ignore_index=True)
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")

                # Ortalama işlevsellik skoru
                avg_func = round(df["score"].mean(), 2)
                st.metric("Ort. İşlevsellik", avg_func)

                # Son skor
                latest_score = df["score"].values[-1]
                st.markdown(f"### Son İşlevsellik Skoru: {latest_score}/100")

                # Skor değerlendirmesi
                if latest_score < 40:
                    st.error("Düşük İşlevsellik: Günlük yaşam aktivitelerini sürdürmede önemli zorluklar yaşıyor.")
                elif latest_score < 70:
                    st.warning("Orta İşlevsellik: Bazı alanlarda zorluklar yaşasa da temel işlevleri sürdürebiliyor.")
                else:
                    st.success("Yüksek İşlevsellik: İşlevsellikte önemli bir sorun gözlenmiyor.")                # İşlevsellik trendi (tüm süreci gösteren grafik)🐇🐇
                st.markdown("### İşlevsellik Trendi")
                st.line_chart(df.set_index("date")["score"], use_container_width=True)
            else:
                st.info("İşlevsellik verileri henüz mevcut değil.")
        else:
            st.info("İşlevsellik dosyası bulunamadı.")
    else:
        st.info("İşlevsellik klasörü mevcut değil.")
    
# Sidebar'da klinik özet ve yapay zeka analizi gösterimi🐇🐇
if selected:    # Klinik özet (ham verilerle)
    clinical_overview = generate_clinical_overview(mood_df, nlp_summary, video_json, form_scores, avg_steps, avg_sleep, functioning_score, avg_func)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<div style='white-space: normal; font-size:15px; text-align:left'>{clinical_overview}</div>", unsafe_allow_html=True)# Yapay zeka analizi butonuna basılırsa GPT'den analiz al ve kaydet🐇🐇🐇
    if st.session_state.get("analiz_triggered", False):
        # Eğer daha önce kaydedilmiş analiz yoksa GPT'den al
        ai_summary = load_clinical_summary(selected)
        if not ai_summary:
            ai_summary = generate_clinical_summary(mood_df, nlp_summary, video_json, form_scores, avg_steps, avg_sleep, functioning_score, patient_id=selected)
            save_clinical_summary(selected, ai_summary)
        st.session_state.clinical_summary = ai_summary
        st.session_state.analiz_triggered = False  # Buton tetiklenmesini sıfırla
        
    # Eğer kaydedilmiş analiz varsa göster🐇🐇🐇🐇
    if st.session_state.clinical_summary:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🤖 Yapay Zeka Klinik Analizi")
        formatted_summary = format_clinical_summary(st.session_state.clinical_summary)
        st.sidebar.markdown(f"<div style='white-space: normal; font-size:15px; text-align:left'>{formatted_summary}</div>", unsafe_allow_html=True)
        st.sidebar.markdown("### 📚 DSBIT Referans Listesi")
        for title, url in DSBIT_REFERENCES:
            st.sidebar.markdown(f"- [{title}]({url})")

