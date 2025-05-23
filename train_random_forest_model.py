import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = "data/records"
FORM_WEEKLY = ["PHQ9", "GAD7", "PSS10"]
FORM_MONTHLY = ["PSQI", "IESR"]

# Özellikleri ve etiketleri çıkar🐇🐇🐇🐇
def extract_features_from_all_patients():
    data = []
    for pid in os.listdir(BASE_DIR):
        base = os.path.join(BASE_DIR, pid)
        try:
            # Mood
            mood_file = sorted(os.listdir(os.path.join(base, "mood_tracking")))[-1]
            mood_df = pd.read_csv(os.path.join(base, "mood_tracking", mood_file))
            mood_avg = mood_df["average"].values[0]

            # Formlar
            form_scores = {}
            for form in FORM_WEEKLY + FORM_MONTHLY:
                form_path = os.path.join(base, f"forms/{form}")
                if os.path.exists(form_path):
                    form_files = sorted(os.listdir(form_path))
                    if form_files:
                        with open(os.path.join(form_path, form_files[-1]), "r", encoding="utf-8") as f:
                            j = eval(f.read())
                            form_scores[form] = j["score"]
                    else:
                        form_scores[form] = 0
                else:
                    form_scores[form] = 0

            # Adım & uyku
            health_files = sorted([f for f in os.listdir(os.path.join(base, "healthkit")) if f.endswith(".csv")])
            df = pd.concat([pd.read_csv(os.path.join(base, "healthkit", f)) for f in health_files])
            steps = df["steps"].mean()
            sleep = df["hours"].mean()

            # İşlevsellik
            func_file = sorted(os.listdir(os.path.join(base, "functioning_score")))[-1]
            func_df = pd.read_csv(os.path.join(base, "functioning_score", func_file))
            func_score = func_df["score"].values[0]

            # Risk label
            if func_score < 40:
                label = "V"
            elif func_score < 60:
                label = "IV"
            elif func_score < 70:
                label = "III"
            else:
                label = "II"

            row = {
                "mood_avg": mood_avg,
                "steps": steps,
                "sleep": sleep,
                "PHQ9": form_scores.get("PHQ9", 0),
                "GAD7": form_scores.get("GAD7", 0),
                "PSS10": form_scores.get("PSS10", 0),
                "PSQI": form_scores.get("PSQI", 0),
                "IESR": form_scores.get("IESR", 0),
                "functioning": func_score,
                "risk": label
            }
            data.append(row)
        except Exception as e:
            print(f"Error in {pid}: {e}")
    return pd.DataFrame(data)

# Eğit ve kaydet
df = extract_features_from_all_patients()
X = df.drop(columns=["risk"])
y = df["risk"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Kaydet
with open("random_forest_risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Sınıf dağılımı:", pd.Series(y).value_counts())
print("✅ Model başarıyla eğitildi ve 'random_forest_risk_model.pkl' olarak kaydedildi.")
