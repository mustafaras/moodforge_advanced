![image](https://github.com/user-attachments/assets/772ba161-4ef0-4320-bab6-b9e044735856)

# 🧠 MoodForge Advanced- AI Supported-Psychiatry-Psychology

**MoodForge Advanced** is a high-fidelity, multimodal simulation engine for **digital psychiatry** and **AI-supported psychiatric risk modeling**. It creates synthetic patients, simulates their daily psychological profiles, and evaluates them using natural language processing, emotion recognition, psychometric scores, and physical activity metrics.

---

## 🔬 What Does It Do?

- Simulates psychiatric patients with disorders like:
  - Depression, Anxiety, PTSD, Bipolar, Psychosis, OCD
  - As well as normal (Grade I-II) profiles
- Tracks daily mood variations and computes functioning scores
- Analyzes sentiment and emotions from journal entries, voice logs, and facial expressions
- Generates clinically structured psychometric forms (PHQ-9, GAD-7, etc.)
- Scores psychiatric risk (Grade I–V) using machine learning
- Visualizes changes with interactive Streamlit dashboards
- Explains model outputs using SHAP (Explainable ML)

---
![image](https://github.com/user-attachments/assets/63748a96-cfff-48fc-8bf0-78b4f7256642)
![image](https://github.com/user-attachments/assets/d7ee612b-43de-4e7f-a280-5636f92f4f35)


## ⚙️ Technologies Used

- **Python 3.11+**
- **Streamlit** – User interface
- **scikit-learn** – Machine learning (Random Forest)
- **SHAP** – Explainability
- **OpenAI GPT-4 Turbo** – Clinical AI summaries
- **Plotly / Matplotlib** – Data visualization
- **Pandas / NumPy** – Data processing

---

## 🤖 Machine Learning & Risk Grading

MoodForge uses a trained **Random Forest Classifier** to predict psychiatric risk based on:

| Feature        | Description |
|----------------|-------------|
| `mood_avg`     | Average mood score (1–5) |
| `steps`        | Daily physical steps |
| `sleep`        | Daily sleep duration |
| `PHQ9`         | Depression score |
| `GAD7`         | Generalized anxiety |
| `PSS10`        | Perceived stress |
| `PSQI`         | Sleep quality |
| `IESR`         | PTSD symptom load |
| `functioning`  | Combined score (0–100) |

### 🏥 Output: Risk Grades (I–V)

- Grade I–II: Normal to mild symptoms
- Grade III–V: Clinical risk increases

![image](https://github.com/user-attachments/assets/1df3c908-bf9b-44fb-aa57-0bc53140265e)
![image](https://github.com/user-attachments/assets/cca6ad6d-1c89-4cec-9845-de67e6603e1e)
![image](https://github.com/user-attachments/assets/02b22056-5ee8-4bfa-ab93-ce98e183d33b)

### 🔍 Explainability with SHAP

- SHAP Waterfall and Bar plots are generated
- OpenAI explains feature contributions (e.g., "High PHQ-9 increased Grade IV risk")
- Clinically interpretable for psychiatrists and researchers

### 🛠️ Train the Model

```bash
python train_random_forest_model.py
```

The model is saved as:

```text
random_forest_risk_model.pkl
```

---

## 📈 Clinical Assessment Pipeline

1. **Data Collection**
   - Simulated mood, emotion, audio, journal, healthkit data
2. **Risk Evaluation**
   - Grade I–V via ML
3. **Visualization**
   - Mood heatmaps, function graphs, form score plots
4. **AI Summary**
   - GPT-based narrative report for each patient

---

## 🧬 Supported Psychometric Tools

- **PHQ-9**: Depression
- **GAD-7**: Anxiety
- **PSS-10**: Stress
- **PSQI**: Sleep quality
- **IES-R**: PTSD

Each score includes:
- Numerical score
- Severity level (normal, mild, moderate, severe)

---

## 🧠 Clinical AI Commentary

MoodForge provides GPT-4-generated clinical reports based on each patient's multimodal data including:

- Mood changes
- Emotion trends
- Functioning score
- Form severity
- NLP-derived sentiment

---

## 🧪 Scientific Basis

MoodForge is built upon the principles of:

- **RDoC (Research Domain Criteria)**
- **DSM-5 Diagnostic Framework**
- **Digital Phenotyping** (Torous et al., 2020)
- **Emotion research** (Ekman, Scherer)
- **ML in Psychiatry** (Luxton, 2020)

References used in simulation logic:
- APA (2013), WHO ICD-11 (2021)
- Kroenke et al. (2010), Spitzer et al. (2006)
- Buysse et al. (1989), Cohen et al. (1983)
- Insel et al. (2010), Linehan (2018)

---

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/mustafaras/moodforge_advanced_psychiatry_psychology.git
cd cd moodforge_advanced_psychiatry_psychology
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key in `.env`:
```env
OPENAI_API_KEY=your_openai_key_here
```

4. Run the app:
```bash
streamlit run moodforge_main.py
```

---

## 📁 Directory Structure

```text
├── moodforge_main.py           # Main simulation logic
├── projection.py               # Intervention modeling
├── templates/                  # NLP and affective response templates
├── train_random_forest_model.py
├── test_grade.py               # Grade evaluation CLI
├── requirements.txt
└── data/
    └── records/
        └── <patient_id>/
```

---


# 🌲 Random Forest Risk Modeling in MoodForge

This document provides an in-depth explanation of how the **Random Forest classifier** is used in MoodForge to assess psychiatric risk levels based on behavioral, emotional, and psychometric features.

---

## 📊 What is Random Forest?

Random Forest is a powerful **ensemble learning algorithm** that combines multiple decision trees to improve classification accuracy and reduce overfitting. It’s particularly useful in mental health applications where data is noisy, multi-dimensional, and non-linear.

---

## 🧠 Why Random Forest for Psychiatry?

- Handles **non-linear interactions** between symptoms
- Resistant to **overfitting** on small datasets
- Allows for **feature importance extraction**
- Integrates well with **SHAP** for explainability
- Performs well even with **imperfect or simulated data**

---

## 🎯 Features Used for Classification

| Feature         | Description                            |
|----------------|----------------------------------------|
| `mood_avg`      | Daily average mood score (1–5)         |
| `steps`         | Physical activity (steps per day)      |
| `sleep`         | Sleep duration (hours)                 |
| `PHQ9`          | Depression score                       |
| `GAD7`          | Generalized anxiety score              |
| `PSS10`         | Perceived stress score                 |
| `PSQI`          | Sleep quality score                    |
| `IESR`          | PTSD-related symptom score             |
| `functioning`   | Behavioral-functional score (0–100)    |

Each patient is represented as a vector of these 9 features.

---

## 🩺 Model Output: Risk Grades

The Random Forest model predicts a **psychiatric risk grade** from I (low risk) to V (critical condition). These grades are used to:

- Trigger **AI-based clinical summaries**
- Generate **projections for therapeutic interventions**
- Enable **monitoring dashboards** and heatmaps

---

## 🔍 Explainability with SHAP

To make clinical decisions transparent, **SHAP (SHapley Additive exPlanations)** is integrated. It enables:

- Visual understanding of feature contributions (bar and waterfall plots)
- AI commentary on risk factors influencing the classification
- Identification of dominant clinical factors for each patient

Example:
> "High PHQ-9 and low functioning were the main contributors to this patient's Grade IV risk score."

---

## 🛠️ Training the Model

To train the model on available data:

```bash
python train_random_forest_model.py
```

This script:
- Extracts features from all available synthetic patients
- Trains a `RandomForestClassifier` with 100 trees
- Saves the model as `random_forest_risk_model.pkl`

---

## 🧪 Performance Notes

- Suitable for simulated or real-world small-scale mental health datasets
- SHAP outputs enhance trust and interpretability for clinical users
- Tunable with more trees, different depths, or feature selection

---

## 📚 Reference Concepts

- Breiman, L. (2001). Random forests. *Machine Learning*
- Luxton, D. D. (2020). Artificial Intelligence in Behavioral and Mental Health Care.
- Insel et al. (2010). RDoC Framework
- Torous et al. (2020). Digital Phenotyping and Mobile Sensing

---

This model enables scientific rigor and practical clinical utility in psychiatric simulations. For further integration, one could extend this model with logistic regression, gradient boosting, or deep learning layers.

## 🧠 Disclaimer

This is a research prototype intended for **simulation, education, and experimentation** in computational psychiatry. It does **not** replace clinical diagnosis or treatment. Use responsibly.

---

## 👨‍⚕️ Developed By

MoodForge Advanced is part of a broader research initiative on **AI-assisted psychiatry**, **digital mental health**, and **neuro-informatics**. [/mustafarasit](https://github.com/mustafaras)

Feel free to contribute, cite, or fork!


