# 🧭 Marketing Campaign Predictor & ROI Analyzer  
### *Sistem inteligent pentru predicția succesului și a ROI-ului campaniilor de marketing*

## 📘 1. Descriere generală a proiectului

**Scopul proiectului:**  
Dezvoltarea unei aplicații inteligente de tip *Marketing Intelligence Platform* care:
- analizează date istorice din campanii de marketing (folosind datasetul [Kaggle](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign)),
- **prezice succesul unei campanii** în funcție de parametri (buget, canal, audiență etc.),
- **estimează ROI-ul (Return on Investment)** pentru fiecare campanie,
- și **identifică segmente de clienți** cu cel mai mare potențial de conversie.

**Public-țintă:**  
Marketeri, analiști de business, și echipe care doresc să optimizeze campaniile în funcție de date.

## 🧱 2. Arhitectura generală a aplicației

| Componentă | Descriere | Tehnologie recomandată |
|-------------|------------|------------------------|
| **Model ML** | Antrenat pe datasetul Kaggle pentru clasificare și regresie | `scikit-learn`, `XGBoost` |
| **API / Backend** | Primește parametrii de campanie și returnează predicțiile | `FastAPI` sau `Flask` |
| **Interfață web (frontend)** | Formular interactiv pentru introducerea datelor | `Streamlit` (pentru MVP) sau `React` |
| **Bază de date (opțional)** | Stocare rezultate și campanii simulate | `SQLite` sau `PostgreSQL` |

## ⚙️ 3. Fluxul logic al aplicației

```text
[Utilizator] ➜ Introduce parametri campanie
            |
            ▼
[Backend/API] ➜ Trimite datele către modelul ML
            |
            ▼
[Model ML] ➜ Prezice succesul, ROI-ul, și segmentele țintă
            |
            ▼
[Frontend] ➜ Afișează rezultatele dinamice (grafice, scoruri, recomandări)
```

## 🧠 4. Datele de bază (Dataset Kaggle)

### 🔹 Sursa:
📦 [Marketing Campaign Dataset – Kaggle](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign)

### 🔹 Variabile existente utile:
| Coloană | Descriere | Tip |
|----------|------------|-----|
| `Age` | Vârsta clientului | Numeric |
| `Income` | Venitul anual | Numeric |
| `Education`, `Marital_Status` | Profil demografic | Categorical |
| `MntWines`, `MntFruits`, `MntGoldProds` | Sume cheltuite pe categorii de produse | Numeric |
| `NumDealsPurchases`, `NumWebPurchases`, `Recency` | Activitate recentă de cumpărare | Numeric |
| `Response` | Răspuns la campanie (Succes / Eșec) | Boolean (label) |

### 🔹 Coloane suplimentare propuse:
| Nume | Descriere |
|------|------------|
| `Budget` | Bugetul alocat campaniei |
| `Channel` | Canalul de promovare (`Email`, `Social Media`, `TV`, etc.) |
| `Campaign_Type` | Tipul campaniei (`Discount`, `Product Launch`, etc.) |
| `ROI` | Return on Investment (profit / cost) |
| `Age_min`, `Age_max` | Intervalul de vârstă al audienței țintă |

## 🧩 5. Modelele de învățare automată

| Model | Tip problemă | Variabilă țintă | Scop |
|--------|----------------|-----------------|------|
| **Model 1** | Clasificare | `Response` | Prezicerea succesului campaniei |
| **Model 2** | Regresie | `ROI` | Estimarea ROI-ului |
| **Model 3 (opțional)** | Clustering (KMeans) | — | Identificarea segmentelor de audiență |

### 🔹 Exemple de features:
`Budget`, `Channel`, `Age_min`, `Age_max`, `Income`, `Education`, `MntWines`, `Recency`, `Campaign_Type`

## 💡 6. Scenariu de utilizare dinamic

### 🔸 Input (date introduse de utilizator):
```text
Age_min = 25
Age_max = 40
Income = 60000
Education = Graduate
Channel = Email
Budget = 2000 €
Campaign_Type = Discount
Recency = 20
```

### 🔸 Output (predicție generată de aplicație):
```text
Predicție: Campania are 82% șanse de succes.
ROI estimat: 1.45 (adică +45% profit).
Segment țintă recomandat: clienți 25–40 ani, venit 50k–70k, cumpărători online activi.
```

### 🔸 Afișare:
Dashboard cu:
- grafic ROI estimat,
- distribuție pe segmente de vârstă,
- top canale cu șanse de succes mai mari.

## 🧮 7. Etapele de implementare

### **Etapa 1 – Preprocesare și curățare**
- Încarcă datasetul Kaggle cu `pandas`.
- Elimină valorile lipsă (`NaN`).
- Normalizează coloanele numerice.
- Encode variabilele categorice (`Education`, `Channel` etc.).
- Calculează media vârstei:  
  `Age_mean = (Age_min + Age_max) / 2`.

### **Etapa 2 – Antrenarea modelelor**
- **Clasificare:** `RandomForestClassifier` pentru `Response`.
- **Regresie:** `XGBoostRegressor` sau `LinearRegression` pentru `ROI`.
- **Clustering:** `KMeans` pentru segmentare automată a clienților.

### **Etapa 3 – Construirea API-ului**
- Endpoint `/predict` → primește JSON cu datele campaniei.
- Returnează predicția (`success_probability`, `estimated_roi`, `target_segment`).
- API implementat în `FastAPI`.

### **Etapa 4 – Interfață web (frontend)**
- Creată cu `Streamlit` (rapid pentru MVP).
- Form cu slideri și dropdown-uri:
  ```python
  age_min = st.slider("Vârsta minimă", 18, 70, 25)
  age_max = st.slider("Vârsta maximă", 18, 70, 40)
  channel = st.selectbox("Canal de promovare", ["Email", "Social Media", "TV", "Influencers"])
  budget = st.number_input("Buget campanie (€)", 500, 10000, 2000)
  ```
- Afișare rezultate în timp real cu grafice (`Plotly` / `matplotlib`).

### **Etapa 5 – Testare și optimizare**
- Split train/test (80/20).
- Metrici de evaluare:
  - Clasificare: `accuracy`, `F1-score`.
  - Regresie: `R²`, `MAE`, `RMSE`.
- Ajustare hiperparametri cu `GridSearchCV`.

## 🌐 8. Tehnologii recomandate

| Rol | Tehnologie | Justificare |
|------|-------------|-------------|
| Model ML | `scikit-learn`, `XGBoost` | Simplitate + performanță |
| API | `FastAPI` | Rapid, modern, documentație automată Swagger |
| Frontend | `Streamlit` | Ușor pentru prototipare interactivă |
| Vizualizare | `Plotly`, `matplotlib`, `seaborn` | Grafice dinamice |
| Bază de date | `SQLite` (MVP) / `PostgreSQL` (scalabil) | Persistență opțională |
| Mediu | `Python 3.11+`, `pandas`, `numpy`, `joblib` | Ecosistem complet pentru ML |

## 🧭 9. Arhitectura logică a sistemului

```text
+----------------------+
|   User Interface     | ← Streamlit UI
|  (input: age, etc.)  |
+----------+-----------+
           |
           ▼
+----------+-----------+
|     FastAPI Backend  |
| (route /predict)     |
+----------+-----------+
           |
           ▼
+----------+-----------+
| Machine Learning Core|
| - Classifier (success)
| - Regressor (ROI)
| - Clustering (segments)
+----------+-----------+
           |
           ▼
+----------+-----------+
|     Database (optional)
|   Save results & logs |
+----------------------+
```

## 🧩 10. Extensii viitoare

- 🔸 **Optimizare buget:** recomandare automată pentru alocarea bugetului între canale.
- 🔸 **Analiză text (NLP):** scor de performanță pentru texte de reclame.
- 🔸 **Real-time learning:** actualizarea modelului cu rezultate noi.
- 🔸 **Recomandări personalizate:** “Ce tip de campanie are cel mai mare ROI pentru segmentul X”.

## ✅ 11. Concluzie

Acest proiect demonstrează cum **datele și învățarea automată pot ghida decizii de marketing**.  
Prin combinarea clasificării, regresiei și clusteringului, aplicația oferă:
- predicții de succes pentru campanii noi,  
- estimări realiste ale ROI-ului,  
- și înțelegerea comportamentului diferitelor segmente de clienți.

Este o bază solidă pentru un **MVP (Minimum Viable Product)** care poate fi ulterior extins într-o platformă completă de *AI Marketing Analytics*.
