# 🧱 FAZA 1 – Preprocesare date & Feature Engineering
📅 **Durată estimată:** 2 săptămâni  
🎯 **Output final:** dataset curat, pregătit pentru antrenarea modelelor ML

---

## ⚙️ 1️⃣ Obiectivul principal

Pregătirea datelor brute (ex. din [Kaggle – Marketing Campaign Dataset](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign)) pentru modele de:
- **Clasificare** → prezicerea succesului (`Response`)
- **Regresie** → estimarea ROI (`ROI`)
- **Clustering** → segmentarea audienței (`KMeans`)

---

## 🧩 2️⃣ Structura fișierelor implicate

**Input:**  
`data/raw/marketing_campaign.csv` (fișier original Kaggle)

**Output:**  
`data/processed/marketing_campaign_clean.csv`  
`data/processed/features_selected.csv`

---

## 🧹 3️⃣ Etapele principale de preprocesare

### 🔸 a) Import și analiză inițială

```python
import pandas as pd

df = pd.read_csv("data/raw/marketing_campaign.csv")
print(df.info())
print(df.describe())
print(df.isna().sum())
```

➡️ Scop: înțelegerea structurii datelor, tipurile de coloane, valorile lipsă și distribuțiile.

---

### 🔸 b) Curățarea datelor

1. **Elimină valorile lipsă (NaN):**
```python
df = df.dropna(subset=["Income", "Education", "Marital_Status"])
```

2. **Elimină duplicatele:**
```python
df = df.drop_duplicates()
```

3. **Corectează formatele:**
```python
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")
df["Year_Birth"] = df["Year_Birth"].astype(int)
```

4. **Filtrează vârste aberante:**
```python
df = df[(df["Year_Birth"] > 1940) & (df["Year_Birth"] < 2005)]
```

---

### 🔸 c) Crearea de coloane derivate (Feature Engineering)

1. **Calculează vârsta clientului:**
```python
from datetime import datetime
df["Age"] = datetime.now().year - df["Year_Birth"]
```

2. **Număr total de cumpărături:**
```python
df["TotalPurchases"] = df["NumDealsPurchases"] + df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
```

3. **Cheltuieli totale:**
```python
df["TotalSpent"] = df[["MntWines", "MntFruits", "MntGoldProds", "MntMeatProducts", "MntSweetProducts", "MntFishProducts"]].sum(axis=1)
```

4. **Categorii de vârstă:**
```python
df["AgeGroup"] = pd.cut(df["Age"], bins=[18, 30, 45, 60, 80], labels=["Young", "Adult", "Mature", "Senior"])
```

5. **Timp de la ultima cumpărare (Recency bucket):**
```python
df["RecencyGroup"] = pd.cut(df["Recency"], bins=[0, 30, 60, 120, 365], labels=["Recent", "Mid", "Old", "Dormant"])
```

---

### 🔸 d) Curățare categorii și encoding

1. **Normalizare text:**
```python
df["Education"] = df["Education"].str.strip().replace({"PhD": "Doctor", "2n Cycle": "Graduate"})
df["Marital_Status"] = df["Marital_Status"].str.title()
```

2. **Encoding pentru variabile categorice:**
```python
df = pd.get_dummies(df, columns=["Education", "Marital_Status", "AgeGroup", "RecencyGroup"], drop_first=True)
```

---

### 🔸 e) Feature scaling (standardizare numerică)

Pentru modele bazate pe distanță (KMeans, regresie liniară etc.):

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
num_cols = ["Income", "TotalPurchases", "TotalSpent", "Recency", "Age"]
df[num_cols] = scaler.fit_transform(df[num_cols])
```

Salvează scalerul pentru reutilizare:
```python
import joblib
joblib.dump(scaler, "data/models/scaler.joblib")
```

---

### 🔸 f) Salvare dataset curat

```python
df.to_csv("data/processed/marketing_campaign_clean.csv", index=False)
```

---

## 📈 4️⃣ Validare și verificare calitate date

| Verificare | Scop | Exemplu de cod |
|-------------|------|----------------|
| Lipsa valorilor nule | Asigură completitudinea datelor | `df.isna().sum()` |
| Distribuția numerică | Detectează anomalii | `df[num_cols].describe()` |
| Corelații | Identifică relații utile | `df.corr()` |
| Dimensiunea finală | Confirmă consistența | `df.shape` |

---

## 🧠 5️⃣ Rezultate finale

După faza 1, obții:
- ✅ date curate, fără valori lipsă/aberante,  
- ✅ coloane derivate relevante (`TotalSpent`, `TotalPurchases`, `AgeGroup` etc.),  
- ✅ variabile normalizate,  
- ✅ set salvat în `data/processed/`.

---

## 💡 6️⃣ (Opțional) Notebook pentru faza 1

Creează un Jupyter notebook în `notebooks/EDA.ipynb` cu etapele de mai sus, incluzând:
- histogramă pentru distribuția vârstei,  
- grafic pentru cheltuieli totale,  
- heatmap pentru corelații (`seaborn.heatmap(df.corr())`).

---

📄 **Rezultat livrabil:** `docs/Faza1_Preprocesare_Date.md` + `data/processed/marketing_campaign_clean.csv`
