# 📁 Structura proiectului: Marketing Campaign Predictor & ROI Analyzer

## 🧭 Prezentare generală

Această structură de directoare este concepută pentru a organiza clar componentele aplicației — modele ML, API, interfață web și documentație — într-un mod scalabil și ușor de întreținut.

---

## 🗂️ Structură completă a proiectului

```bash
marketing-campaign-predictor/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
├── 📄 setup.py                # (opțional, pentru pachetizare)
│
├── 📂 data/
│   ├── raw/                   # seturi de date brute (ex: Kaggle dataset)
│   ├── processed/             # date curățate și normalizate
│   └── models/                # modele antrenate (ex: .pkl, .joblib)
│
├── 📂 notebooks/
│   ├── EDA.ipynb              # explorare și analiză date
│   ├── Model_Training.ipynb   # antrenare modele ML
│   └── Feature_Engineering.ipynb
│
├── 📂 src/
│   ├── 📂 ml_core/            # logica pentru modele ML
│   │   ├── train_classifier.py
│   │   ├── train_regressor.py
│   │   ├── cluster_analysis.py
│   │   └── utils_ml.py
│   │
│   ├── 📂 api/                # backend FastAPI
│   │   ├── main.py            # entry point API (FastAPI app)
│   │   ├── routes/            
│   │   │   ├── predict.py     # endpoint /predict
│   │   │   ├── healthcheck.py # endpoint /health
│   │   │   └── __init__.py
│   │   ├── schemas.py         # modele Pydantic pentru input/output
│   │   ├── services.py        # funcții logice intermediare
│   │   ├── config.py          # configurări (DB, env vars)
│   │   └── __init__.py
│   │
│   ├── 📂 database/           # persistenta datelor
│   │   ├── db_connection.py   # conexiune la SQLite/PostgreSQL
│   │   ├── models.py          # ORM (SQLAlchemy)
│   │   ├── crud.py            # operațiuni CRUD
│   │   └── __init__.py
│   │
│   ├── 📂 frontend/           # aplicația Streamlit
│   │   ├── app.py             # fișierul principal Streamlit
│   │   ├── components/        # grafice, formulare, carduri UI
│   │   │   ├── charts.py
│   │   │   ├── forms.py
│   │   │   └── __init__.py
│   │   ├── styles/            # CSS personalizat
│   │   │   ├── main.css
│   │   │   └── colors.css
│   │   └── utils_ui.py
│   │
│   ├── 📂 utils/              # funcții generale reutilizabile
│   │   ├── logger.py
│   │   ├── validators.py
│   │   ├── constants.py
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── 📂 tests/
│   ├── test_api.py
│   ├── test_ml_core.py
│   ├── test_end_to_end.py
│   └── __init__.py
│
├── 📂 configs/
│   ├── settings.yaml          # configurări generale
│   ├── logging.conf
│   └── env.example            # variabile de mediu
│
├── 📂 docs/
│   ├── Business_Requirements.md
│   ├── Technical_Design.md
│   └── Architecture_Diagram.png
│
└── 📂 deployment/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── start.sh
    └── nginx.conf
```

---

## 🧱 Explicație pe module

### 🧠 `src/ml_core/`
Conține toate scripturile de machine learning:
- **train_classifier.py** – pentru predicția succesului (`Response`),
- **train_regressor.py** – pentru estimarea ROI,
- **cluster_analysis.py** – pentru segmentarea clienților.
- Rezultatele antrenării (modelele salvate `.pkl` sau `.joblib`) se stochează în `data/models/`.

### ⚙️ `src/api/`
Backend-ul bazat pe **FastAPI**, care expune endpoint-uri REST:
- `/predict` pentru generarea predicțiilor,
- `/train` pentru reantrenarea modelelor,
- `/health` pentru verificarea statusului API.

### 💻 `src/frontend/`
Aplicația **Streamlit**, care:
- colectează datele de la utilizator (input),
- trimite cererea către API,
- afișează rezultatele în grafice interactive.

### 🧩 `src/database/`
- Gestionarea conexiunii la baza de date (SQLite / PostgreSQL),
- Definirea modelelor ORM cu SQLAlchemy,
- Persistența rezultatelor și logurilor de campanii.

### 🧮 `src/utils/`
Funcții ajutătoare reutilizabile:
- logare evenimente,
- validare date,
- constante comune.

### 🧾 `docs/`
Documentația proiectului: cerințe, design tehnic, diagrame UML, arhitectură logică.

### 🧪 `tests/`
Teste unitare și de integrare (API, modele ML, frontend).

### 🌍 `deployment/`
- **Dockerfile** – imaginea de rulare a aplicației,
- **docker-compose.yml** – orchestrarea serviciilor (API + UI + DB),
- **nginx.conf** – reverse proxy (opțional).

---

## 📦 `requirements.txt` minimal

```txt
fastapi==0.115.0
uvicorn==0.30.1
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
xgboost==2.1.0
joblib==1.4.2
streamlit==1.37.0
plotly==5.23.0
sqlalchemy==2.0.31
python-dotenv==1.0.1
```

---

## ✅ Concluzie

Această structură de directoare oferă o bază solidă pentru un proiect complet — de la colectarea datelor și antrenarea modelelor ML, până la expunerea rezultatelor prin API și interfață grafică.  
Este potrivită atât pentru un **proiect academic demonstrativ**, cât și pentru un **prototip comercial (MVP)**.
