# 🧭 Marketing Campaign Predictor & ROI Analyzer
### *Business & Functional Requirements Document (BFRD)*

## 📘 1. Scopul general al proiectului

Proiectul **Marketing Campaign Predictor & ROI Analyzer** își propune să ofere o soluție inteligentă de **analiză predictivă pentru campaniile de marketing**, bazată pe modele de învățare automată.  

Aplicația va permite:
- evaluarea **șanselor de succes** ale unei campanii înainte de lansare,
- estimarea **ROI-ului** (Return on Investment),
- și **recomandarea automată a segmentelor de clienți** cu cel mai mare potențial.

Scopul final este de a **crește eficiența bugetelor de marketing** și de a oferi un **instrument decizional data-driven** pentru profesioniștii din domeniu.

## 💡 2. Context și motivație

În mediul actual, campaniile de marketing sunt tot mai complexe și implică un volum uriaș de date: demografice, comportamentale, financiare etc.  
Companiile investesc resurse considerabile în campanii care adesea nu produc randamentul așteptat.  

Prin combinarea **analizei datelor istorice** cu **modele de predicție și regresie**, acest proiect urmărește:
- reducerea pierderilor financiare,
- creșterea ROI-ului global,
- și generarea de insight-uri valoroase pentru segmentarea clienților.

## 💼 3. Obiective de business (Business Objectives)

| ID | Obiectiv | Indicator de succes | Beneficiu principal |
|----|-----------|--------------------|---------------------|
| BO-1 | Automatizarea analizei campaniilor | ≥ 80% acuratețe în predicții | Reducerea timpului de analiză manuală |
| BO-2 | Creșterea eficienței investițiilor | ROI mediu > 1.25 | Alocarea bugetelor mai eficient |
| BO-3 | Optimizarea segmentării | Identificarea a cel puțin 3 segmente de audiență distincte | Targetare precisă și personalizată |
| BO-4 | Democratizarea deciziilor bazate pe date | Utilizatori non-tehnici pot genera predicții în < 2 minute | Accesibilitate și adoptare largă |
| BO-5 | Scalabilitate pentru campanii multiple | Suport pentru 100+ campanii simultane | Performanță și fiabilitate |

## 👥 4. Stakeholderi principali și nevoi

| Rol | Nevoie principală | Valoare adăugată |
|------|--------------------|------------------|
| **Marketeri** | Evaluarea șanselor de succes înainte de campanie | Evitarea pierderilor și planificare mai bună |
| **Analiști de business** | Vizualizarea impactului bugetelor asupra ROI | Înțelegerea relației cost–beneficiu |
| **Manageri** | Decizii rapide bazate pe rapoarte predictive | Creșterea rentabilității investițiilor |
| **Dezvoltatori ML/AI** | Platformă ușor de antrenat și extins | Posibilitatea de a îmbunătăți modelele |
| **Stakeholderi externi (clienți)** | Predicții pentru campanii proprii | Serviciu valoros de consultanță AI |

## 🧭 5. Domeniul de aplicare

Platforma va oferi funcționalități de tip **predictive analytics** și **prescriptive insights**, fiind destinată:
- agențiilor de marketing digitale,
- departamentelor interne de marketing,
- companiilor care doresc să optimizeze costurile de promovare,
- startup-urilor din zona *MarTech* (Marketing Technology).

Proiectul este conceput ca **MVP extensibil**, care poate fi ulterior transformat într-un produs SaaS (Software-as-a-Service).

## ⚙️ 6. Cerințe funcționale detaliate (Functional Requirements)

| ID | Cerință | Descriere detaliată | Prioritate | Modul responsabil |
|----|----------|---------------------|-------------|-------------------|
| FR-1 | Introducere date campanie | Utilizatorul introduce parametri (buget, canal, audiență, tip campanie) prin interfața web | High | UI |
| FR-2 | Validare input | Verificarea completitudinii și formatului datelor (ex: buget numeric, canale valide) | High | Backend |
| FR-3 | Predicție succes campanie | Modelul de clasificare prezice probabilitatea de succes | High | ML Core |
| FR-4 | Estimare ROI | Modelul de regresie calculează ROI-ul pe baza inputului | High | ML Core |
| FR-5 | Segmentare automată | KMeans identifică clustere de audiență similare | Medium | ML Core |
| FR-6 | Vizualizare interactivă | Rezultatele sunt afișate în grafice dinamice | High | UI |
| FR-7 | Export rapoarte | Utilizatorul poate descărca rezultatele în format PDF/CSV | Medium | Backend |
| FR-8 | Istoric campanii | Toate predicțiile sunt salvate cu timestamp | Low | Database |
| FR-9 | API extern | Expune endpoint `/predict` pentru integrare externă | Medium | API Layer |

## 🔒 7. Cerințe non-funcționale (NFR)

| ID | Tip | Descriere | Prag minim de performanță |
|----|------|------------|---------------------------|
| NFR-1 | Performanță | Predicție completă < 2 secunde / request | 2s |
| NFR-2 | Disponibilitate | Sistemul trebuie să funcționeze 99% din timp | 99% uptime |
| NFR-3 | Securitate | Transmiterea datelor prin HTTPS / validare JWT | Obligatoriu |
| NFR-4 | UX/UI | Design responsive, minimalist, accesibil | PWA-ready |
| NFR-5 | Mentenabilitate | Arhitectură modulară pe componente independente | Respectarea SOLID |
| NFR-6 | Portabilitate | Compatibil cu Windows, macOS și Linux | Testat pe 3 platforme |
| NFR-7 | Logging & Monitoring | Evenimentele majore logate în fișier local | Logrotate activat |

## 🧮 8. Metrici și KPI-uri de evaluare

| Metrică | Tip | Valoare țintă | Metodă de evaluare |
|----------|------|---------------|--------------------|
| Acuratețe model clasificare | ML Metric | ≥ 80% | Test set 20% |
| R² pentru regresie ROI | ML Metric | ≥ 0.75 | Cross-validation |
| Timp de răspuns API | Performanță | < 2 secunde | Benchmark |
| Grad de adopție | Business | ≥ 70% dintre utilizatori repetă folosirea | UAT Feedback |
| Satisfacție UI | UX | ≥ 4/5 | Survey intern |

## 🎯 9. Scenarii de utilizare (Use Case Scenarios)

### **UC-1: Predicția succesului unei campanii**
1. Utilizatorul introduce parametrii campaniei în formular.  
2. Sistemul validează datele.  
3. Modelul ML calculează probabilitatea de succes.  
4. Rezultatele sunt afișate sub formă de grafice și scoruri.

### **UC-2: Estimarea ROI-ului**
1. Utilizatorul setează bugetul și canalul de promovare.  
2. Modelul de regresie estimează ROI-ul.  
3. Se afișează profitul estimat și recomandări pentru optimizare.

### **UC-3: Segmentarea audienței**
1. Sistemul rulează un model KMeans pe datele istorice.  
2. Se afișează segmentele (Cluster A, B, C) cu caracteristici cheie.  
3. Marketerul decide asupra canalelor potrivite pentru fiecare cluster.

## 🧩 10. Arhitectură conceptuală

```text
+---------------------------+
|  UI Layer (Streamlit)     |
|  - Form input             |
|  - Charts (Plotly)        |
+------------+--------------+
             |
             ▼
+------------+--------------+
| Backend/API (FastAPI)     |
|  - /predict endpoint       |
|  - Data validation         |
+------------+--------------+
             |
             ▼
+------------+--------------+
| ML Engine (scikit-learn)  |
|  - Classifier (success)    |
|  - Regressor (ROI)         |
|  - Clustering (KMeans)     |
+------------+--------------+
             |
             ▼
+------------+--------------+
| Database (SQLite/Postgres)|
|  - Campaign logs           |
|  - Results storage         |
+---------------------------+
```

## 🌐 11. Roadmap de dezvoltare

| Etapă | Descriere | Durată estimată | Output |
|--------|------------|----------------|---------|
| Faza 1 | Preprocesare date + feature engineering | 2 săptămâni | Set de date curat |
| Faza 2 | Antrenare modele ML | 3 săptămâni | Modele salvate `.pkl` |
| Faza 3 | Implementare API | 1 săptămână | Endpoint `/predict` |
| Faza 4 | UI & Vizualizare | 2 săptămâni | Dashboard interactiv |
| Faza 5 | Testare & optimizare | 1 săptămână | MVP funcțional |

## 🔮 12. Extensii viitoare

- **Integrare cu Google Ads / Meta Ads API**  
  → pentru import automat de campanii reale;  
- **Optimizare bugetară automată**  
  → recomandare de alocare procentuală pe canale;  
- **NLP pentru analiză text**  
  → scor de calitate pentru descrierea campaniei;  
- **AutoML retraining**  
  → modelul se reantrenează periodic pe date noi.

## ✅ 13. Concluzie

Acest document definește în mod detaliat cerințele de **business**, **funcționale** și **non-funcționale** pentru un proiect conceptual, dar realist, de tip *AI Marketing Analytics Platform*.  
Deși sistemul nu este încă implementat, structura oferă o **bază solidă pentru dezvoltarea unui MVP funcțional**, care poate fi extins ulterior în produs comercial.
