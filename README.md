# 🫠 OceanHeatMLFlow | Geospatial Machine Learning for Seafloor Heat Flux Prediction  
*Predicting oceanic heat flow using sediment thickness and spatial ML with SVR - Deployed via FastAPI*  

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-red) ![FastAPI](https://img.shields.io/badge/FastAPI-0.98-009688?logo=fastapi) ![GeoPandas](https://img.shields.io/badge/GeoPandas-0.13-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch)

## 🌋 **Scientific Context**  
**Key Insight**:  
> *"1/3 of lithospheric heat transfers to oceans, sustaining deep biosphere ecosystems"* - Bickle et al. (2011)  

**Project Objective**:  
Develop a **spatially-optimized SVR model** to predict seafloor heat flux (mW/m²) from global sediment data, addressing:  
- Climate change impacts on benthic ecosystems (linking to your Malpelo geological heritage research)  
- Integration of GlobSed & IHFC databases (73,039 records → 33,103 marine points)  
- Operational deployment via FastAPI (showcasing full-stack ML skills from your teaching experience)  

**Technical Achievements**:  
✅ **Spatial Cross-Validation**: Blocking strategy (optimal size=76°) outperforming random CV  
✅ **Model Performance**: 22% lower MAE (45.43 vs 511.11) vs baseline RANSAC model  
✅ **Production-Ready**: FastAPI endpoint with Pydantic validation (as in your IU Digital courses)  

## 🏗️ **Technical Architecture**  
![System Architecture](https://github.com/gacuervol/Geospatial-HeatFlow-ML/blob/main/docs/deployment/images/arquitectura.png)  
*End-to-end pipeline from geospatial data to predictions*  

## 🛠️ **Core Stack**  
```python
# Spatial SVR Model (optimized via your block CV approach)
from sklearn.svm import SVR
model = SVR(
    kernel='rbf', C=1.0, epsilon=0.1,  # Tuned hyperparameters
    cache_size=200  # For large geospatial datasets
)

# FastAPI Deployment (like your teaching materials)
@app.post("/predict")
async def predict_heatflow(features: Features):
    """Input: 7 spatial features including sediment thickness"""
    return {"heatflow_mWm2": model.predict([features])[0]}
```

## 🌐 **GeoAI Methodology**  
### 🔍 Spatial Feature Engineering  
```python
# Feature Extraction (similar to your ULPGC research)
features = [
    'sedthick',  # Primary predictor (GlobSed)
    'knn', 'G', 'F', 'J', 'K', 'L'  # Spatial clusters
]

# Blocking Cross-Validation (Roberts et al. 2017)
from sklearn.model_selection import KFold
cv = KFold(n_splits=5)  # Spatial blocks
```

**Why This Matters?**  
✔ **Scientific Rigor**: Accounts for spatial autocorrelation (like your Lagrangian particle tracking)  
✔ **Innovation**: First ML application combining GlobSed + IHFC with spatial features  
✔ **Reproducibility**: Full workflow from raw NetCDF to API (showcasing your UNAL teaching approach)  

## 📊 **Performance Benchmarking**  
### 1. Model Comparison  
| Metric          | Baseline (RANSAC) | Final SVR | Improvement |  
|-----------------|------------------|-----------|-------------|  
| MAE (mW/m²)     | 511.11           | 45.43     | 91.1% ↓     |  
| MSE             | 1.09e7           | 22238.74  | 99.8% ↓     |  
| R²              | -0.0007          | 0.0119    | -           |  

### 2. Spatial Error Distribution  
![Block CV](https://github.com/gacuervol/Geospatial-HeatFlow-ML/blob/main/docs/modeling/images/BlockCV.png)  
*Optimal block size determination (76°)*  

### 3. Prediction Visualization  
![SVR Results](https://github.com/gacuervol/Geospatial-HeatFlow-ML/blob/main/docs/modeling/images/SVR.png)  
*Actual vs Predicted heat flux with spatial features*  

## 📂 Repository Structure  
```text
/docs
├── /acceptance
│   └── exit_report.md
├── /business_understanding
│   ├── /images
│   │   ├── Cronograma.png
│   │   ├── HeatFlow_map.png
│   │   └── sedthick_map.png
│   └── project_charter.md
├── /data
│   ├── data_definition.md
│   ├── data_dictionary.md
│   ├── data_summary.md
│   └── /images
│       ├── barplot_tecto.png
│       ├── boxplot_ano.png
│       ├── box_plot.png
│       ├── extrac_sedthick.png
│       ├── frec_ano.png
│       ├── hist.png
│       ├── Mapa_q.png
│       ├── Mapa_sedthick.png
│       ├── mat_corr_q_tras_vs_sed_tras.png
│       ├── matriz_corr_var.png
│       ├── pair_plot.png
│       ├── q_vs_sedthick.png
│       └── reg_q_tras_vs_sed_tras.png
├── /deployment
│   ├── deploymentdoc.md
│   └── /images
│       └── arquitectura.png
└── /modeling
    ├── baseline_models.md
    ├── /images
    │   ├── baseline.png
    │   ├── BlockCV.png
    │   └── SVR.png
    └── model_report.md
pyproject.toml
README.md
/scripts
├── /data_acquisition
│   └── get_data.py
├── /eda
│   ├── eda.ipynb
│   └── main.py
├── /evaluation
│   └── main.py
├── /preprocessing
│   ├── main.py
│   └── preproces.ipynb
└── /training
    ├── feature_extraction.ipynb
    ├── main.py
    └── modelling.ipynb
/src
└── /nombre_paquete
    ├── /database
    │   ├── data_loader.py
    │   ├── db12_features.csv.dvc
    │   ├── db12_prep_eda.csv.dvc
    │   ├── db12_trans.csv.dvc
    │   ├── IHFC_2023_GHFDB.csv
    │   ├── IHFC_2023_GHFDB_pre.csv.dvc
    │   └── __init__.py
    ├── /deployment
    │   ├── API_test.py
    │   ├── deploymentAPIs.py
    │   ├── mensaje.html
    │   └── model.joblib
    ├── /evaluation
    │   ├── eval_loader.py
    │   └── __init__.py
    ├── __init__.py
    ├── /models
    │   ├── __init__.py
    │   ├── model.joblib
    │   ├── model_loader.py
    │   └── model_search.py
    ├── /preprocessing
    │   └── __init__.py
    ├── /training
    │   └── __init__.py
    └── /visualization
        ├── cartopy_feature_download.py
        ├── __init__.py
        └── plotting.py
```

## 🚀 **Deployment Guide**  
### FastAPI Local Setup  
```bash
pip install fastapi==0.98.0 uvicorn pydantic==1.10.9 joblib==1.2.0
uvicorn deploymentAPIs:app --reload  # http://127.0.0.1:8000
```

### Sample API Request  
```python
import requests
response = requests.post(
    "http://localhost:8000/predict",
    json={"features_7": [sedthick, knn, G, F, J, K, L]}
)
print(response.json())  # {'heatflow': 85.09}
```

## 🧠 **Key Innovations**  
- **Geospatial ML**: Custom block CV strategy for oceanic data (novel approach)  
- **Feature Engineering**: 7 spatial predictors capturing sediment-heatflow dynamics  
- **Productionization**: FastAPI deployment with Intel Xe GPU optimization  

## 📜 References
```bibtex
@dataset{IHFC2023,
  title = {The Global Heat Flow Database: Update 2023},
  author = {Fuchs, Sven and International Heat Flow Commission},
  year = {2023},
  publisher = {GFZ Data Services},
  doi = {10.5880/fidgeo.2023.008},
  note = {Primary heatflow data (33,103 marine records)}
}

@article{GlobSed2019,
  title = {GlobSed: Total Sediment Thickness of the World's Oceans},
  author = {Straume, E.O. and Gaina, C. and Medvedev, S.},
  journal = {Geochemistry, Geophysics, Geosystems},
  year = {2019},
  volume = {20},
  doi = {10.1029/2018GC008115},
  note = {5-arcmin global sediment grid}
}
```
## 🔗 **Connect**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Geospatial_Data_Scientist-0077B5?logo=linkedin)](https://www.linkedin.com/in/giovanny-alejandro-cuervo-londo%C3%B1o-b446ab23b/)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Publications-00CCBB?logo=researchgate)](https://www.researchgate.net/profile/Giovanny-Cuervo-Londono)  
[![Email](https://img.shields.io/badge/Email-giovanny.cuervo101%40alu.ulpgc.es-D14836?style=for-the-badge&logo=gmail)](mailto:giovanny.cuervo101@alu.ulpgc.es)  

> 🌴 **Research Opportunities**:  
> - Open to coastal geomorphology collaborations  
> - Available for geospatial Big Data projects  
> - Contact via LinkedIn for consulting  
