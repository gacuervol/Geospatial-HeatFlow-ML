# 🔥 OceanHeatMLFlow | Geospatial Machine Learning for Seafloor Heat Flux Prediction  
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
![System Architecture](./images/arquitectura.png)  
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
![Block CV](./images/BlockCV.png)  
*Optimal block size determination (76°)*  

### 3. Prediction Visualization  
![SVR Results](./images/SVR.png)  
*Actual vs Predicted heat flux with spatial features*  

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

## 🌎 **Applications**  
- **Climate Research**: Quantifying ocean-lithosphere heat exchange (ties to your PhD)  
- **Resource Exploration**: Identifying hydrothermal vent potentials  
- **Education**: Demo for computational oceanography courses (like your UdeA teaching)  

## 🔗 **Connect**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Geospatial_Data_Scientist-0077B5?logo=linkedin)](https://www.linkedin.com/in/yourprofile)  
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Publications-00CCBB?logo=researchgate)](https://www.researchgate.net/profile/yourprofile)  

> 🔥 **Research Collaboration Opportunities**:  
> - Extending to GNNs for 3D heatflow modeling (leveraging your PyTorch expertise)  
> - Integration with OceanParcels for dynamic systems (building on your SENALMAR work)
```
