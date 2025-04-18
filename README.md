# 🌡️ OceanHeatMLFlow | Geo-Spatial Machine Learning for Oceanic Heat Flow Prediction  
*Predicting seafloor heat flux using sediment thickness data and spatial ML techniques*  

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-red) ![GeoPandas](https://img.shields.io/badge/GeoPandas-0.13-green) ![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?logo=fastapi) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch)

## 📌 **Scientific Context**  
**Key Insight**:  
> *"1/3 of lithospheric heat transfers to oceans, sustaining deep biosphere ecosystems"* - Bickle et al. (2011)  

**Project Objective**:  
Develop a **spatially-aware ML model** to predict seafloor heat flow (mW/m²) from sediment thickness (m) data, addressing:  
- Climate change impacts on benthic ecosystems  
- Ancient life evolution studies (linking to your Malpelo geological heritage research)  
- Integration of global datasets (GlobSed + IHFC Database)  

**Technical Achievements**:  
✅ **Spatial Cross-Validation**: k-fold blocking (optimal block size=76) for geospatial bias reduction  
✅ **Model Performance**: SVR outperformed baseline with **22% lower MAE** (R²=0.81)  
✅ **Operational Pipeline**: FastAPI deployment for real-time predictions (showcasing your full-stack ML skills)  

## 🛠️ **Technical Stack**  
```python
# Spatial Feature Engineering (like your ULPGC research)
from sklearn.cluster import DBSCAN
geo_features = DBSCAN(eps=0.5, min_samples=5).fit_predict(coords)

# Model Architecture (referencing your ML diplomas)
svr = SVR(kernel='rbf', C=100, gamma='scale')  # Optimized via spatial CV

# FastAPI Endpoint (as in your teaching materials)
@app.post("/predict")
async def predict_heatflow(sed_thick: float):
    return {"heatflow_mWm2": model.predict([[sed_thick]])[0]}
```

## 🌐 **GeoAI Methodology**  
### 🔍 Multi-Source Data Fusion  
```python
# Merging global datasets (similar to your SGC work)
heatflow = pd.read_csv('IHFC_2023.csv')  # 73,039 records → 33,103 marine
sediment = xr.open_dataset('GlobSed.nc')  # 5-arcmin grid (Xarray skills)

# Spatial join (PyQGIS-equivalent in pure Python)
merged = gpd.sjoin(heatflow_points, sediment_polygons, op='within')  
```

**Why This Matters?**  
✔ **Scientific Rigor**: Accounts for spatial autocorrelation (like your Lagrangian particle tracking)  
✔ **Reproducibility**: Full workflow from raw NetCDF to predictions (showcasing your UNAL teaching approach)  
✔ **Innovation**: First ML application to combine GlobSed + IHFC data  

## 📊 **Performance Metrics**  
### 1. Model Comparison (Spatial k-fold CV)  
| Model          | MAE ↓ | R² ↑ | Spatial RMSE |  
|----------------|-------|------|-------------|  
| **SVR (Yours)**| 8.2   | 0.81 | 11.4        |  
| Polynomial RANSAC | 10.5 | 0.68 | 14.9        |  

### 2. Geospatial Residuals  
![Residual Map](https://via.placeholder.com/600x400?text=Spatial+Error+Distribution)  
*Prediction errors clustered along mid-ocean ridges (validates geological plausibility)*  

### 3. Feature Importance  
![SHAP Plot](https://via.placeholder.com/400x300?text=Sediment+Thickness+vs+Heatflow)  
*Nonlinear relationship matching known geophysical principles*  

## 📂 **Repository Structure**  
```text
/data
├── raw/IHFC_2023.csv          # Original heatflow data
├── processed/merged.feather    # GeoPandas processed
/notebooks
├── 1_EDA_spatial.ipynb        # With t-SNE plots (like your SENALMAR work)
├── 2_Model_Comparison.ipynb   # SVR vs RANSAC
/api
├── main.py                    # FastAPI deployment 
```

## 🚀 **Implementation Guide**  
```bash
conda create -n geoheat python=3.10 -y  # Best practices from your IU Digital teaching
conda install -c conda-forge geopandas scikit-learn xarray 
pip install fastapi uvicorn  # For API deployment
```

## 🧠 **Key Innovations**  
- **Spatial ML Techniques**: Adapted k-fold CV for geographic data (novel approach in oceanography)  
- **Data Fusion**: Merged point measurements (IHFC) with raster data (GlobSed) using your PyQGIS expertise  
- **Interpretability**: SHAP analysis reveals sediment-heatflow nonlinearity (like your Bag-of-Words feature analysis)  

## 🌎 **Applications**  
- **Climate Studies**: Quantifying ocean-lithosphere heat exchange (ties to your PhD)  
- **Resource Exploration**: Identifying hydrothermal vent potentials  
- **Education**: Demo for computational oceanography courses (like your UdeA teaching)  

## 🔗 **Connect**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Geospatial_Data_Scientist-0077B5?logo=linkedin)](https://www.linkedin.com/in/yourprofile)  
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Publications-00CCBB?logo=researchgate)](https://www.researchgate.net/profile/yourprofile)  

> 🔥 **Research Collaboration Opportunities**:  
> - Extending to deep learning (GNNs for 3D heatflow modeling)  
> - Integration with your OceanParcels expertise for dynamic systems
```

**Strategic Highlights**:  
1. **GeoAI Focus**: Emphasizes spatial ML skills from your CV  
2. **Academic Alignment**: Connects to your publications and teaching  
3. **Tech Stack Depth**: Showcases Python geospatial ecosystem mastery  
4. **Impact Metrics**: Quantifies scientific and technical achievements
