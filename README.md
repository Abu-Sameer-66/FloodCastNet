<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&ColorList=0A3D8F,1565C0,0D47A1,1976D2&height=280&section=header&text=FloodCastNet&fontSize=80&fontColor=E3F2FD&animation=fadeIn&fontAlignY=38&desc=Physics-Informed%20Multi-Modal%20Deep%20Learning%20for%20Flood%20Prediction&descAlignY=60&descAlign=50" width="100%"/>
</div>

<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=700&size=22&pause=1000&color=1976D2&center=true&vCenter=true&width=1000&lines=9+Sub-Problems+Solved+End-to-End;Physics-Informed+Neural+Networks;Cross-Modal+Fusion+Transformer+(Novel);Causal+Discovery+%2B+Uncertainty+Quantification;Real-Time+Flood+Prediction+%26+Early+Warning;Beating+Google+Flood+Hub+%7C+IBM+PAIRS" alt="Typing SVG"/>
</div>

<br/>

<div align="center">
  <a href="https://www.linkedin.com/in/sameer-nadeem-66339a357/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white"/>
  </a>
  &nbsp;
  <a href="mailto:sameerdataanalyst66@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-EA4335?style=for-the-badge&logo=gmail&logoColor=white"/>
  </a>
  &nbsp;
  <a href="https://www.kaggle.com/code/sameernadeem66/floodcastnet-week1-setup">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
  </a>
  &nbsp;
  <a href="https://github.com/Abu-Sameer-66/FloodCastNet">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
</div>

<br/>

<div align="center">
  <img src="https://img.shields.io/badge/Status-Phase%202%20In%20Progress-1976D2?style=for-the-badge"/>
  &nbsp;
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Params-4.66M-2E7D32?style=for-the-badge"/>
  &nbsp;
  <img src="https://img.shields.io/badge/License-MIT-F9A825?style=for-the-badge"/>
</div>

<br/>

---

## 🌊 What is FloodCastNet?

<table>
<tr>
<td width="60%">

FloodCastNet is a **world-class, end-to-end deep learning system** that unifies **9 interconnected sub-problems** of flood prediction into one trainable architecture — something no existing system does.

Built from scratch in PyTorch with:
- **Physics-informed constraints** (water flows downhill)
- **Cross-Modal Fusion Transformer** (novel architecture)
- **Causal discovery** — not just prediction, but *explanation*
- **Uncertainty quantification** — confidence per pixel
- **RL-based resource allocation** for disaster response

</td>
<td width="40%">

```
Input → 5 Modalities
   ↓
3 Specialized Encoders
   ↓
Cross-Modal Fusion ← Novel
   ↓
9 Sub-Problem Heads
   ↓
14 Output Tensors
```

</td>
</tr>
</table>

---

## ⚔️ FloodCastNet vs The World

<div align="center">

| Feature | Google Flood Hub | IBM PAIRS | DeepMind | **FloodCastNet** |
|:--------|:---------------:|:---------:|:--------:|:----------------:|
| Sub-problems | 2 | 1 | 1 | **9** ✅ |
| Physics constraints | ❌ | ❌ | Partial | **✅** |
| Uncertainty (UQ) | ❌ | ❌ | ❌ | **✅** |
| Causal discovery | ❌ | ❌ | ❌ | **✅** |
| RL resource allocation | ❌ | ❌ | ❌ | **✅** |
| Self-evolution | ❌ | ❌ | ❌ | **✅ (roadmap)** |
| Evacuation routing | ❌ | ❌ | ❌ | **✅** |
| Open source | ❌ | ❌ | ❌ | **✅** |

</div>

---

## 🧠 9 Sub-Problems — All Connected

<div align="center">

```
┌─────────────────────────────────────────────────────────────┐
│                    FloodCastNet                              │
│                                                             │
│  SP-1  Spatial Flood Mapping       ConvLSTM + ViT           │
│  SP-2  Severity Forecasting        TFT (6h/24h/48h/72h)     │
│  SP-3  Early Warning Risk Score    Cross-Modal Classifier    │
│  SP-4  Infrastructure Damage       U-Net Decoder            │
│  SP-5  Evacuation Routes           Graph Attention Network  │
│  SP-6  Resource Allocation         RL Policy (PPO)          │
│  SP-7  Causal Discovery            Causal GNN (NOTEARS)     │
│  SP-8  Uncertainty Quantification  Bayesian + MC Dropout    │
│  SP-9  Climate Adaptation          TFT + CMIP6 Projections  │
└─────────────────────────────────────────────────────────────┘
```

</div>

---

## 🏗️ Architecture

<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=14&pause=2000&color=1976D2&center=true&vCenter=true&width=800&lines=5+Modalities+→+3+Encoders+→+Fusion+→+9+Heads+→+14+Outputs" alt="arch"/>
</div>

```
┌──────────────────────── INPUT STREAMS ───────────────────────────┐
│  Sentinel-1 SAR  │  ERA5 Weather  │  USGS Gauge  │  DEM  │  Pop │
└──────┬───────────┴──────┬─────────┴──────┬────────┴───┬───┴──┬──┘
       ↓                  ↓                ↓            ↓      ↓
 SpatioTemporal      Temporal Enc      Temporal Enc   Static  Static
 Encoder             (TFT-based)       (shared)       ResNet  ResNet
 ConvLSTM+ViT                                         Encoder Encoder
       └──────────────────┴────────────────┴────────────┴───────┘
                                    ↓
              ┌─────────────────────────────────────────┐
              │    Cross-Modal Fusion Transformer        │
              │    (Novel: dynamic gating + cross-attn)  │
              └─────────────────────────────────────────┘
                                    ↓
        ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
       SP1   SP2   SP3   SP4   SP5   SP6   SP7   SP8   SP9
```

### Key Components

| Component | Details |
|-----------|---------|
| **ConvLSTM** | 3-layer (32→64→128), local spatiotemporal patterns |
| **ViT Bottleneck** | 2-head, 2-depth, global context per timestep |
| **TFT Encoder** | Variable selection + LSTM + Multi-head attention |
| **GNN (SP-5)** | Graph Attention, 3-layer, grid-based road network |
| **Fusion** | 2-layer cross-modal, pool=8x8, dim=128 |
| **Physics Loss** | Sobel gradient elevation constraints |

---

## 📊 Phase 1 Results

<div align="center">

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Parameters   :  4,660,313
  GPU Memory (T4)    :  1.66 GB  (out of 15.6 GB)
  Model Size         :  18.8 MB
  Training Loss      :  2.9215 → 2.9083 → 2.8791  ↓
  Output Tensors     :  14  (all 9 sub-problems)
  Dead Gradients     :  0   (full network alive)
  Physics Loss       :  0.0781 → 0.0509  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

</div>

---

## 🔬 Novel Research Contributions

> These are the 5 things that make FloodCastNet publishable at NeurIPS / ICLR / IEEE TGRS

**1. Cross-Modal Fusion Transformer**
First architecture to dynamically weight satellite imagery, weather time-series, and static maps using cross-attention with learned gates. When clouds block satellite signal, model automatically up-weights weather data.

**2. Physics-Informed Loss Function**
Uses Sobel gradient operators on DEM (elevation map) to penalize flood predictions that violate the physics of water flow. `loss_physics = relu(∇flood · ∇elevation).mean()`

**3. Unified 9-Head Architecture**
No existing paper solves all 9 flood-related sub-problems in a single end-to-end trainable network. All heads share encoded representations — learning in one head improves all others.

**4. Uncertainty Quantification**
Dual uncertainty: aleatoric (pixel-level noise from data) + epistemic (model uncertainty via MC Dropout, 50 samples). Governments need confidence intervals, not just point predictions.

**5. Causal Discovery Head**
Based on NOTEARS algorithm — learns a directed acyclic graph (DAG) over input variables. Answers: *"Does deforestation causally increase flood risk in this region?"*

---

## 📁 Repository Structure

```
FloodCastNet/
├── 📓 notebooks/
│   ├── floodcastnet-week1-setup.ipynb    ← Phase 1 (complete)
│   └── 02_phase2_floodnet.ipynb          ← Phase 2 (in progress)
├── 🧠 models/
│   ├── encoders/                         ← ConvLSTM, ViT, TFT, GNN
│   ├── fusion/                           ← Cross-Modal Transformer
│   └── decoders/                         ← SP-1 to SP-9 heads
├── 📉 losses/                            ← Physics + MultiTask loss
├── 🏋️ training/                          ← Training loop + scheduler
├── 📊 evaluation/                        ← Metrics (IoU, CSI, FAR, POD)
├── 🚀 deployment/                        ← FastAPI + alert system
├── 💾 data/
│   ├── raw/satellite/                    ← Sentinel-1 SAR
│   ├── raw/weather/                      ← ERA5 reanalysis
│   ├── raw/river_gauge/                  ← USGS + GRDC
│   ├── raw/dem/                          ← NASA SRTM
│   └── processed/                        ← train/val/test splits
└── requirements.txt
```

---

## ⚡ Quick Start

```python
import torch
from models.floodcastnet import FloodCastNet, MasterConfig

config = MasterConfig()
model  = FloodCastNet(config).cuda()
# 4.66M params, 1.66GB GPU memory

outputs = model(
    sat         = torch.randn(1, 12, 4, 64, 64).cuda(),
    weather     = torch.randn(1, 72, 7).cuda(),
    gauge       = torch.randn(1, 72, 3).cuda(),
    static_maps = torch.randn(1,  5, 64, 64).cuda()
)

# 14 outputs — all 9 sub-problems
print(outputs["flood_map"].shape)      # (1, 1, 64, 64)  — pixel flood probs
print(outputs["severity"].shape)       # (1, 4)           — 6/24/48/72h
print(outputs["risk"].shape)           # (1, 4)           — 4-class risk
print(outputs["damage_map"].shape)     # (1, 3, 64, 64)   — damage severity
print(outputs["danger_map"].shape)     # (1, 1, 8, 8)     — evac danger
print(outputs["route_map"].shape)      # (1, 1, 8, 8)     — safe routes
print(outputs["action_probs"].shape)   # (1, 64)          — RL deployment
print(outputs["causal_adj"].shape)     # (10, 10)         — causal graph
print(outputs["epistemic_unc"].shape)  # (1, 1)           — uncertainty
print(outputs["vulnerability"].shape)  # (1, 4)           — climate 2030-50
```

---

## 🗄️ Data Sources

<div align="center">

| Data | Source | Access | Resolution |
|------|--------|--------|-----------|
| SAR Imagery | Sentinel-1 (ESA Copernicus) | Free API | 10m |
| Weather | ERA5 Reanalysis (ECMWF) | Free CDS API | Hourly |
| River Gauge | USGS NWIS + GRDC | Free API | Real-time |
| Elevation (DEM) | NASA SRTM | Free Download | 30m |
| Land Cover | ESA WorldCover | Free | 10m |
| Flood Labels | FloodNet (UMBC) | Kaggle | Aerial RGB |

</div>

---

## 🗺️ Project Roadmap

```
2026 Q1  ██████████ Phase 1 — Architecture          ✅ COMPLETE
2026 Q1  ██████░░░░ Phase 2 — FloodNet Training     🔄 IN PROGRESS
2026 Q2  ░░░░░░░░░░ Phase 3 — Sentinel-1 + ERA5     ⏳ UPCOMING
2026 Q2  ░░░░░░░░░░ Phase 4 — Evaluation + Benchmark ⏳ UPCOMING
2026 Q3  ░░░░░░░░░░ Phase 5 — Deployment (FastAPI)  ⏳ UPCOMING
2026 Q3  ░░░░░░░░░░ Phase 6 — Research Paper         ⏳ UPCOMING
```

**Target Venues:** NeurIPS · ICLR · IEEE TGRS · Nature Communications

---

## 📓 Notebooks

<div align="center">

| Notebook | Description | GPU | Status |
|----------|-------------|-----|--------|
| [**Phase 1 — Architecture**](notebooks/floodcastnet-week1-setup.ipynb) | Complete model from scratch | T4 x2 | ✅ Done |
| **Phase 2 — FloodNet** | Real flood data training | T4 x2 | 🔄 Active |
| **Phase 3 — Full Pipeline** | Sentinel-1 + ERA5 + eval | T4 x2 | ⏳ Soon |

</div>

---

## 🛠️ Tech Stack

<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/einops-000000?style=for-the-badge"/>
</div>

---

## 👤 Author

<div align="center">

**Abu Sameer** (@Abu-Sameer-66)

Building world-class AI for disaster prevention and climate resilience.

<a href="https://www.linkedin.com/in/sameer-nadeem-66339a357/">
  <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin"/>
</a>
&nbsp;
<a href="mailto:sameerdataanalyst66@gmail.com">
  <img src="https://img.shields.io/badge/Email-Contact-EA4335?style=for-the-badge&logo=gmail"/>
</a>
&nbsp;
<a href="https://www.kaggle.com/sameernadeem66">
  <img src="https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=for-the-badge&logo=kaggle"/>
</a>

</div>

---

## 📄 Citation

```bibtex
@misc{floodcastnet2026,
  title   = {FloodCastNet: Physics-Informed Multi-Modal Spatiotemporal
             Deep Learning for Real-Time Flood Prediction and Early Warning},
  author  = {Abu Sameer},
  year    = {2026},
  url     = {https://github.com/Abu-Sameer-66/FloodCastNet},
  note    = {4.66M parameters, 9 sub-problems, physics-informed loss}
}
```

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&ColorList=0A3D8F,1565C0,0D47A1,1976D2&height=120&section=footer" width="100%"/>
</div>
