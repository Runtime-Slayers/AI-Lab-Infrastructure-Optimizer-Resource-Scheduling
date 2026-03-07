# BT55 — Solar-Powered Air Quality Monitoring Network with Federated Learning: Real-Time Campus-Wide Environmental Intelligence

**Domain:** Electronics / Environmental Monitoring / IoT / Machine Learning  
**Date:** 2026-02-27  
**Status:** Brainstorming  
**Novelty Level:** ★★★★☆ (High)  
**Feasibility:** ★★★★★ (Very achievable)

---

## PART A — WHAT & WHY

### A1. The Problem

University campuses are micro-environments with dramatic spatial and temporal variation in air quality — from vehicle exhaust at parking lots, to construction dust, to lab chemical fumes, to indoor CO₂ buildup in crowded lecture halls. Yet most campuses have zero or at most 1-2 air quality monitoring stations, providing no spatial resolution. Students with asthma, allergies, or chemical sensitivities have no way to plan routes or choose study locations based on real-time air quality.

**The gap:** No campus-scale air quality system combines (1) solar-powered autonomous nodes for zero-maintenance outdoor deployment, (2) indoor/outdoor unified monitoring, (3) federated ML for privacy-preserving multi-campus models, and (4) student-facing real-time exposure scoring.

### A2. Why It Matters

| Stakeholder | Pain Point |
|---|---|
| Students with asthma/allergies | No real-time air quality data to avoid triggers |
| Campus facilities | Cannot identify building ventilation failures |
| Environmental health & safety | Lack spatial air quality data for compliance |
| Researchers | No longitudinal campus air quality datasets |
| Administration | No data to correlate air quality with student health outcomes |

### A3. Research Gap

| Existing Work | Limitation |
|---|---|
| PurpleAir network | Consumer-grade, no indoor, no ML prediction |
| Municipal AQ stations | Too sparse (1 per 10 km²) |
| Smart building HVAC sensors | Indoor only, no outdoor context |
| Low-cost AQ sensors | Drift, no calibration correction |
| AirNow (EPA) | Government only, 1-hour resolution, no campus scale |

**Our innovation:** A dual-mode (indoor/outdoor) sensor network with solar-powered outdoor nodes, LoRa mesh connectivity, federated learning for cross-campus model training without sharing raw data, and a Personal Air Quality Exposure Index (PAQEI) delivered to students via mobile app.

### A4. Core Hypothesis

> *A solar-powered air quality mesh network with federated machine learning can achieve PM2.5 prediction accuracy within ±5 µg/m³ of reference instruments, provide 15-minute spatial resolution across a 1 km² campus, and enable personal exposure scoring that correlates with student respiratory health outcomes (r > 0.5).*

---

## PART B — TECHNICAL APPROACH

### B1. Mathematical Framework

#### Air Quality Index Calculation

**PM2.5 to AQI conversion (EPA breakpoints):**

$$AQI_{PM2.5} = \frac{I_{high} - I_{low}}{C_{high} - C_{low}} \cdot (C - C_{low}) + I_{low}$$

Where $C$ is the measured concentration, $[C_{low}, C_{high}]$ is the concentration breakpoint, $[I_{low}, I_{high}]$ is the AQI breakpoint.

**Multi-pollutant composite index:**

$$CAQI = \max(AQI_{PM2.5}, AQI_{PM10}, AQI_{O_3}, AQI_{NO_2}, AQI_{CO})$$

#### Sensor Calibration (Drift Correction)

**Low-cost sensor correction model:**

$$C_{corrected} = \alpha C_{raw} + \beta T + \gamma RH + \delta T \cdot RH + \epsilon$$

Coefficients updated via online gradient descent with reference colocation data.

#### Federated Learning

**FedAvg update rule:**

$$w_{global}^{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_k^{t+1}$$

Where $K$ is number of campus nodes, $n_k$ is local dataset size, $w_k^{t+1}$ is locally trained model weights.

**Differential privacy guarantee:**

$$\tilde{g}_k = g_k + \mathcal{N}(0, \sigma^2 S^2 \mathbf{I})$$

Where $S$ is gradient clipping bound, $\sigma$ is noise multiplier for $(\epsilon, \delta)$-DP.

#### Personal Air Quality Exposure Index (PAQEI)

$$PAQEI = \frac{1}{T}\int_0^T AQI(x(t), t) \cdot w_{activity}(t) \, dt$$

Where $x(t)$ is user location trajectory, $w_{activity}$ is ventilation-rate weight (higher during exercise).

### B2. System Architecture

```
┌─────────────────────────────────────────────────┐
│              OUTDOOR SOLAR NODE                  │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐           │
│  │PM2.5 │ │PM10  │ │ O₃   │ │NO₂   │ Sensors   │
│  │PMS5003│ │      │ │MQ131 │ │MICS- │           │
│  └──┬───┘ └──┬───┘ └──┬───┘ │2714  │           │
│     └────────┴────────┴──────┴──┬───┘           │
│  ┌──────────────────────────────▼───┐           │
│  │   STM32WL (LoRa SoC + Cortex-M4)│           │
│  │   + Edge ML inference            │           │
│  └──────────────────────────┬───────┘           │
│  ┌──────────────────────────▼───────┐           │
│  │   LoRa Radio (868/915 MHz)       │ → Mesh    │
│  └──────────────────────────────────┘           │
│  ┌──────────┐  ┌──────────────┐                 │
│  │ Solar 5W │→ │ LiFePO₄ 3Ah │ Power           │
│  └──────────┘  └──────────────┘                 │
└─────────────────────────────────────────────────┘
          │ LoRa Mesh
          ▼
┌─────────────────────────────────────────────────┐
│              CAMPUS GATEWAY                      │
│  LoRa ←→ Ethernet/WiFi ←→ Campus Server         │
│  + GPS time sync + OTA firmware updates          │
└──────────────────┬──────────────────────────────┘
                   │ HTTPS
                   ▼
┌─────────────────────────────────────────────────┐
│         FEDERATED LEARNING SERVER                │
│  ┌──────────────────────────────────┐           │
│  │ Campus A model  ←─┐              │           │
│  │ Campus B model  ←──┼── FedAvg    │           │
│  │ Campus C model  ←─┘  aggregation │           │
│  └──────────────┬───────────────────┘           │
│                 ▼                                 │
│  ┌──────────────────────────────────┐           │
│  │  Global prediction model         │           │
│  │  + Anomaly detection             │           │
│  │  + 15-min AQ forecast            │           │
│  └──────────────────────────────────┘           │
│  Real-time API → Mobile App (PAQEI)              │
└─────────────────────────────────────────────────┘
```

### B3. Python Implementation

```python
"""
BT55 - Solar-Powered Campus Air Quality Network
with Federated Learning and Personal Exposure Index
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AQSensorNode:
    """Air quality sensor node (indoor or outdoor)."""
    node_id: str
    lat: float
    lon: float
    indoor: bool = False
    solar_panel_W: float = 5.0
    battery_Wh: float = 11.1  # 3Ah × 3.7V LiFePO4
    
    # Sensor readings with drift simulation
    pm25_bias: float = 0.0
    pm25_noise_std: float = 2.0
    drift_rate: float = 0.01  # µg/m³ per day
    
    # State
    battery_soc: float = 0.8
    days_deployed: int = 0
    readings_count: int = 0


class AirQualityModel:
    """Simulate realistic campus air quality patterns."""
    
    def __init__(self, campus_size_km: float = 1.0, n_sources: int = 5):
        self.campus_size = campus_size_km
        
        # Pollution sources (Gaussian plumes)
        np.random.seed(42)
        self.sources = []
        for _ in range(n_sources):
            self.sources.append({
                'lat': np.random.uniform(0, campus_size_km),
                'lon': np.random.uniform(0, campus_size_km),
                'type': np.random.choice(['traffic', 'construction',
                                          'kitchen', 'lab', 'hvac']),
                'strength': np.random.uniform(10, 50),  # µg/m³ at source
                'spread': np.random.uniform(0.05, 0.2)  # km
            })
    
    def get_true_aq(self, lat: float, lon: float, hour: float,
                     indoor: bool = False) -> Dict[str, float]:
        """Calculate true air quality at a location and time."""
        # Background PM2.5 (diurnal pattern)
        background_pm25 = 8 + 5 * np.sin(2 * np.pi * (hour - 8) / 24)
        
        # Add source contributions (Gaussian plume)
        source_pm25 = 0
        for src in self.sources:
            dist = np.sqrt((lat - src['lat'])**2 + (lon - src['lon'])**2)
            
            # Time-dependent source activity
            if src['type'] == 'traffic':
                activity = 1.5 if (7 <= hour <= 9 or 16 <= hour <= 18) else 0.5
            elif src['type'] == 'construction':
                activity = 1.0 if 8 <= hour <= 17 else 0.0
            elif src['type'] == 'kitchen':
                activity = 1.5 if (11 <= hour <= 13 or 17 <= hour <= 19) else 0.3
            else:
                activity = 0.5 + 0.5 * np.random.random()
            
            contribution = src['strength'] * activity * np.exp(
                -dist**2 / (2 * src['spread']**2))
            source_pm25 += contribution
        
        total_pm25 = background_pm25 + source_pm25
        
        if indoor:
            # Indoor/outdoor ratio depends on ventilation
            io_ratio = 0.4 + 0.3 * np.random.random()
            total_pm25 *= io_ratio
            # But add indoor sources (CO₂, VOC)
            co2_ppm = 400 + np.random.uniform(0, 800)
        else:
            co2_ppm = 400 + np.random.uniform(0, 50)
        
        # Other pollutants (correlated with PM2.5)
        pm10 = total_pm25 * (1.5 + 0.5 * np.random.random())
        o3_ppb = 30 + 20 * np.sin(2 * np.pi * (hour - 14) / 24)  # Peaks afternoon
        no2_ppb = 15 + 10 * (total_pm25 / 30)
        
        return {
            'pm25': max(0, total_pm25 + np.random.normal(0, 1)),
            'pm10': max(0, pm10 + np.random.normal(0, 2)),
            'o3_ppb': max(0, o3_ppb + np.random.normal(0, 3)),
            'no2_ppb': max(0, no2_ppb + np.random.normal(0, 2)),
            'co2_ppm': max(350, co2_ppm + np.random.normal(0, 20)),
            'temperature_C': 22 + 8 * np.sin(2 * np.pi * (hour - 15) / 24),
            'humidity_pct': 55 + 15 * np.sin(2 * np.pi * (hour - 6) / 24)
        }


class SensorCalibrator:
    """Online calibration of low-cost sensors against reference."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
        # Correction model: C_corr = α*C_raw + β*T + γ*RH + δ*T*RH + ε
        self.alpha = 1.0
        self.beta = 0.0
        self.gamma = 0.0
        self.delta = 0.0
        self.epsilon = 0.0
        
        self.n_updates = 0
        self.mse_history = []
    
    def correct(self, raw_pm25: float, temperature: float, 
                humidity: float) -> float:
        """Apply correction model."""
        return (self.alpha * raw_pm25 + 
                self.beta * temperature +
                self.gamma * humidity +
                self.delta * temperature * humidity +
                self.epsilon)
    
    def update(self, raw_pm25: float, reference_pm25: float,
               temperature: float, humidity: float):
        """Online gradient descent update with reference data."""
        prediction = self.correct(raw_pm25, temperature, humidity)
        error = prediction - reference_pm25
        
        # Gradient descent
        self.alpha -= self.lr * error * raw_pm25
        self.beta -= self.lr * error * temperature
        self.gamma -= self.lr * error * humidity
        self.delta -= self.lr * error * temperature * humidity
        self.epsilon -= self.lr * error
        
        self.n_updates += 1
        self.mse_history.append(error ** 2)
    
    def get_rmse(self, window: int = 50) -> float:
        if not self.mse_history:
            return float('inf')
        recent = self.mse_history[-window:]
        return np.sqrt(np.mean(recent))


class AQICalculator:
    """Calculate Air Quality Index from pollutant concentrations."""
    
    # EPA PM2.5 breakpoints (24-hour)
    PM25_BREAKPOINTS = [
        (0.0, 12.0, 0, 50),       # Good
        (12.1, 35.4, 51, 100),     # Moderate
        (35.5, 55.4, 101, 150),    # Unhealthy for sensitive
        (55.5, 150.4, 151, 200),   # Unhealthy
        (150.5, 250.4, 201, 300),  # Very unhealthy
        (250.5, 500.4, 301, 500),  # Hazardous
    ]
    
    @classmethod
    def pm25_to_aqi(cls, pm25: float) -> int:
        """Convert PM2.5 concentration to AQI."""
        for c_low, c_high, i_low, i_high in cls.PM25_BREAKPOINTS:
            if c_low <= pm25 <= c_high:
                aqi = (i_high - i_low) / (c_high - c_low) * (pm25 - c_low) + i_low
                return int(round(aqi))
        return 500  # Off scale
    
    @classmethod
    def aqi_category(cls, aqi: int) -> str:
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 150: return "Unhealthy (Sensitive)"
        elif aqi <= 200: return "Unhealthy"
        elif aqi <= 300: return "Very Unhealthy"
        else: return "Hazardous"


class FederatedLearningSimulator:
    """Simulate federated learning across campus nodes."""
    
    def __init__(self, n_campuses: int = 3, model_dim: int = 10):
        self.n_campuses = n_campuses
        self.model_dim = model_dim
        
        # Each campus has a local model (simulated as weight vector)
        self.local_models = [np.random.randn(model_dim) * 0.1 
                            for _ in range(n_campuses)]
        self.global_model = np.zeros(model_dim)
        
        # Local dataset sizes
        self.local_data_sizes = [
            np.random.randint(500, 5000) for _ in range(n_campuses)
        ]
        
        # True model (what we're trying to learn)
        self.true_model = np.random.randn(model_dim)
        
        # Privacy parameters
        self.noise_multiplier = 1.0
        self.clip_norm = 1.0
        self.epsilon_budget = 10.0
        self.epsilon_spent = 0.0
        
        # Metrics
        self.round_losses = []
    
    def local_train(self, campus_idx: int, n_steps: int = 10,
                     lr: float = 0.01) -> np.ndarray:
        """Simulate local training on campus data."""
        w = self.local_models[campus_idx].copy()
        n = self.local_data_sizes[campus_idx]
        
        for _ in range(n_steps):
            # Simulated gradient (noisy version of true gradient)
            X = np.random.randn(min(32, n), self.model_dim)
            noise = np.random.randn(min(32, n)) * 0.5
            y = X @ self.true_model + noise
            
            # Linear regression gradient
            pred = X @ w
            grad = -2 / len(y) * X.T @ (y - pred)
            
            w -= lr * grad
        
        self.local_models[campus_idx] = w
        return w
    
    def federated_average(self, apply_dp: bool = True) -> float:
        """FedAvg aggregation with optional differential privacy."""
        total_n = sum(self.local_data_sizes)
        
        # Weighted average of local models
        new_global = np.zeros(self.model_dim)
        
        for k in range(self.n_campuses):
            weight = self.local_data_sizes[k] / total_n
            model_update = self.local_models[k] - self.global_model
            
            if apply_dp:
                # Clip gradient norm
                norm = np.linalg.norm(model_update)
                if norm > self.clip_norm:
                    model_update = model_update * self.clip_norm / norm
                
                # Add Gaussian noise
                noise = np.random.normal(0, self.noise_multiplier * self.clip_norm,
                                         self.model_dim)
                model_update += noise
                
                # Track privacy budget (simplified)
                self.epsilon_spent += 1 / self.noise_multiplier
            
            new_global += weight * (self.global_model + model_update)
        
        self.global_model = new_global
        
        # Broadcast global model to all campuses
        for k in range(self.n_campuses):
            self.local_models[k] = self.global_model.copy()
        
        # Calculate loss
        loss = np.mean((self.global_model - self.true_model) ** 2)
        self.round_losses.append(loss)
        
        return loss
    
    def train_rounds(self, n_rounds: int = 50, 
                      local_steps: int = 10) -> List[float]:
        """Run multiple federated learning rounds."""
        losses = []
        
        for r in range(n_rounds):
            # Local training on each campus
            for k in range(self.n_campuses):
                self.local_train(k, n_steps=local_steps)
            
            # Global aggregation
            loss = self.federated_average()
            losses.append(loss)
        
        return losses


class PersonalExposureTracker:
    """Calculate Personal Air Quality Exposure Index (PAQEI)."""
    
    def __init__(self, aq_model: AirQualityModel, nodes: List[AQSensorNode]):
        self.aq_model = aq_model
        self.nodes = nodes
        self.aqi_calc = AQICalculator()
    
    def interpolate_aq(self, lat: float, lon: float, hour: float) -> float:
        """Interpolate AQI at arbitrary location from sensor readings."""
        # IDW (Inverse Distance Weighting) interpolation
        aqi_values = []
        weights = []
        
        for node in self.nodes:
            dist = np.sqrt((lat - node.lat)**2 + (lon - node.lon)**2)
            dist = max(dist, 0.001)  # Avoid division by zero
            
            aq = self.aq_model.get_true_aq(node.lat, node.lon, hour, node.indoor)
            aqi = self.aqi_calc.pm25_to_aqi(aq['pm25'])
            
            aqi_values.append(aqi)
            weights.append(1 / dist ** 2)
        
        weights = np.array(weights)
        weights /= weights.sum()
        
        return np.average(aqi_values, weights=weights)
    
    def calculate_paqei(self, trajectory: List[Tuple[float, float, float, float]],
                        ) -> Dict:
        """
        Calculate PAQEI from user trajectory.
        trajectory: list of (lat, lon, hour, activity_level)
        activity_level: 0=rest, 0.5=light, 1.0=exercise
        """
        exposures = []
        aqi_timeline = []
        
        # Activity-based ventilation rate multipliers
        vent_multiplier = {0: 0.5, 0.5: 1.0, 1.0: 2.5}
        
        for lat, lon, hour, activity in trajectory:
            aqi = self.interpolate_aq(lat, lon, hour)
            weight = vent_multiplier.get(activity, 1.0)
            
            exposures.append(aqi * weight)
            aqi_timeline.append(aqi)
        
        paqei = np.mean(exposures) if exposures else 0
        
        # Risk categorization
        if paqei <= 30:
            risk = "Low"
        elif paqei <= 60:
            risk = "Moderate"
        elif paqei <= 100:
            risk = "Elevated"
        else:
            risk = "High"
        
        return {
            'paqei': paqei,
            'risk_level': risk,
            'max_aqi': max(aqi_timeline) if aqi_timeline else 0,
            'min_aqi': min(aqi_timeline) if aqi_timeline else 0,
            'hours_above_moderate': sum(1 for a in aqi_timeline if a > 50),
            'peak_exposure_hour': trajectory[np.argmax(aqi_timeline)][2] if aqi_timeline else 0
        }


class SolarPowerModel:
    """Model solar panel energy harvesting for outdoor nodes."""
    
    def __init__(self, panel_watts: float = 5.0, latitude_deg: float = 35.0):
        self.panel_watts = panel_watts
        self.latitude = latitude_deg
    
    def daily_harvest_Wh(self, day_of_year: int, 
                          cloud_cover: float = 0.3) -> float:
        """Calculate daily solar energy harvest."""
        # Simplified solar irradiance model
        declination = 23.45 * np.sin(np.radians((284 + day_of_year) * 360 / 365))
        
        # Day length (hours)
        cos_hour_angle = -np.tan(np.radians(self.latitude)) * np.tan(np.radians(declination))
        cos_hour_angle = np.clip(cos_hour_angle, -1, 1)
        day_length = 2 * np.degrees(np.arccos(cos_hour_angle)) / 15
        
        # Peak irradiance (W/m²)
        peak_irradiance = 1000 * np.cos(np.radians(abs(self.latitude - declination)))
        peak_irradiance *= (1 - 0.75 * cloud_cover)
        
        # Average daily insolation (Wh/m²)
        daily_insolation = peak_irradiance * day_length * 0.6  # Integration factor
        
        # Panel output (assuming 20% efficiency, 0.025 m² panel)
        panel_area = self.panel_watts / 200  # 200 W/m² at STC
        daily_Wh = daily_insolation * panel_area * 0.20
        
        return daily_Wh
    
    def can_sustain(self, daily_consumption_Wh: float, 
                     worst_day: int = 355) -> bool:
        """Check if solar can sustain node on worst day (winter solstice)."""
        harvest = self.daily_harvest_Wh(worst_day, cloud_cover=0.7)
        return harvest > daily_consumption_Wh


def run_full_simulation():
    """Execute complete air quality monitoring simulation."""
    print("=" * 70)
    print("BT55: SOLAR-POWERED CAMPUS AIR QUALITY NETWORK")
    print("         with Federated Learning & Personal Exposure Index")
    print("=" * 70)
    
    # Setup campus
    campus_size = 1.0  # km
    aq_model = AirQualityModel(campus_size_km=campus_size, n_sources=5)
    
    # Deploy sensor nodes
    np.random.seed(123)
    nodes = []
    n_outdoor = 12
    n_indoor = 8
    
    for i in range(n_outdoor):
        nodes.append(AQSensorNode(
            node_id=f"OUT-{i:02d}",
            lat=np.random.uniform(0, campus_size),
            lon=np.random.uniform(0, campus_size),
            indoor=False
        ))
    
    for i in range(n_indoor):
        nodes.append(AQSensorNode(
            node_id=f"IN-{i:02d}",
            lat=np.random.uniform(0.2, 0.8),
            lon=np.random.uniform(0.2, 0.8),
            indoor=True
        ))
    
    print(f"\nDeployed {n_outdoor} outdoor + {n_indoor} indoor nodes")
    
    # Simulate 24-hour monitoring
    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║        24-HOUR AIR QUALITY MONITORING                    ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    
    aqi_calc = AQICalculator()
    hourly_data = []
    
    for hour in range(24):
        aqi_values = []
        pm25_values = []
        
        for node in nodes:
            aq = aq_model.get_true_aq(node.lat, node.lon, float(hour), node.indoor)
            # Add sensor noise and drift
            measured_pm25 = aq['pm25'] + node.pm25_bias + np.random.normal(0, node.pm25_noise_std)
            measured_pm25 = max(0, measured_pm25)
            pm25_values.append(measured_pm25)
            aqi_values.append(aqi_calc.pm25_to_aqi(measured_pm25))
        
        avg_aqi = np.mean(aqi_values)
        max_aqi = np.max(aqi_values)
        avg_pm25 = np.mean(pm25_values)
        
        cat = aqi_calc.aqi_category(int(avg_aqi))
        hourly_data.append({
            'hour': hour, 'avg_aqi': avg_aqi, 'max_aqi': max_aqi,
            'avg_pm25': avg_pm25, 'category': cat
        })
        
        bar = "█" * int(avg_aqi / 5)
        print(f"║ {hour:02d}:00 │ AQI: {avg_aqi:>5.0f} │ PM2.5: {avg_pm25:>5.1f} │"
              f" {cat:<22s} ║")
    
    print(f"╚══════════════════════════════════════════════════════════╝")
    
    # Sensor calibration demo
    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║         SENSOR CALIBRATION (Online Learning)             ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    
    calibrator = SensorCalibrator(learning_rate=0.001)
    
    for i in range(200):
        true_pm25 = 15 + 20 * np.random.random()
        temp = 20 + 10 * np.random.random()
        humidity = 40 + 30 * np.random.random()
        
        # Simulated raw sensor (biased)
        raw_pm25 = true_pm25 * 1.3 + 5 + np.random.normal(0, 3)
        
        calibrator.update(raw_pm25, true_pm25, temp, humidity)
    
    rmse_after = calibrator.get_rmse(50)
    print(f"║ Calibration samples:     {calibrator.n_updates:>6d}                      ║")
    print(f"║ RMSE (last 50):          {rmse_after:>6.2f} µg/m³                 ║")
    print(f"║ Correction coefficients:                                 ║")
    print(f"║   α (raw PM2.5): {calibrator.alpha:>8.4f}                              ║")
    print(f"║   β (temperature): {calibrator.beta:>8.4f}                            ║")
    print(f"║   γ (humidity):   {calibrator.gamma:>8.4f}                             ║")
    print(f"║   δ (T×RH):      {calibrator.delta:>8.4f}                             ║")
    print(f"║   ε (offset):    {calibrator.epsilon:>8.4f}                             ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    
    # Federated learning
    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║        FEDERATED LEARNING (3 Campuses)                   ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    
    fl = FederatedLearningSimulator(n_campuses=3, model_dim=10)
    losses = fl.train_rounds(n_rounds=30, local_steps=5)
    
    print(f"║ {'Round':>6} │ {'Loss':>10} │ {'ε spent':>10} │ {'Status':>12}    ║")
    print(f"║{'─' * 6}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 12}────║")
    
    for r in [0, 4, 9, 14, 19, 24, 29]:
        status = "Converging" if losses[r] > 0.1 else "Converged"
        print(f"║ {r + 1:>6d} │ {losses[r]:>10.4f} │ {fl.epsilon_spent * (r + 1) / 30:>10.2f} │"
              f" {status:>12s}    ║")
    
    print(f"╠══════════════════════════════════════════════════════════╣")
    print(f"║ Final model loss:        {losses[-1]:>8.4f}                      ║")
    print(f"║ Privacy budget used:     {fl.epsilon_spent:>8.2f} / {fl.epsilon_budget:.1f}             ║")
    print(f"║ Privacy preserved:       {'✓ YES' if fl.epsilon_spent < fl.epsilon_budget else '✗ NO':>8s}                       ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    
    # Personal exposure index
    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║     PERSONAL AIR QUALITY EXPOSURE INDEX (PAQEI)          ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    
    tracker = PersonalExposureTracker(aq_model, nodes)
    
    # Simulated student day
    student_trajectory = [
        (0.3, 0.4, 8.0, 0.5),   # Walk to class
        (0.5, 0.5, 9.0, 0.0),   # Lecture (indoor)
        (0.5, 0.5, 10.0, 0.0),  # Lecture continued
        (0.4, 0.6, 11.0, 0.5),  # Walk to next building
        (0.4, 0.6, 12.0, 0.0),  # Indoor study
        (0.3, 0.3, 13.0, 0.5),  # Lunch walk
        (0.6, 0.7, 14.0, 1.0),  # Outdoor exercise
        (0.5, 0.5, 15.0, 0.0),  # Lab work
        (0.5, 0.5, 16.0, 0.0),  # Lab continued
        (0.3, 0.4, 17.0, 0.5),  # Walk home
    ]
    
    paqei = tracker.calculate_paqei(student_trajectory)
    
    print(f"║ Student daily PAQEI:     {paqei['paqei']:>6.1f}                        ║")
    print(f"║ Risk level:              {paqei['risk_level']:<10s}                    ║")
    print(f"║ Max AQI encountered:     {paqei['max_aqi']:>6.0f}                        ║")
    print(f"║ Min AQI encountered:     {paqei['min_aqi']:>6.0f}                        ║")
    print(f"║ Hours above 'Moderate':  {paqei['hours_above_moderate']:>6d}                        ║")
    print(f"║ Peak exposure at:        {paqei['peak_exposure_hour']:>5.0f}:00                      ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    
    # Solar sustainability test
    print(f"\n--- Solar Power Sustainability ---\n")
    solar = SolarPowerModel(panel_watts=5.0, latitude_deg=35)
    
    # Node power consumption
    node_daily_Wh = 0.5  # Sleep mostly, 500 mWh/day
    
    seasons = [(80, "Spring Equinox"), (172, "Summer Solstice"),
               (266, "Autumn Equinox"), (355, "Winter Solstice")]
    
    for day, name in seasons:
        for cloud in [0.0, 0.3, 0.7]:
            harvest = solar.daily_harvest_Wh(day, cloud_cover=cloud)
            sustainable = "✓" if harvest > node_daily_Wh else "✗"
            print(f"  {name:<20s} Cloud:{cloud:.0%}  → {harvest:.2f} Wh/day "
                  f"(need {node_daily_Wh:.1f}) {sustainable}")
    
    return hourly_data, paqei


if __name__ == '__main__':
    hourly_data, paqei = run_full_simulation()
```

---

## PART C — EXPECTED RESULTS

### C1. Air Quality Monitoring Performance

| Metric | Target | Expected |
|---|---|---|
| PM2.5 accuracy (vs. reference) | ±5 µg/m³ | ±3-4 µg/m³ (after calibration) |
| Spatial resolution | 100m | 50-100m (20 nodes/km²) |
| Temporal resolution | 15 min | 5-15 min (adaptive) |
| Network uptime | > 99% | 99.5% (solar + battery backup) |
| LoRa range | > 500m | 800m-2km (campus environment) |
| Sensor drift correction | < 5% error/month | 2-3% error/month |

### C2. Federated Learning Performance

| Metric | Centralized | Federated (no DP) | Federated + DP |
|---|---|---|---|
| Model convergence rounds | 15 | 20 | 30 |
| Final RMSE (PM2.5) | 3.2 µg/m³ | 3.5 µg/m³ | 4.1 µg/m³ |
| Data privacy | None | Partial | (ε=10)-DP |
| Communication cost | 100% | 5-10% | 5-10% |

### C3. Solar Power Budget

| Condition | Solar Harvest (Wh) | Node Consumption (Wh) | Margin |
|---|---|---|---|
| Summer, clear | 4.5 | 0.5 | 9× surplus |
| Summer, cloudy | 2.0 | 0.5 | 4× surplus |
| Winter, clear | 2.5 | 0.5 | 5× surplus |
| Winter, cloudy | 0.8 | 0.5 | 1.6× surplus |
| Worst case (3 rain days) | 0 (battery) | 0.5 | 22h backup |

---

## PART D — COMPARISON WITH EXISTING WORK

| Feature | PurpleAir | Clarity Node | AirBeam | Municipal AQ | **BT55 (Ours)** |
|---|---|---|---|---|---|
| Solar powered | ✗ (wall plug) | ✓ | ✗ (battery) | ✓ | **✓** |
| Indoor + Outdoor | ✗ | ✗ | ✓ | ✗ | **✓** |
| ML calibration | Basic | ✓ | ✗ | ✓ | **✓ (federated)** |
| Privacy-preserving | ✗ | ✗ | ✗ | N/A | **✓ (DP)** |
| Personal exposure | ✗ | ✗ | App-based | ✗ | **✓ (PAQEI)** |
| Mesh networking | ✗ (WiFi) | ✗ (cellular) | ✗ (BLE) | N/A | **✓ (LoRa mesh)** |
| Cost per node | $250 | $3000+ | $250 | $50K+ | **$80-120** |
| Multi-campus learning | ✗ | ✗ | ✗ | ✗ | **✓** |

---

## PART E — TOOLS & RESOURCES

### E1. Hardware Bill of Materials

| Component | Specific Part | Est. Cost |
|---|---|---|
| PM2.5/PM10 sensor | PMS5003 (Plantower) | $15 |
| O₃ sensor | MQ131 (Winsen) | $8 |
| NO₂ sensor | MICS-2714 (SGX) | $10 |
| CO₂ sensor (indoor) | SCD41 (Sensirion) | $15 |
| Temp/Humidity | BME280 (Bosch) | $3 |
| MCU + LoRa | STM32WLE5 (ST) | $8 |
| Solar panel (5W) | Polycrystalline 6V/5W | $8 |
| LiFePO₄ battery | 3.2V 3000mAh | $6 |
| MPPT charger | CN3791 | $1 |
| Weatherproof enclosure | IP67 ABS box | $5 |
| **Outdoor node total** | | **~$80** |
| **Indoor node total** | (no solar/battery) | **~$55** |

### E2. Software Stack

| Tool | Purpose |
|---|---|
| Python + NumPy/SciPy | Simulation & data analysis |
| TensorFlow Federated | Federated learning framework |
| InfluxDB | Time-series sensor data storage |
| Grafana | Real-time dashboards |
| Flutter | Cross-platform mobile app (PAQEI) |
| Zephyr RTOS | Embedded node firmware |
| LoRaWAN (ChirpStack) | Network server |
| OpenSenseMap | Public data sharing platform |

### E3. Publication Targets

| Venue | Type | Fit |
|---|---|---|
| Environmental Science & Technology | Journal | ★★★★★ |
| IEEE Internet of Things Journal | Journal | ★★★★★ |
| ACM SenSys | Conference | ★★★★☆ |
| Atmospheric Environment | Journal | ★★★★☆ |
| Nature Sustainability | Journal | ★★★☆☆ |

### E4. Summary Metrics

| Dimension | Rating |
|---|---|
| Effort | 🟡 Medium (hardware + software + deployment) |
| Difficulty | 🟡 Medium (sensor calibration is main challenge) |
| Novelty | 🟢 High (federated + solar + personal exposure) |
| Impact | 🟢 High (campus health + environmental justice) |
| Time to Prototype | 2-3 months |
| Time to Publication | 6-8 months |
