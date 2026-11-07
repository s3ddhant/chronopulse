# ChronoPulse: Real-Time Streaming Telemetry Anomaly Detector & Incident State Machine

> **Academic-Grade Portfolio Artifact for Master of Science in Computer Science Admissions**  
> *Track: Distributed Systems, Reliability Engineering & Applied Statistics*  
> **Author:** Siddhant Bhandare ([siddhantbhandare1@gmail.com](mailto:siddhantbhandare1@gmail.com))

---

## 1. Project Overview

**ChronoPulse** is a 100% offline, self-contained streaming telemetry anomaly detector and incident finite state machine. Designed to emulate real-time transaction processing systems (TPS) and critical infrastructure health monitors, ChronoPulse addresses the two most persistent failure modes of standard monitoring stacks:
1. **Catastrophic Cancellation & Memory Bloat in Streaming Metrics:** Replaced naive sliding-window statistics with **Welford's One-Pass Recurrence Algorithm** ($\mathcal{O}(1)$ time, $\mathcal{O}(1)$ space).
2. **Alert Fatigue & Anomaly Masking in Heavy-Tailed Distributions:** Replaced standard 3-sigma thresholds (which collapse under single extreme outliers) with an adaptive envelope derived from the **Median Absolute Deviation (MAD)** ($50\%$ breakdown point), coupled with a **Debounce Finite State Machine (FSM)**.

---

## 2. Theoretical Foundations

### 2.1 Welford's One-Pass Algorithm (1962)
Computing sample variance $\sigma^2$ across high-throughput telemetry streams using the naive formula:
$$\sigma^2 = \frac{1}{n-1} \left( \sum_{i=1}^n x_i^2 - \frac{(\sum_{i=1}^n x_i)^2}{n} \right)$$
suffers from catastrophic floating-point cancellation when variance is small relative to $\bar{x}^2$. 

Welford maintains the running mean $\bar{x}_k$ and squared error accumulator $M_{2,k}$ sequentially:
$$\bar{x}_k = \bar{x}_{k-1} + \frac{x_k - \bar{x}_{k-1}}{k}$$
$$M_{2,k} = M_{2,k-1} + (x_k - \bar{x}_{k-1})(x_k - \bar{x}_k)$$
$$s^2 = \frac{M_{2,k}}{k - 1} \quad \text{for } k > 1$$

- **Time Complexity:** $\mathcal{O}(1)$ per ingested tick.
- **Space Complexity:** $\mathcal{O}(1)$ total memory consumption.

---

### 2.2 Robust Statistics: Median Absolute Deviation (MAD)
Standard deviation has a **$0\%$ breakdown point**: a single massive DDoS spike or null connection drop corrupts both $\mu$ and $\sigma$, widening the bounds to infinity and blinding subsequent alarms (masking effect).

MAD provides an optimal **$50\%$ breakdown point**:
$$\text{MAD} = \text{median}\left( |x_i - \text{median}(X)| \right)$$
$$\hat{\sigma}_{\text{MAD}} = 1.4826 \cdot \text{MAD} \quad (\text{normal distribution consistency factor})$$
$$\text{Envelope} = \left[ \text{median}(X) - 3\hat{\sigma}_{\text{MAD}}, \; \text{median}(X) + 3\hat{\sigma}_{\text{MAD}} \right]$$

---

### 2.3 Alarm Debounce Finite State Machine (FSM)
To prevent alert flapping from transient network hiccups, state transitions follow a debounced hysteresis curve:

```
               +-------------------------------------------------+
               |                                                 |
               v                                                 | (Metric returns to
        +--------------+   1 Outlier   +------------------+      |  envelope)
------> |   NOMINAL    | ------------> |    EVALUATING    | -----+
        +--------------+               +------------------+
               ^                         | (3 consecutive
               |                         |  anomalous ticks)
               |                         v
               | (Acknowledge)         +------------------+
               +---------------------- |      FIRING      |
               |                       +------------------+
               | (Safe telemetry)        |
               |                         v
        +--------------+               +------------------+
        |   NOMINAL    | <------------ |    RECOVERED     |
        +--------------+               +------------------+
```

---

## 3. UI/UX Design System (Warm Sunrise Aesthetic)

- **Canvas Gradient:** Luminous linear gradient (`linear-gradient(135deg, #FFD3DF 0%, #FFF8E7 45%, #FDF396 100%)`).
- **Floating Diagnostic Card:** Elevated pure white container (`background: #FFFFFF; border-radius: 28px; box-shadow: 0 22px 45px rgba(180, 120, 140, 0.15)`).
- **Pill Controls:** Soft strawberry pink primary pill (`#FFB3C6`), frosted white secondary pills.
- **Plotly Chart:** 100% transparent background (`rgba(0,0,0,0)`), midnight navy signal trace (`#1E293B`), soft pastel envelope (`rgba(255, 179, 198, 0.35)`), coral-pink trigger nodes (`#F43F5E`).
- **Dynamic Bottom Dock:** Floating matte black container (`#111827`) with pastel yellow fault trigger (`#FDF18A`).

---

## 4. How to Run 100% Offline (Python Streamlit)

### Requirements
- Python 3.10+
- Libraries: `streamlit`, `numpy`, `scipy`, `pandas`, `plotly`

### Setup and Launch
```bash
# 1. Install dependencies (standard offline packages)
pip install streamlit numpy scipy pandas plotly

# 2. Launch the Streamlit application
streamlit run app.py
```
The application will open immediately in your default browser at `http://localhost:8501`.
No API keys, external database connections, or internet access are required.
