"""
ChronoPulse: Real-Time Streaming Telemetry Anomaly Detector & Incident State Machine
===================================================================================
Master's in Computer Science Admissions Portfolio Artifact
Author: Siddhant Bhandare (siddhantbhandare1@gmail.com)
Date: 2026
License: Apache-2.0

Architecture Overview:
----------------------
1. 100% Offline Synthetic Telemetry Stream:
   - Continuous diurnal sine-wave pattern simulating transaction throughput (TPS).
   - Low-amplitude stochastic Gaussian noise N(0, sigma^2).
   - Real-time interactive fault injection hooks (Traffic Surge, Service Outage, Baseline Reset).

2. Online Statistical Engine:
   - Welford's Algorithm (1962): O(1) time and O(1) space per tick for streaming mean and sample variance.
     Numerically stable against catastrophic cancellation.
   - Robust Outlier Scoring: Rolling Median Absolute Deviation (MAD) with scale factor c=1.4826 for
     asymptotic normality. High breakdown point (50%), resisting heavy-tailed skew and masking.
     Adaptive dynamic envelope: [Rolling_Median - 3 * MAD, Rolling_Median + 3 * MAD].

3. Alarm Debounce Finite State Machine (FSM):
   - Discrete State Space: {NOMINAL, EVALUATING, FIRING, RECOVERED}
   - Debounce Threshold: Requires k=3 consecutive anomalous ticks to transition EVALUATING -> FIRING,
     mitigating transient telemetry spikes and alert fatigue.
   - Self-healing recovery path upon telemetry returning within the MAD envelope.

4. UI/UX:
   - Warm Sunrise gradient aesthetic (#FFD3DF -> #FFF8E7 -> #FDF396).
   - Elevated floating flashcards, custom pill controls, and transparent Plotly charts.
   - Floating matte black status dock with pastel yellow action trigger.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import random
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# 1. STREAMLIT CONFIGURATION & CUSTOM STYLING (WARM SUNRISE PALETTE)
# =============================================================================

st.set_page_config(
    page_title="ChronoPulse | Telemetry Anomaly Detector & FSM",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Base canvas reset & luminous sunrise gradient */
html, body, [data-testid="stAppViewContainer"], .main {
    background: linear-gradient(135deg, #FFD3DF 0%, #FFF8E7 45%, #FDF396 100%) fixed !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #111827 !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Hide default streamlit elements */
#MainMenu, footer, header {
    visibility: hidden;
}

/* Main Container Constraint */
.block-container {
    max-width: 920px !important;
    padding-top: 2rem !important;
    padding-bottom: 7rem !important;
}

/* Floating Hero Card */
.hero-card {
    background: #FFFFFF;
    border-radius: 28px;
    box-shadow: 0 22px 45px rgba(180, 120, 140, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.8);
    padding: 32px;
    margin-bottom: 24px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.hero-card:hover {
    box-shadow: 0 26px 50px rgba(180, 120, 140, 0.20);
}

/* Micro-tags & Category Headers */
.category-tag {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.category-tag .pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10B981;
    display: inline-block;
    box-shadow: 0 0 8px #10B981;
    animation: pulse 1.8s infinite;
}

.category-tag .pulse-dot.firing {
    background-color: #F43F5E;
    box-shadow: 0 0 10px #F43F5E;
}

.category-tag .pulse-dot.evaluating {
    background-color: #F59E0B;
    box-shadow: 0 0 10px #F59E0B;
}

@keyframes pulse {
    0% { transform: scale(0.95); opacity: 0.8; }
    50% { transform: scale(1.25); opacity: 1; }
    100% { transform: scale(0.95); opacity: 0.8; }
}

/* Confident Title */
.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: #111827;
    line-height: 1.15;
    margin-bottom: 12px;
    letter-spacing: -0.02em;
}

/* Metric Display Highlight */
.metric-large {
    font-family: 'JetBrains Mono', monospace;
    font-size: 46px;
    font-weight: 800;
    color: #111827;
    letter-spacing: -0.03em;
    line-height: 1;
}

.metric-unit {
    font-size: 20px;
    font-weight: 600;
    color: #6B7280;
    margin-left: 6px;
}

/* Status Badge */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    border-radius: 9999px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.02em;
}

.status-pill.nominal {
    background: #DCFCE7;
    color: #15803D;
}

.status-pill.evaluating {
    background: #FEF3C7;
    color: #B45309;
}

.status-pill.firing {
    background: #FFE4E6;
    color: #BE123C;
}

.status-pill.recovered {
    background: #E0E7FF;
    color: #4338CA;
}

/* Subtext */
.card-subtext {
    font-size: 14px;
    color: #6B7280;
    margin-top: 10px;
    line-height: 1.5;
}

/* Floating Bottom Status Dock */
.bottom-dock {
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%);
    background: #111827;
    color: #FFFFFF;
    border-radius: 24px;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 20px 40px rgba(17, 24, 39, 0.35);
    z-index: 9999;
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.bottom-dock-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    color: #E5E7EB;
}

/* Streamlit Button Styling Overrides */
div.stButton > button {
    border-radius: 9999px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    padding: 10px 24px !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

/* Primary Button (Soft Strawberry Pink) */
div.stButton > button[kind="primary"] {
    background: #FFB3C6 !important;
    color: #111827 !important;
    box-shadow: 0 4px 14px rgba(255, 179, 198, 0.4) !important;
}

div.stButton > button[kind="primary"]:hover {
    background: #FFA2B9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(255, 179, 198, 0.6) !important;
}

/* Secondary Button (Frosted White) */
div.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #E5E7EB !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
}

div.stButton > button[kind="secondary"]:hover {
    background: #F9FAFB !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
}

/* Yellow Action Button */
.yellow-pill-btn {
    background: #FDF18A;
    color: #111827;
    font-weight: 700;
    border-radius: 9999px;
    padding: 6px 16px;
    font-size: 13px;
    text-decoration: none;
    display: inline-block;
    border: none;
    cursor: pointer;
    box-shadow: 0 2px 6px rgba(253, 241, 138, 0.4);
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# 2. STATISTICAL ENGINE & ALGORITHMS (O(1) WELFORD + ROBUST MAD)
# =============================================================================

class WelfordAccumulator:
    """
    Numerically stable online computation of streaming mean and sample variance
    using Welford's 1962 one-pass recurrence relations.

    Formulas:
        M_1 = x_1,  S_1 = 0
        M_k = M_{k-1} + (x_k - M_{k-1}) / k
        S_k = S_{k-1} + (x_k - M_{k-1}) * (x_k - M_k)
        Sample Variance s^2 = S_k / (k - 1)

    Complexity:
        Time: O(1) per update
        Space: O(1) total memory
    """
    def __init__(self):
        self.count: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0

    def update(self, x: float) -> Tuple[float, float]:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        variance = self.M2 / (self.count - 1) if self.count > 1 else 0.0
        return self.mean, variance

    @property
    def variance(self) -> float:
        return self.M2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def std_dev(self) -> float:
        return math.sqrt(self.variance)


def compute_mad_bounds(values: np.ndarray, k: float = 3.0, c: float = 1.4826) -> Tuple[float, float, float, float]:
    """
    Robust non-parametric outlier bounding using Median Absolute Deviation (MAD).

    Unlike mean and standard deviation which have a 0% breakdown point (a single
    infinite outlier corrupts the estimator), MAD boasts a 50% breakdown point.

    Definitions:
        MAD = Median(|X_i - Median(X)|)
        Robust Sigma = c * MAD (where c = 1.4826 for consistency with normal distribution)
        Envelope = [Median - k * Robust_Sigma, Median + k * Robust_Sigma]

    Returns:
        (median, lower_bound, upper_bound, robust_sigma)
    """
    if len(values) == 0:
        return 100.0, 70.0, 130.0, 10.0

    median = float(np.median(values))
    deviations = np.abs(values - median)
    mad = float(np.median(deviations))

    # Guard against zero MAD in quantized or synthetic degenerate flatlines
    robust_sigma = c * mad if mad > 1e-4 else 2.5
    lower_bound = max(0.0, median - k * robust_sigma)
    upper_bound = median + k * robust_sigma

    return median, lower_bound, upper_bound, robust_sigma


# =============================================================================
# 3. INCIDENT DEBOUNCE FINITE STATE MACHINE (FSM)
# =============================================================================

class IncidentState(str, Enum):
    NOMINAL = "NOMINAL"
    EVALUATING = "EVALUATING"
    FIRING = "FIRING"
    RECOVERED = "RECOVERED"


@dataclass
class IncidentFSM:
    """
    Finite State Machine implementing debounce filtering over consecutive anomalous ticks.
    Prevents false-alarm flapping caused by transient single-tick network glitches.

    Transition Invariants:
        NOMINAL -> EVALUATING: Single out-of-bounds observation.
        EVALUATING -> FIRING: When consecutive_anomalies >= debounce_limit (default: 3).
        EVALUATING -> NOMINAL: When telemetry returns to bounds before debounce limit.
        FIRING -> RECOVERED: When telemetry returns within safe MAD envelope.
        RECOVERED -> NOMINAL: Operator acknowledgment or sustained stability.
    """
    state: IncidentState = IncidentState.NOMINAL
    consecutive_anomalies: int = 0
    debounce_limit: int = 3
    incident_history: List[Dict[str, Any]] = field(default_factory=list)

    def transition(self, is_anomalous: bool, tick: int, value: float) -> IncidentState:
        prev_state = self.state

        if self.state == IncidentState.NOMINAL:
            if is_anomalous:
                self.consecutive_anomalies = 1
                self.state = IncidentState.EVALUATING
            else:
                self.consecutive_anomalies = 0

        elif self.state == IncidentState.EVALUATING:
            if is_anomalous:
                self.consecutive_anomalies += 1
                if self.consecutive_anomalies >= self.debounce_limit:
                    self.state = IncidentState.FIRING
                    self.incident_history.append({
                        "tick": tick,
                        "value": value,
                        "type": "ALARM_ACTIVATED",
                        "timestamp": time.time(),
                    })
            else:
                # Debounce reset: transient glitch absorbed without incident escalation
                self.consecutive_anomalies = 0
                self.state = IncidentState.NOMINAL

        elif self.state == IncidentState.FIRING:
            if not is_anomalous:
                self.consecutive_anomalies = 0
                self.state = IncidentState.RECOVERED
                self.incident_history.append({
                    "tick": tick,
                    "value": value,
                    "type": "RECOVERY_SIGNAL",
                    "timestamp": time.time(),
                })
            else:
                self.consecutive_anomalies += 1

        elif self.state == IncidentState.RECOVERED:
            if is_anomalous:
                # Immediate re-entry into FIRING
                self.state = IncidentState.FIRING
                self.consecutive_anomalies = 1
            else:
                # Self-healing closure
                self.consecutive_anomalies = 0
                self.state = IncidentState.NOMINAL

        return self.state

    def acknowledge(self):
        """Operator manual acknowledgment resets state machine to NOMINAL."""
        self.state = IncidentState.NOMINAL
        self.consecutive_anomalies = 0


# =============================================================================
# 4. STREAMING GENERATOR & STATE INITIALIZATION
# =============================================================================

def generate_telemetry_point(tick: int, fault_injection: str = "NONE") -> float:
    """
    Generates continuous synthetic transaction throughput (Transactions Per Second).
    Baseline: Diurnal sinusoidal pattern + Gaussian stochastic noise.
    """
    diurnal_period = 50.0  # Ticks per diurnal period
    base_tps = 95.0 + 20.0 * math.sin(2.0 * math.pi * (tick % diurnal_period) / diurnal_period)
    stochastic_noise = np.random.normal(0.0, 3.2)
    sample = base_tps + stochastic_noise

    # Fault Injections
    if fault_injection == "SURGE":
        # Sudden steep spike (e.g. flash-crowd DDoS or viral traffic surge)
        sample += 82.0 + np.random.uniform(5.0, 15.0)
    elif fault_injection == "OUTAGE":
        # Instant service outage / catastrophic database connection pool exhaustion
        sample = max(0.0, np.random.uniform(0.0, 3.5))

    return round(float(max(0.0, sample)), 2)


# Initialize Session State
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.tick = 0
    st.session_state.fault_mode = "NONE"
    st.session_state.history_values: List[float] = []
    st.session_state.history_ticks: List[int] = []
    st.session_state.history_medians: List[float] = []
    st.session_state.history_lowers: List[float] = []
    st.session_state.history_uppers: List[float] = []
    st.session_state.history_anomalies: List[bool] = []
    st.session_state.welford = WelfordAccumulator()
    st.session_state.fsm = IncidentFSM()
    st.session_state.window_size = 40

    # Seed initial 35 baseline points
    for i in range(35):
        val = generate_telemetry_point(i, "NONE")
        st.session_state.tick = i
        st.session_state.history_values.append(val)
        st.session_state.history_ticks.append(i)
        st.session_state.welford.update(val)

        # Baseline window
        window = np.array(st.session_state.history_values[-st.session_state.window_size:])
        med, low, up, _ = compute_mad_bounds(window)
        st.session_state.history_medians.append(med)
        st.session_state.history_lowers.append(low)
        st.session_state.history_uppers.append(up)
        st.session_state.history_anomalies.append(False)


def advance_tick(fault: str = None):
    """Generates the next stream item, updates statistics, and triggers FSM."""
    if fault is not None:
        st.session_state.fault_mode = fault

    st.session_state.tick += 1
    t = st.session_state.tick
    val = generate_telemetry_point(t, st.session_state.fault_mode)

    # Online Welford Update (O(1))
    st.session_state.welford.update(val)

    # Rolling MAD Envelope (Robust statistics over sliding window)
    st.session_state.history_values.append(val)
    st.session_state.history_ticks.append(t)

    recent_window = np.array(st.session_state.history_values[-st.session_state.window_size:])
    median, lower, upper, _ = compute_mad_bounds(recent_window)

    st.session_state.history_medians.append(median)
    st.session_state.history_lowers.append(lower)
    st.session_state.history_uppers.append(upper)

    # Anomaly condition
    is_anomaly = (val < lower) or (val > upper)
    st.session_state.history_anomalies.append(is_anomaly)

    # FSM Transition
    st.session_state.fsm.transition(is_anomaly, t, val)


# =============================================================================
# 5. UI LAYOUT & HERO FLASHCARD
# =============================================================================

# Top Hero Header
col_header_a, col_header_b = st.columns([3, 1])
with col_header_a:
    state_class = st.session_state.fsm.state.value.lower()
    st.markdown(
        f'<div class="category-tag"><span class="pulse-dot {state_class}"></span> TELEMETRY AUDITOR · REAL-TIME STREAMING</div>'
        f'<div class="hero-title">System Heartbeat</div>',
        unsafe_allow_html=True,
    )

# Current metric and bounds
current_val = st.session_state.history_values[-1]
current_med = st.session_state.history_medians[-1]
current_low = st.session_state.history_lowers[-1]
current_up = st.session_state.history_uppers[-1]
current_state = st.session_state.fsm.state

state_badge_html = {
    IncidentState.NOMINAL: '<span class="status-pill nominal">● Stable Flow (Nominal)</span>',
    IncidentState.EVALUATING: f'<span class="status-pill evaluating">▲ Evaluating ({st.session_state.fsm.consecutive_anomalies}/3)</span>',
    IncidentState.FIRING: '<span class="status-pill firing">■ Incident Firing</span>',
    IncidentState.RECOVERED: '<span class="status-pill recovered">↺ Stabilizing (Recovered)</span>',
}[current_state]

# Main Diagnostic Hero Card
st.markdown(
    f"""
    <div class="hero-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 18px;">
            <div>
                <div class="category-tag">METRIC DRIFT & RUNTIME TELEMETRY</div>
                <div class="metric-large">{current_val:.1f}<span class="metric-unit">TPS</span></div>
            </div>
            <div>
                {state_badge_html}
            </div>
        </div>
        <div class="card-subtext">
            Dynamic MAD envelope actively bounding <strong>{st.session_state.window_size}</strong> observations.
            Safe Range: <code>[{current_low:.1f} - {current_up:.1f} TPS]</code> · Welford Mean: <code>{st.session_state.welford.mean:.1f} TPS</code> (σ = {st.session_state.welford.std_dev:.2f})
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pill Action Buttons
btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.2, 1.4, 1.2, 1.2])

with btn_col1:
    if st.button("Next Step ➔", type="primary", use_container_width=True):
        advance_tick()
        st.rerun()

with btn_col2:
    if st.button("Acknowledge Alert", type="secondary", use_container_width=True):
        st.session_state.fsm.acknowledge()
        st.rerun()

with btn_col3:
    if st.button("Surge Fault", type="secondary", use_container_width=True):
        advance_tick(fault="SURGE")
        st.rerun()

with btn_col4:
    if st.button("Outage Fault", type="secondary", use_container_width=True):
        advance_tick(fault="OUTAGE")
        st.rerun()


# =============================================================================
# 6. EMBEDDED DYNAMIC PLOTLY TELEMETRY CHART
# =============================================================================

view_window = 60
display_ticks = st.session_state.history_ticks[-view_window:]
display_vals = st.session_state.history_values[-view_window:]
display_low = st.session_state.history_lowers[-view_window:]
display_up = st.session_state.history_uppers[-view_window:]
display_anom = st.session_state.history_anomalies[-view_window:]

fig = go.Figure()

# 1. Shaded MAD Adaptive Envelope
# Upper bound trace (invisible line)
fig.add_trace(go.Scatter(
    x=display_ticks,
    y=display_up,
    mode='lines',
    line=dict(color='rgba(255, 179, 198, 0.4)', width=1, dash='dot'),
    showlegend=False,
    hoverinfo='skip'
))

# Lower bound trace with fill
fig.add_trace(go.Scatter(
    x=display_ticks,
    y=display_low,
    mode='lines',
    line=dict(color='rgba(255, 179, 198, 0.4)', width=1, dash='dot'),
    fill='tonexty',
    fillcolor='rgba(255, 179, 198, 0.30)',
    showlegend=False,
    hoverinfo='skip'
))

# 2. Primary Telemetry Signal Line
fig.add_trace(go.Scatter(
    x=display_ticks,
    y=display_vals,
    mode='lines',
    line=dict(color='#1E293B', width=2.6, shape='spline', smoothing=0.7),
    name='Throughput (TPS)',
    hovertemplate='Tick %{x}<br>TPS: <b>%{y:.2f}</b><extra></extra>'
))

# 3. Anomaly Trigger Points
anom_x = [t for t, a in zip(display_ticks, display_anom) if a]
anom_y = [v for v, a in zip(display_vals, display_anom) if a]

if anom_x:
    fig.add_trace(go.Scatter(
        x=anom_x,
        y=anom_y,
        mode='markers',
        marker=dict(
            color='#F43F5E',
            size=10,
            line=dict(color='#FFFFFF', width=2),
            symbol='circle'
        ),
        name='Anomaly Trigger',
        hovertemplate='ANOMALY ALERT<br>Tick %{x}<br>TPS: %{y:.2f}<extra></extra>'
    ))

# Plotly styling according to Warm Sunrise spec
fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=10, r=10, t=10, b=10),
    height=340,
    showlegend=False,
    xaxis=dict(
        showgrid=True,
        gridcolor='rgba(229, 231, 235, 0.6)',
        zeroline=False,
        showline=False,
        tickfont=dict(family='JetBrains Mono', size=11, color='#6B7280'),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(229, 231, 235, 0.6)',
        zeroline=False,
        showline=False,
        tickfont=dict(family='JetBrains Mono', size=11, color='#6B7280'),
    ),
    hoverlabel=dict(
        bgcolor='#111827',
        font=dict(family='Plus Jakarta Sans', color='#FFFFFF', size=12),
        bordercolor='transparent'
    )
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# 7. MATHEMATICAL RIGOR & INCIDENT AUDIT LOG
# =============================================================================

with st.expander("🔬 Academic Rigor: Welford Recurrence & Robust MAD Formulation", expanded=False):
    st.markdown(r"""
    ### 1. Welford's Streaming Algorithm
    Traditional two-pass sample variance calculates $s^2 = \frac{1}{n-1}\left(\sum x_i^2 - \frac{(\sum x_i)^2}{n}\right)$, which is vulnerable to catastrophic floating-point cancellation when variance is tiny relative to mean squared.

    Welford updates mean $\bar{x}_k$ and sum of squared differences $M_2$:
    $$\bar{x}_k = \bar{x}_{k-1} + \frac{x_k - \bar{x}_{k-1}}{k}$$
    $$M_{2,k} = M_{2,k-1} + (x_k - \bar{x}_{k-1})(x_k - \bar{x}_k)$$
    $$\sigma^2 = \frac{M_{2,k}}{k - 1} \quad (\mathcal{O}(1)\text{ time, }\mathcal{O}(1)\text{ space})$$

    ### 2. Median Absolute Deviation (MAD) Breakdown Point
    Standard deviation has a **$0\%$ breakdown point** (a single extreme value pushes threshold to infinity, blinding anomaly detection). MAD exhibits an optimal **$50\%$ breakdown point**:
    $$\text{MAD} = \text{median}(|x_i - \tilde{x}|)$$
    $$\hat{\sigma}_{\text{MAD}} = 1.4826 \cdot \text{MAD}$$
    $$\text{Thresholds} = [\tilde{x} - 3\hat{\sigma}_{\text{MAD}}, \tilde{x} + 3\hat{\sigma}_{\text{MAD}}]$$
    """)

# Reset normal baseline trigger
if st.session_state.fault_mode != "NONE":
    if st.button("Reset Normal Baseline ⟲", type="secondary"):
        st.session_state.fault_mode = "NONE"
        st.rerun()


# =============================================================================
# 8. FLOATING BOTTOM STATUS DOCK (MATTE BLACK CONTAINER) & ATTRIBUTION
# =============================================================================

dock_step_text = f"Step {st.session_state.tick} / 100"
fault_tag = f" · Fault: {st.session_state.fault_mode}" if st.session_state.fault_mode != "NONE" else ""

st.markdown(
    f"""
    <div class="bottom-dock">
        <span class="bottom-dock-text">⚡ ChronoPulse · {dock_step_text}{fault_tag}</span>
        <span class="bottom-dock-text" style="color: #9CA3AF;">FSM: {st.session_state.fsm.state.value}</span>
        <span class="bottom-dock-text" style="color: #FDF18A; font-weight: 700; font-size: 12px; border-left: 1px solid #374151; padding-left: 14px;">
            Project by siddhantbhandare1@gmail.com
        </span>
    </div>
    <div style="text-align: center; margin-top: 40px; margin-bottom: 20px; font-size: 13px; font-weight: 600; color: #4B5563;">
        Project by <a href="mailto:siddhantbhandare1@gmail.com" style="color: #111827; font-weight: 700; text-decoration: underline;">siddhantbhandare1@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True,
)

