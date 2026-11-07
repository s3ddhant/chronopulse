/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * 
 * ChronoPulse: Real-Time Streaming Telemetry Anomaly Detector & Incident State Machine
 * Academic-Grade CS Portfolio Artifact
 */

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  Activity,
  AlertTriangle,
  Flame,
  CheckCircle2,
  RefreshCw,
  Play,
  Pause,
  SkipForward,
  Copy,
  Check,
  Download,
  Code2,
  BookOpen,
  Zap,
  ShieldAlert,
  ArrowRight,
  Sliders,
  ExternalLink,
} from 'lucide-react';

// ============================================================================
// 1. STATISTICAL ENGINE & ALGORITHMS (O(1) WELFORD + ROBUST MAD)
// ============================================================================

/**
 * Welford's Algorithm (1962)
 * Numerically stable, one-pass streaming calculation of mean and sample variance.
 * Time: O(1) per update, Space: O(1) total.
 */
class WelfordAccumulator {
  count: number = 0;
  mean: number = 0;
  M2: number = 0;

  update(x: number): { mean: number; variance: number; stdDev: number } {
    this.count += 1;
    const delta = x - this.mean;
    this.mean += delta / this.count;
    const delta2 = x - this.mean;
    this.M2 += delta * delta2;
    const variance = this.count > 1 ? this.M2 / (this.count - 1) : 0;
    const stdDev = Math.sqrt(variance);
    return { mean: this.mean, variance, stdDev };
  }

  getVariance(): number {
    return this.count > 1 ? this.M2 / (this.count - 1) : 0;
  }

  getStdDev(): number {
    return Math.sqrt(this.getVariance());
  }

  reset() {
    this.count = 0;
    this.mean = 0;
    this.M2 = 0;
  }
}

/**
 * Robust Outlier Scoring via Median Absolute Deviation (MAD)
 * Breakdown point = 50% (resists extreme values without shifting baseline bounds).
 * Consistency factor c = 1.4826 for asymptotically normal distribution.
 */
function computeMADEnvelope(
  windowValues: number[],
  k: number = 3.0,
  c: number = 1.4826
): { median: number; lowerBound: number; upperBound: number; robustSigma: number; mad: number } {
  if (windowValues.length === 0) {
    return { median: 100, lowerBound: 70, upperBound: 130, robustSigma: 10, mad: 7 };
  }

  // Calculate Median
  const sorted = [...windowValues].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;

  // Calculate Absolute Deviations from Median
  const absDeviations = windowValues.map((v) => Math.abs(v - median)).sort((a, b) => a - b);
  const madMid = Math.floor(absDeviations.length / 2);
  const mad =
    absDeviations.length % 2 !== 0
      ? absDeviations[madMid]
      : (absDeviations[madMid - 1] + absDeviations[madMid]) / 2;

  // Consistency multiplier for normal distribution
  const robustSigma = mad > 1e-4 ? c * mad : 2.5;
  const lowerBound = Math.max(0, median - k * robustSigma);
  const upperBound = median + k * robustSigma;

  return { median, lowerBound, upperBound, robustSigma, mad };
}

// ============================================================================
// 2. INCIDENT DEBOUNCE FINITE STATE MACHINE (FSM)
// ============================================================================

export type IncidentState = 'NOMINAL' | 'EVALUATING' | 'FIRING' | 'RECOVERED';

interface TelemetryPoint {
  tick: number;
  value: number;
  median: number;
  lowerBound: number;
  upperBound: number;
  isAnomaly: boolean;
  state: IncidentState;
  fault: 'NONE' | 'SURGE' | 'OUTAGE';
}

interface IncidentRecord {
  id: string;
  tick: number;
  value: number;
  type: 'ALERT_TRIGGERED' | 'RECOVERY_DETECTED' | 'OPERATOR_ACK';
  timestamp: string;
}

// Full Python source code for app.py
const PYTHON_APP_SOURCE = `"""
ChronoPulse: Real-Time Streaming Telemetry Anomaly Detector & Incident State Machine
===================================================================================
Master's in Computer Science Admissions Portfolio Artifact
Author: Siddhant Bhandare (siddhantbhandare1@gmail.com)
Date: 2026
License: Apache-2.0
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="ChronoPulse | Telemetry Anomaly Detector",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = \"\"\"
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], .main {
    background: linear-gradient(135deg, #FFD3DF 0%, #FFF8E7 45%, #FDF396 100%) fixed !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #111827 !important;
}

.hero-card {
    background: #FFFFFF;
    border-radius: 28px;
    box-shadow: 0 22px 45px rgba(180, 120, 140, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.8);
    padding: 32px;
    margin-bottom: 24px;
}
.metric-large {
    font-family: 'JetBrains Mono', monospace;
    font-size: 46px;
    font-weight: 800;
    color: #111827;
}
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
}
</style>
\"\"\"
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

class WelfordAccumulator:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> Tuple[float, float]:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        variance = self.M2 / (self.count - 1) if self.count > 1 else 0.0
        return self.mean, variance

def compute_mad_bounds(values: np.ndarray, k: float = 3.0, c: float = 1.4826):
    median = float(np.median(values))
    deviations = np.abs(values - median)
    mad = float(np.median(deviations))
    robust_sigma = c * mad if mad > 1e-4 else 2.5
    lower_bound = max(0.0, median - k * robust_sigma)
    upper_bound = median + k * robust_sigma
    return median, lower_bound, upper_bound, robust_sigma
`;

export default function App() {
  // Streaming simulation state
  const [tick, setTick] = useState<number>(35);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(600); // ms per tick
  const [faultMode, setFaultMode] = useState<'NONE' | 'SURGE' | 'OUTAGE'>('NONE');
  const [consecutiveAnomalies, setConsecutiveAnomalies] = useState<number>(0);
  const [fsmState, setFsmState] = useState<IncidentState>('NOMINAL');
  const [history, setHistory] = useState<TelemetryPoint[]>([]);
  const [incidents, setIncidents] = useState<IncidentRecord[]>([]);
  const [windowSize, setWindowSize] = useState<number>(40);
  const [kMultiplier, setKMultiplier] = useState<number>(3.0);
  const [activeTab, setActiveTab] = useState<'monitor' | 'theory' | 'source'>('monitor');
  const [copiedCode, setCopiedCode] = useState<boolean>(false);
  const [hoveredPoint, setHoveredPoint] = useState<TelemetryPoint | null>(null);

  // Welford Accumulator Instance
  const welfordRef = useRef(new WelfordAccumulator());

  // Synthetic Generator Function
  const generateDataPoint = useCallback(
    (t: number, fault: 'NONE' | 'SURGE' | 'OUTAGE') => {
      const diurnalPeriod = 50.0;
      const diurnal = 95.0 + 20.0 * Math.sin((2.0 * Math.PI * (t % diurnalPeriod)) / diurnalPeriod);
      // Box-Muller transform for stochastic Gaussian noise N(0, 3.2^2)
      const u1 = Math.max(1e-6, Math.random());
      const u2 = Math.random();
      const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      const stochasticNoise = z0 * 3.2;

      let val = diurnal + stochasticNoise;

      if (fault === 'SURGE') {
        val += 85.0 + Math.random() * 15.0;
      } else if (fault === 'OUTAGE') {
        val = Math.max(0, Math.random() * 3.5);
      }

      return Math.round(Math.max(0, val) * 10) / 10;
    },
    []
  );

  // Initial Baseline Population (35 samples)
  useEffect(() => {
    const initialPoints: TelemetryPoint[] = [];
    welfordRef.current.reset();

    for (let i = 0; i <= 35; i++) {
      const v = generateDataPoint(i, 'NONE');
      welfordRef.current.update(v);

      const recentWindow = initialPoints.map((p) => p.value).concat([v]);
      const { median, lowerBound, upperBound } = computeMADEnvelope(
        recentWindow.slice(-windowSize),
        kMultiplier
      );

      initialPoints.push({
        tick: i,
        value: v,
        median,
        lowerBound,
        upperBound,
        isAnomaly: false,
        state: 'NOMINAL',
        fault: 'NONE',
      });
    }

    setHistory(initialPoints);
    setTick(35);
  }, [generateDataPoint, windowSize, kMultiplier]);

  // Step Forward Logic (Single Tick)
  const stepForward = useCallback(
    (overrideFault?: 'NONE' | 'SURGE' | 'OUTAGE') => {
      const activeFault = overrideFault !== undefined ? overrideFault : faultMode;
      const nextTick = tick + 1;
      const nextVal = generateDataPoint(nextTick, activeFault);

      // 1. Welford's streaming update
      welfordRef.current.update(nextVal);

      // 2. Compute Robust MAD envelope over sliding window
      const currentValues = history.map((p) => p.value).concat([nextVal]);
      const windowSlice = currentValues.slice(-windowSize);
      const { median, lowerBound, upperBound } = computeMADEnvelope(windowSlice, kMultiplier);

      const isAnomaly = nextVal < lowerBound || nextVal > upperBound;

      // 3. Finite State Machine Transitions
      let nextState: IncidentState = fsmState;
      let nextConsecutive = consecutiveAnomalies;

      if (fsmState === 'NOMINAL') {
        if (isAnomaly) {
          nextConsecutive = 1;
          nextState = 'EVALUATING';
        } else {
          nextConsecutive = 0;
        }
      } else if (fsmState === 'EVALUATING') {
        if (isAnomaly) {
          nextConsecutive += 1;
          if (nextConsecutive >= 3) {
            nextState = 'FIRING';
            setIncidents((prev) => [
              {
                id: `inc-${Date.now()}-${nextTick}`,
                tick: nextTick,
                value: nextVal,
                type: 'ALERT_TRIGGERED',
                timestamp: new Date().toLocaleTimeString(),
              },
              ...prev.slice(0, 9),
            ]);
          }
        } else {
          // Debounce reset: Glitch absorbed without firing
          nextConsecutive = 0;
          nextState = 'NOMINAL';
        }
      } else if (fsmState === 'FIRING') {
        if (!isAnomaly) {
          nextConsecutive = 0;
          nextState = 'RECOVERED';
          setIncidents((prev) => [
            {
              id: `rec-${Date.now()}-${nextTick}`,
              tick: nextTick,
              value: nextVal,
              type: 'RECOVERY_DETECTED',
              timestamp: new Date().toLocaleTimeString(),
            },
            ...prev.slice(0, 9),
          ]);
        } else {
          nextConsecutive += 1;
        }
      } else if (fsmState === 'RECOVERED') {
        if (isAnomaly) {
          nextConsecutive = 1;
          nextState = 'FIRING';
        } else {
          nextConsecutive = 0;
          nextState = 'NOMINAL';
        }
      }

      setFsmState(nextState);
      setConsecutiveAnomalies(nextConsecutive);
      setTick(nextTick);

      const newPoint: TelemetryPoint = {
        tick: nextTick,
        value: nextVal,
        median,
        lowerBound,
        upperBound,
        isAnomaly,
        state: nextState,
        fault: activeFault,
      };

      setHistory((prev) => [...prev.slice(-80), newPoint]);
    },
    [tick, faultMode, generateDataPoint, history, windowSize, kMultiplier, fsmState, consecutiveAnomalies]
  );

  // Playback timer loop
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      stepForward();
    }, playbackSpeed);
    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, stepForward]);

  // Manual Operator Acknowledgment
  const acknowledgeAlert = () => {
    setFsmState('NOMINAL');
    setConsecutiveAnomalies(0);
    setIncidents((prev) => [
      {
        id: `ack-${Date.now()}-${tick}`,
        tick,
        value: history[history.length - 1]?.value || 0,
        type: 'OPERATOR_ACK',
        timestamp: new Date().toLocaleTimeString(),
      },
      ...prev.slice(0, 9),
    ]);
  };

  // Download app.py
  const downloadPythonFile = () => {
    const blob = new Blob([PYTHON_APP_SOURCE], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'app.py';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Copy code handler
  const copyPythonCode = () => {
    navigator.clipboard.writeText(PYTHON_APP_SOURCE);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2200);
  };

  // Current telemetry point metrics
  const currentPoint = history[history.length - 1] || {
    tick: 0,
    value: 95.0,
    median: 95.0,
    lowerBound: 70.0,
    upperBound: 120.0,
    isAnomaly: false,
    state: 'NOMINAL',
    fault: 'NONE',
  };

  // Chart view window
  const visiblePoints = useMemo(() => history.slice(-50), [history]);

  // Scaled coordinates for SVG chart matching Plotly spec
  const chartHeight = 260;
  const chartWidth = 720;
  const padding = { top: 24, bottom: 32, left: 45, right: 24 };

  const { minY, maxY, minX, maxX } = useMemo(() => {
    if (visiblePoints.length === 0) return { minY: 0, maxY: 200, minX: 0, maxX: 50 };
    const allVals = visiblePoints.flatMap((p) => [p.value, p.lowerBound, p.upperBound]);
    const minVal = Math.max(0, Math.min(...allVals) - 10);
    const maxVal = Math.max(...allVals) + 15;
    const minT = visiblePoints[0].tick;
    const maxT = visiblePoints[visiblePoints.length - 1].tick;
    return {
      minY: minVal,
      maxY: Math.max(minVal + 30, maxVal),
      minX: minT,
      maxX: Math.max(minT + 10, maxT),
    };
  }, [visiblePoints]);

  const scaleX = useCallback(
    (t: number) => {
      const rangeX = maxX - minX || 1;
      return padding.left + ((t - minX) / rangeX) * (chartWidth - padding.left - padding.right);
    },
    [minX, maxX, chartWidth, padding.left, padding.right]
  );

  const scaleY = useCallback(
    (v: number) => {
      const rangeY = maxY - minY || 1;
      return chartHeight - padding.bottom - ((v - minY) / rangeY) * (chartHeight - padding.top - padding.bottom);
    },
    [minY, maxY, chartHeight, padding.top, padding.bottom]
  );

  // SVG Paths
  const { linePath, areaPath } = useMemo(() => {
    if (visiblePoints.length === 0) return { linePath: '', areaPath: '' };

    // Primary Signal Line
    const lineSegments = visiblePoints.map((p, idx) => {
      const x = scaleX(p.tick);
      const y = scaleY(p.value);
      return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    });

    // Shaded Envelope Area (Upper bounds left-to-right, then lower bounds right-to-left)
    const upperPoints = visiblePoints.map((p) => `${scaleX(p.tick).toFixed(1)} ${scaleY(p.upperBound).toFixed(1)}`);
    const lowerPoints = [...visiblePoints]
      .reverse()
      .map((p) => `${scaleX(p.tick).toFixed(1)} ${scaleY(p.lowerBound).toFixed(1)}`);

    const areaSegments = `M ${upperPoints[0]} L ${upperPoints.join(' L ')} L ${lowerPoints.join(' L ')} Z`;

    return {
      linePath: lineSegments.join(' '),
      areaPath: areaSegments,
    };
  }, [visiblePoints, scaleX, scaleY]);

  // State Pills
  const stateBadgeInfo = {
    NOMINAL: {
      label: 'Stable Flow',
      bg: 'bg-emerald-100 text-emerald-800 border-emerald-200',
      dot: 'bg-emerald-500 shadow-emerald-400',
    },
    EVALUATING: {
      label: `Evaluating (${consecutiveAnomalies}/3)`,
      bg: 'bg-amber-100 text-amber-900 border-amber-200',
      dot: 'bg-amber-500 shadow-amber-400',
    },
    FIRING: {
      label: 'Anomaly Detected',
      bg: 'bg-rose-100 text-rose-800 border-rose-200 animate-pulse',
      dot: 'bg-rose-600 shadow-rose-500',
    },
    RECOVERED: {
      label: 'Self-Healing (Recovered)',
      bg: 'bg-indigo-100 text-indigo-800 border-indigo-200',
      dot: 'bg-indigo-500 shadow-indigo-400',
    },
  }[fsmState];

  return (
    <div className="min-h-screen text-slate-900 px-4 py-8 md:py-12 pb-32 flex flex-col items-center">
      <div className="w-full max-w-4xl space-y-6">

        {/* Top Header & Navigation Tabs */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-bold tracking-widest text-slate-500 uppercase">
              <span className={`w-2.5 h-2.5 rounded-full inline-block shadow-sm ${stateBadgeInfo.dot} animate-ping`} />
              <span>TELEMETRY AUDITOR · REAL-TIME</span>
            </div>
            <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-slate-900 mt-1">
              System Heartbeat
            </h1>
          </div>

          <div className="flex items-center gap-2 bg-white/70 backdrop-blur-md p-1.5 rounded-full border border-white/60 shadow-sm self-start md:self-auto">
            <button
              onClick={() => setActiveTab('monitor')}
              className={`px-4 py-1.5 rounded-full text-xs font-bold transition-all ${
                activeTab === 'monitor'
                  ? 'bg-slate-900 text-white shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Live Telemetry
            </button>
            <button
              onClick={() => setActiveTab('theory')}
              className={`px-4 py-1.5 rounded-full text-xs font-bold transition-all ${
                activeTab === 'theory'
                  ? 'bg-slate-900 text-white shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Academic Theory
            </button>
            <button
              onClick={() => setActiveTab('source')}
              className={`px-4 py-1.5 rounded-full text-xs font-bold transition-all flex items-center gap-1.5 ${
                activeTab === 'source'
                  ? 'bg-slate-900 text-white shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Code2 className="w-3.5 h-3.5" />
              <span>app.py Source</span>
            </button>
          </div>
        </div>

        {/* ================================================================ */}
        {/* TAB 1: LIVE MONITOR (Hero Card + Transparent Dynamic Chart)      */}
        {/* ================================================================ */}
        {activeTab === 'monitor' && (
          <div className="space-y-6">
            {/* The Floating Hero Card */}
            <div className="floating-hero-card p-6 md:p-8">
              <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 mb-4">
                <div>
                  <div className="text-xs font-bold tracking-wider text-slate-400 uppercase mb-1">
                    Metric Drift & Throughput
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className="font-mono text-4xl md:text-5xl font-extrabold tracking-tight text-slate-900">
                      {currentPoint.value.toFixed(1)}
                    </span>
                    <span className="text-lg md:text-xl font-bold text-slate-400">TPS</span>
                    {faultMode !== 'NONE' && (
                      <span className="ml-3 px-2.5 py-0.5 rounded-md text-xs font-extrabold bg-rose-100 text-rose-700 tracking-wide uppercase animate-pulse">
                        Fault: {faultMode}
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <div className={`flex items-center gap-2 px-4 py-2 rounded-full border text-xs font-bold shadow-sm ${stateBadgeInfo.bg}`}>
                    <span className={`w-2 h-2 rounded-full ${stateBadgeInfo.dot}`} />
                    <span>{stateBadgeInfo.label}</span>
                  </div>
                </div>
              </div>

              {/* Subtext description according to prompt */}
              <p className="text-xs md:text-sm text-slate-500 leading-relaxed">
                Dynamic MAD envelope actively bounding <strong className="text-slate-800">{windowSize}</strong> observations.
                Safe Range: <code className="bg-slate-100 px-1.5 py-0.5 rounded font-mono text-slate-800">[{currentPoint.lowerBound.toFixed(1)} - {currentPoint.upperBound.toFixed(1)} TPS]</code> ·
                Welford Rolling Mean: <code className="bg-slate-100 px-1.5 py-0.5 rounded font-mono text-slate-800">{welfordRef.current.mean.toFixed(1)} TPS</code> (σ = {welfordRef.current.getStdDev().toFixed(2)})
              </p>

              {/* Action Buttons: Next Step & Acknowledge Alert */}
              <div className="mt-6 pt-5 border-t border-slate-100 flex flex-wrap items-center gap-3">
                <button
                  onClick={() => stepForward()}
                  className="pill-primary px-7 py-3 text-sm flex items-center gap-2 cursor-pointer"
                >
                  <SkipForward className="w-4 h-4" />
                  <span>Next Step</span>
                </button>

                <button
                  onClick={acknowledgeAlert}
                  disabled={fsmState === 'NOMINAL'}
                  className={`pill-secondary px-6 py-3 text-sm flex items-center gap-2 cursor-pointer ${
                    fsmState === 'NOMINAL' ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  <span>Acknowledge Alert</span>
                </button>

                <div className="flex items-center gap-2 ml-auto">
                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="p-3 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-700 transition-colors cursor-pointer"
                    title={isPlaying ? 'Pause auto-streaming' : 'Auto-stream continuous ticks'}
                  >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-slate-700" />}
                  </button>

                  <select
                    value={playbackSpeed}
                    onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                    className="bg-slate-100 text-xs font-semibold px-3 py-2.5 rounded-full border-none text-slate-700 cursor-pointer"
                  >
                    <option value={1000}>1.0x (1.0s)</option>
                    <option value={600}>1.5x (0.6s)</option>
                    <option value={300}>3.0x (0.3s)</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Embedded Dynamic Telemetry Chart (Plotly Transparent Aesthetic) */}
            <div className="floating-hero-card p-6 md:p-8">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-sm font-bold uppercase tracking-wider text-slate-700 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-slate-800" />
                    <span>Real-Time Ingestion Stream & MAD Envelope</span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Charcoal line is live signal; shaded pink area is dynamic [Median ± 3·MAD] threshold
                  </p>
                </div>

                {/* Legend Chips */}
                <div className="hidden sm:flex items-center gap-4 text-xs font-medium text-slate-500">
                  <div className="flex items-center gap-1.5">
                    <span className="w-3 h-0.5 bg-slate-800 inline-block rounded-full" />
                    <span>Signal (TPS)</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-3 h-2 bg-[#FFB3C6]/60 inline-block rounded-sm" />
                    <span>MAD Envelope</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#F43F5E] inline-block border border-white" />
                    <span>Anomaly</span>
                  </div>
                </div>
              </div>

              {/* Transparent Chart Canvas */}
              <div className="relative w-full overflow-hidden bg-transparent rounded-2xl">
                <svg
                  viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                  className="w-full h-auto max-h-[340px] overflow-visible"
                >
                  {/* Subtle Gridlines */}
                  {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => {
                    const y = padding.top + pct * (chartHeight - padding.top - padding.bottom);
                    const valAtY = maxY - pct * (maxY - minY);
                    return (
                      <g key={i}>
                        <line
                          x1={padding.left}
                          y1={y}
                          x2={chartWidth - padding.right}
                          y2={y}
                          stroke="#E5E7EB"
                          strokeDasharray="3 3"
                          strokeWidth="1"
                        />
                        <text
                          x={padding.left - 8}
                          y={y + 3}
                          fontSize="10"
                          fontFamily="JetBrains Mono"
                          fill="#9CA3AF"
                          textAnchor="end"
                        >
                          {valAtY.toFixed(0)}
                        </text>
                      </g>
                    );
                  })}

                  {/* 1. Shaded MAD Adaptive Envelope */}
                  {areaPath && (
                    <path
                      d={areaPath}
                      fill="rgba(255, 179, 198, 0.35)"
                      stroke="rgba(255, 179, 198, 0.8)"
                      strokeWidth="1"
                      strokeDasharray="2 2"
                    />
                  )}

                  {/* 2. Primary Signal Trace: Deep Charcoal / Midnight Navy */}
                  {linePath && (
                    <path
                      d={linePath}
                      fill="none"
                      stroke="#1E293B"
                      strokeWidth="2.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  )}

                  {/* 3. Anomaly Trigger Points (Bright coral-pink circular markers #F43F5E) */}
                  {visiblePoints.map((pt) => {
                    const cx = scaleX(pt.tick);
                    const cy = scaleY(pt.value);
                    if (pt.isAnomaly) {
                      return (
                        <g key={`anom-${pt.tick}`}>
                          <circle
                            cx={cx}
                            cy={cy}
                            r="9"
                            fill="#F43F5E"
                            fillOpacity="0.25"
                            className="animate-ping"
                          />
                          <circle
                            cx={cx}
                            cy={cy}
                            r="5.5"
                            fill="#F43F5E"
                            stroke="#FFFFFF"
                            strokeWidth="2"
                          />
                        </g>
                      );
                    }
                    return null;
                  })}

                  {/* Hover Interactivity & Crosshair */}
                  {visiblePoints.map((pt) => {
                    const cx = scaleX(pt.tick);
                    const cy = scaleY(pt.value);
                    return (
                      <circle
                        key={`hover-${pt.tick}`}
                        cx={cx}
                        cy={cy}
                        r="12"
                        fill="transparent"
                        className="cursor-pointer"
                        onMouseEnter={() => setHoveredPoint(pt)}
                        onMouseLeave={() => setHoveredPoint(null)}
                      />
                    );
                  })}

                  {/* Tick axis numbers */}
                  {visiblePoints
                    .filter((_, idx) => idx % 8 === 0 || idx === visiblePoints.length - 1)
                    .map((pt) => (
                      <text
                        key={`tick-num-${pt.tick}`}
                        x={scaleX(pt.tick)}
                        y={chartHeight - 10}
                        fontSize="10"
                        fontFamily="JetBrains Mono"
                        fill="#9CA3AF"
                        textAnchor="middle"
                      >
                        t={pt.tick}
                      </text>
                    ))}
                </svg>

                {/* Tooltip Card overlay on hover */}
                {hoveredPoint && (
                  <div className="absolute top-3 right-3 bg-slate-900/95 text-white text-xs px-3.5 py-2.5 rounded-xl shadow-xl backdrop-blur-md border border-slate-700/50 pointer-events-none">
                    <div className="font-mono text-amber-300 font-bold">Tick #{hoveredPoint.tick}</div>
                    <div className="mt-1 flex items-baseline gap-2">
                      <span className="text-base font-extrabold font-mono">{hoveredPoint.value.toFixed(1)} TPS</span>
                      {hoveredPoint.isAnomaly && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-rose-500/80 font-bold text-white uppercase">
                          Anomaly
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-slate-300 mt-1">
                      Safe Bounds: [{hoveredPoint.lowerBound.toFixed(1)} - {hoveredPoint.upperBound.toFixed(1)}]
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Interactive Fault Injection Panel */}
            <div className="floating-hero-card p-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div>
                  <h4 className="text-xs font-bold tracking-wider text-slate-500 uppercase">
                    Interactive Fault Injection Suite
                  </h4>
                  <p className="text-xs text-slate-500">
                    Inject chaos events into the synthetic telemetry stream to verify Debounce FSM transitions
                  </p>
                </div>

                {faultMode !== 'NONE' && (
                  <button
                    onClick={() => {
                      setFaultMode('NONE');
                      stepForward('NONE');
                    }}
                    className="self-start sm:self-auto px-4 py-1.5 rounded-full text-xs font-bold bg-slate-900 text-white hover:bg-slate-800 transition-colors flex items-center gap-1.5 cursor-pointer shadow-sm"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    <span>Reset Normal Baseline</span>
                  </button>
                )}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <button
                  onClick={() => {
                    setFaultMode('SURGE');
                    stepForward('SURGE');
                  }}
                  className={`p-4 rounded-2xl border transition-all text-left cursor-pointer ${
                    faultMode === 'SURGE'
                      ? 'bg-rose-50 border-rose-300 ring-2 ring-rose-400'
                      : 'bg-white hover:bg-slate-50 border-slate-200'
                  }`}
                >
                  <div className="flex items-center gap-2 text-rose-600 font-bold text-sm mb-1">
                    <Flame className="w-4 h-4" />
                    <span>Traffic Surge</span>
                  </div>
                  <p className="text-xs text-slate-500 leading-snug">
                    Simulates flash-crowd DDoS or viral traffic spiking TPS by +85
                  </p>
                </button>

                <button
                  onClick={() => {
                    setFaultMode('OUTAGE');
                    stepForward('OUTAGE');
                  }}
                  className={`p-4 rounded-2xl border transition-all text-left cursor-pointer ${
                    faultMode === 'OUTAGE'
                      ? 'bg-amber-50 border-amber-300 ring-2 ring-amber-400'
                      : 'bg-white hover:bg-slate-50 border-slate-200'
                  }`}
                >
                  <div className="flex items-center gap-2 text-amber-700 font-bold text-sm mb-1">
                    <ShieldAlert className="w-4 h-4" />
                    <span>Service Outage</span>
                  </div>
                  <p className="text-xs text-slate-500 leading-snug">
                    Simulates database connection pool exhaustion flatlining throughput to ~0 TPS
                  </p>
                </button>

                <button
                  onClick={() => {
                    setFaultMode('NONE');
                    stepForward('NONE');
                  }}
                  className={`p-4 rounded-2xl border transition-all text-left cursor-pointer ${
                    faultMode === 'NONE'
                      ? 'bg-emerald-50 border-emerald-300 ring-2 ring-emerald-400'
                      : 'bg-white hover:bg-slate-50 border-slate-200'
                  }`}
                >
                  <div className="flex items-center gap-2 text-emerald-700 font-bold text-sm mb-1">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Normal Baseline</span>
                  </div>
                  <p className="text-xs text-slate-500 leading-snug">
                    Resets to continuous diurnal sine wave with stochastic Gaussian noise
                  </p>
                </button>
              </div>
            </div>

            {/* State Machine Transition Invariants Card */}
            <div className="floating-hero-card p-6">
              <h4 className="text-xs font-bold tracking-wider text-slate-500 uppercase mb-3">
                Debounce State Machine Execution State
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
                <div
                  className={`p-3 rounded-2xl border transition-all ${
                    fsmState === 'NOMINAL'
                      ? 'bg-emerald-50 border-emerald-300 shadow-sm font-bold text-emerald-900'
                      : 'bg-slate-50/60 border-slate-200 text-slate-400'
                  }`}
                >
                  <div className="text-xs uppercase tracking-wider mb-0.5">State 0</div>
                  <div className="text-sm font-extrabold">NOMINAL</div>
                  <div className="text-[10px] mt-1 text-slate-500">Steady Baseline</div>
                </div>

                <div
                  className={`p-3 rounded-2xl border transition-all ${
                    fsmState === 'EVALUATING'
                      ? 'bg-amber-50 border-amber-300 shadow-sm font-bold text-amber-900'
                      : 'bg-slate-50/60 border-slate-200 text-slate-400'
                  }`}
                >
                  <div className="text-xs uppercase tracking-wider mb-0.5">State 1</div>
                  <div className="text-sm font-extrabold">EVALUATING</div>
                  <div className="text-[10px] mt-1 text-slate-500">{consecutiveAnomalies}/3 Debounce</div>
                </div>

                <div
                  className={`p-3 rounded-2xl border transition-all ${
                    fsmState === 'FIRING'
                      ? 'bg-rose-50 border-rose-300 shadow-sm font-bold text-rose-900 animate-pulse'
                      : 'bg-slate-50/60 border-slate-200 text-slate-400'
                  }`}
                >
                  <div className="text-xs uppercase tracking-wider mb-0.5">State 2</div>
                  <div className="text-sm font-extrabold">FIRING</div>
                  <div className="text-[10px] mt-1 text-slate-500">Active Incident</div>
                </div>

                <div
                  className={`p-3 rounded-2xl border transition-all ${
                    fsmState === 'RECOVERED'
                      ? 'bg-indigo-50 border-indigo-300 shadow-sm font-bold text-indigo-900'
                      : 'bg-slate-50/60 border-slate-200 text-slate-400'
                  }`}
                >
                  <div className="text-xs uppercase tracking-wider mb-0.5">State 3</div>
                  <div className="text-sm font-extrabold">RECOVERED</div>
                  <div className="text-[10px] mt-1 text-slate-500">Self-Healing</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ================================================================ */}
        {/* TAB 2: ACADEMIC THEORY & STATISTICAL PROOFS                      */}
        {/* ================================================================ */}
        {activeTab === 'theory' && (
          <div className="space-y-6">
            <div className="floating-hero-card p-6 md:p-8 space-y-6">
              <div>
                <span className="text-xs font-bold tracking-wider text-rose-500 uppercase">
                  Master's in CS Portfolio Technical Dossier
                </span>
                <h2 className="text-2xl font-extrabold text-slate-900 mt-1">
                  Streaming Statistical Inference & Robust Estimation
                </h2>
              </div>

              {/* Welford's Section */}
              <div className="p-5 rounded-2xl bg-amber-50/60 border border-amber-100 space-y-3">
                <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                  <Zap className="w-4 h-4 text-amber-600" />
                  <span>1. Online Welford Algorithm (1962)</span>
                </h3>
                <p className="text-xs md:text-sm text-slate-600 leading-relaxed">
                  Naive variance calculation requires two passes: first computing sample mean, then computing deviations.
                  The computational one-pass formula <code className="font-mono bg-white px-1.5 py-0.5 rounded text-amber-900">Var = (Σx² - (Σx)²/n)/(n-1)</code> suffers from severe catastrophic cancellation when variance is small relative to the squared mean.
                </p>
                <div className="bg-white p-4 rounded-xl border border-amber-200 font-mono text-xs md:text-sm text-slate-800 space-y-1">
                  <div>M₁ = x₁, S₁ = 0</div>
                  <div>M_k = M_(k-1) + (x_k - M_(k-1)) / k</div>
                  <div>S_k = S_(k-1) + (x_k - M_(k-1)) · (x_k - M_k)</div>
                  <div className="text-amber-800 font-bold pt-1">Sample Variance: s² = S_k / (k - 1)  [O(1) Time · O(1) Space]</div>
                </div>
              </div>

              {/* MAD Section */}
              <div className="p-5 rounded-2xl bg-rose-50/60 border border-rose-100 space-y-3">
                <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-rose-600" />
                  <span>2. Median Absolute Deviation (MAD) Breakdown Point</span>
                </h3>
                <p className="text-xs md:text-sm text-slate-600 leading-relaxed">
                  Classical 3-sigma rules rely on standard deviation, which has a <strong>0% breakdown point</strong>: a single sufficiently large outlier corrupts the mean and blows up the variance, hiding subsequent anomalies (masking effect). MAD provides an optimal <strong>50% breakdown point</strong>:
                </p>
                <div className="bg-white p-4 rounded-xl border border-rose-200 font-mono text-xs md:text-sm text-slate-800 space-y-1">
                  <div>MAD = median( | x_i - median(X) | )</div>
                  <div>σ_hat = 1.4826 · MAD  (asymptotically normal consistency factor)</div>
                  <div className="text-rose-800 font-bold pt-1">Dynamic Safe Envelope: [Median - 3·σ_hat, Median + 3·σ_hat]</div>
                </div>
              </div>

              {/* FSM Debounce Section */}
              <div className="p-5 rounded-2xl bg-indigo-50/60 border border-indigo-100 space-y-3">
                <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                  <ShieldAlert className="w-4 h-4 text-indigo-600" />
                  <span>3. Alarm Debounce Finite State Machine</span>
                </h3>
                <p className="text-xs md:text-sm text-slate-600 leading-relaxed">
                  In distributed telemetry, isolated transient spikes (network re-transmissions, micro-bursts) routinely occur without indicating sustained infrastructure failure. Debounce filtering enforces hysteresis:
                </p>
                <ul className="text-xs md:text-sm text-slate-600 list-disc list-inside space-y-1.5 pl-2">
                  <li><strong>NOMINAL → EVALUATING:</strong> Triggered on 1st out-of-envelope anomaly.</li>
                  <li><strong>EVALUATING → FIRING:</strong> Requires 3 consecutive anomalous observations.</li>
                  <li><strong>EVALUATING → NOMINAL:</strong> Automatically rolls back if metric returns within envelope before 3 ticks.</li>
                  <li><strong>FIRING → RECOVERED:</strong> Metric recovers within bounds; self-healing lifecycle.</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* ================================================================ */}
        {/* TAB 3: STANDALONE PYTHON APP.PY SOURCE CODE INSPECTOR            */}
        {/* ================================================================ */}
        {activeTab === 'source' && (
          <div className="space-y-6">
            <div className="floating-hero-card p-6 md:p-8 space-y-5">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <div className="text-xs font-bold tracking-wider text-slate-400 uppercase">
                    Standalone Streamlit Script
                  </div>
                  <h2 className="text-2xl font-extrabold text-slate-900">app.py Code Inspector</h2>
                  <p className="text-xs text-slate-500 mt-1">
                    Ready to run offline with: <code className="bg-slate-100 px-2 py-0.5 rounded font-mono text-slate-800">streamlit run app.py</code>
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={copyPythonCode}
                    className="pill-secondary px-4 py-2.5 text-xs flex items-center gap-1.5 cursor-pointer"
                  >
                    {copiedCode ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>{copiedCode ? 'Copied!' : 'Copy Code'}</span>
                  </button>
                  <button
                    onClick={downloadPythonFile}
                    className="pill-primary px-4 py-2.5 text-xs flex items-center gap-1.5 cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download app.py</span>
                  </button>
                </div>
              </div>

              {/* Execution instructions banner */}
              <div className="p-4 rounded-xl bg-slate-900 text-white font-mono text-xs flex flex-col md:flex-row md:items-center justify-between gap-2">
                <div>
                  <span className="text-emerald-400 font-bold">$</span> pip install streamlit numpy scipy pandas plotly
                  <br />
                  <span className="text-emerald-400 font-bold">$</span> streamlit run app.py
                </div>
                <span className="text-slate-400 text-[11px]">100% Offline · Zero API Keys</span>
              </div>

              {/* Code display */}
              <pre className="p-5 rounded-2xl bg-slate-950 text-slate-200 font-mono text-xs overflow-x-auto max-h-[500px] border border-slate-800 leading-relaxed">
                <code>{PYTHON_APP_SOURCE}</code>
              </pre>
            </div>
          </div>
        )}

      </div>

      {/* ================================================================== */}
      {/* 4. DYNAMIC STATUS DOCK / BOTTOM BAR (FLOATING MATTE BLACK CONTAINER)*/}
      {/* ================================================================== */}
      <div className="fixed bottom-5 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 z-50 max-w-[95vw]">
        <div className="floating-bottom-dock flex items-center gap-3 sm:gap-5 px-5 py-3 border border-white/10 shadow-2xl">
          <div className="flex items-center gap-2 font-mono text-xs sm:text-sm font-semibold text-slate-100">
            <Zap className="w-4 h-4 text-yellow-300" />
            <span>Step {tick} / 100</span>
            {faultMode !== 'NONE' && (
              <span className="hidden sm:inline text-rose-400 text-xs font-mono">
                · {faultMode}
              </span>
            )}
          </div>

          <div className="h-4 w-px bg-slate-700 hidden sm:block" />

          <div className="hidden md:flex items-center gap-1.5 text-xs text-slate-300 font-mono">
            <span className="text-slate-400">FSM:</span>
            <span className="font-bold text-white">{fsmState}</span>
          </div>

          <div className="h-4 w-px bg-slate-700 hidden lg:block" />

          <div className="hidden lg:flex items-center text-xs text-slate-300 font-medium">
            <span>Project by&nbsp;</span>
            <a
              href="mailto:siddhantbhandare1@gmail.com"
              className="text-[#FDF18A] hover:underline font-semibold"
            >
              siddhantbhandare1@gmail.com
            </a>
          </div>

          {/* Quick Anomaly Injection yellow accent button */}
          <button
            onClick={() => {
              const nextFault = faultMode === 'NONE' ? 'SURGE' : 'NONE';
              setFaultMode(nextFault);
              stepForward(nextFault);
            }}
            className="pill-dock-yellow px-4 py-1.5 text-xs font-extrabold flex items-center gap-1.5 cursor-pointer shadow-md"
          >
            <Flame className="w-3.5 h-3.5" />
            <span>{faultMode === 'NONE' ? 'Trigger Anomaly' : 'Clear Fault'}</span>
          </button>
        </div>

        {/* Mobile & Tablet Visible Attribution Pill */}
        <div className="lg:hidden text-[11px] font-semibold text-slate-700 bg-white/80 backdrop-blur-md px-3.5 py-1 rounded-full border border-white/80 shadow-sm flex items-center gap-1.5">
          <span>Project by</span>
          <a
            href="mailto:siddhantbhandare1@gmail.com"
            className="text-slate-900 font-bold hover:underline"
          >
            siddhantbhandare1@gmail.com
          </a>
        </div>
      </div>

      {/* Page Base Footer */}
      <footer className="w-full text-center pt-8 pb-10 text-xs font-medium text-slate-500">
        <span>Project by </span>
        <a
          href="mailto:siddhantbhandare1@gmail.com"
          className="text-slate-800 hover:text-slate-950 font-bold underline underline-offset-4 decoration-slate-300"
        >
          siddhantbhandare1@gmail.com
        </a>
      </footer>
    </div>
  );
}
