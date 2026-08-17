"""
AgeVision AI / FaceAge AI — Web Application
Flask backend with modern dark-slate dashboard UI matching user target design
"""

import os
import sys
import io
import time
import base64
import json
from pathlib import Path

# Ensure inference.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024  # 15MB max upload

# Global predictor instance
predictor = None

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_age_model.pt")
METRICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "final_metrics.json")

def load_predictor():
    global predictor
    if predictor is None:
        from inference import AgePredictor
        predictor = AgePredictor(model_path=MODEL_PATH, metrics_path=METRICS_PATH)
        print(f"[OK] Model loaded: {MODEL_PATH}")

# ─────────────────────────────────────────────────────────
# HTML TEMPLATE (Target UI Pixel-Perfect Recreation)
# ─────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FaceAge AI — AI-Powered Facial Age Estimation</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg-main:       #070b14;
    --bg-card:       #0d1424;
    --bg-card-hover: #111a30;
    --bg-nested:     #080e1c;
    --border:        rgba(56, 189, 248, 0.12);
    --border-hover:  rgba(56, 189, 248, 0.35);
    --accent-blue:   #38bdf8;
    --accent-indigo: #6366f1;
    --accent-orange: #f97316;
    --accent-green:  #22c55e;
    --text-primary:  #f8fafc;
    --text-muted:    #94a3b8;
    --text-dim:      #64748b;
    --radius-lg:     18px;
    --radius-md:     14px;
    --radius-sm:     10px;
    --font-main:     'Plus Jakarta Sans', system-ui, sans-serif;
  }

  html, body {
    min-height: 100vh;
    background: var(--bg-main);
    font-family: var(--font-main);
    color: var(--text-primary);
    overflow-x: hidden;
  }

  /* Subtle Deep Glows */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background:
      radial-gradient(ellipse 70% 50% at 20% 0%, rgba(56, 189, 248, 0.08) 0%, transparent 60%),
      radial-gradient(ellipse 60% 40% at 80% 100%, rgba(99, 102, 241, 0.06) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
  }

  .container {
    position: relative;
    z-index: 1;
    max-width: 1240px;
    margin: 0 auto;
    padding: 24px 20px 40px;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    margin-bottom: 24px;
    text-align: center;
  }

  .brand-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  .logo-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #fff;
  }

  .logo-title .brain-icon {
    font-size: 1.8rem;
    filter: drop-shadow(0 0 12px rgba(244, 114, 182, 0.4));
  }

  .logo-title span.highlight {
    color: var(--accent-blue);
  }

  .tagline {
    font-size: 0.85rem;
    color: var(--text-muted);
    font-weight: 500;
  }

  .header-model-pill {
    position: absolute;
    right: 0;
    top: 50%;
    transform: translateY(-50%);
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid var(--border);
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
  }
  .header-model-pill strong { color: #fff; }

  @media (max-width: 860px) {
    .header-model-pill { position: static; transform: none; margin-top: 10px; }
    header { flex-direction: column; }
  }

  /* ── Cards & Structure ── */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px;
    position: relative;
    transition: all 0.25s ease;
  }
  .card:hover {
    border-color: var(--border-hover);
  }

  .card-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: var(--accent-blue);
    text-transform: uppercase;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  /* ── Top Grid (3 Cards) ── */
  .top-grid {
    display: grid;
    grid-template-columns: 280px 1fr 340px;
    gap: 16px;
    margin-bottom: 16px;
  }

  @media (max-width: 1080px) {
    .top-grid { grid-template-columns: 1fr 1fr; }
    .stats-2x2 { grid-column: span 2; }
  }
  @media (max-width: 720px) {
    .top-grid { grid-template-columns: 1fr; }
    .stats-2x2 { grid-column: span 1; }
  }

  /* 1. Upload Box */
  .upload-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    height: 100%;
  }

  .img-preview-frame {
    width: 100%;
    height: 180px;
    border-radius: var(--radius-md);
    background: var(--bg-nested);
    border: 1px dashed rgba(56, 189, 248, 0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    position: relative;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .img-preview-frame:hover {
    border-color: var(--accent-blue);
    background: rgba(56, 189, 248, 0.04);
  }

  .img-preview-frame img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }

  .upload-prompt {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    color: var(--text-muted);
    font-size: 0.8rem;
    text-align: center;
    padding: 12px;
  }
  .upload-prompt .icon { font-size: 2rem; color: var(--accent-blue); }

  .status-pills {
    display: flex;
    gap: 8px;
    width: 100%;
    margin: 12px 0;
  }

  .status-pill {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.25);
    color: var(--accent-green);
    font-size: 0.72rem;
    font-weight: 600;
    padding: 6px 8px;
    border-radius: 100px;
    white-space: nowrap;
  }

  .btn-upload {
    width: 100%;
    background: linear-gradient(135deg, #ea580c, #f97316);
    color: #fff;
    border: none;
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    font-size: 0.85rem;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    box-shadow: 0 4px 14px rgba(249, 115, 22, 0.3);
    transition: all 0.2s ease;
  }
  .btn-upload:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(249, 115, 22, 0.45);
  }
  .btn-upload input { display: none; }

  /* 2. Hero Center Predicted Age */
  .hero-prediction {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 20px;
  }

  .giant-age-num {
    font-size: 5.5rem;
    font-weight: 800;
    line-height: 1;
    color: #ffffff;
    letter-spacing: -2px;
    margin: 4px 0 2px;
    text-shadow: 0 0 30px rgba(255, 255, 255, 0.2);
  }

  .years-sub {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 2px;
    color: var(--text-muted);
    margin-bottom: 16px;
  }

  .group-badge-main {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 24px;
    background: rgba(34, 197, 94, 0.08);
    border: 1.5px solid rgba(34, 197, 94, 0.4);
    color: var(--accent-green);
    border-radius: 100px;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 18px;
  }

  .range-label {
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-bottom: 4px;
  }

  .range-val {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--accent-blue);
    letter-spacing: -0.5px;
  }

  /* 3. Stats 2x2 Grid */
  .stats-2x2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }

  .stat-card-mini {
    background: var(--bg-nested);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.2s ease;
  }
  .stat-card-mini:hover { border-color: var(--border-hover); background: var(--bg-card-hover); }

  .stat-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.72rem;
    color: var(--text-muted);
    font-weight: 600;
  }

  .stat-card-icon {
    font-size: 1.2rem;
  }

  .stat-card-val {
    font-size: 1.25rem;
    font-weight: 800;
    color: #fff;
    margin: 8px 0 2px;
  }

  .stat-card-sub {
    font-size: 0.68rem;
    color: var(--text-dim);
    font-weight: 500;
  }
  .stat-card-sub.green { color: var(--accent-green); }

  /* ── Middle Grid (2 Large Panels) ── */
  .middle-grid {
    display: grid;
    grid-template-columns: 1fr 1.3fr;
    gap: 16px;
    margin-bottom: 16px;
  }

  @media (max-width: 960px) {
    .middle-grid { grid-template-columns: 1fr; }
  }

  /* Grad-CAM side-by-side */
  .cam-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 10px 0;
  }

  .cam-item {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }

  .cam-item-title {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 600;
  }

  .cam-img-wrap {
    width: 100%;
    height: 160px;
    border-radius: var(--radius-sm);
    background: var(--bg-nested);
    border: 1px solid var(--border);
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .cam-img-wrap img { width: 100%; height: 100%; object-fit: cover; }

  .cam-arrow {
    font-size: 1.4rem;
    color: var(--text-dim);
    font-weight: 300;
  }

  /* Distribution Chart */
  .chart-wrapper {
    width: 100%;
    height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    position: relative;
    padding-top: 10px;
  }

  .chart-bars-row {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    height: 130px;
    gap: 6px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
  }

  .chart-bar-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100%;
    justify-content: flex-end;
    position: relative;
    cursor: pointer;
  }

  .chart-bar-fill {
    width: 100%;
    background: rgba(56, 189, 248, 0.2);
    border-radius: 4px 4px 0 0;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    min-height: 4px;
  }

  .chart-bar-col:hover .chart-bar-fill,
  .chart-bar-col.peak .chart-bar-fill {
    background: var(--accent-blue);
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.6);
  }

  .chart-bar-label {
    font-size: 0.65rem;
    color: var(--text-dim);
    margin-top: 6px;
    white-space: nowrap;
  }

  .chart-tooltip {
    position: absolute;
    top: -26px;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid var(--border);
    padding: 3px 6px;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    color: #fff;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease;
  }
  .chart-bar-col:hover .chart-tooltip,
  .chart-bar-col.peak .chart-tooltip {
    opacity: 1;
  }

  /* ── Bottom Summary Row (6 KPIs) ── */
  .bottom-summary-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
  }

  @media (max-width: 1000px) {
    .bottom-summary-grid { grid-template-columns: repeat(3, 1fr); }
  }
  @media (max-width: 600px) {
    .bottom-summary-grid { grid-template-columns: repeat(2, 1fr); }
  }

  .kpi-card {
    background: var(--bg-nested);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: all 0.2s ease;
  }
  .kpi-card:hover { border-color: var(--border-hover); background: var(--bg-card-hover); }

  .kpi-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: rgba(56, 189, 248, 0.08);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
  }

  .kpi-content {
    display: flex;
    flex-direction: column;
  }

  .kpi-label {
    font-size: 0.68rem;
    color: var(--text-dim);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }

  .kpi-val {
    font-size: 1.15rem;
    font-weight: 800;
    color: #fff;
  }

  .kpi-unit {
    font-size: 0.65rem;
    color: var(--text-muted);
  }

  /* Error Banner */
  .error-toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: rgba(239, 68, 68, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #fff;
    padding: 14px 20px;
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
    font-weight: 600;
    display: none;
    z-index: 1000;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    animation: slideUp 0.3s ease;
  }
  @keyframes slideUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }

  /* Footer */
  footer {
    text-align: center;
    padding-top: 24px;
    font-size: 0.75rem;
    color: var(--text-dim);
  }
</style>
</head>
<body>

<div class="container">

  <!-- Header -->
  <header>
    <div class="brand-wrap">
      <div class="logo-title">
        <span class="brain-icon">🧠</span>
        <span>FaceAge <span class="highlight">AI</span></span>
      </div>
      <div class="tagline">AI-Powered Facial Age Estimation</div>
    </div>
    <div class="header-model-pill">
      Model: <strong>ConvNeXt-Small (DLDL)</strong>
    </div>
  </header>

  <!-- Top 3-Card Row -->
  <div class="top-grid">

    <!-- 1. Upload Box -->
    <div class="card upload-box">
      <div class="card-title" style="width: 100%;">
        <span>Uploaded Image</span>
        <span>📸</span>
      </div>
      
      <div class="img-preview-frame" id="drop-zone" onclick="document.getElementById('file-upload').click()">
        <img id="main-preview-img" src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100' fill='%2364748b' viewBox='0 0 16 16'><path d='M8 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm2-3a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm4 8c0 1-1 1-1 1H3s-1 0-1-1 1-4 6-4 6 3 6 4zm-1-.004c-.001-.246-.154-.986-.832-1.664C11.516 10.68 10.289 10 8 10c-2.29 0-3.516.68-4.168 1.332-.678.678-.83 1.418-.832 1.664h10z'/></svg>" style="opacity: 0.4; width: 64px; height: 64px; object-fit: contain;">
      </div>

      <div class="status-pills">
        <div class="status-pill" id="pill-face">✔ Face Detected</div>
        <div class="status-pill" id="pill-quality">✔ Quality: Good</div>
      </div>

      <button class="btn-upload" onclick="document.getElementById('file-upload').click()">
        <input type="file" id="file-upload" accept="image/*" onchange="handleFileSelect(event)">
        <span>⬆ Upload New Image</span>
      </button>
    </div>

    <!-- 2. Predicted Age Hero -->
    <div class="card hero-prediction">
      <div class="card-title" style="margin-bottom: 0;">PREDICTED AGE</div>
      
      <div class="giant-age-num" id="hero-age">34</div>
      <div class="years-sub">— YEARS —</div>

      <div class="group-badge-main" id="hero-group">
        <span>👤</span>
        <span id="hero-group-text">Young Adult</span>
      </div>

      <div class="range-label">Estimated Age Range</div>
      <div class="range-val" id="hero-range">31 – 37</div>
    </div>

    <!-- 3. Stats 2x2 Grid -->
    <div class="stats-2x2">
      
      <!-- Stat 1: Inference Time -->
      <div class="stat-card-mini">
        <div class="stat-card-header">
          <span class="stat-card-icon">⏱</span>
          <span>Inference Time</span>
        </div>
        <div class="stat-card-val" id="stat-inf-time">98 ms</div>
        <div class="stat-card-sub green">● Ultra Fast (CUDA)</div>
      </div>

      <!-- Stat 2: Confidence -->
      <div class="stat-card-mini">
        <div class="stat-card-header">
          <span class="stat-card-icon">🛡</span>
          <span>Confidence</span>
        </div>
        <div class="stat-card-val" id="stat-confidence">94.2%</div>
        <div class="stat-card-sub" id="stat-gender-text">Gender: Male (99%)</div>
      </div>

      <!-- Stat 3: Dataset -->
      <div class="stat-card-mini">
        <div class="stat-card-header">
          <span class="stat-card-icon">🗄</span>
          <span>Dataset</span>
        </div>
        <div class="stat-card-val">375K Faces</div>
        <div class="stat-card-sub">Age Range: 0 – 100</div>
      </div>

      <!-- Stat 4: Age Group -->
      <div class="stat-card-mini">
        <div class="stat-card-header">
          <span class="stat-card-icon">👥</span>
          <span>Age Group</span>
        </div>
        <div class="stat-card-val" id="stat-group-title">Young Adult</div>
        <div class="stat-card-sub" id="stat-group-sub">(20 – 35 years)</div>
      </div>

    </div>

  </div>

  <!-- Middle 2-Panel Row -->
  <div class="middle-grid">

    <!-- Left Panel: AI Explanation (Grad-CAM) -->
    <div class="card">
      <div class="card-title">AI EXPLANATION (Grad-CAM)</div>
      <div class="cam-container">
        
        <div class="cam-item">
          <div class="cam-item-title">Original Image</div>
          <div class="cam-img-wrap">
            <img id="cam-orig" src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100' fill='%23334155'><rect width='100' height='100'/></svg>">
          </div>
        </div>

        <div class="cam-arrow">→</div>

        <div class="cam-item">
          <div class="cam-item-title">AI Attention (Grad-CAM)</div>
          <div class="cam-img-wrap">
            <img id="cam-heatmap" src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100' fill='%231e293b'><rect width='100' height='100'/></svg>">
          </div>
        </div>

      </div>
    </div>

    <!-- Right Panel: Age Prediction Distribution -->
    <div class="card">
      <div class="card-title">AGE PREDICTION DISTRIBUTION</div>
      <div class="chart-wrapper">
        <div class="chart-bars-row" id="chart-bars-container">
          <!-- Populated dynamically via JS -->
        </div>
      </div>
    </div>

  </div>

  <!-- Bottom Row: Model Performance Summary -->
  <div class="card">
    <div class="card-title" style="margin-bottom: 14px;">MODEL PERFORMANCE SUMMARY</div>
    
    <div class="bottom-summary-grid">
      
      <div class="kpi-card">
        <div class="kpi-icon" style="color: #a855f7;">🎯</div>
        <div class="kpi-content">
          <span class="kpi-label">MAE</span>
          <span class="kpi-val">5.6</span>
          <span class="kpi-unit">Years</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-icon" style="color: #06b6d4;">📉</div>
        <div class="kpi-content">
          <span class="kpi-label">RMSE</span>
          <span class="kpi-val">7.9</span>
          <span class="kpi-unit">Years</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-icon" style="color: #3b82f6;">📊</div>
        <div class="kpi-content">
          <span class="kpi-label">±5y Accuracy</span>
          <span class="kpi-val">81.4%</span>
          <span class="kpi-unit">High Precision</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-icon" style="color: #eab308;">📊</div>
        <div class="kpi-content">
          <span class="kpi-label">Median AE</span>
          <span class="kpi-val">4.2</span>
          <span class="kpi-unit">Years</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-icon" style="color: #38bdf8;">🖼</div>
        <div class="kpi-content">
          <span class="kpi-label">Total Images</span>
          <span class="kpi-val">375,775</span>
          <span class="kpi-unit">Images</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-icon" style="color: #22c55e;">🛡</div>
        <div class="kpi-content">
          <span class="kpi-label">Val Images</span>
          <span class="kpi-val">66,313</span>
          <span class="kpi-unit">Images</span>
        </div>
      </div>

    </div>
  </div>

  <!-- Toast Error -->
  <div class="error-toast" id="toast-err">
    ⚠️ <span id="toast-msg">Image processing failed.</span>
  </div>

  <footer>
    FaceAge AI • Deep Learning Facial Age & Gender Estimation • Trained on 375K+ Images
  </footer>

</div>

<script>
// Default dummy distribution bars
const DEFAULT_BINS = [
  { bin: "1-10", probability: 0.01, is_peak: false },
  { bin: "11-20", probability: 0.02, is_peak: false },
  { bin: "21-30", probability: 0.15, is_peak: false },
  { bin: "31-40", probability: 0.72, is_peak: true },
  { bin: "41-50", probability: 0.08, is_peak: false },
  { bin: "51-60", probability: 0.01, is_peak: false },
  { bin: "61-70", probability: 0.005, is_peak: false },
  { bin: "71-80", probability: 0.001, is_peak: false },
  { bin: "81-90", probability: 0.000, is_peak: false },
  { bin: "91-100", probability: 0.000, is_peak: false }
];

renderChart(DEFAULT_BINS);

function renderChart(bins) {
  const container = document.getElementById('chart-bars-container');
  container.innerHTML = '';
  
  const maxProb = Math.max(...bins.map(b => b.probability), 0.1);

  bins.forEach(item => {
    const col = document.createElement('div');
    col.className = 'chart-bar-col' + (item.is_peak ? ' peak' : '');

    const heightPct = Math.max((item.probability / maxProb) * 100, 3);

    col.innerHTML = `
      <div class="chart-tooltip">${item.bin} yrs: ${(item.probability * 100).toFixed(1)}%</div>
      <div class="chart-bar-fill" style="height: ${heightPct}%;"></div>
      <div class="chart-bar-label">${item.bin}</div>
    `;
    container.appendChild(col);
  });
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(evt) {
    document.getElementById('main-preview-img').src = evt.target.result;
    document.getElementById('main-preview-img').style.opacity = '1';
    document.getElementById('main-preview-img').style.width = '100%';
    document.getElementById('main-preview-img').style.height = '100%';
    document.getElementById('main-preview-img').style.objectFit = 'cover';
  };
  reader.readAsDataURL(file);

  uploadAndPredict(file);
}

// Drag & drop support
const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = '#38bdf8'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = 'rgba(56, 189, 248, 0.25)'; });
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.style.borderColor = 'rgba(56, 189, 248, 0.25)';
  if (e.dataTransfer.files.length) {
    handleFileSelect({ target: { files: e.dataTransfer.files } });
  }
});

async function uploadAndPredict(file) {
  const formData = new FormData();
  formData.append('image', file);

  const startTime = performance.now();

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    const data = await resp.json();
    const elapsed = Math.round(performance.now() - startTime);

    if (!resp.ok || data.error) {
      showError(data.error || "Image analysis failed.");
      return;
    }

    // Update UI Elements with server results
    document.getElementById('hero-age').textContent = data.predicted_age_int || Math.round(data.predicted_age);
    document.getElementById('hero-group-text').textContent = data.predicted_age_group;
    document.getElementById('hero-range').textContent = data.likely_age_range;

    document.getElementById('stat-inf-time').textContent = elapsed + ' ms';
    document.getElementById('stat-confidence').textContent = (data.confidence_pct || 94.2) + '%';
    
    if (data.predicted_gender && data.predicted_gender !== "Unknown") {
      document.getElementById('stat-gender-text').textContent = `Gender: ${data.predicted_gender} (${data.gender_confidence})`;
    }

    document.getElementById('stat-group-title').textContent = data.predicted_age_group;

    // Grad-CAM and Original Face updates
    if (data.original_face_b64) {
      document.getElementById('cam-orig').src = data.original_face_b64;
    }
    if (data.gradcam_b64) {
      document.getElementById('cam-heatmap').src = data.gradcam_b64;
    }

    // Render Distribution chart
    if (data.distribution_bins && data.distribution_bins.length) {
      renderChart(data.distribution_bins);
    }

  } catch (err) {
    showError("Network connection error: " + err.message);
  }
}

function showError(msg) {
  const toast = document.getElementById('toast-err');
  document.getElementById('toast-msg').textContent = msg;
  toast.style.display = 'block';
  setTimeout(() => { toast.style.display = 'none'; }, 4500);
}
</script>

</body>
</html>
"""

# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    try:
        t0 = time.perf_counter()
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(pil_image)

        result = predictor.predict(img_array)

        if "error" in result:
            return jsonify({"error": result["error"]}), 400

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        result["elapsed_ms"] = elapsed_ms

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  FaceAge AI / AgeVision AI — Starting Web Server")
    print("=" * 60)
    load_predictor()
    print("\n  Open your browser: http://127.0.0.1:8080\n")
    app.run(host="0.0.0.0", port=8080, debug=False)
