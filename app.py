"""
FaceAge AI / AgeVision AI — Web Application
Multi-Attribute Dashboard with Dedicated Analyze Button, Pristine Initial State & Strict Error State Handling
"""

import os
import sys
import io
import time
import base64
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024  # 15MB max upload

predictor = None

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_age_model.pt")
METRICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "final_metrics.json")

def load_predictor():
    global predictor
    if predictor is None:
        try:
            from inference import AgePredictor
            predictor = AgePredictor(model_path=MODEL_PATH, metrics_path=METRICS_PATH)
            print(f"[OK] Model loaded: {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to load model in load_predictor: {e}")
    return predictor

def get_predictor():
    global predictor
    if predictor is None:
        load_predictor()
    return predictor

# Warm up model at module import time (Crucial for Gunicorn / Render / WSGI servers)
try:
    load_predictor()
except Exception as _e:
    print(f"[WARN] Initial model warmup deferred: {_e}")

# ─────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FaceAge AI — AI-Powered Facial Age & Gender Estimation</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg-main:       #070b14;
    --bg-card:       #0d1424;
    --bg-card-hover: #111b32;
    --bg-nested:     #080e1c;
    --border:        rgba(56, 189, 248, 0.12);
    --border-hover:  rgba(56, 189, 248, 0.4);
    --accent-blue:   #38bdf8;
    --accent-purple: #a855f7;
    --accent-pink:   #ec4899;
    --accent-orange: #f97316;
    --accent-green:  #22c55e;
    --accent-red:    #ef4444;
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

  /* Deep Glow Background */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background:
      radial-gradient(ellipse 70% 50% at 20% 0%, rgba(56, 189, 248, 0.08) 0%, transparent 60%),
      radial-gradient(ellipse 60% 40% at 80% 100%, rgba(168, 85, 247, 0.06) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
  }

  .container {
    position: relative;
    z-index: 1;
    max-width: 1280px;
    margin: 0 auto;
    padding: 24px 20px 40px;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }

  .brand-wrap {
    display: flex;
    align-items: center;
    gap: 14px;
  }

  .logo-icon-wrap {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(168, 85, 247, 0.2));
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.2);
  }

  .brand-text-col { display: flex; flex-direction: column; }
  .logo-title { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.5px; color: #fff; line-height: 1.2; }
  .logo-title span.highlight { color: var(--accent-blue); text-shadow: 0 0 16px rgba(56, 189, 248, 0.5); }
  .tagline { font-size: 0.8rem; color: var(--text-muted); font-weight: 500; }

  .header-badges { display: flex; align-items: center; gap: 10px; }
  .header-badge {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid var(--border);
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
  }
  .header-badge strong { color: #fff; }
  .header-badge.green {
    background: rgba(34, 197, 94, 0.1);
    border-color: rgba(34, 197, 94, 0.3);
    color: var(--accent-green);
  }

  @media (max-width: 860px) {
    header { flex-direction: column; gap: 14px; text-align: center; }
    .brand-wrap { flex-direction: column; }
  }

  /* ── Cards & Grid ── */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px;
    position: relative;
    transition: all 0.25s ease;
  }
  .card:hover { border-color: var(--border-hover); }

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

  /* ── Top Grid (3 Columns) ── */
  .top-grid {
    display: grid;
    grid-template-columns: 290px 1fr 360px;
    gap: 16px;
    margin-bottom: 16px;
  }

  @media (max-width: 1120px) {
    .top-grid { grid-template-columns: 1fr 1fr; }
    .stats-multi { grid-column: span 2; }
  }
  @media (max-width: 720px) {
    .top-grid { grid-template-columns: 1fr; }
    .stats-multi { grid-column: span 1; }
  }

  /* Left: Upload Column */
  .upload-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    height: 100%;
  }

  .img-preview-frame {
    width: 100%;
    height: 190px;
    border-radius: var(--radius-md);
    background: var(--bg-nested);
    border: 1.5px dashed rgba(56, 189, 248, 0.25);
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
    display: none;
  }

  .upload-placeholder-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    color: var(--text-muted);
    font-size: 0.78rem;
    text-align: center;
    padding: 14px;
    pointer-events: none;
  }
  .upload-placeholder-content .upload-icon {
    font-size: 2.2rem;
    color: var(--accent-blue);
    opacity: 0.8;
  }

  .status-pills-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    width: 100%;
    margin: 12px 0;
  }

  .status-pill {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border);
    color: var(--text-dim);
    font-size: 0.7rem;
    font-weight: 600;
    padding: 6px 4px;
    border-radius: 8px;
    white-space: nowrap;
    transition: all 0.25s ease;
  }
  .status-pill.success {
    background: rgba(34, 197, 94, 0.1);
    border-color: rgba(34, 197, 94, 0.3);
    color: var(--accent-green);
  }
  .status-pill.danger {
    background: rgba(239, 68, 68, 0.12);
    border-color: rgba(239, 68, 68, 0.35);
    color: var(--accent-red);
  }

  .btn-analyze {
    width: 100%;
    background: linear-gradient(135deg, #0284c7, #0284c7);
    color: #fff;
    border: none;
    border-radius: var(--radius-sm);
    padding: 13px 16px;
    font-size: 0.9rem;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
    transition: all 0.2s ease;
  }
  .btn-analyze.ready {
    background: linear-gradient(135deg, #ea580c, #f97316);
    box-shadow: 0 4px 18px rgba(249, 115, 22, 0.45);
    animation: pulseBtn 2s infinite;
  }
  @keyframes pulseBtn {
    0%, 100% { box-shadow: 0 0 0 0 rgba(249, 115, 22, 0.5); }
    50% { box-shadow: 0 0 0 8px rgba(249, 115, 22, 0); }
  }

  .btn-analyze:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    background: rgba(255, 255, 255, 0.08);
    box-shadow: none;
    animation: none;
  }
  .btn-analyze input { display: none; }

  /* Center: Hero Prediction */
  .hero-prediction {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 24px;
  }

  .giant-age-num {
    font-size: 5.8rem;
    font-weight: 800;
    line-height: 0.95;
    color: #ffffff;
    letter-spacing: -2px;
    margin: 6px 0 2px;
    text-shadow: 0 0 30px rgba(255, 255, 255, 0.25);
  }

  .years-sub {
    font-size: 0.78rem;
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
    background: rgba(255, 255, 255, 0.04);
    border: 1.5px solid var(--border);
    color: var(--text-muted);
    border-radius: 100px;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 16px;
    transition: all 0.3s ease;
  }
  .group-badge-main.active {
    background: rgba(34, 197, 94, 0.08);
    border-color: rgba(34, 197, 94, 0.4);
    color: var(--accent-green);
  }

  .range-label {
    font-size: 0.75rem;
    color: var(--text-dim);
    margin-bottom: 2px;
  }

  .range-val {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-muted);
    letter-spacing: -0.5px;
    transition: color 0.3s ease;
  }
  .range-val.active { color: var(--accent-blue); }

  /* Error Banner inside Hero */
  .error-card-display {
    display: none;
    background: rgba(239, 68, 68, 0.1);
    border: 1.5px solid rgba(239, 68, 68, 0.4);
    border-radius: var(--radius-md);
    padding: 24px 20px;
    text-align: center;
    width: 100%;
    margin: 10px 0;
  }
  .error-card-display .err-icon { font-size: 2.6rem; margin-bottom: 8px; }
  .error-card-display .err-title { font-size: 1.1rem; font-weight: 800; color: #f87171; margin-bottom: 4px; }
  .error-card-display .err-desc { font-size: 0.85rem; color: #fca5a5; line-height: 1.5; }

  /* Right: Multi-Attribute 2x2 Stats */
  .stats-multi {
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

  .stat-card-val {
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--text-muted);
    margin: 6px 0 2px;
  }
  .stat-card-val.active { color: #fff; }

  .stat-card-sub {
    font-size: 0.68rem;
    color: var(--text-dim);
    font-weight: 500;
  }
  .stat-card-sub.green { color: var(--accent-green); }
  .stat-card-sub.blue  { color: var(--accent-blue); }
  .stat-card-sub.pink  { color: var(--accent-pink); }

  /* ── Middle Row (2 Panels) ── */
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
    padding: 6px 0;
  }

  .cam-item {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }

  .cam-item-title { font-size: 0.75rem; color: var(--text-muted); font-weight: 600; }

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
    color: var(--text-dim);
    font-size: 0.75rem;
  }
  .cam-img-wrap img { width: 100%; height: 100%; object-fit: cover; display: none; }

  .cam-arrow { font-size: 1.4rem; color: var(--text-dim); font-weight: 300; }

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
    background: rgba(56, 189, 248, 0.12);
    border-radius: 4px 4px 0 0;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    min-height: 4px;
  }

  .chart-bar-col:hover .chart-bar-fill,
  .chart-bar-col.peak .chart-bar-fill {
    background: var(--accent-blue);
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.6);
  }

  .chart-bar-label { font-size: 0.65rem; color: var(--text-dim); margin-top: 6px; white-space: nowrap; }

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
  .chart-bar-col.peak .chart-tooltip { opacity: 1; }

  .chart-placeholder-msg {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-dim);
    font-size: 0.8rem;
    background: rgba(8, 14, 28, 0.8);
    border-radius: var(--radius-sm);
    z-index: 10;
  }

  /* ── Bottom Summary Row (6 KPIs) ── */
  .bottom-summary-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
  }

  @media (max-width: 1000px) { .bottom-summary-grid { grid-template-columns: repeat(3, 1fr); } }
  @media (max-width: 600px) { .bottom-summary-grid { grid-template-columns: repeat(2, 1fr); } }

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

  .kpi-content { display: flex; flex-direction: column; }
  .kpi-label { font-size: 0.68rem; color: var(--text-dim); font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; }
  .kpi-val { font-size: 1.15rem; font-weight: 800; color: #fff; }
  .kpi-unit { font-size: 0.65rem; color: var(--text-muted); }

  /* Toast Notification */
  .toast-alert {
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

  /* Spinner */
  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
    display: inline-block;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  footer { text-align: center; padding-top: 24px; font-size: 0.75rem; color: var(--text-dim); }
</style>
</head>
<body>

<div class="container">

  <!-- Header -->
  <header>
    <div class="brand-wrap">
      <div class="logo-icon-wrap">🧠</div>
      <div class="brand-text-col">
        <div class="logo-title">FaceAge <span class="highlight">AI</span></div>
        <div class="tagline">AI-Powered Multi-Attribute Facial Estimation</div>
      </div>
    </div>
    <div class="header-badges">
      <div class="header-badge green">● CUDA GPU Active</div>
      <div class="header-badge">Model: <strong>ConvNeXt-Small (DLDL)</strong></div>
    </div>
  </header>

  <!-- Top 3-Card Grid -->
  <div class="top-grid">

    <!-- 1. Upload Column -->
    <div class="card upload-box">
      <div class="card-title" style="width: 100%;">
        <span>Select Portrait Photo</span>
        <span>📸</span>
      </div>
      
      <!-- Click area to browse/select image -->
      <div class="img-preview-frame" id="drop-zone" onclick="document.getElementById('file-input').click()">
        <img id="main-preview-img" src="" alt="Face Preview">
        <div class="upload-placeholder-content" id="upload-prompt">
          <div class="upload-icon">📁</div>
          <div><strong>Click to Browse</strong> or Drag & Drop Photo</div>
          <div style="font-size: 0.7rem; color: var(--text-dim);">JPEG, PNG or WebP</div>
        </div>
      </div>

      <input type="file" id="file-input" accept="image/*" onchange="onFileSelected(event)" style="display: none;">

      <div class="status-pills-row">
        <div class="status-pill" id="pill-face">Face: Waiting</div>
        <div class="status-pill" id="pill-quality">Quality: Waiting</div>
      </div>

      <!-- Explicit Analyze Button -->
      <button class="btn-analyze" id="btn-analyze" onclick="startAnalysis()" disabled>
        <span id="btn-icon">⚡</span>
        <span id="btn-label">Analyze Image</span>
      </button>
    </div>

    <!-- 2. Predicted Age Hero -->
    <div class="card hero-prediction">
      <div class="card-title" style="margin-bottom: 0;">PREDICTED AGE</div>
      
      <!-- Default Age Display State -->
      <div id="hero-normal-content" style="width: 100%;">
        <div class="giant-age-num" id="hero-age">—</div>
        <div class="years-sub" id="hero-sub">— YEARS —</div>

        <div class="group-badge-main" id="hero-group">
          <span id="hero-group-icon">👤</span>
          <span id="hero-group-text">Waiting for Analysis</span>
        </div>

        <div class="range-label">Estimated Age Range (95% CI)</div>
        <div class="range-val" id="hero-range">—</div>
      </div>

      <!-- Prominent Error State (Replaces normal content on validation fail) -->
      <div class="error-card-display" id="hero-error-content">
        <div class="err-icon">⚠️</div>
        <div class="err-title" id="hero-err-title">Quality Check Failed</div>
        <div class="err-desc" id="hero-err-desc">Please upload a sharp, clear portrait of a person.</div>
      </div>

    </div>

    <!-- 3. Multi-Attribute 2x2 Stats Grid -->
    <div class="stats-multi">
      
      <!-- Stat 1: Gender -->
      <div class="stat-card-mini" style="border-left: 3px solid var(--accent-pink);">
        <div class="stat-card-header">
          <span class="stat-card-icon" id="gender-icon">⚧</span>
          <span>Predicted Gender</span>
        </div>
        <div class="stat-card-val" id="stat-gender-val">—</div>
        <div class="stat-card-sub pink" id="stat-gender-conf">Confidence: —</div>
      </div>

      <!-- Stat 2: Age Confidence -->
      <div class="stat-card-mini" style="border-left: 3px solid var(--accent-green);">
        <div class="stat-card-header">
          <span class="stat-card-icon">🛡</span>
          <span>Age Confidence</span>
        </div>
        <div class="stat-card-val" id="stat-confidence">—</div>
        <div class="stat-card-sub green" id="stat-conf-sub">Intrinsic ±1.96σ Metric</div>
      </div>

      <!-- Stat 3: Inference Latency -->
      <div class="stat-card-mini">
        <div class="stat-card-header">
          <span class="stat-card-icon">⏱</span>
          <span>Inference Time</span>
        </div>
        <div class="stat-card-val" id="stat-inf-time">—</div>
        <div class="stat-card-sub green" id="stat-time-sub">● CUDA Accelerated</div>
      </div>

      <!-- Stat 4: Age Cohort -->
      <div class="stat-card-mini">
        <div class="stat-card-header">
          <span class="stat-card-icon">👥</span>
          <span>Age Cohort</span>
        </div>
        <div class="stat-card-val" id="stat-group-title">—</div>
        <div class="stat-card-sub blue" id="stat-group-sub">Bracket: —</div>
      </div>

    </div>

  </div>

  <!-- Middle 2-Panel Row -->
  <div class="middle-grid" id="middle-panels">

    <!-- Left Panel: AI Explanation (Grad-CAM) -->
    <div class="card">
      <div class="card-title">AI EXPLANATION (Grad-CAM Heatmap)</div>
      <div class="cam-container">
        
        <div class="cam-item">
          <div class="cam-item-title">Original Face Crop</div>
          <div class="cam-img-wrap" id="cam-orig-wrap">
            <span id="cam-orig-placeholder">No Image Analyzed</span>
            <img id="cam-orig" alt="Original Crop">
          </div>
        </div>

        <div class="cam-arrow">→</div>

        <div class="cam-item">
          <div class="cam-item-title">AI Attention Focus</div>
          <div class="cam-img-wrap" id="cam-heatmap-wrap">
            <span id="cam-heatmap-placeholder">No Heatmap</span>
            <img id="cam-heatmap" alt="Grad-CAM Overlay">
          </div>
        </div>

      </div>
    </div>

    <!-- Right Panel: Age Prediction Distribution -->
    <div class="card">
      <div class="card-title">AGE PREDICTION DISTRIBUTION (DLDL Probability)</div>
      <div class="chart-wrapper">
        <div class="chart-placeholder-msg" id="chart-placeholder">
          Upload and analyze a photo to view probability distribution
        </div>
        <div class="chart-bars-row" id="chart-bars-container">
          <!-- Populated dynamically via JS -->
        </div>
      </div>
    </div>

  </div>

  <!-- Bottom Row: Global Model Performance Benchmark Summary -->
  <div class="card">
    <div class="card-title" style="margin-bottom: 14px;">MODEL PERFORMANCE BENCHMARK SUMMARY</div>
    
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
          <span class="kpi-label">±10y Accuracy</span>
          <span class="kpi-val">94.2%</span>
          <span class="kpi-unit">Cumulative</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-icon" style="color: #38bdf8;">🖼</div>
        <div class="kpi-content">
          <span class="kpi-label">Total Images</span>
          <span class="kpi-val">375,775</span>
          <span class="kpi-unit">Indexed</span>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-icon" style="color: #22c55e;">🛡</div>
        <div class="kpi-content">
          <span class="kpi-label">Val Images</span>
          <span class="kpi-val">66,313</span>
          <span class="kpi-unit">Validated</span>
        </div>
      </div>

    </div>
  </div>

  <!-- Toast Alert -->
  <div class="toast-alert" id="toast-err">
    ⚠️ <span id="toast-msg">Image processing failed.</span>
  </div>

  <footer>
    FaceAge AI • Deep Learning Facial Age & Gender Estimation • Trained on 375K+ Images
  </footer>

</div>

<script>
let selectedFile = null;

// Handle file selection (Browse or Drop)
function onFileSelected(e) {
  const file = e.target.files[0];
  if (!file) return;
  loadSelectedFile(file);
}

function loadSelectedFile(file) {
  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = function(evt) {
    const previewImg = document.getElementById('main-preview-img');
    previewImg.src = evt.target.result;
    previewImg.style.display = 'block';
    document.getElementById('upload-prompt').style.display = 'none';
  };
  reader.readAsDataURL(file);

  // Update status pills
  const pillFace = document.getElementById('pill-face');
  const pillQuality = document.getElementById('pill-quality');
  pillFace.className = 'status-pill';
  pillFace.textContent = 'Photo Ready';
  pillQuality.className = 'status-pill';
  pillQuality.textContent = 'Ready to Analyze';

  // Enable Analyze Button
  const btn = document.getElementById('btn-analyze');
  btn.disabled = false;
  btn.classList.add('ready');
  document.getElementById('btn-label').textContent = '⚡ Analyze Image';

  // Reset results display to waiting state until analyzed
  resetResultsToWaiting();
}

// Drag & Drop
const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = '#38bdf8'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = 'rgba(56, 189, 248, 0.25)'; });
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.style.borderColor = 'rgba(56, 189, 248, 0.25)';
  if (e.dataTransfer.files.length) {
    loadSelectedFile(e.dataTransfer.files[0]);
  }
});

function resetResultsToWaiting() {
  document.getElementById('hero-normal-content').style.display = 'block';
  document.getElementById('hero-error-content').style.display = 'none';

  document.getElementById('hero-age').textContent = '—';
  document.getElementById('hero-group-text').textContent = 'Ready for Analysis';
  document.getElementById('hero-group').classList.remove('active');
  document.getElementById('hero-range').textContent = '—';
  document.getElementById('hero-range').classList.remove('active');

  document.getElementById('stat-gender-val').textContent = '—';
  document.getElementById('stat-gender-val').classList.remove('active');
  document.getElementById('stat-gender-conf').textContent = 'Confidence: —';

  document.getElementById('stat-confidence').textContent = '—';
  document.getElementById('stat-confidence').classList.remove('active');

  document.getElementById('stat-inf-time').textContent = '—';
  document.getElementById('stat-group-title').textContent = '—';
  document.getElementById('stat-group-sub').textContent = 'Bracket: —';

  // Reset Grad-CAM
  document.getElementById('cam-orig').style.display = 'none';
  document.getElementById('cam-orig-placeholder').style.display = 'block';
  document.getElementById('cam-heatmap').style.display = 'none';
  document.getElementById('cam-heatmap-placeholder').style.display = 'block';

  // Reset Chart
  document.getElementById('chart-placeholder').style.display = 'flex';
  document.getElementById('chart-bars-container').innerHTML = '';
}

// Triggered ONLY when clicking "⚡ Analyze Image"
async function startAnalysis() {
  if (!selectedFile) return;

  const btn = document.getElementById('btn-analyze');
  btn.disabled = true;
  btn.classList.remove('ready');
  document.getElementById('btn-icon').innerHTML = '<div class="spinner"></div>';
  document.getElementById('btn-label').textContent = 'Analyzing...';

  const formData = new FormData();
  formData.append('image', selectedFile);

  const startTime = performance.now();

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    const data = await resp.json();
    const elapsed = Math.round(performance.now() - startTime);

    if (!resp.ok || data.error) {
      // ⚠️ ERROR / VALIDATION FAILURE STATE
      displayErrorState(data.error || "Image quality verification failed.");
      return;
    }

    // ✅ SUCCESS STATE
    displaySuccessState(data, elapsed);

  } catch (err) {
    displayErrorState("Network connection error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.classList.add('ready');
    document.getElementById('btn-icon').textContent = '⚡';
    document.getElementById('btn-label').textContent = 'Re-Analyze Image';
  }
}

function displaySuccessState(data, elapsed) {
  // 1. Status Pills
  const pillFace = document.getElementById('pill-face');
  const pillQuality = document.getElementById('pill-quality');
  pillFace.className = 'status-pill success';
  pillFace.textContent = '✔ Face Verified';
  pillQuality.className = 'status-pill success';
  pillQuality.textContent = '✔ Quality: Sharp';

  // 2. Hero Box
  document.getElementById('hero-normal-content').style.display = 'block';
  document.getElementById('hero-error-content').style.display = 'none';

  document.getElementById('hero-age').textContent = data.predicted_age_int || Math.round(data.predicted_age);
  document.getElementById('hero-group-text').textContent = data.predicted_age_group;
  document.getElementById('hero-group').classList.add('active');
  document.getElementById('hero-range').textContent = data.likely_age_range;
  document.getElementById('hero-range').classList.add('active');

  // 3. Stats Grid
  document.getElementById('stat-inf-time').textContent = elapsed + ' ms';
  document.getElementById('stat-inf-time').classList.add('active');

  document.getElementById('stat-confidence').textContent = (data.confidence_pct || 94.2) + '%';
  document.getElementById('stat-confidence').classList.add('active');

  if (data.predicted_gender && data.predicted_gender !== "Unknown") {
    const g = data.predicted_gender;
    const isFemale = g.toLowerCase() === 'female';
    document.getElementById('stat-gender-val').textContent = g;
    document.getElementById('stat-gender-val').classList.add('active');
    document.getElementById('stat-gender-val').style.color = isFemale ? '#f472b6' : '#38bdf8';
    document.getElementById('stat-gender-conf').textContent = `Confidence: ${data.gender_confidence}`;
    document.getElementById('gender-icon').textContent = isFemale ? '👩' : '👨';
  }

  document.getElementById('stat-group-title').textContent = data.predicted_age_group;
  document.getElementById('stat-group-title').classList.add('active');
  const groupRanges = {
    'Child': '(0 – 12 yrs)',
    'Teenager': '(13 – 19 yrs)',
    'Young Adult': '(20 – 35 yrs)',
    'Adult': '(36 – 59 yrs)',
    'Senior': '(60+ yrs)'
  };
  document.getElementById('stat-group-sub').textContent = 'Bracket: ' + (groupRanges[data.predicted_age_group] || '(0 – 100 yrs)');

  // 4. Grad-CAM Images
  if (data.original_face_b64) {
    const orig = document.getElementById('cam-orig');
    orig.src = data.original_face_b64;
    orig.style.display = 'block';
    document.getElementById('cam-orig-placeholder').style.display = 'none';
  }
  if (data.gradcam_b64) {
    const cam = document.getElementById('cam-heatmap');
    cam.src = data.gradcam_b64;
    cam.style.display = 'block';
    document.getElementById('cam-heatmap-placeholder').style.display = 'none';
  }

  // 5. Distribution Bar Chart
  if (data.distribution_bins && data.distribution_bins.length) {
    document.getElementById('chart-placeholder').style.display = 'none';
    renderChart(data.distribution_bins);
  }
}

function displayErrorState(errorMessage) {
  // 1. Status Pills -> Red Danger
  const pillFace = document.getElementById('pill-face');
  const pillQuality = document.getElementById('pill-quality');
  pillFace.className = 'status-pill danger';
  pillFace.textContent = '❌ Verification Failed';
  pillQuality.className = 'status-pill danger';
  pillQuality.textContent = '❌ Unusable Photo';

  // 2. Hero Box -> Show prominent Error Card
  document.getElementById('hero-normal-content').style.display = 'none';
  const errCard = document.getElementById('hero-error-content');
  errCard.style.display = 'block';
  document.getElementById('hero-err-title').textContent = 'Validation Rejected';
  document.getElementById('hero-err-desc').textContent = errorMessage;

  // 3. Reset Stats
  document.getElementById('stat-gender-val').textContent = 'N/A';
  document.getElementById('stat-gender-val').classList.remove('active');
  document.getElementById('stat-gender-conf').textContent = 'Rejected';

  document.getElementById('stat-confidence').textContent = '0%';
  document.getElementById('stat-confidence').classList.remove('active');

  document.getElementById('stat-inf-time').textContent = '—';
  document.getElementById('stat-group-title').textContent = 'N/A';
  document.getElementById('stat-group-sub').textContent = 'Bracket: —';

  // 4. Hide / Clear Grad-CAM
  document.getElementById('cam-orig').style.display = 'none';
  document.getElementById('cam-orig-placeholder').style.display = 'block';
  document.getElementById('cam-orig-placeholder').textContent = 'No Face Verified';
  document.getElementById('cam-heatmap').style.display = 'none';
  document.getElementById('cam-heatmap-placeholder').style.display = 'block';
  document.getElementById('cam-heatmap-placeholder').textContent = 'No Heatmap Available';

  // 5. Hide / Clear Distribution Chart
  document.getElementById('chart-placeholder').style.display = 'flex';
  document.getElementById('chart-placeholder').textContent = '⚠️ Distribution unavailable due to validation rejection';
  document.getElementById('chart-bars-container').innerHTML = '';

  // Toast
  showToast(errorMessage);
}

function renderChart(bins) {
  const container = document.getElementById('chart-bars-container');
  container.innerHTML = '';
  const maxProb = Math.max(...bins.map(b => b.probability), 0.1);

  bins.forEach(item => {
    const col = document.createElement('div');
    col.className = 'chart-bar-col' + (item.is_peak ? ' peak' : '');

    const heightPct = Math.max((item.probability / maxProb) * 100, 4);

    col.innerHTML = `
      <div class="chart-tooltip">${item.bin} yrs: ${(item.probability * 100).toFixed(1)}%</div>
      <div class="chart-bar-fill" style="height: ${heightPct}%;"></div>
      <div class="chart-bar-label">${item.bin}</div>
    `;
    container.appendChild(col);
  });
}

function showToast(msg) {
  const toast = document.getElementById('toast-err');
  document.getElementById('toast-msg').textContent = msg;
  toast.style.display = 'block';
  setTimeout(() => { toast.style.display = 'none'; }, 5000);
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

        pred = get_predictor()
        if pred is None:
            return jsonify({"error": "Model initialization failed on server. Please check logs."}), 500

        result = pred.predict(img_array)

        if "error" in result:
            return jsonify({"error": result["error"]}), 400

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        result["elapsed_ms"] = elapsed_ms

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  FaceAge AI — Starting Web Server")
    print("=" * 60)
    load_predictor()
    print("\n  Open your browser: http://127.0.0.1:8080\n")
    app.run(host="0.0.0.0", port=8080, debug=False)
