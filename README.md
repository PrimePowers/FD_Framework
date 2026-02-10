# FD_Framework – A Spectral Stability Framework for the Riemann Hypothesis

## 📖 Overview

This repository implements the **Fast Diagonalization (FD) framework**, a novel mathematical methodology that reformulates the Riemann Hypothesis (RH) as a spectral stability problem. Derived from the Weil explicit formula, the framework transforms arithmetic prime data into finite, windowed Gram matrices in logarithmic coordinates, where the height parameter \( T \) acts as an **exponential amplifier** rather than a numerical obstacle.

The core insight is that any deviation from the critical line (\( \beta \neq 1/2 \)) produces signals growing as \( e^{(\beta-1/2)T} \), making violations increasingly detectable at large heights – a conceptual reversal from classical verification approaches.

## 🧬 The FD Hierarchy: FD₃ → FD₂ → FD₁

The framework is organized around three progressively stronger stability regimes:

- **FD₃** – **Numerical diagnostics**: Probabilistic diagonal dominance, directly testable on existing zero data
- **FD₂** – **Density-one stability**: Fast diagonalization holds for almost all heights, a verifiable analytic milestone  
- **FD₁** – **Uniform stability**: Equivalent to RH, represents maximal spectral stability of the Gram system

This repository provides computational tools for each level of the hierarchy.

## 🏗️ Repository Structure


---

## 🔧 Core Scripts

# Validate the baseline numerical stability
python FD_number_validation.py

# Run the R3 Stress Test (Scaling Law analysis)
python R3_stresstest_suite_v2.py

# Test topological winding stability
python FD_topology.py

### 1. `FD_number_validation.py` – Arithmetic-to-Spectral Validation

**Purpose**: Validates the numerical pipeline from arithmetic data to finite spectral systems.

**What it does**:
- Constructs spectral objects from prime/proxy data in logarithmic coordinates
- Applies window functions (Gaussian, Hann, bump) for localization
- Measures diagonal dominance and spectral concentration
- Provides consistency checks for the arithmetic → spectral transformation

**Key outputs**:
- Diagonal/off-diagonal energy ratios \( r_F(T) \)
- Hermiticity error diagnostics
- Window-robustness statistics

**Usage**:
```bash
python FD_number_validation.py

### 2. `R3_stresstest_suite_v2.py` – No-Hiding and Detection Diagnostics (FD₃ / FD₂)

**Purpose**: Implements the **R3 No-Hiding Stress Suite**, empirically testing the amplifier mechanism and the "No-Hiding" principle.

**What it does**:
- Injects controlled off-critical perturbations ($\varepsilon$) into windowed spectral signals
- Conducts adversarial testing using phase noise to simulate destructive interference
- Measures the detection sensitivity of the FD functional against naive energy-based methods
- Auto-calibrates detection thresholds ($\tau$) based on baseline null-models

**Key outputs**:
- **Scaling Law Plots**: Empirically confirms $\varepsilon_{\min} \sim T^{-1}$ (observed $\approx T^{-0.93}$)
- **Cancellation Scores**: Quantifies the inability of phase noise to conceal off-critical signals
- **Detection Heatmaps**: Visualizes the phase transition from stability to detection

**Usage**:
```bash
python R3_stresstest_suite_v2.py

### 1. `FD_number_validation.py` – Arithmetic-to-Spectral Validation

**Purpose**: Validates the numerical pipeline from arithmetic data to finite spectral systems.

**What it does**:
- Constructs spectral objects from prime/proxy data in logarithmic coordinates
- Applies window functions (Gaussian, Hann, bump) for localization
- Measures diagonal dominance and spectral concentration
- Provides consistency checks for the arithmetic → spectral transformation

**Key outputs**:
- Diagonal/off-diagonal energy ratios $r_F(T)$
- Hermiticity error diagnostics
- Window-robustness statistics

**Usage**:
```bash
python FD_number_validation.py





