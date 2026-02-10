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
bash
python FD_number_validation.py

---

### 2. `R3_stresstest_suite_v2.py` – No-Hiding and Detection Diagnostics (FD₃ / FD₂)
**Purpose**: Implements the **R3 No-Hiding Stress Suite** to empirically validate the "Amplifier Mechanism."
- **What it does**: 
    - Injects controlled off-critical perturbations ($\varepsilon$) and adversarial phase noise.
    - Measures detection rates to prove that deviations cannot be hidden by interference.
    - Extracts the asymptotic scaling law of the detection threshold.
- **Key Outputs**: 
    - **Scaling Law**: Confirms $\varepsilon_{\min} \sim T^{-1}$ (observed $\approx T^{-0.933}$).
    - **Cancellation Score**: Proves that noise cannot conceal energy shifts (Score $\approx 0.055$).
- **Usage**: `python R3_stresstest_suite_v2.py`

---

### 3. `FD_topology.py` – Topological Stability and Winding Diagnostics (FD₁)
**Purpose**: Probes the **topological layer** of the framework, linking spectral stability to geometric winding invariants.
- **What it does**: 
    - Maps the spectral signal into the complex plane to analyze its trajectory.
    - Computes **normalized winding numbers** ($W$) as global invariants.
    - Identifies the **Topological Bifurcation Point** where phase-locking collapses.
- **Key Outputs**: 
    - **Phase-Drift Diagrams**: Captures the discontinuous "radian-jump" ($> \pi$) at the detection limit.
    - **Bifurcation Analysis**: Maps the transition from stable order to topological chaos.
- **Usage**: `python FD_topology.py`

---

## 📊 Summary of Empirical Evidence

| Validation Pillar | Observed Metric | Framework Level | Status |
| :--- | :--- | :--- | :--- |
| **Structural** | Hermiticity $\approx 10^{-16}$ | FD₃ (Baseline) | ✅ Verified |
| **Asymptotic** | Scaling $\alpha \approx -0.933$ | FD₂ (Dynamics) | ✅ Verified |
| **Topological** | Phase Drift $> \pi$ | FD₁ (Structure) | ✅ Verified |

---

## 📖 Conceptual Mapping to the Paper

The scripts in this repository provide the reproducibility unit for the manuscript **"The Weil Explicit Formula as a Statistical Amplifier"**:

1. **Arithmetic → Logarithmic coordinates**: Handled by `FD_number_validation.py`.
2. **Amplifier / No-Hiding (FD₃, FD₂)**: Validated by `R3_stresstest_suite_v2.py`.
3. **Topological Rigidity (FD₁)**: Established by `FD_topology.py`.

---

## 📝 Disclaimer
This repository implements a **diagnostic methodology**, not a standalone proof. It provides numerical, structural, and topological criteria whose uniform satisfaction is shown in the accompanying paper to be equivalent to spectral confinement to the critical line.

---


---



