"""
FD-Framework Numerical Validation 
With proper hermiticity error measurement 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.linalg import eigh
from scipy.interpolate import interp1d
import mpmath as mp
import pandas as pd
import time
import os
from typing import Dict, List, Tuple, Optional

# ========== CONFIGURATION & REPRODUCIBILITY ==========
class Config:
    """Central configuration for reproducibility"""
    # mpmath precision
    MP_DPS = 50
    
    # Window parameters
    XI_MAX = 12
    N_XI = 801
    N_U = 2001
    
    # Sampling parameters
    T_MIN = 100
    T_MAX = 20000
    N_T_POINTS = 40
    
    # Numerical thresholds
    HERM_THRESHOLD = 1e-10
    ZERO_THRESHOLD = 1e-12
    MIN_LOG_VALUE = 1e-300  # For safe log computations
    
    @classmethod
    def setup(cls):
        """Setup global configuration"""
        mp.mp.dps = cls.MP_DPS
        np.random.seed(42)  # For any random operations
        print(f"✓ Configuration: mpmath dps={cls.MP_DPS}, "
              f"T∈[{cls.T_MIN}, {cls.T_MAX}], {cls.N_T_POINTS} points")

# ========== 1. ARRAY-SAFE WINDOW FUNCTIONS ==========
class RobustWindow:
    """Computes ĝ(ξ) = ∫ w²(u) e^{-iξu} du efficiently and robustly"""
    
    def __init__(self, window_type: str = 'bump'):
        self.type = window_type
        
        # Discretization grids
        self.u_grid = np.linspace(-1, 1, Config.N_U)
        self.xi_grid = np.linspace(-Config.XI_MAX, Config.XI_MAX, Config.N_XI)
        self.du = self.u_grid[1] - self.u_grid[0]
        
        # Precompute w² on u-grid
        self.w_sq_grid = self._w_sq(self.u_grid)
        
        # Compute Fourier transform via vectorized trapezoidal rule
        self._compute_fourier_transform()
        
    def _w(self, u: np.ndarray) -> np.ndarray:
        """Window function on [-1,1] (fully array-safe)"""
        u = np.asarray(u)
        out = np.zeros_like(u, dtype=float)
        mask = np.abs(u) < 1
        
        if not np.any(mask):
            return out
        
        x = u[mask]
        if self.type == 'bump':
            # C∞ compact support - reference window
            out[mask] = np.exp(-1/(1 - x**2))
        elif self.type == 'gaussian':
            # Analytic window - for comparison
            out[mask] = np.exp(-(x/0.5)**2)
        elif self.type == 'parzen':
            # C² window - stress test
            out[mask] = 1 - np.abs(x)
        elif self.type == 'cosine':
            # C¹ window - stress test
            out[mask] = np.cos(np.pi * x / 2)
        else:
            raise ValueError(f"Unknown window type: {self.type}")
        
        return out
    
    def _w_sq(self, u: np.ndarray) -> np.ndarray:
        """w²(u) for kernel (array-safe)"""
        return self._w(u) ** 2
    
    def _compute_fourier_transform(self):
        """Compute ĝ(ξ) via vectorized integration"""
        # Create phase matrix using outer product for efficiency
        phase_matrix = np.exp(-1j * np.outer(self.u_grid, self.xi_grid))
        integrand_matrix = self.w_sq_grid[:, None] * phase_matrix
        
        # Integrate over u using trapezoidal rule
        self.g_hat_vals = trapezoid(integrand_matrix, self.u_grid, axis=0)
        
        # Create linear interpolators
        self.g_hat_interp_real = interp1d(
            self.xi_grid, self.g_hat_vals.real, 
            kind='linear',
            bounds_error=False, 
            fill_value=0.0
        )
        self.g_hat_interp_imag = interp1d(
            self.xi_grid, self.g_hat_vals.imag,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
    
    def g_hat(self, xi: np.ndarray) -> np.ndarray:
        """Ĝ(ξ) for scalar or array inputs"""
        xi = np.asarray(xi)
        real_part = self.g_hat_interp_real(xi)
        imag_part = self.g_hat_interp_imag(xi)
        return real_part + 1j * imag_part
    
    def fourier_decay_rate(self) -> float:
        """Estimate exponential decay rate |ĝ(ξ)| ~ exp(-α|ξ|) with safe log"""
        # Fit to tail of Fourier transform
        tail_mask = self.xi_grid > 5
        if not np.any(tail_mask):
            return 0.0
        
        # Safe logarithm computation
        abs_vals = np.abs(self.g_hat_vals[tail_mask])
        safe_vals = np.maximum(abs_vals, Config.MIN_LOG_VALUE)
        log_vals = np.log(safe_vals)
        xi_vals = self.xi_grid[tail_mask]
        
        # Only use points with reasonable values
        valid_mask = np.isfinite(log_vals)
        if np.sum(valid_mask) < 3:
            return 0.0
            
        coeffs = np.polyfit(xi_vals[valid_mask], log_vals[valid_mask], 1)
        return max(0.0, -coeffs[0])  # decay rate (non-negative)

# ========== 2. EFFICIENT GRAM MATRIX CALCULATOR ==========
class EfficientGramCalculator:
    """Computes Gram matrix with O(N·bandwidth) complexity"""
    
    def __init__(self, all_gammas: np.ndarray, window_type: str = 'bump', 
                 H_factor: float = 0.5):
        self.all_gammas = np.asarray(all_gammas)
        self.window = RobustWindow(window_type)
        self.H_factor = H_factor
        self.window_type = window_type
        
        # Ensure sorted for searchsorted
        if not np.all(np.diff(self.all_gammas) >= 0):
            self.all_gammas = np.sort(self.all_gammas)
    
    def _local_spacing(self, gammas: np.ndarray) -> float:
        """Estimate local spacing for bandwidth calculation"""
        if len(gammas) > 20:
            # Use median spacing in the local window
            spacings = np.diff(gammas)
            if len(spacings) > 0:
                return float(np.median(spacings))
        
        # Fallback: theoretical spacing at center of window
        if len(gammas) > 0:
            T_center = np.mean(gammas) if len(gammas) > 0 else 100
            # Δ(T) ≈ 2π / log(T/2π)
            return 2 * np.pi / np.log(max(T_center/(2*np.pi), 1.1))
        
        return 1.0  # Default fallback
    
    def window_indices(self, T: float, H: float) -> Tuple[int, int]:
        """
        Find indices of zeros in window [T-H, T+H]
        Uses searchsorted for O(log N) complexity
        """
        i0 = np.searchsorted(self.all_gammas, T - H, side='left')
        i1 = np.searchsorted(self.all_gammas, T + H, side='right')
        return i0, i1
    
    def compute_gram_matrix(self, T: float
                           ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Efficient computation of G_{mn} = H·e^{i(γ_m-γ_n)T}·ĝ(H(γ_m-γ_n))
        
        Returns:
            G: Hermitian Gram matrix (after symmetrization)
            indices: indices in original array
            H: actual window width
            herm_err_raw: TRUE hermiticity error BEFORE symmetrization
        """
        H = self.H_factor * np.log(max(T, 100))
        
        # Get indices in window
        i0, i1 = self.window_indices(T, H)
        indices = np.arange(i0, i1)
        gammas = self.all_gammas[indices]
        N = len(gammas)
        
        if N == 0:
            G = np.zeros((0, 0), dtype=complex)
            return G, indices, H, 0.0
        
        # Estimate local spacing for bandwidth calculation
        local_spacing = self._local_spacing(gammas)
        
        # Convert frequency cutoff to index cutoff
        freq_cutoff = 10.0  # |ξ| < 10 corresponds to |Δγ| < 10/H
        max_band_idx = int(np.ceil(freq_cutoff / (H * local_spacing))) + 3
        
        # Create RAW unsymmetrized matrix (we fill BOTH directions independently)
        G_raw = np.zeros((N, N), dtype=complex)
        
        # Process in batches for efficiency
        batch_size = min(100, N)
        
        for i_start in range(0, N, batch_size):
            i_end = min(i_start + batch_size, N)
            i_range = np.arange(i_start, i_end)
            
            for i in i_range:
                j_start = max(0, i - max_band_idx)
                j_end = min(N, i + max_band_idx + 1)
                
                if j_start >= j_end:
                    continue
                
                j_indices = np.arange(j_start, j_end)
                
                # Compute all valid pairs in this band, regardless of i<j or i>j
                # We'll compute (i,j) for all j in the band
                dgamma = gammas[i] - gammas[j_indices]
                xi = H * dgamma
                
                # Apply cutoff
                cutoff_mask = np.abs(xi) < 10
                if not np.any(cutoff_mask):
                    continue
                
                valid_j = j_indices[cutoff_mask]
                d = dgamma[cutoff_mask]
                xi_valid = xi[cutoff_mask]
                
                # Compute raw values G_raw(i,j)
                fourier_vals = self.window.g_hat(xi_valid)
                phase = np.exp(1j * d * T)
                vals = H * phase * fourier_vals
                
                # Fill G_raw(i, j) for all valid j
                G_raw[i, valid_j] = vals
        
        # NOW compute TRUE raw hermiticity error (before any symmetrization)
        # This compares G_raw(i,j) with conj(G_raw(j,i)) for all i,j
        norm_raw = np.linalg.norm(G_raw, 'fro')
        if norm_raw > 0:
            herm_err_raw = np.linalg.norm(G_raw - G_raw.conj().T, 'fro') / norm_raw
        else:
            herm_err_raw = 0.0
        
        # Symmetrize to get proper Hermitian matrix
        G = 0.5 * (G_raw + G_raw.conj().T)
        
        # Remove numerical noise
        max_abs = np.max(np.abs(G))
        threshold = max_abs * Config.ZERO_THRESHOLD if max_abs > 0 else 0.0
        if threshold > 0:
            G[np.abs(G) < threshold] = 0
        
        return G, indices, H, herm_err_raw

# ========== 3. FD METRICS WITH ROBUST STATISTICS ==========
def compute_fd_metrics(G: np.ndarray, herm_err_raw: float = 0.0) -> Dict:
    """
    Compute all FD metrics with numerical robustness
    
    Args:
        G: Hermitian Gram matrix (after symmetrization)
        herm_err_raw: TRUE hermiticity error before symmetrization
    """
    N = G.shape[0]
    if N < 2:
        return {
            'r_F': np.nan, 'N': N, 'bandwidth_95': 0,
            'off_diag_ratio': np.nan, 'eig_range': (np.nan, np.nan),
            'herm_err_raw': herm_err_raw, 'herm_err_post': 0.0,
            'diag_std_rel': 0.0, 'cond_est': np.nan,
            'is_positive_definite': False
        }
    
    # 1. Post-symmetrization hermiticity check (should be tiny due to symmetrization)
    norm_G = np.linalg.norm(G, 'fro')
    herm_err_post = (np.linalg.norm(G - G.conj().T, 'fro') / norm_G 
                    if norm_G > 0 else 0.0)
    
    # 2. Diagonal statistics
    diag = np.diag(G)
    diag_abs = np.abs(diag)
    diag_mean = np.mean(diag_abs) if len(diag_abs) > 0 else 0.0
    diag_std = np.std(diag_abs) if len(diag_abs) > 1 else 0.0
    diag_energy = np.sum(diag_abs**2)
    
    # 3. Total and off-diagonal energy
    total_energy = np.sum(np.abs(G)**2)
    off_diag_energy = total_energy - diag_energy
    
    # 4. r_F with protection
    if diag_energy > Config.ZERO_THRESHOLD:
        r_F = np.sqrt(off_diag_energy) / np.sqrt(diag_energy)
        off_diag_ratio = off_diag_energy / total_energy
    else:
        r_F = np.inf
        off_diag_ratio = np.nan
    
    # 5. Bandwidth containing 95% of off-diagonal energy
    bandwidth_95 = N
    if off_diag_energy > 0:
        cum_energy = 0.0
        max_k = min(100, N)
        for k in range(1, max_k):
            band_energy = (np.sum(np.abs(np.diag(G, k))**2) + 
                          np.sum(np.abs(np.diag(G, -k))**2))
            cum_energy += band_energy
            if cum_energy / off_diag_energy >= 0.95:
                bandwidth_95 = k
                break
    
    # 6. Eigenvalue statistics (if matrix is reasonable size)
    eig_range = (np.nan, np.nan)
    cond_est = np.nan
    is_positive_definite = False
    
    if N <= 2000 and diag_energy > 0:
        try:
            eigvals = eigh(G, eigvals_only=True, check_finite=False)
            # Filter numerical noise
            eig_abs_max = np.max(np.abs(eigvals))
            valid_mask = np.abs(eigvals) > Config.ZERO_THRESHOLD * eig_abs_max
            eigvals_valid = eigvals[valid_mask]
            
            if len(eigvals_valid) >= 2:
                eig_min = np.min(eigvals_valid)
                eig_max = np.max(eigvals_valid)
                eig_range = (float(eig_min), float(eig_max))
                
                if eig_min > 0:
                    cond_est = eig_max / eig_min
                    is_positive_definite = True
        except Exception:
            # Silently skip if eigenvalue computation fails
            pass
    
    return {
        'r_F': float(r_F),
        'N': N,
        'diag_energy': float(diag_energy),
        'diag_std_rel': float(diag_std / (diag_mean + 1e-14)),
        'off_diag_ratio': float(off_diag_ratio),
        'bandwidth_95': bandwidth_95,
        'eig_range': eig_range,
        'cond_est': float(cond_est) if not np.isnan(cond_est) else None,
        'is_positive_definite': is_positive_definite,
        'herm_err_raw': float(herm_err_raw),    # TRUE error before symmetrization
        'herm_err_post': float(herm_err_post)   # Verification (should be tiny)
    }

# ========== 4. ZETA ZEROS LOADING ==========
def load_zeta_zeros(n_zeros: int = 3000, cache_file: str = None) -> np.ndarray:
    """
    Load zeta zeros, using cache if available
    """
    if cache_file is None:
        cache_file = f'zeta_zeros_{n_zeros}_dps{Config.MP_DPS}.npy'
    
    if os.path.exists(cache_file):
        print(f"Loading zeros from cache: {cache_file}")
        zeros = np.load(cache_file)
        if len(zeros) >= n_zeros:
            return zeros[:n_zeros]
    
    print(f"Computing first {n_zeros} zeta zeros via mpmath (dps={Config.MP_DPS})...")
    zeros = np.zeros(n_zeros)
    
    for n in range(1, n_zeros + 1):
        if n % 500 == 0:
            print(f"  {n}/{n_zeros}")
        zeros[n-1] = mp.im(mp.zetazero(n))
    
    zeros = np.sort(zeros)
    
    # Cache for future use
    np.save(cache_file, zeros)
    print(f"✓ Zeros saved to cache: {cache_file}")
    
    return zeros

# ========== 5. STATISTICAL ANALYSIS ==========
class StatisticalAnalyzer:
    """Analyzes statistical properties of FD metrics"""
    
    @staticmethod
    def compute_quantiles(data: List[float], quantiles: List[float] = None) -> Dict:
        """Compute quantiles of data"""
        if quantiles is None:
            quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        
        data_array = np.array(data)
        data_array = data_array[~np.isnan(data_array)]
        
        if len(data_array) == 0:
            return {q: np.nan for q in quantiles}
        
        return {q: float(np.quantile(data_array, q)) for q in quantiles}
    
    @staticmethod
    def power_law_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """Fit y ~ a * x^b, return (a, b, r^2)"""
        valid = (~np.isnan(x)) & (~np.isnan(y)) & (y > 0) & (x > 0)
        if np.sum(valid) < 3:
            return np.nan, np.nan, np.nan
        
        log_x = np.log(x[valid])
        log_y = np.log(y[valid])
        
        # Linear regression
        coeffs = np.polyfit(log_x, log_y, 1)
        b = coeffs[0]  # exponent
        log_a = coeffs[1]
        a = np.exp(log_a)
        
        # R² calculation
        y_pred = a * x[valid] ** b
        y_mean = np.mean(y[valid])
        ss_res = np.sum((y[valid] - y_pred) ** 2)
        ss_tot = np.sum((y[valid] - y_mean) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        
        return a, b, r2
    
    @staticmethod
    def summarize_robustness(results: Dict[str, List[Dict]], T_min: float = 500) -> Dict:
        """Compute statistical summary of robustness claims"""
        summary = {}
        
        for wt in results:
            # Filter by T >= T_min
            T_vals = np.array([m['T'] for m in results[wt]])
            N_vals = np.array([m['N'] for m in results[wt]])
            rF_vals = np.array([m['r_F'] for m in results[wt]])
            bw_vals = np.array([m['bandwidth_95'] for m in results[wt]])
            herm_raw_vals = np.array([m['herm_err_raw'] for m in results[wt]])
            herm_post_vals = np.array([m['herm_err_post'] for m in results[wt]])
            
            mask_herm = (T_vals >= T_min)
            mask_fd = (T_vals >= T_min) & (N_vals >= 2) & np.isfinite(rF_vals) & np.isfinite(bw_vals)

            if np.sum(mask_herm) == 0:
                continue

            summary[wt] = {
                'rF_median': float(np.nanmedian(rF_vals[mask_fd])) if np.sum(mask_fd) else np.nan,
                'rF_max': float(np.nanmax(rF_vals[mask_fd])) if np.sum(mask_fd) else np.nan,
                'rF_q95': float(np.nanquantile(rF_vals[mask_fd], 0.95)) if np.sum(mask_fd) else np.nan,

                'bw_median': float(np.nanmedian(bw_vals[mask_fd])) if np.sum(mask_fd) else np.nan,
                'bw_max': float(np.nanmax(bw_vals[mask_fd])) if np.sum(mask_fd) else np.nan,
                'bw_q95': float(np.nanquantile(bw_vals[mask_fd], 0.95)) if np.sum(mask_fd) else np.nan,

                'herm_raw_median': float(np.nanmedian(herm_raw_vals[mask_herm])),
                'herm_raw_max': float(np.nanmax(herm_raw_vals[mask_herm])),
                'herm_raw_q99': float(np.nanquantile(herm_raw_vals[mask_herm], 0.99)),

                'herm_post_max': float(np.nanmax(herm_post_vals[mask_herm])),
                'herm_post_q99': float(np.nanquantile(herm_post_vals[mask_herm], 0.99)),

                'n_samples': int(np.sum(mask_fd))
            }

        
        return summary

# ========== 6. THEORETICAL PREDICTIONS ==========
def theoretical_window_count(T: float, H_factor: float = 0.5) -> float:
    """
    Theoretical prediction for number of zeros in window [T-H, T+H]
    
    For zeros up to height T: N(T) ≈ (T/2π) log(T/2π) - (T/2π) + ...
    Density: dN/dT ≈ (1/2π) log(T/2π)
    
    In window of width 2H: N_window ≈ 2H × (1/2π) log(T/2π)
    """
    H = H_factor * np.log(max(T, 100))
    density = (1/(2*np.pi)) * np.log(max(T/(2*np.pi), 1.0))
    return 2 * H * density

# ========== 7. MAIN PRODUCTION DEMO ==========
def run_production_demo() -> Tuple[Dict, pd.DataFrame, Dict]:
    """Run complete production demo with professional outputs"""
    print("="*80)
    print("FD-FRAMEWORK: NUMERICAL VALIDATION DEMO (FINAL CORRECTED)")
    print("="*80)
    
    # Setup configuration
    Config.setup()
    
    # 1. Load zeros
    n_zeros = 3000
    gammas = load_zeta_zeros(n_zeros)
    
    print(f"\nLoaded {len(gammas)} zeros")
    print(f"Range: [{gammas[0]:.2f}, {gammas[-1]:.2f}]")
    
    # Limit T-range to what the cached zeros can actually support
    T_max_data = float(gammas[-1])
    T_max_eff = min(Config.T_MAX, 0.95 * T_max_data)  # safety margin
    if T_max_eff < Config.T_MAX:
        print(f"⚠️ Limiting T_MAX from {Config.T_MAX} to {T_max_eff:.2f} (zero range)")

    
    # Compute normalized spacings for GUE comparison
    if len(gammas) > 1000:
        spacings_raw = np.diff(gammas[:1000])
        spacings_normalized = spacings_raw / np.mean(spacings_raw)
        print(f"Mean spacing: {np.mean(spacings_raw):.4f}")
        print(f"Normalized spacing std: {np.std(spacings_normalized):.4f}")
    else:
        spacings_normalized = None
    
    # 2. Setup calculators
    window_types = ['bump', 'gaussian']
    calculators = {wt: EfficientGramCalculator(gammas, window_type=wt) 
                   for wt in window_types}
    
    # 3. T-values
    T_values = np.logspace(np.log10(Config.T_MIN), np.log10(T_max_eff),
                           Config.N_T_POINTS)

    
    
    # 4. Collect metrics
    print("\nComputing FD metrics across heights...")
    results = {wt: [] for wt in window_types}
    sample_matrices = {}
    
    start_time = time.time()
    
    for wt in window_types:
        print(f"\n  Window: {wt}")
        calc = calculators[wt]
        
        for i, T in enumerate(T_values):
            if i % 8 == 0:
                print(f"    T={T:.0f}", end=' ', flush=True)
            
            # Compute Gram matrix with TRUE raw hermiticity error
            G, idx, H, herm_err_raw = calc.compute_gram_matrix(T)
            
            # Compute all metrics
            metrics = compute_fd_metrics(G, herm_err_raw)
            metrics.update({
                'T': T,
                'H': H,
                'N_theory_window': theoretical_window_count(T, calc.H_factor)
            })
            results[wt].append(metrics)
            
            # Store sample matrices for visualization
            if wt == 'bump':
                if abs(T - 1000) < 50 and 'T1000' not in sample_matrices:
                    sample_matrices['T1000'] = (G[:100, :100], T, H)
                if abs(T - 5000) < 100 and 'T5000' not in sample_matrices:
                    sample_matrices['T5000'] = (G[:100, :100], T, H)
        
        print()
    
    elapsed = time.time() - start_time
    print(f"\n✓ Computation completed in {elapsed:.1f}s")
    
    # 5. Statistical analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    analyzer = StatisticalAnalyzer()
    robustness_summary = analyzer.summarize_robustness(results, T_min=500)
    
    for wt in window_types:
        if wt in robustness_summary:
            s = robustness_summary[wt]
            print(f"\n{wt.upper()} window (T ≥ 500):")
            print(f"  r_F: median={s['rF_median']:.4f}, max={s['rF_max']:.4f}, "
                  f"95% ≤ {s['rF_q95']:.4f}")
            print(f"  Bandwidth: median={s['bw_median']:.1f}, max={s['bw_max']:.1f}, "
                  f"95% ≤ {s['bw_q95']:.1f}")
            print(f"  Hermiticity error (raw): median={s['herm_raw_median']:.2e}, "
                  f"max={s['herm_raw_max']:.2e}, 99% ≤ {s['herm_raw_q99']:.2e}")
            print(f"  Hermiticity error (post): max={s['herm_post_max']:.2e}, "
                  f"99% ≤ {s['herm_post_q99']:.2e} (symmetrization verification)")
    
    # 6. Power law fits
    print("\n" + "="*80)
    print("SCALING ANALYSIS")
    print("="*80)
    
    for wt in window_types:
        T_vals = np.array([m['T'] for m in results[wt]])
        rF_vals = np.array([m['r_F'] for m in results[wt]])
        
        a, b, r2 = analyzer.power_law_fit(T_vals, rF_vals)
        if not np.isnan(b):
            print(f"\n{wt.upper()} window: r_F(T) ∝ T^{b:.3f}")
            print(f"  R² = {r2:.3f}")
            
            # Interpret exponent
            if b < -0.2:
                print(f"  → Strong diagonal dominance improvement with height")
            elif b < 0:
                print(f"  → Mild improvement with height")
            else:
                print(f"  → No significant improvement with height")
    
    # 7. Create plots
    create_production_plots(results, sample_matrices, gammas, window_types, 
                           spacings_normalized)
    
    # 8. Create summary table
    summary_df = create_summary_table(results, window_types)
    
    return results, summary_df, robustness_summary

def create_production_plots(results: Dict, sample_matrices: Dict, 
                           gammas: np.ndarray, window_types: List[str],
                           normalized_spacings: Optional[np.ndarray] = None):
    """Generate professional publication-quality plots"""
    # Use default style for maximum compatibility
    plt.style.use('default')
    fig = plt.figure(figsize=(18, 14))
    
    colors = {'bump': '#1f77b4', 'gaussian': '#ff7f0e'}
    markers = {'bump': 'o', 'gaussian': 's'}
    linestyles = {'bump': '-', 'gaussian': '--'}
    
    # ----- Plot 1: r_F(T) evolution -----
    ax1 = plt.subplot(3, 4, 1)
    for wt in window_types:
        T_vals = np.array([m['T'] for m in results[wt]])
        rF_vals = np.array([m['r_F'] for m in results[wt]])
        
        # Filter valid points
        valid = ~np.isnan(rF_vals) & (rF_vals > 0)
        if np.sum(valid) > 3:
            ax1.plot(T_vals[valid], rF_vals[valid], 
                    marker=markers[wt], markersize=4,
                    color=colors[wt], label=wt, 
                    linewidth=1.5, alpha=0.8, linestyle=linestyles[wt])
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Height $T$', fontsize=12)
    ax1.set_ylabel('$r_F(T) = \\|\\text{off-diag}\\|_F / \\|\\text{diag}\\|_F$', 
                   fontsize=12)
    ax1.set_title('(A) FD-Hierarchy: Diagonal Dominance', 
                  fontsize=13, fontweight='bold', pad=12)
    ax1.legend(frameon=True, fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_ylim(1e-3, 0.5)
    
    # ----- Plot 2: Bandwidth evolution -----
    ax2 = plt.subplot(3, 4, 2)
    for wt in window_types:
        T_vals = [m['T'] for m in results[wt]]
        bw_vals = [m['bandwidth_95'] for m in results[wt]]
        ax2.plot(T_vals, bw_vals, marker=markers[wt], markersize=3,
                color=colors[wt], label=wt, linewidth=1, alpha=0.7,
                linestyle=linestyles[wt])
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Height $T$', fontsize=12)
    ax2.set_ylabel('Bandwidth (95% energy)', fontsize=12)
    ax2.set_title('(B) Spectral Localization', 
                  fontsize=13, fontweight='bold', pad=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # ----- Plot 3: Hermiticity errors (CORRECTED) -----
    ax3 = plt.subplot(3, 4, 3)
    for wt in window_types:
        T_vals = [m['T'] for m in results[wt]]
        herm_raw_vals = [m['herm_err_raw'] for m in results[wt]]
        herm_post_vals = [m['herm_err_post'] for m in results[wt]]
        
        ax3.semilogy(T_vals, herm_raw_vals, marker='.', markersize=2,
                    color=colors[wt], label=f'{wt} (raw)', linewidth=1, alpha=0.7)
        ax3.semilogy(T_vals, herm_post_vals, marker='x', markersize=2,
                    color=colors[wt], label=f'{wt} (post)', linewidth=1, 
                    alpha=0.7, linestyle=':')
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Height $T$', fontsize=12)
    ax3.set_ylabel('Hermiticity error', fontsize=12)
    ax3.set_title('(C) Numerical Stability\n(raw: before symmetrization)', 
                  fontsize=13, fontweight='bold', pad=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, ncol=2)
    ax3.set_ylim(1e-16, 1)
    
    # Add explanation
    ax3.text(0.02, 0.98, 'Note: raw error measures\nactual numerical deviation\nfrom hermiticity',
            transform=ax3.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ----- Plot 4: Heatmap at T=1000 -----
    ax4 = plt.subplot(3, 4, 4)
    if 'T1000' in sample_matrices:
        G_sample, T_val, H_val = sample_matrices['T1000']
        im = ax4.imshow(np.log10(np.abs(G_sample) + 1e-12), 
                       cmap='viridis', aspect='auto',
                       interpolation='nearest')
        ax4.set_title(f'(D) Gram Matrix: $T={T_val:.0f}$, $H={H_val:.1f}$', 
                     fontsize=13, fontweight='bold', pad=12)
        ax4.set_xlabel('Index $j$', fontsize=12)
        ax4.set_ylabel('Index $i$', fontsize=12)
        cbar = plt.colorbar(im, ax=ax4, pad=0.01)
        cbar.set_label('$\\log_{10}|G_{ij}|$', fontsize=12)
    
    # ----- Plot 5: Heatmap at T=5000 -----
    ax5 = plt.subplot(3, 4, 5)
    if 'T5000' in sample_matrices:
        G_sample, T_val, H_val = sample_matrices['T5000']
        im = ax5.imshow(np.log10(np.abs(G_sample) + 1e-12),
                       cmap='viridis', aspect='auto',
                       interpolation='nearest')
        ax5.set_title(f'(E) Gram Matrix: $T={T_val:.0f}$, $H={H_val:.1f}$', 
                     fontsize=13, fontweight='bold', pad=12)
        ax5.set_xlabel('Index $j$', fontsize=12)
        ax5.set_ylabel('Index $i$', fontsize=12)
        cbar = plt.colorbar(im, ax=ax5, pad=0.01)
        cbar.set_label('$\\log_{10}|G_{ij}|$', fontsize=12)
    
    # ----- Plot 6: Band energy decay -----
    ax6 = plt.subplot(3, 4, 6)
    if 'T1000' in sample_matrices:
        G_sample, T_val, _ = sample_matrices['T1000']
        N_small = min(50, G_sample.shape[0])
        band_energies = []
        
        for k in range(N_small):
            if k == 0:
                energy = np.sum(np.abs(np.diag(G_sample))**2)
            else:
                energy = (np.sum(np.abs(np.diag(G_sample, k))**2) + 
                         np.sum(np.abs(np.diag(G_sample, -k))**2))
            band_energies.append(energy)
        
        total_energy = np.sum(band_energies)
        if total_energy > 0:
            band_energies = np.array(band_energies) / total_energy
        
        ax6.bar(range(N_small), band_energies, color='#2ca02c', alpha=0.7)
        ax6.set_xlabel('Band offset $k$', fontsize=12)
        ax6.set_ylabel('Normalized energy', fontsize=12)
        ax6.set_title('(F) Energy Decay', 
                     fontsize=13, fontweight='bold', pad=12)
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3, axis='y')
    
    # ----- Plot 7: Zero spacing distribution (with GUE) -----
    ax7 = plt.subplot(3, 4, 7)
    if normalized_spacings is not None and len(normalized_spacings) > 10:
        n, bins, patches = ax7.hist(normalized_spacings, bins=30, density=True, 
                                   alpha=0.7, color='#9467bd', edgecolor='black')
        ax7.set_xlabel('Normalized spacing $s = (\\gamma_{n+1}-\\gamma_n)/\\langle s\\rangle$', 
                       fontsize=12)
        ax7.set_ylabel('Density $p(s)$', fontsize=12)
        ax7.set_title('(G) Zero Spacing vs GUE', 
                     fontsize=13, fontweight='bold', pad=12)
        ax7.grid(True, alpha=0.3)
        
        # GUE prediction (Wigner surmise)
        s = np.linspace(0, 5, 200)
        gue_pred = (32/np.pi**2) * s**2 * np.exp(-4*s**2/np.pi)
        ax7.plot(s, gue_pred, 'r-', linewidth=2.5, alpha=0.8, label='GUE prediction')
        
        # Poisson prediction (uncorrelated)
        poisson_pred = np.exp(-s)
        ax7.plot(s, poisson_pred, 'g--', linewidth=2, alpha=0.6, label='Poisson')
        
        ax7.legend(fontsize=11, frameon=True)
        ax7.set_xlim(0, 4)
    
    # ----- Plot 8: Window comparison in Fourier domain -----
    ax8 = plt.subplot(3, 4, 8)
    xi_test = np.linspace(-10, 10, 401)
    
    for wt in window_types:
        window = RobustWindow(wt)
        g_hat = window.g_hat(xi_test)
        ax8.plot(xi_test, np.abs(g_hat), label=wt, linewidth=2.5, alpha=0.8)
    
    ax8.set_xlabel('$\\xi$', fontsize=12)
    ax8.set_ylabel('$|\\hat{g}(\\xi)|$', fontsize=12)
    ax8.set_title('(H) Fourier Transforms of $w^2$', 
                 fontsize=13, fontweight='bold', pad=12)
    ax8.set_yscale('log')
    ax8.legend(fontsize=11, frameon=True)
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(1e-4, 2)
    
    # ----- Plot 9: N(T) in window vs theory -----
    ax9 = plt.subplot(3, 4, 9)
    for wt in window_types:
        T_vals = [m['T'] for m in results[wt]]
        N_vals = [m['N'] for m in results[wt]]
        ax9.plot(T_vals, N_vals, marker='.', markersize=3,
                color=colors[wt], label=f'{wt} measured', linewidth=1, alpha=0.7)
        
        # Theoretical prediction for this window type
        H_factor = 0.5  # Same as calculator
        T_theory = np.logspace(np.log10(Config.T_MIN), np.log10(Config.T_MAX), 100)
        N_theory = [theoretical_window_count(T, H_factor) for T in T_theory]
        ax9.plot(T_theory, N_theory, '--', color=colors[wt], 
                alpha=0.5, linewidth=1.5, label=f'{wt} theory')
    
    ax9.set_xscale('log')
    ax9.set_xlabel('Height $T$', fontsize=12)
    ax9.set_ylabel('Zeros in window $N_{\\text{window}}(T)$', fontsize=12)
    ax9.set_title('(I) Window Content vs Theory\n$N_{\\text{window}} \\approx 2H \\cdot \\frac{1}{2\\pi}\\log(T/2\\pi)$', 
                 fontsize=13, fontweight='bold', pad=12)
    ax9.grid(True, alpha=0.3)
    ax9.legend(fontsize=9, ncol=2)
    
    # ----- Plot 10: Off-diagonal ratio -----
    ax10 = plt.subplot(3, 4, 10)
    for wt in window_types:
        T_vals = [m['T'] for m in results[wt]]
        ratio_vals = [m['off_diag_ratio'] for m in results[wt]]
        ax10.plot(T_vals, ratio_vals, marker='.', markersize=3,
                 color=colors[wt], label=wt, linewidth=1, alpha=0.7,
                 linestyle=linestyles[wt])
    
    ax10.set_xscale('log')
    ax10.set_yscale('log')
    ax10.set_xlabel('Height $T$', fontsize=12)
    ax10.set_ylabel('Off-diagonal energy ratio', fontsize=12)
    ax10.set_title('(J) Off-diagonal Energy Fraction', 
                  fontsize=13, fontweight='bold', pad=12)
    ax10.grid(True, alpha=0.3)
    ax10.legend(fontsize=11)
    
    # ----- Plot 11: Diagonal uniformity -----
    ax11 = plt.subplot(3, 4, 11)
    for wt in window_types:
        T_vals = [m['T'] for m in results[wt]]
        diag_std_vals = [m['diag_std_rel'] for m in results[wt]]
        ax11.plot(T_vals, diag_std_vals, marker='.', markersize=3,
                 color=colors[wt], label=wt, linewidth=1, alpha=0.7,
                 linestyle=linestyles[wt])
    
    ax11.set_xscale('log')
    ax11.set_xlabel('Height $T$', fontsize=12)
    ax11.set_ylabel('Relative diagonal std', fontsize=12)
    ax11.set_title('(K) Diagonal Uniformity', 
                  fontsize=13, fontweight='bold', pad=12)
    ax11.grid(True, alpha=0.3)
    ax11.legend(fontsize=11)
    
    # ----- Plot 12: Summary statistics -----
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Create summary text
    summary_text = []
    summary_text.append("SUMMARY STATISTICS (T ≥ 500)")
    summary_text.append("=" * 40)
    
    # Get robustness summary
    analyzer = StatisticalAnalyzer()
    robustness_summary = analyzer.summarize_robustness(results, T_min=500)
    
    for wt in window_types:
        if wt in robustness_summary:
            s = robustness_summary[wt]
            summary_text.append(f"\n{wt.upper()}:")
            summary_text.append(f"  r_F: {s['rF_median']:.3f} (med)")
            summary_text.append(f"       {s['rF_q95']:.3f} (95%ile)")
            summary_text.append(f"  BW:  {s['bw_median']:.0f} (med)")
            summary_text.append(f"  herm_raw: {s['herm_raw_median']:.1e}")
    
    summary_text.append("\n" + "=" * 40)
    summary_text.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    ax12.text(0.05, 0.95, "\n".join(summary_text),
              transform=ax12.transAxes,
              fontsize=9,
              family='monospace',
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('FD_Framework_Production_Results.png', dpi=300, bbox_inches='tight')
    plt.savefig('FD_Framework_Production_Results.pdf', bbox_inches='tight')
    plt.show()

def create_summary_table(results: Dict, window_types: List[str]) -> pd.DataFrame:
    """Create scientific summary table"""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    # Select representative T values
    T_selected = [100, 500, 1000, 2000, 5000, 10000, 20000]
    
    table_data = []
    for T_target in T_selected:
        row = {'T': T_target}
        
        for wt in window_types:
            # Find nearest computed T value
            T_vals = np.array([m['T'] for m in results[wt]])
            if len(T_vals) == 0:
                continue
                
            idx = np.argmin(np.abs(T_vals - T_target))
            metrics = results[wt][idx]
            
            row[f'{wt}_N'] = metrics['N']
            row[f'{wt}_rF'] = f"{metrics['r_F']:.4f}"
            row[f'{wt}_bw'] = metrics['bandwidth_95']
            row[f'{wt}_off_ratio'] = f"{metrics['off_diag_ratio']:.4f}"
            row[f'{wt}_herm_raw'] = f"{metrics['herm_err_raw']:.2e}"
            row[f'{wt}_herm_post'] = f"{metrics['herm_err_post']:.2e}"
        
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Reorder and rename columns
    columns = ['T']
    for wt in window_types:
        columns.extend([f'{wt}_N', f'{wt}_rF', f'{wt}_bw', 
                       f'{wt}_off_ratio', f'{wt}_herm_raw', f'{wt}_herm_post'])
    
    df = df[columns]
    
    rename_dict = {'T': 'T'}
    for wt in window_types:
        rename_dict.update({
            f'{wt}_N': f'{wt[:3]}.N',
            f'{wt}_rF': f'{wt[:3]}.r_F',
            f'{wt}_bw': f'{wt[:3]}.BW',
            f'{wt}_off_ratio': f'{wt[:3]}.off_ratio',
            f'{wt}_herm_raw': f'{wt[:3]}.herm_raw',
            f'{wt}_herm_post': f'{wt[:3]}.herm_post'
        })
    
    df = df.rename(columns=rename_dict)
    
    # Print formatted table
    print("\nEmpirical FD metrics for selected heights T:")
    print("-"*120)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" 
                      if isinstance(x, float) else str(x)))
    
    # Save outputs
    df.to_csv('fd_framework_summary.csv', index=False, float_format='%.6f')
    
    latex_str = df.to_latex(index=False, 
                           float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
                           column_format='l' + 'r' * (len(df.columns) - 1))
    with open('fd_framework_summary.tex', 'w') as f:
        f.write(latex_str)
    
    print("\n✓ Table saved as:")
    print("  - fd_framework_summary.csv")
    print("  - fd_framework_summary.tex (LaTeX)")
    
    return df

# ========== 8. MAIN EXECUTION ==========
if __name__ == "__main__":
    print("FD-Framework Numerical Validation v3.1 - FINAL CORRECTED")
    print("="*80)
    print("With proper hermiticity error measurement:")
    print("  • Raw error: true numerical deviation before symmetrization")
    print("  • Post error: verification after symmetrization")
    print("  • All other fixes maintained")
    print("="*80)
    
    try:
        results, summary_df, robustness_summary = run_production_demo()
        
        print("\n" + "="*80)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nKey empirical observations (from measured data):")
        for wt in robustness_summary:
            s = robustness_summary[wt]
            print(f"\n{wt.upper()} window:")
            print(f"  • r_F(T) < {s['rF_q95']:.3f} for 95% of T ≥ 500")
            print(f"  • Bandwidth < {s['bw_q95']:.0f} for 95% of T ≥ 500")
            print(f"  • Raw hermiticity errors ~ {s['herm_raw_median']:.1e} (median)")
            print(f"  • Post-symmetrization errors < {s['herm_post_q99']:.1e} (99%ile)")
        
        print("\n" + "="*80)
        print("Generated outputs:")
        print("  - FD_Framework_Production_Results.png (12-panel plot)")
        print("  - FD_Framework_Production_Results.pdf (vector graphic)")
        print("  - fd_framework_summary.csv (numerical data)")
        print("  - fd_framework_summary.tex (LaTeX table)")
        print("  - zeta_zeros_*.npy (cached zeros for reproducibility)")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()