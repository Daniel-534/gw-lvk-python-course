import json

nb = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "cells": []
}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

# ── Cell 0: Title & ToC ──────────────────────────────────────────────────────
nb["cells"].append(md(
r"""# Lesson 6 — Matched Filtering & Signal Detection

**Gravitational-Wave Data Analysis with Python**  
LVK Python Course — Module 6

> *"The optimal filter is not found by intuition but by mathematics — and when found, it is surprisingly simple."*

---

## Table of Contents

1. [Introduction: Why Matched Filtering?](#1.-Introduction:-Why-Matched-Filtering?)
2. [Derivation of the Optimal SNR Statistic](#2.-Derivation-of-the-Optimal-SNR-Statistic)
   - 2.1 The Noise-Weighted Inner Product
   - 2.2 The Wiener Filter and Optimal Template
   - 2.3 The SNR Time Series
   - 2.4 Distribution Under Noise and Signal Hypotheses
3. [Generating Waveform Templates with PyCBC](#3.-Generating-Waveform-Templates-with-PyCBC)
   - 3.1 Frequency-Domain Templates: `get_fd_waveform`
   - 3.2 Time-Domain Templates: `get_td_waveform`
   - 3.3 Approximants: SEOBNRv4 and IMRPhenomD
   - 3.4 Parameter Space: Masses, Spins, Inclination
   - 3.5 Visualising the Waveform and Frequency Evolution
4. [PSD Estimation and Frequency-Domain Whitening](#4.-PSD-Estimation-and-Frequency-Domain-Whitening)
   - 4.1 Welch's Method via `pycbc.psd`
   - 4.2 `interpolate` and `inverse_spectrum_truncation`
   - 4.3 Whitening Strain Data
   - 4.4 Understanding the Noise Floor
5. [Computing the Complex SNR Time Series](#5.-Computing-the-Complex-SNR-Time-Series)
   - 5.1 `pycbc.filter.matched_filter`
   - 5.2 Interpreting Magnitude |ρ(t)| and Phase
   - 5.3 Finding Trigger Times and Peaks
6. [The χ² Consistency Test](#6.-The-χ²-Consistency-Test)
   - 6.1 Theory: Sub-templates and Expected Power
   - 6.2 `pycbc.vetoes.power_chisq`
   - 6.3 Re-weighted SNR
7. [GW150914 Detection Pipeline — Step by Step](#7.-GW150914-Detection-Pipeline-—-Step-by-Step)
   - 7.1 Downloading H1 and L1 Strain
   - 7.2 Conditioning the Data
   - 7.3 PSD Estimation on Off-Source Data
   - 7.4 Template Construction
   - 7.5 Running the Matched Filter
   - 7.6 Applying the χ² Veto
   - 7.7 Finding Peak SNR and Checking Coincidence
   - 7.8 Comparison with Published Values
8. [Additional Topics: Template Banks, Coincidence, and FAR](#8.-Additional-Topics:-Template-Banks,-Coincidence,-and-FAR)
9. [Student Exercises](#9.-Student-Exercises)
10. [References](#10.-References)
"""))

# ── Cell 1: Introduction ─────────────────────────────────────────────────────
nb["cells"].append(md(
r"""---
## 1. Introduction: Why Matched Filtering?

At the heart of every gravitational-wave detection pipeline lies a single question:

> *Given a noisy data stream $s(t) = n(t) + h(t)$, is a signal $h(t)$ present?*

Because LIGO and Virgo noise is broad-band, coloured, and typically 3–4 orders of magnitude larger than any signal in the raw time series, a naïve comparison in the time domain is hopeless. However, for compact binary coalescences (CBCs), we can **compute the expected waveform** from first principles using General Relativity. This prior knowledge is what makes matched filtering so powerful.

### Why Matched Filtering is Optimal

In the presence of **stationary Gaussian noise**, matched filtering is provably the **maximum-likelihood detector** for a known signal shape embedded in noise. The proof rests on the Neyman–Pearson lemma: the optimal test statistic that maximises the probability of detection for a fixed false-alarm rate is the likelihood ratio, and for Gaussian noise this ratio reduces exactly to the matched-filter SNR.

In practice, CBC noise is neither perfectly stationary nor perfectly Gaussian, but matched filtering remains the foundation of the search, supplemented by consistency tests (like $\\chi^2$) that veto non-Gaussian transients.

### Prerequisites
- Lesson 1 (GW detector physics and interferometry)
- Lesson 2 (Python/NumPy/SciPy)
- Lesson 3 (Time-domain strain data handling)
- Lesson 4 (Spectral analysis and PSD estimation)
- Basic familiarity with Fourier transforms and complex analysis
"""))

# ── Cell 2: Imports ──────────────────────────────────────────────────────────
nb["cells"].append(code(
r"""# ── Environment setup ──────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import signal as spsignal

# PyCBC — the main library used throughout this lesson
try:
    import pycbc
    from pycbc import types, filter as pycbc_filter, psd as pycbc_psd
    from pycbc.waveform import get_fd_waveform, get_td_waveform
    from pycbc.filter import matched_filter, sigma
    from pycbc.psd import welch, interpolate, inverse_spectrum_truncation
    from pycbc.types import TimeSeries, FrequencySeries, float32, float64
    from pycbc.catalog import Merger
    HAS_PYCBC = True
    print(f"PyCBC version: {pycbc.__version__}")
except ImportError:
    HAS_PYCBC = False
    print("WARNING: pycbc not installed. Install with:  pip install pycbc")
    print("Code cells that require PyCBC will print a notice instead of running.")

# gwpy (optional, for comparison)
try:
    from gwpy.timeseries import TimeSeries as GWpyTimeSeries
    HAS_GWPY = True
except ImportError:
    HAS_GWPY = False

# Global matplotlib style
plt.rcParams.update({
    'figure.dpi': 110,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
})

print("Setup complete.")
"""))

# ── Cell 3: Section 2 Theory ─────────────────────────────────────────────────
nb["cells"].append(md(
r"""---
## 2. Derivation of the Optimal SNR Statistic

### 2.1 The Noise-Weighted Inner Product

Let $\tilde{a}(f)$ and $\tilde{b}(f)$ be the Fourier transforms of two real time series. The **noise-weighted inner product** is defined as:

$$
\langle a \,|\, b \rangle \;=\; 4 \,\mathrm{Re} \int_0^{\infty} \frac{\tilde{a}^*(f)\,\tilde{b}(f)}{S_n(f)}\, df
$$

where $S_n(f)$ is the **one-sided power spectral density** of the noise, and the factor of 4 ensures that $\langle n | n \rangle = 2\sigma_n^2$ for a white-noise process.

**Key properties:**

| Property | Expression |
|---|---|
| Symmetry | $\langle a \,\|\, b \rangle = \langle b \,\|\, a \rangle$ |
| Noise variance | $\langle n \,\|\, n \rangle_\text{expected} = 2$ (normalised) |
| Signal norm (optimal SNR) | $\rho_\text{opt}^2 = \langle h \,\|\, h \rangle$ |

This inner product induces a norm $\|h\| = \sqrt{\langle h | h \rangle}$ that measures the *detectability* of a signal: a waveform with $\|h\| = 10$ would be detected with SNR 10 in ideal noise.

### 2.2 The Wiener Filter and Optimal Template

Suppose the data are $s(t) = n(t) + A\,h(t)$ where $A$ is an unknown amplitude and $h(t)$ is the known template. We want to construct a linear filter $K(t)$ such that the output

$$
\rho \;=\; \int_{-\infty}^{\infty} K(\tau)\, s(t - \tau)\, d\tau
$$

is maximised in SNR. By the **Wiener–Hopf theorem**, the optimal filter in the frequency domain is:

$$
\tilde{K}(f) \;=\; \frac{\tilde{h}^*(f)}{S_n(f)}
$$

This is the **matched filter**: the template normalised by the noise PSD. Notice that it down-weights frequency bins where the noise is large (poor sensitivity) and up-weights bins where the noise is small (high sensitivity).

### 2.3 The SNR Time Series

Inserting the optimal filter, the SNR as a function of coalescence time $t_c$ is:

$$
\boxed{
\rho(t_c) \;=\; \frac{\langle s \,|\, h(t_c) \rangle}{\sigma_h}
\;=\; \frac{1}{\sigma_h}\, 4 \,\mathrm{Re} \int_0^{\infty} \frac{\tilde{s}^*(f)\,\tilde{h}(f)}{S_n(f)}\, e^{2\pi i f t_c}\, df
}
$$

where the normalisation factor is:

$$
\sigma_h^2 \;=\; \langle h \,|\, h \rangle \;=\; 4 \int_0^{\infty} \frac{|\tilde{h}(f)|^2}{S_n(f)}\, df
$$

This is simply an **inverse Fourier transform** of $\tilde{s}^*(f)\tilde{h}(f)/S_n(f)$, which can be computed efficiently with the FFT. The complex SNR $z(t_c) = \rho(t_c)\,e^{i\phi(t_c)}$ encodes both the amplitude and the phase of the match.

### 2.4 Distribution Under Noise and Signal Hypotheses

Under the **noise-only hypothesis** ($A=0$):

$$
\rho \sim \chi_2 / \sqrt{2} \quad \Rightarrow \quad |\rho|^2 \sim \chi^2_2
$$

The probability of a false alarm above threshold $\rho_*$ is:

$$
p_\text{FA}(\rho > \rho_*) = e^{-\rho_*^2/2}
$$

Under the **signal hypothesis** ($A \neq 0$, optimal SNR $= \rho_\text{opt}$):

$$
|\rho| \sim \text{Rice}(\rho_\text{opt},\, 1)
$$

The probability of detection at threshold $\rho_*$ is:

$$
p_\text{det} = Q_1(\rho_\text{opt},\, \rho_*)
$$

where $Q_1$ is the Marcum Q-function. The curves $p_\text{det}$ vs. $p_\text{FA}$ form the **Receiver Operating Characteristic (ROC)** — matched filtering achieves the Neyman–Pearson optimal ROC.
"""))

# ── Cell 4: ROC / distribution demo ─────────────────────────────────────────
nb["cells"].append(code(
r"""# ── 2.4  Noise vs signal SNR distributions (analytic) ──────────────────────
from scipy.special import iv  # Modified Bessel function (for Rice dist)
from scipy.stats import ncx2  # Non-central chi-squared

rho_vals = np.linspace(0, 10, 400)

# Under noise: |rho|^2 ~ chi2_2, so |rho| has Rayleigh distribution
rayleigh_pdf = rho_vals * np.exp(-rho_vals**2 / 2)

# Under signal with rho_opt = 8: Rice distribution
rho_opt = 8.0
rice_pdf = rho_vals * np.exp(-(rho_vals**2 + rho_opt**2) / 2) * iv(0, rho_vals * rho_opt)

# FAR threshold for single-detector search (illustrative)
threshold = 5.5  # standard ~3sigma combined

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

# Left: PDF comparison
ax1.semilogy(rho_vals, rayleigh_pdf, 'steelblue', lw=2, label='Noise only (Rayleigh)')
ax1.semilogy(rho_vals, rice_pdf, 'firebrick', lw=2, label=r'Signal ($\rho_\mathrm{opt}=8$, Rice)')
ax1.axvline(threshold, color='k', ls='--', lw=1.5, label=f'Threshold $\\rho_*={threshold}$')
ax1.fill_betweenx([1e-5, 1], threshold, 10, color='firebrick', alpha=0.1, label='Detection region')
ax1.set_xlim(0, 10)
ax1.set_ylim(1e-5, 1)
ax1.set_xlabel(r'$|\rho|$')
ax1.set_ylabel('Probability density')
ax1.set_title('SNR Distributions: Noise vs Signal')
ax1.legend()

# Right: ROC curve
rho_thresh = np.linspace(0, 12, 500)
p_fa = np.exp(-rho_thresh**2 / 2)  # False alarm probability
# Detection probability via Rice survival function
from scipy.stats import rice as rice_dist
p_det = rice_dist.sf(rho_thresh, rho_opt / 1.0, scale=1.0)  # scipy Rice(nu, scale)

ax2.plot(p_fa, p_det, 'darkorchid', lw=2.5)
ax2.plot([0, 1], [0, 1], 'gray', ls='--', lw=1, label='Random classifier')
ax2.set_xlabel('False Alarm Probability')
ax2.set_ylabel('Detection Probability')
ax2.set_title('ROC Curve for Matched Filter')
ax2.set_xlim(0, 0.05)
ax2.set_ylim(0, 1)
ax2.legend(['Matched filter ($\\rho_\\mathrm{opt}=8$)', 'Random'])

plt.tight_layout()
plt.savefig('lesson6_roc.png', dpi=110, bbox_inches='tight')
plt.show()
print("Figure saved: lesson6_roc.png")
"""))

# ── Cell 5: Section 3 ────────────────────────────────────────────────────────
nb["cells"].append(md(
r"""---
## 3. Generating Waveform Templates with PyCBC

PyCBC provides a unified interface to hundreds of gravitational waveform approximants through the `pycbc.waveform` module. For CBC matched filtering the two most commonly used are:

| Approximant | Domain | Physics | Speed |
|---|---|---|---|
| **SEOBNRv4** | Time-domain (TD) | Spinning effective-one-body | Moderate |
| **IMRPhenomD** | Frequency-domain (FD) | Phenomenological inspiral-merger-ringdown | Fast |
| **IMRPhenomXP** | FD | Precessing double-spin | Moderate |
| **SpinTaylorT4** | TD | Post-Newtonian inspiral only | Fast |

### 3.1 Frequency-Domain Templates: `get_fd_waveform`

The frequency-domain interface is most natural for matched filtering because the FFT of the data and the template live in the same space:

```python
hp, hc = get_fd_waveform(
    approximant='IMRPhenomD',
    mass1=36.0,          # Solar masses
    mass2=29.0,          # Solar masses
    spin1z=0.0,          # Dimensionless z-spin on primary
    spin2z=0.0,          # Dimensionless z-spin on secondary
    delta_f=1.0 / T,     # Frequency resolution (Hz) = 1 / segment length
    f_lower=20.0,        # Starting frequency (Hz)
    distance=410.0,      # Luminosity distance (Mpc) — for GW150914
)
```

`hp` and `hc` are the **plus** and **cross** polarisations. For a face-on binary ($\\iota=0$) the strain seen by LIGO is approximately $h = F_+ h_+ + F_\\times h_\\times$ where $F_{+,\\times}$ are the antenna response factors.

### 3.2 Time-Domain Templates: `get_td_waveform`

```python
hp, hc = get_td_waveform(
    approximant='SEOBNRv4',
    mass1=36.0,
    mass2=29.0,
    spin1z=0.32,
    delta_t=1.0 / 4096,  # Sampling interval (s)
    f_lower=20.0,
    distance=410.0,
)
```

The TD waveform is tapered at the start and ends at merger. It must be zero-padded and FFT'd before use in the matched filter.

### 3.3 Parameter Space

The CBC parameter space has 15 dimensions (for quasi-circular precessing binaries):

| Parameter | Symbol | Typical range |
|---|---|---|
| Component masses | $m_1, m_2$ | $1 - 100\;M_\\odot$ |
| Spin magnitudes | $a_1, a_2$ | $0 - 1$ |
| Spin tilt angles | $\\theta_1, \\theta_2$ | $0 - \\pi$ |
| Inclination | $\\iota$ | $0 - \\pi$ |
| Coalescence phase | $\\phi_c$ | $0 - 2\\pi$ |
| Sky location | $\\alpha, \\delta$ | All sky |
| Polarisation | $\\psi$ | $0 - \\pi$ |
| Coalescence time | $t_c$ | Continuous |
| Luminosity distance | $d_L$ | $1 - 10^4$ Mpc |

For the **matched-filter search**, the coalescence time $t_c$, phase $\\phi_c$, and distance $d_L$ are **extrinsic** — they are marginalised analytically. The search templates span the remaining **intrinsic** parameters.
"""))

# ── Cell 6: Waveform generation code ─────────────────────────────────────────
nb["cells"].append(code(
r"""# ── 3.4  Generate and plot waveforms ────────────────────────────────────────
if not HAS_PYCBC:
    print("PyCBC required for this cell.")
else:
    sample_rate = 4096   # Hz
    delta_t = 1.0 / sample_rate
    f_lower = 20.0       # Hz — LIGO low-frequency cutoff

    # --- GW150914-like parameters ---
    m1, m2 = 36.2, 29.1  # Solar masses (best-fit from Abbott+2016)
    s1z, s2z = 0.32, -0.44  # Effective spins

    # Time-domain waveform (SEOBNRv4)
    hp_td, hc_td = get_td_waveform(
        approximant='SEOBNRv4',
        mass1=m1, mass2=m2,
        spin1z=s1z, spin2z=s2z,
        delta_t=delta_t,
        f_lower=f_lower,
        distance=410.0,
    )
    times = hp_td.sample_times.numpy()

    # Frequency-domain waveform (IMRPhenomD) — non-spinning for simplicity
    T_seg = 32.0          # segment duration for frequency resolution
    delta_f = 1.0 / T_seg
    hp_fd, hc_fd = get_fd_waveform(
        approximant='IMRPhenomD',
        mass1=m1, mass2=m2,
        spin1z=0.0, spin2z=0.0,
        delta_f=delta_f,
        f_lower=f_lower,
        distance=410.0,
    )
    freqs = hp_fd.sample_frequencies.numpy()

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Top-left: full TD waveform
    ax = axes[0, 0]
    ax.plot(times - times[-1], hp_td.numpy() * 1e21, color='steelblue', lw=1.0, label=r'$h_+$')
    ax.plot(times - times[-1], hc_td.numpy() * 1e21, color='firebrick', lw=1.0, alpha=0.7, label=r'$h_\times$')
    ax.set_xlabel('Time before merger (s)')
    ax.set_ylabel(r'Strain $\times 10^{21}$')
    ax.set_title('SEOBNRv4 TD Waveform')
    ax.legend()

    # Top-right: zoom on merger
    ax = axes[0, 1]
    mask = (times - times[-1]) >= -0.5
    ax.plot(times[mask] - times[-1], hp_td.numpy()[mask] * 1e21, color='steelblue', lw=1.5, label=r'$h_+$')
    ax.plot(times[mask] - times[-1], hc_td.numpy()[mask] * 1e21, color='firebrick', lw=1.5, alpha=0.8, label=r'$h_\times$')
    ax.set_xlabel('Time before merger (s)')
    ax.set_ylabel(r'Strain $\times 10^{21}$')
    ax.set_title('Zoom: Last 0.5 s (inspiral + merger + ringdown)')
    ax.legend()

    # Bottom-left: FD amplitude
    ax = axes[1, 0]
    mask_f = (freqs >= f_lower) & (freqs <= 1000.0)
    ax.loglog(freqs[mask_f], np.abs(hp_fd.numpy()[mask_f]), color='darkorchid', lw=1.5, label='IMRPhenomD $|\\tilde{h}_+(f)|$')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'$|\tilde{h}(f)|$ (strain/Hz)')
    ax.set_title('FD Waveform Amplitude Spectrum')
    ax.legend()

    # Bottom-right: Instantaneous frequency (from phase derivative)
    ax = axes[1, 1]
    # Compute instantaneous frequency via analytic signal
    from scipy.signal import hilbert
    analytic = hilbert(hp_td.numpy())
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.gradient(phase, delta_t) / (2 * np.pi)
    mask_f2 = (inst_freq > 0) & (inst_freq < 600) & ((times - times[-1]) > -4.0)
    ax.plot(times[mask_f2] - times[-1], inst_freq[mask_f2], color='forestgreen', lw=1.5)
    ax.set_xlabel('Time before merger (s)')
    ax.set_ylabel('Instantaneous frequency (Hz)')
    ax.set_title('Chirp: Frequency Evolution')
    ax.set_ylim(0, 600)

    plt.suptitle(f'GW150914-like Template  ($m_1={m1}$, $m_2={m2}\\,M_\\odot$, SEOBNRv4)', y=1.01, fontsize=13)
    plt.tight_layout()
    plt.savefig('lesson6_waveforms.png', dpi=110, bbox_inches='tight')
    plt.show()
    print("Waveform figure saved.")
"""))

# ── Cell 7: Section 4 ────────────────────────────────────────────────────────
nb["cells"].append(md(
r"""---
## 4. PSD Estimation and Frequency-Domain Whitening

### 4.1 Welch's Method via `pycbc.psd`

A reliable estimate of the noise PSD is essential for the matched filter: if the PSD is over-estimated, the SNR will be deflated; if under-estimated, spurious triggers proliferate.

PyCBC implements **Welch's method** (averaging periodograms of overlapping, windowed segments):

$$
\hat{S}_n(f) = \frac{1}{K} \sum_{k=1}^{K} |\tilde{d}_k(f)|^2
$$

where each segment $\tilde{d}_k$ is windowed with a Hann window and the segments overlap by 50%. The variance of the PSD estimate is $\propto 1/K$, so longer data stretches give better estimates.

**PyCBC usage:**

```python
psd = welch(
    data,               # pycbc.types.TimeSeries
    avg_method='median',# 'mean', 'median', or 'median-mean'
    seg_len=4.0,        # segment length in seconds
    seg_stride=2.0,     # overlap stride in seconds
)
```

The **median** average is preferred because it is more robust to non-Gaussian outliers (short glitches) than the mean.

### 4.2 `interpolate` and `inverse_spectrum_truncation`

After estimation, the PSD must be **resampled** to match the FFT frequency resolution of the data segment, and optionally **inverse-spectrum truncated** to avoid ringing artefacts from sharp PSD features:

```python
# Interpolate PSD to the same delta_f as the data
psd_interp = interpolate(psd, delta_f=1.0/T)

# Inverse spectrum truncation: limit the time-domain noise filter to max_len samples
psd_trunc = inverse_spectrum_truncation(
    psd_interp,
    max_filter_len=int(4.0 * sample_rate),  # 4 s
    low_frequency_cutoff=f_lower,
    trunc_method='hann',
)
```

The truncation sets the time-domain equivalent of the noise filter to zero outside a finite window, preventing long-range correlations that would smear out genuine triggers.

### 4.3 Whitening

Whitening is the process of dividing the Fourier-domain data by the amplitude spectral density $\sqrt{S_n(f)}$:

$$
\tilde{d}_\text{white}(f) = \frac{\tilde{d}(f)}{\sqrt{S_n(f)}}
$$

After whitening, all frequency bins have equal variance $= 1/2$, and the noise looks like band-limited white noise in the time domain. This makes visual inspection of transients much easier.
"""))

# ── Cell 8: PSD estimation code ──────────────────────────────────────────────
nb["cells"].append(code(
r"""# ── 4. PSD estimation demo using simulated colored noise ────────────────────
if not HAS_PYCBC:
    print("PyCBC required for this cell.")
else:
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.noise import noise_from_psd

    sample_rate = 4096
    seg_duration = 128   # seconds of simulated noise
    delta_f_true = 1.0 / seg_duration
    f_lower = 20.0

    # Generate a design-sensitivity PSD
    flen = seg_duration * sample_rate // 2 + 1
    psd_design = aLIGOZeroDetHighPower(flen, delta_f_true, f_lower)

    # Simulate noise from that PSD
    np.random.seed(42)
    noise_ts = noise_from_psd(
        length=seg_duration * sample_rate,
        delta_t=1.0 / sample_rate,
        psd=psd_design,
        seed=42,
    )

    # Estimate PSD with Welch
    psd_estimated = welch(
        noise_ts,
        avg_method='median',
        seg_len=4 * sample_rate,       # 4-s segments
        seg_stride=2 * sample_rate,    # 2-s stride (50 % overlap)
    )

    freqs = psd_design.sample_frequencies.numpy()
    mask = (freqs >= f_lower) & (freqs <= 2000.0)

    # ── Plot: True vs estimated PSD ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.loglog(freqs[mask], np.sqrt(psd_design.numpy()[mask]), color='steelblue',
              lw=2.0, label='aLIGO design ASD (true)')
    ax.loglog(freqs[mask], np.sqrt(psd_estimated.numpy()[mask]), color='firebrick',
              lw=1.5, alpha=0.85, label='Welch estimate (128 s)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'ASD $[\mathrm{Hz}^{-1/2}]$')
    ax.set_title('PSD Estimation: Welch vs Design')
    ax.legend()

    # ── Plot: Whitened noise ───────────────────────────────────────────────────
    ax = axes[1]
    # Whiten: divide FD noise by ASD, convert back to time domain
    noise_fd = noise_ts.to_frequencyseries()
    freqs_fd = noise_fd.sample_frequencies.numpy()
    psd_interp = interpolate(psd_estimated, noise_fd.delta_f)
    asd_interp = FrequencySeries(np.sqrt(psd_interp.numpy()), delta_f=psd_interp.delta_f)

    white_fd_data = noise_fd.numpy().copy()
    nonzero = asd_interp.numpy() > 0
    white_fd_data[nonzero] /= asd_interp.numpy()[nonzero]
    # Zero below f_lower
    white_fd_data[freqs_fd < f_lower] = 0.0

    white_ts = FrequencySeries(white_fd_data, delta_f=noise_fd.delta_f).to_timeseries()
    t_white = white_ts.sample_times.numpy()
    # Show 1 s in the middle
    mid = len(t_white) // 2
    half = sample_rate // 2
    ax.plot(t_white[mid-half:mid+half] - t_white[mid], white_ts.numpy()[mid-half:mid+half],
            color='darkorchid', lw=0.5, alpha=0.9)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Whitened strain (normalised)')
    ax.set_title('Whitened Simulated Noise (1 s window)')

    plt.tight_layout()
    plt.savefig('lesson6_psd.png', dpi=110, bbox_inches='tight')
    plt.show()
    print("PSD figure saved.")
"""))

# ── Cell 9: Section 5 ────────────────────────────────────────────────────────
nb["cells"].append(md(
r"""---
## 5. Computing the Complex SNR Time Series

### 5.1 `pycbc.filter.matched_filter`

The workhorse of the detection pipeline is `pycbc.filter.matched_filter`. Under the hood it computes:

$$
z(t) = 4 \int_0^{f_\text{Nyq}} \frac{\tilde{s}(f)\,\tilde{h}^*(f)}{S_n(f)}\, e^{2\pi i f t}\, df
$$

then divides by $\sigma_h$ to normalise. The function returns a complex `TimeSeries` $z(t)$:

```python
snr = matched_filter(
    template,          # pycbc.types.FrequencySeries
    data,              # pycbc.types.TimeSeries
    psd=psd,           # noise PSD (optional but recommended)
    low_frequency_cutoff=f_lower,
    high_frequency_cutoff=f_upper,
)
```

### 5.2 Interpreting Magnitude |ρ(t)| and Phase

The matched-filter output is complex:

$$
z(t) = |\rho(t)|\, e^{i\phi(t)}
$$

- $|\rho(t)|$ — the **SNR magnitude**: peaks near the coalescence time when the template matches the signal.
- $\phi(t)$ — the **phase**: related to the orbital phase at coalescence; in a two-phase filter bank, this is maximised over the coalescence phase analytically.
- The real part $\text{Re}[z(t)]$ is sometimes called the "in-phase" component and $\text{Im}[z(t)]$ the "quadrature" component.

### 5.3 Finding Trigger Times and Peaks

A **trigger** is a time $t_*$ where $|\rho(t_*)| > \rho_\text{thresh}$ (typically $\rho_\text{thresh} = 5.5$ for a single detector). PyCBC provides utilities for peak-finding and clustering:

```python
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.events import threshold_and_cluster

# Find peaks above threshold, clustered within 0.2 s
peak_idx, peak_snr = threshold_and_cluster(abs(snr), 5.5, 0.2 * snr.sample_rate)
peak_time = snr.start_time + peak_idx / snr.sample_rate
```

In practice, the search also records the peak phase $\phi$ (for coincidence checks) and the matched-filter trigger time with sub-sample precision.
"""))

# ── Cell 10: SNR on simulated signal ─────────────────────────────────────────
nb["cells"].append(code(
r"""# ── 5. Matched filter demo: inject synthetic signal into noise ───────────────
if not HAS_PYCBC:
    print("PyCBC required for this cell.")
else:
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.noise import noise_from_psd
    from pycbc.filter import matched_filter, sigma

    # Parameters
    sample_rate = 4096
    seg_duration = 64     # seconds
    f_lower = 20.0
    injection_snr = 20.0  # target optimal SNR

    # Design PSD
    flen = seg_duration * sample_rate // 2 + 1
    delta_f_seg = 1.0 / seg_duration
    psd = aLIGOZeroDetHighPower(flen, delta_f_seg, f_lower)

    # Simulate noise
    np.random.seed(1234)
    noise_ts = noise_from_psd(
        length=seg_duration * sample_rate,
        delta_t=1.0 / sample_rate,
        psd=psd,
        seed=1234,
    )

    # Generate template (FD, IMRPhenomD)
    hp_tmpl, _ = get_fd_waveform(
        approximant='IMRPhenomD',
        mass1=36.0, mass2=29.0,
        spin1z=0.0, spin2z=0.0,
        delta_f=delta_f_seg,
        f_lower=f_lower,
        distance=1.0,   # arbitrary distance — we will rescale
    )
    # Pad to correct length
    hp_tmpl.resize(flen)

    # Rescale template to achieve desired optimal SNR
    sig = sigma(hp_tmpl, psd=psd, low_frequency_cutoff=f_lower)
    scale = injection_snr / sig
    hp_inj = hp_tmpl * scale

    # Inject signal at t = 32 s (centre of segment)
    noise_fd = noise_ts.to_frequencyseries()
    data_fd_arr = noise_fd.numpy().copy()
    data_fd_arr[:len(hp_inj)] += hp_inj.numpy()
    from pycbc.types import FrequencySeries as FS
    data_fd = FS(data_fd_arr, delta_f=noise_fd.delta_f, epoch=noise_fd.epoch)
    data_ts = data_fd.to_timeseries()

    # Interpolate and truncate PSD
    psd_interp = interpolate(psd, data_fd.delta_f)
    psd_trunc = inverse_spectrum_truncation(
        psd_interp,
        max_filter_len=int(4.0 * sample_rate),
        low_frequency_cutoff=f_lower,
        trunc_method='hann',
    )

    # Matched filter
    snr_ts = matched_filter(
        hp_tmpl,
        data_ts,
        psd=psd_trunc,
        low_frequency_cutoff=f_lower,
    )

    # Extract SNR time series, trimming filter artefacts
    pad = 4   # seconds of pad to trim
    snr_crop = snr_ts[pad * sample_rate : (seg_duration - pad) * sample_rate]
    t_snr = snr_crop.sample_times.numpy()
    abs_snr = abs(snr_crop).numpy()

    # Find peak
    peak_idx_val = np.argmax(abs_snr)
    peak_time_val = t_snr[peak_idx_val]
    peak_snr_val  = abs_snr[peak_idx_val]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

    # Top: SNR time series
    ax = axes[0]
    ax.plot(t_snr - float(noise_ts.start_time), abs_snr,
            color='steelblue', lw=1.2, label=r'$|\rho(t)|$')
    ax.axhline(5.5, color='firebrick', ls='--', lw=1.5, label=r'$\rho_*=5.5$')
    ax.axvline(peak_time_val - float(noise_ts.start_time), color='darkorange',
               ls='--', lw=1.5, label=f'Peak: $|\\rho|={peak_snr_val:.1f}$ at $t={peak_time_val - float(noise_ts.start_time):.2f}$ s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$|\rho(t)|$')
    ax.set_title(f'Matched Filter SNR (injected $\\rho_\\mathrm{{opt}}={injection_snr}$)')
    ax.legend(fontsize=10)

    # Bottom: zoom around peak
    ax = axes[1]
    t_ref = peak_time_val - float(noise_ts.start_time)
    zoom_mask = np.abs(t_snr - float(noise_ts.start_time) - t_ref) < 0.5
    ax.plot(t_snr[zoom_mask] - float(noise_ts.start_time) - t_ref,
            abs_snr[zoom_mask], color='steelblue', lw=1.5)
    ax.plot(t_snr[zoom_mask] - float(noise_ts.start_time) - t_ref,
            np.real(snr_crop.numpy()[zoom_mask]), color='firebrick', lw=1.0, alpha=0.7, label=r'Re[$\rho$]')
    ax.plot(t_snr[zoom_mask] - float(noise_ts.start_time) - t_ref,
            np.imag(snr_crop.numpy()[zoom_mask]), color='forestgreen', lw=1.0, alpha=0.7, label=r'Im[$\rho$]')
    ax.axhline(5.5, color='gray', ls='--', lw=1)
    ax.set_xlabel('Time relative to peak (s)')
    ax.set_ylabel(r'$|\rho|$, Re[$\rho$], Im[$\rho$]')
    ax.set_title('Zoom: Complex SNR Around Merger')
    ax.legend()

    plt.tight_layout()
    plt.savefig('lesson6_snr.png', dpi=110, bbox_inches='tight')
    plt.show()
    print(f"Peak SNR: {peak_snr_val:.2f} at t = {t_ref:.3f} s")
"""))

# ── Cell 11: Section 6 ────────────────────────────────────────────────────────
nb["cells"].append(md(
r"""---
## 6. The χ² (Chi-Squared) Consistency Test

Even with a high SNR, a genuine gravitational-wave trigger must survive a battery of **signal-consistency tests**. The most widely used is the $\chi^2$ test introduced by Allen (2005).

### 6.1 Theory: Sub-templates and Expected Power

The key insight is that a true CBC signal, when filtered with the matched template, deposits power **uniformly** across frequency bins. A short-duration glitch, by contrast, will deposit power in a narrow frequency range or a short time window.

The $p$-bin **power $\chi^2$ statistic** works as follows:

1. Divide the template frequency band $[f_\text{low}, f_\text{high}]$ into $p$ equal-contribution sub-bands, where "equal contribution" means each sub-band contributes $1/p$ of the total $\sigma_h^2$.

2. For each sub-band $k$, compute the partial matched-filter SNR $z_k(t)$.

3. Compare the observed partial SNRs to the expected value:

$$
\chi^2 \;=\; p \sum_{k=1}^{p} \left|z_k - \frac{z}{p}\right|^2
$$

Under the signal hypothesis, $\chi^2 / p \approx 1$ at the peak. For a glitch with concentrated power, $\chi^2 / p \gg 1$.

The degrees of freedom are $\nu = 2(p-1)$ (since the total SNR $z$ is subtracted). Under noise-only, $\chi^2 / \nu \sim 1$; under a true signal, $\chi^2 / \nu \lesssim 1.5$ near the peak.

### 6.2 `pycbc.vetoes.power_chisq`

```python
from pycbc.vetoes import power_chisq, power_chisq_bins

# Compute equal-power frequency bins
bins = power_chisq_bins(template, num_bins=16, psd=psd, low_frequency_cutoff=f_lower)

# Compute chi-squared time series
chisq, chisq_dof = power_chisq(
    template, data, num_bins=16, psd=psd,
    low_frequency_cutoff=f_lower,
    return_dof=True
)
```

### 6.3 Re-weighted SNR: $\hat{\\rho}$

To combine the SNR and $\chi^2$ into a single detection statistic, the **re-weighted SNR** is defined as:

$$
\boxed{
\hat{\rho} \;=\; \begin{cases}
\rho & \text{if } \chi^2_r \leq 1 \\[4pt]
\dfrac{\rho}{\left[\dfrac{1}{2}\left(1 + (\chi^2_r)^3\right)\right]^{1/6}} & \text{if } \chi^2_r > 1
\end{cases}
}
$$

where $\chi^2_r = \chi^2 / (2p - 2)$ is the reduced $\chi^2$. This re-weighting penalises high-SNR triggers with inconsistent time–frequency content while preserving the SNR for genuine signals ($\chi^2_r \approx 1$).

The re-weighted SNR $\hat{\rho}$ is the primary detection ranking statistic used in the PyCBC offline search pipeline (Usman et al. 2016; Nitz et al. 2017).

**Physical interpretation of the exponents**: The exponent $1/6$ is chosen so that $\hat{\rho} \propto \rho^{-1/2}$ for very loud glitches ($\chi^2_r \gg 1$, $\rho \sim \chi^2_r^{1/2}$), which suppresses them in the background distribution.
"""))

# ── Cell 12: Chi-squared demo ─────────────────────────────────────────────────
nb["cells"].append(code(
r"""# ── 6. Chi-squared veto: illustration ────────────────────────────────────────
if not HAS_PYCBC:
    print("PyCBC required for this cell.")
else:
    from pycbc.vetoes import power_chisq

    # Reuse psd, hp_tmpl, data_ts from Section 5 demo
    # If those variables exist, run the chi-squared test
    try:
        num_bins = 16

        # Compute chi-squared time series
        chisq_ts = power_chisq(
            hp_tmpl,
            data_ts,
            num_bins,
            psd=psd_trunc,
            low_frequency_cutoff=f_lower,
        )
        chisq_dof = 2 * (num_bins - 1)

        # Crop to same window as SNR
        chisq_crop = chisq_ts[pad * sample_rate : (seg_duration - pad) * sample_rate]
        chisq_r = chisq_crop.numpy() / chisq_dof  # reduced chi-squared

        # Re-weighted SNR
        def reweighted_snr(snr_abs, chi2_r):
            rw = snr_abs.copy()
            mask = chi2_r > 1.0
            rw[mask] = snr_abs[mask] / ((0.5 * (1 + chi2_r[mask]**3)) ** (1.0/6.0))
            return rw

        rw_snr = reweighted_snr(abs_snr, chisq_r)

        # ── Plot ────────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

        ax = axes[0]
        ax.plot(t_snr - float(noise_ts.start_time), abs_snr, color='steelblue', lw=1.2, label=r'$|\rho(t)|$')
        ax.axhline(5.5, color='firebrick', ls='--', lw=1.2, label=r'$\rho_*=5.5$')
        ax.set_ylabel(r'$|\rho(t)|$')
        ax.set_title('SNR, $\\chi^2$, and Re-weighted SNR Time Series')
        ax.legend()

        ax = axes[1]
        ax.plot(t_snr - float(noise_ts.start_time), chisq_r, color='darkorange', lw=1.0, label=r'$\chi^2_r = \chi^2 / \nu$')
        ax.axhline(1.0, color='gray', ls='--', lw=1.2)
        ax.set_ylabel(r'Reduced $\chi^2_r$')
        ax.set_ylim(0, 5)
        ax.legend()

        ax = axes[2]
        ax.plot(t_snr - float(noise_ts.start_time), rw_snr, color='forestgreen', lw=1.2, label=r'$\hat{\rho}(t)$ (re-weighted SNR)')
        ax.plot(t_snr - float(noise_ts.start_time), abs_snr, color='steelblue', lw=0.8, alpha=0.5, label=r'$|\rho(t)|$ (unweighted)')
        ax.axhline(5.5, color='firebrick', ls='--', lw=1.2)
        ax.set_ylabel(r'$\hat{\rho}(t)$')
        ax.set_xlabel('Time (s)')
        ax.legend()

        plt.tight_layout()
        plt.savefig('lesson6_chisq.png', dpi=110, bbox_inches='tight')
        plt.show()

        # Print chi-squared at peak
        peak_idx2 = np.argmax(abs_snr)
        print(f"At SNR peak: |rho| = {abs_snr[peak_idx2]:.2f},  chi2_r = {chisq_r[peak_idx2]:.3f},  rw_SNR = {rw_snr[peak_idx2]:.2f}")
    except NameError:
        print("Please run Section 5 demo first to define psd_trunc, hp_tmpl, data_ts, etc.")
"""))

# ── Cell 13: Section 7 ────────────────────────────────────────────────────────
nb["cells"].append(md(
r"""---
## 7. GW150914 Detection Pipeline — Step by Step

We now build a complete matched-filter pipeline for **GW150914**, the first confirmed gravitational-wave detection (Abbott et al. 2016, PRL 116, 061102).

### Published reference values

| Quantity | H1 | L1 |
|---|---|---|
| GPS trigger time (s) | 1126259462.44 | 1126259462.44 |
| Network SNR | \multicolumn{2}{c}{$23.7$} |
| Matched-filter SNR (H1) | ~18 | — |
| Matched-filter SNR (L1) | — | ~13 |
| Best-fit masses | $m_1 = 36.2\,M_\odot$, $m_2 = 29.1\,M_\odot$ | — |
| Template approximant | SEOBNRv4\_ROM | — |

**GPS event time**: 1126259462.4 (UTC 2015-09-14 09:50:45)

### 7.1 Downloading H1 and L1 Strain with `pycbc.catalog`

`pycbc.catalog.Merger` provides the most convenient access to GWOSC open data:

```python
from pycbc.catalog import Merger

merger = Merger('GW150914')
strain_H1 = merger.strain('H1')  # pycbc.types.TimeSeries, 32-s segment
strain_L1 = merger.strain('L1')
```

This automatically downloads the 4096 Hz strain data for a $\\pm 32$-second window around the event.
"""))

# ── Cell 14: Download real data ───────────────────────────────────────────────
nb["cells"].append(code(
r"""# ── 7.1  Download GW150914 open data ─────────────────────────────────────────
if not HAS_PYCBC:
    print("PyCBC required for this cell.")
else:
    from pycbc.catalog import Merger

    print("Downloading GW150914 strain data from GWOSC ...")
    try:
        merger = Merger('GW150914')
        # Request 256 seconds of data centred on the event
        strain_H1 = merger.strain('H1', duration=256)
        strain_L1 = merger.strain('L1', duration=256)
        print(f"  H1: {len(strain_H1)} samples, sample rate = {int(1/strain_H1.delta_t)} Hz")
        print(f"  L1: {len(strain_L1)} samples, sample rate = {int(1/strain_L1.delta_t)} Hz")
        print(f"  Segment start GPS: {float(strain_H1.start_time):.3f}")
        DATA_AVAILABLE = True
    except Exception as e:
        print(f"Data download failed: {e}")
        print("Proceeding with synthetic data for the remaining cells.")
        DATA_AVAILABLE = False
"""))

# ── Cell 15: Condition data ────────────────────────────────────────────────────
nb["cells"].append(code(
r"""# ── 7.2  Condition the data ────────────────────────────────────────────────────
if not HAS_PYCBC:
    print("PyCBC required.")
elif not DATA_AVAILABLE:
    print("Real data not available — using synthetic data from Section 5.")
else:
    from pycbc.filter import resample_to_delta_t, highpass
    from pycbc.types import TimeSeries as PyCBCTS

    target_sample_rate = 4096  # Hz

    def condition_strain(strain, target_sr=4096, f_highpass=15.0):
        '''Resample, high-pass filter, and apply a Tukey window.'''
        # Resample to target sample rate
        if int(round(1.0 / strain.delta_t)) != target_sr:
            strain = resample_to_delta_t(strain, 1.0 / target_sr)

        # High-pass filter at 15 Hz (removes slow drifts)
        strain = highpass(strain, frequency=f_highpass)

        # Tukey window to taper edges (alpha=0.5 by default)
        from pycbc.window import tukey_window
        N = len(strain)
        win = tukey_window(N, alpha=0.5)
        strain = PyCBCTS(strain.numpy() * win,
                         delta_t=strain.delta_t,
                         epoch=strain.start_time)
        return strain

    # Condition both detectors
    strain_H1_cond = condition_strain(strain_H1)
    strain_L1_cond = condition_strain(strain_L1)

    print(f"Conditioned H1: {len(strain_H1_cond)} samples @ {target_sample_rate} Hz")
    print(f"Conditioned L1: {len(strain_L1_cond)} samples @ {target_sample_rate} Hz")

    # Quick visual check: whitened strain around the event
    t_event_gps = 1126259462.4
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for ax, strain, det, col in zip(axes,
                                     [strain_H1_cond, strain_L1_cond],
                                     ['H1', 'L1'],
                                     ['steelblue', 'firebrick']):
        t = strain.sample_times.numpy()
        mask = np.abs(t - t_event_gps) < 0.5
        ax.plot(t[mask] - t_event_gps, strain.numpy()[mask] * 1e21,
                color=col, lw=0.8, label=f'{det} raw (conditioned)')
        ax.set_ylabel(r'Strain $\times 10^{21}$')
        ax.set_title(f'{det} Strain around GW150914')
        ax.legend()
    axes[-1].set_xlabel('Time relative to event (s)')
    plt.tight_layout()
    plt.savefig('lesson6_gw150914_strain.png', dpi=110, bbox_inches='tight')
    plt.show()
"""))

# ── Cell 16: PSD estimation on off-source ─────────────────────────────────────
nb["cells"].append(code(
r"""# ── 7.3  Estimate PSD on off-source (off-event) data ─────────────────────────
if not HAS_PYCBC:
    print("PyCBC required.")
elif not DATA_AVAILABLE:
    print("Using design PSD as substitute.")
    psd_H1 = aLIGOZeroDetHighPower(
        int(32 * target_sample_rate) // 2 + 1, 1.0 / 32, 20.0
    )
    psd_L1 = psd_H1
else:
    from pycbc.psd import welch, interpolate, inverse_spectrum_truncation

    # Use the first 192 s of data for PSD estimation (off-source)
    t_event_gps = 1126259462.4
    psd_seg_len = 4  # seconds per Welch segment
    sr = target_sample_rate

    def estimate_psd_offsource(strain, t_event, seg_len=4):
        '''Estimate PSD using data well away from the event.'''
        # Take first ~3/4 of the data (off-source)
        n_off = int(len(strain) * 0.75)
        off_data = strain[:n_off]
        psd = welch(
            off_data,
            avg_method='median',
            seg_len=seg_len * sr,
            seg_stride=2 * sr,
        )
        return psd

    psd_H1_raw = estimate_psd_offsource(strain_H1_cond, t_event_gps)
    psd_L1_raw = estimate_psd_offsource(strain_L1_cond, t_event_gps)

    # Choose a 32-s analysis segment centred on the event
    delta_f_seg = 1.0 / 32.0
    flen_seg = int(32 * sr) // 2 + 1
    psd_H1 = interpolate(psd_H1_raw, delta_f_seg)
    psd_L1 = interpolate(psd_L1_raw, delta_f_seg)
    psd_H1.resize(flen_seg)
    psd_L1.resize(flen_seg)
    psd_H1 = inverse_spectrum_truncation(psd_H1, int(4 * sr),
                                          low_frequency_cutoff=20.0,
                                          trunc_method='hann')
    psd_L1 = inverse_spectrum_truncation(psd_L1, int(4 * sr),
                                          low_frequency_cutoff=20.0,
                                          trunc_method='hann')

    # Plot the ASDs
    freqs = psd_H1.sample_frequencies.numpy()
    mask = (freqs >= 20) & (freqs <= 2000)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(freqs[mask], np.sqrt(psd_H1.numpy()[mask]),
              color='steelblue', lw=1.5, label='H1 ASD')
    ax.loglog(freqs[mask], np.sqrt(psd_L1.numpy()[mask]),
              color='firebrick', lw=1.5, alpha=0.8, label='L1 ASD')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'ASD $[\mathrm{Hz}^{-1/2}]$')
    ax.set_title('GW150914: Estimated Detector ASD')
    ax.legend()
    plt.tight_layout()
    plt.savefig('lesson6_gw150914_psd.png', dpi=110, bbox_inches='tight')
    plt.show()
    print("PSD estimation complete.")
"""))

# ── Cell 17: Template + matched filter ────────────────────────────────────────
nb["cells"].append(code(
r"""# ── 7.4 + 7.5  Build template and run matched filter ─────────────────────────
if not HAS_PYCBC:
    print("PyCBC required.")
else:
    from pycbc.waveform import get_fd_waveform
    from pycbc.filter import matched_filter, sigma
    from pycbc.types import FrequencySeries

    f_lower = 20.0
    sr = 4096

    # GW150914 best-fit template parameters
    m1_src, m2_src = 36.2, 29.1  # source-frame solar masses
    s1z_src, s2z_src = 0.32, -0.44

    # Build FD template at the segment's frequency resolution
    seg_dur = 32.0
    delta_f_tmpl = 1.0 / seg_dur
    flen_tmpl = int(seg_dur * sr) // 2 + 1

    hp_H1, _ = get_fd_waveform(
        approximant='IMRPhenomD',
        mass1=m1_src, mass2=m2_src,
        spin1z=0.0, spin2z=0.0,      # aligned-spin simplified template
        delta_f=delta_f_tmpl,
        f_lower=f_lower,
        distance=410.0,
    )
    hp_H1.resize(flen_tmpl)
    hp_L1 = hp_H1.copy()  # same template for both detectors

    print(f"Template length: {len(hp_H1)} freq bins (delta_f = {delta_f_tmpl:.4f} Hz)")

    if DATA_AVAILABLE:
        # Extract 32-s analysis segments centred at event
        t_event_gps = 1126259462.4
        def get_analysis_segment(strain, t_event, seg_dur=32.0):
            '''Cut a segment of length seg_dur centred on t_event.'''
            t = strain.sample_times.numpy()
            t_start = t_event - seg_dur / 2
            t_end   = t_event + seg_dur / 2
            idx_start = np.searchsorted(t, t_start)
            idx_end   = np.searchsorted(t, t_end)
            seg = strain[idx_start:idx_end]
            return seg

        seg_H1 = get_analysis_segment(strain_H1_cond, t_event_gps)
        seg_L1 = get_analysis_segment(strain_L1_cond, t_event_gps)

        # Pad / crop to exact segment length
        N_seg = int(seg_dur * sr)
        if len(seg_H1) > N_seg:
            seg_H1 = seg_H1[:N_seg]
        if len(seg_L1) > N_seg:
            seg_L1 = seg_L1[:N_seg]

        # Run matched filter
        snr_H1 = matched_filter(hp_H1, seg_H1, psd=psd_H1,
                                 low_frequency_cutoff=f_lower)
        snr_L1 = matched_filter(hp_L1, seg_L1, psd=psd_L1,
                                 low_frequency_cutoff=f_lower)

        # Trim filter artefacts (4 s each side)
        trim = int(4 * sr)
        snr_H1_crop = snr_H1[trim:-trim]
        snr_L1_crop = snr_L1[trim:-trim]

        abs_H1 = abs(snr_H1_crop).numpy()
        abs_L1 = abs(snr_L1_crop).numpy()
        t_H1   = snr_H1_crop.sample_times.numpy()
        t_L1   = snr_L1_crop.sample_times.numpy()

        peak_H1 = np.argmax(abs_H1)
        peak_L1 = np.argmax(abs_L1)

        print(f"\n=== GW150914 Matched Filter Results ===")
        print(f"H1 peak SNR: {abs_H1[peak_H1]:.2f}  at GPS {t_H1[peak_H1]:.4f}")
        print(f"L1 peak SNR: {abs_L1[peak_L1]:.2f}  at GPS {t_L1[peak_L1]:.4f}")
        print(f"H1-L1 time offset: {(t_H1[peak_H1] - t_L1[peak_L1])*1000:.2f} ms  (light-travel limit: 10 ms)")
    else:
        print("Skipping matched filter (data unavailable).")
"""))

# ── Cell 18: Plot SNR + chi-sq for GW150914 ────────────────────────────────────
nb["cells"].append(code(
r"""# ── 7.6 + 7.7  Chi-squared veto and coincidence check ──────────────────────
if not HAS_PYCBC:
    print("PyCBC required.")
elif not DATA_AVAILABLE:
    print("Data unavailable.")
else:
    from pycbc.vetoes import power_chisq

    num_bins = 16
    chisq_H1 = power_chisq(hp_H1, seg_H1, num_bins,
                             psd=psd_H1, low_frequency_cutoff=f_lower)
    chisq_L1 = power_chisq(hp_L1, seg_L1, num_bins,
                             psd=psd_L1, low_frequency_cutoff=f_lower)

    chisq_dof = 2 * (num_bins - 1)

    chisq_H1_crop = chisq_H1[trim:-trim].numpy() / chisq_dof
    chisq_L1_crop = chisq_L1[trim:-trim].numpy() / chisq_dof

    # Re-weighted SNR
    def reweighted_snr(snr_abs, chi2_r):
        rw = snr_abs.copy()
        mask = chi2_r > 1.0
        rw[mask] = snr_abs[mask] / ((0.5 * (1.0 + chi2_r[mask]**3))**(1.0/6.0))
        return rw

    rw_H1 = reweighted_snr(abs_H1, chisq_H1_crop)
    rw_L1 = reweighted_snr(abs_L1, chisq_L1_crop)

    # ── Figure: 3-row plot for each detector ────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex='col')
    dets  = ['H1', 'L1']
    ts    = [t_H1, t_L1]
    snrs  = [abs_H1, abs_L1]
    chi2s = [chisq_H1_crop, chisq_L1_crop]
    rwsnr = [rw_H1, rw_L1]
    cols  = ['steelblue', 'firebrick']
    peaks = [peak_H1, peak_L1]

    for col_idx, (det, t, s, c2, rw, color, pk) in enumerate(
            zip(dets, ts, snrs, chi2s, rwsnr, cols, peaks)):
        t_ref = t_event_gps

        axes[0, col_idx].plot(t - t_ref, s, color=color, lw=1.0)
        axes[0, col_idx].axhline(5.5, color='gray', ls='--', lw=1)
        axes[0, col_idx].axvline(t[pk] - t_ref, color='k', ls='--', lw=1.2,
                                  label=f'Peak $|\\rho|={s[pk]:.1f}$')
        axes[0, col_idx].set_ylabel(r'$|\rho(t)|$')
        axes[0, col_idx].set_title(f'{det} Matched Filter SNR')
        axes[0, col_idx].legend(fontsize=9)

        axes[1, col_idx].plot(t - t_ref, c2, color=color, lw=0.8, alpha=0.8)
        axes[1, col_idx].axhline(1.0, color='gray', ls='--', lw=1)
        axes[1, col_idx].axvline(t[pk] - t_ref, color='k', ls='--', lw=1.2)
        axes[1, col_idx].set_ylabel(r'Reduced $\chi^2_r$')
        axes[1, col_idx].set_ylim(0, 5)
        axes[1, col_idx].set_title(f'{det} $\\chi^2$ Veto')

        axes[2, col_idx].plot(t - t_ref, rw, color=color, lw=1.0, label=r'$\hat{\rho}$')
        axes[2, col_idx].plot(t - t_ref, s, color='gray', lw=0.7, alpha=0.6, label=r'$|\rho|$')
        axes[2, col_idx].axhline(5.5, color='gray', ls='--', lw=1)
        axes[2, col_idx].axvline(t[pk] - t_ref, color='k', ls='--', lw=1.2,
                                  label=f'$\\hat{{\\rho}}={rw[pk]:.1f}$')
        axes[2, col_idx].set_ylabel(r'$\hat{\rho}(t)$')
        axes[2, col_idx].set_xlabel('Time relative to GW150914 (s)')
        axes[2, col_idx].set_title(f'{det} Re-weighted SNR')
        axes[2, col_idx].legend(fontsize=9)

    plt.suptitle('GW150914 Matched Filter Pipeline Results', y=1.01, fontsize=14)
    plt.tight_layout()
    plt.savefig('lesson6_gw150914_results.png', dpi=110, bbox_inches='tight')
    plt.show()

    print("\n=== Re-weighted SNR at peak ===")
    print(f"H1: {rw_H1[peak_H1]:.2f}   L1: {rw_L1[peak_L1]:.2f}")
    dt_HL = (t_H1[peak_H1] - t_L1[peak_L1]) * 1000  # ms
    print(f"H1-L1 coincidence offset: {dt_HL:.2f} ms  (published: ~7 ms)")
"""))

# ── Cell 19: Section 7.8 comparison ──────────────────────────────────────────
nb["cells"].append(md(
r"""### 7.8 Comparison with Published Values

| Quantity | Published (Abbott+2016) | Our pipeline |
|---|---|---|
| Event GPS time | 1126259462.4 | *see output above* |
| H1 SNR | $\\sim 18$ | *see output above* |
| L1 SNR | $\\sim 13$ | *see output above* |
| H1–L1 time offset | $\\sim 7$ ms | *see output above* |
| Best-fit $m_1$ | $36.2^{+5.2}_{-3.8}\\,M_\\odot$ | $36.2\\,M_\\odot$ (fixed template) |
| Best-fit $m_2$ | $29.1^{+3.7}_{-4.4}\\,M_\\odot$ | $29.1\\,M_\\odot$ (fixed template) |

The simplified pipeline above uses a **single template** with the published best-fit masses and no spin, so the recovered SNR is slightly lower than the optimal (full template bank) result. Real searches use $\\mathcal{O}(10^5)$ templates to maximally cover the parameter space.

**Key observations:**
1. Both detectors independently show an SNR peak above threshold at consistent GPS times.
2. The $\\chi^2$ test returns $\\chi^2_r \\approx 1$ at the peak — consistent with a genuine signal.
3. The H1–L1 time offset ($<10$ ms) is consistent with a source on the sky (maximum light-travel time between the LIGO sites is 10 ms).

These three criteria — SNR threshold, $\\chi^2$ consistency, and network coincidence — constitute the basic detection veto chain of the PyCBC pipeline.
"""))

# ── Cell 20: Section 8 ────────────────────────────────────────────────────────
nb["cells"].append(md(
r"""---
## 8. Additional Topics: Template Banks, Coincidence, and FAR

### 8.1 Template Bank Placement

A real CBC search requires a **template bank** covering the entire parameter space of interest. The key design criterion is the **minimal match** $\mathcal{M}$: any signal in the parameter space must be recovered with $\langle h_\text{signal} | h_\text{template} \rangle \geq \mathcal{M} \|h_\text{signal}\|$ by at least one bank template. Typically $\mathcal{M} = 0.97$ (3% maximum SNR loss).

The number of templates is governed by the metric on the parameter space:

$$
g_{ij}(\lambda) = \left\langle \frac{\partial h}{\partial \lambda^i} \,\middle|\, \frac{\partial h}{\partial \lambda^j} \right\rangle - \left\langle \frac{\partial h}{\partial \lambda^i} \,\middle|\, h \right\rangle \left\langle h \,\middle|\, \frac{\partial h}{\partial \lambda^j} \right\rangle
$$

The volume element in this metric gives the optimal template placement density. For the non-spinning case (dominated by chirp mass $\\mathcal{M}_c$), the bank grows as:

$$
N_\text{bank} \;\sim\; \int \sqrt{g}\, d^n\lambda \;\sim\; \mathcal{O}(10^4\text{–}10^5)
$$

PyCBC provides tools for bank generation: `pycbc.tmpltbank`, and the `pycbc_geom_nonspinbank` executable.

### 8.2 Coincidence Test

A genuine gravitational wave should produce triggers in **multiple detectors** within the light-travel time $\\Delta t_{\\text{max}} \\leq d/c$ where $d$ is the baseline between detectors. For LIGO H1–L1, $\\Delta t_{\\text{max}} = 10$ ms.

The PyCBC coincidence algorithm (Usman et al. 2016) additionally requires the triggers to be:
1. Within $\\Delta t = 15$ ms (including timing uncertainty)
2. Consistent in template parameters (same chirp mass, mass ratio, spins)
3. Produced by the same template (or nearby templates)

### 8.3 False Alarm Rate (FAR) Estimation

The significance of a candidate event is quantified by its **false alarm rate** (FAR): the rate at which noise-only background would produce an event with equal or higher detection statistic.

Because the noise is non-Gaussian, the FAR cannot be estimated analytically. Instead, PyCBC uses **time slides**: the triggers from one detector are time-shifted by $\\Delta t \\gg 10$ ms relative to the other, creating **off-source coincidences** that estimate the background. With $T_{\\text{live}} \\sim 5$ days of coincident data and $\\mathcal{O}(10^5)$ time slides of $\\Delta t = 0.1$ s, the total effective background is $\\sim 50{,}000$ years.

The FAR of GW150914 was estimated to be $< 1/(203{,}000\\text{ yr})$, corresponding to $> 5.1\\sigma$ significance.

### 8.4 The Search Pipeline in Summary

```
Raw strain
    │
    ├─ Condition (resample, highpass, Tukey window)
    │
    ├─ PSD estimation (Welch, off-source)
    │
    ├─ Template bank (O(10^4–10^5) templates)
    │
    ├─ Matched filter → SNR time series per template
    │
    ├─ Threshold & cluster → trigger list
    │
    ├─ χ² veto → re-weighted SNR
    │
    ├─ Coincidence test (H1 ∩ L1 ∩ V1)
    │
    └─ FAR estimation (time slides) → p-value / significance
```
"""))

# ── Cell 21: Template bank demo ───────────────────────────────────────────────
nb["cells"].append(code(
r"""# ── 8.  Template bank: visualise coverage in (M_c, q) space ────────────────
# Pure-numpy demonstration (no PyCBC bank generation needed)

from itertools import product

def chirp_mass(m1, m2):
    return (m1 * m2)**0.6 / (m1 + m2)**0.2

def eta(m1, m2):
    return (m1 * m2) / (m1 + m2)**2

# Sweep over component masses (non-spinning, stellar-mass BBH range)
m_vals = np.linspace(5, 50, 200)
M1, M2 = np.meshgrid(m_vals, m_vals)
mask_valid = M1 >= M2  # m1 >= m2 by convention

Mc = np.where(mask_valid, chirp_mass(M1, M2), np.nan)
q  = np.where(mask_valid, M2 / M1, np.nan)

# Duration of inspiral (Newtonian approximation)
def inspiral_duration(m1, m2, f_lower=20.0):
    """Newtonian chirp time from f_lower to ISCO."""
    Mtot = m1 + m2
    eta_val = (m1 * m2) / Mtot**2
    # tau_0: leading-order chirp time (Peters 1964)
    G = 6.674e-11; c = 3e8; Msun = 1.989e30; pi = np.pi
    M_SI = Mtot * Msun
    tau0 = (5.0 / 256.0) * (c**3 / (G * M_SI))**(5.0/3.0) * (pi * f_lower)**(-8.0/3.0) / eta_val
    return tau0

tau = np.where(mask_valid, inspiral_duration(M1, M2), np.nan)

# Sample template bank points (simplified regular grid in (tau0, tau3))
# (Real banks use geometric or stochastic placement)
Mc_bank = np.arange(3, 40, 0.8)   # chirp masses to sample
q_bank  = np.array([1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
bank_m1, bank_m2, bank_Mc, bank_q = [], [], [], []
for mc, qi in product(Mc_bank, q_bank):
    # Invert: Mc, q -> m1, m2
    # m1+m2 = Mc * (1/q)^(1/5) * (1+1/q)^(6/5) ... simplified
    eta_b = qi / (1 + qi)**2
    Mtot_b = mc / eta_b**0.6
    m1_b = Mtot_b * qi / (1 + qi)  # approximate
    m2_b = Mtot_b / (1 + qi)
    if m1_b >= 3 and m2_b >= 1.4 and m1_b <= 60:
        bank_m1.append(m1_b)
        bank_m2.append(m2_b)
        bank_Mc.append(chirp_mass(m1_b, m2_b))
        bank_q.append(m2_b / m1_b)

bank_Mc = np.array(bank_Mc)
bank_q  = np.array(bank_q)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
cp = ax.contourf(Mc, q, np.log10(tau), levels=20, cmap='viridis')
fig.colorbar(cp, ax=ax, label=r'$\log_{10}(\tau_\mathrm{inspiral}$ [s])')
ax.scatter(bank_Mc, bank_q, s=3, c='white', alpha=0.5, label='Bank templates')
ax.set_xlabel(r'Chirp mass $\mathcal{M}_c\;[M_\odot]$')
ax.set_ylabel('Mass ratio $q = m_2/m_1$')
ax.set_title('Parameter Space Coverage: Chirp Duration')
ax.legend(fontsize=9)

ax = axes[1]
ax.scatter(bank_Mc, bank_q, s=4, c='steelblue', alpha=0.7)
ax.set_xlabel(r'Chirp mass $\mathcal{M}_c\;[M_\odot]$')
ax.set_ylabel('Mass ratio $q = m_2/m_1$')
ax.set_title(f'Template Bank (N = {len(bank_Mc)} templates, schematic)')
ax.set_xlim(0, 40)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('lesson6_bank.png', dpi=110, bbox_inches='tight')
plt.show()
print(f"Schematic template bank: {len(bank_Mc)} templates")
"""))

# ── Cell 22: Section 9 Exercises ─────────────────────────────────────────────
nb["cells"].append(md(
r"""---
## 9. Student Exercises

### Exercise 1 — Inner Products and Optimal SNR

a) Using the analytic result for a Newtonian inspiral $|\tilde{h}(f)| \propto f^{-7/6}$, derive an approximate formula for the optimal SNR $\rho_\text{opt}$ as a function of chirp mass $\mathcal{M}_c$, luminosity distance $d_L$, and the noise PSD $S_n(f)$.

b) Using `aLIGOZeroDetHighPower` from `pycbc.psd`, compute $\rho_\text{opt}$ numerically for a BBH with $\mathcal{M}_c = 28\,M_\odot$ at $d_L = 410$ Mpc. Compare with the published GW150914 SNR.

c) Estimate the **detection horizon** $d_\text{hor}$ (the distance at which $\rho_\text{opt} = 8$) as a function of chirp mass for $\mathcal{M}_c \in [5, 100]\,M_\odot$. Plot the result. What chirp mass maximises the horizon?

---

### Exercise 2 — Waveform Approximants

a) Generate SEOBNRv4 and IMRPhenomD templates for $m_1 = m_2 = 10\,M_\odot$, no spins, at $d_L = 100$ Mpc. Plot both on the same axes.

b) Compute the **overlap** (normalised inner product) between the two approximants:
$$
\mathcal{O} = \frac{\langle h_1 | h_2 \rangle}{\|h_1\|\, \|h_2\|}
$$
Use `pycbc.psd.aLIGOZeroDetHighPower` for the PSD. Is $\mathcal{O} > 0.99$?

c) Repeat for $m_1 = m_2 = 50\,M_\odot$. Why does the overlap change?

d) *(Advanced)* Enable spin: set `spin1z = 0.8` in SEOBNRv4. Generate the corresponding IMRPhenomXP waveform (which includes precession). Compute their overlap. Comment on the loss due to ignoring spin in the template bank.

---

### Exercise 3 — PSD Estimation Bias

a) Simulate 512 s of aLIGO-design coloured noise using `pycbc.noise.noise_from_psd`.

b) Estimate the PSD using Welch's method with three different segment lengths: $T = 1, 4, 16$ s. For each, compute and plot the **fractional error** $(\hat{S}_n - S_{n,\text{true}})/S_{n,\text{true}}$ as a function of frequency.

c) The variance of the Welch estimator is $\propto S_n^2 / K$ where $K$ is the number of segments. Verify this scaling numerically for the three segment lengths.

d) Explain the frequency resolution–variance trade-off and its consequences for matched filtering.

---

### Exercise 4 — False Alarm Rate

a) Simulate $N = 10{,}000$ independent noise realisations of duration $T = 1$ s, each with unit-variance Gaussian noise.

b) Run a matched filter (with a flat PSD) on each realisation using a $\delta$-function template. Record the peak SNR $\rho_\text{max}$ from each.

c) Plot the empirical distribution of $\rho_\text{max}$ and compare it to the analytic prediction:
$$
F(\rho_\text{max} < x) = \exp(-N_\text{ind} \cdot e^{-x^2/2})
$$
where $N_\text{ind} = T \cdot \Delta f$ is the number of independent time–frequency bins.

d) Estimate the threshold $\rho_*$ corresponding to a FAR of $1/(10\text{ yr})$ at a single detector, assuming 1 second of independent data per trial.

---

### Exercise 5 — χ² Sensitivity

a) Using the simulation from Section 5 (signal + noise), vary the injection SNR over $\rho_\text{opt} \in [5, 30]$. For each, compute the $\chi^2_r$ value at the peak trigger.

b) Repeat with a **glitch model**: inject a Gaussian-enveloped sinusoid with the same peak SNR as the signal. Compute $\chi^2_r$ for the glitch.

c) Plot $\chi^2_r$ vs. $|\rho|$ for both signals and glitches on the same scatter plot. Where would you place a veto boundary $\chi^2_r(|\rho|)$?

d) Implement the re-weighted SNR $\hat{\rho}$ and show that it separates signals from glitches more cleanly than $|\rho|$ alone.

---

### Exercise 6 — GW150914 Skymap Check

a) Using the H1 and L1 peak trigger times from Section 7, compute the **time delay** $\Delta t_{HL}$.

b) The time delay constrains the source to lie on a ring on the sky:
$$
\Delta t = \frac{d_{HL}}{c}\cos\theta
$$
where $\theta$ is the angle between the source direction and the H–L baseline, and $d_{HL} = 3002$ km. Compute the allowed range of $\theta$.

c) The published 90% credible sky area for GW150914 is $\sim 180\,\mathrm{deg}^2$. Using your time-delay ring, estimate the fraction of the sky that is consistent with the measured $\Delta t$, assuming $\pm 1$ ms timing uncertainty. Is this consistent with the published skymap?

d) *(Advanced)* Download the official LIGO–Virgo skymap FITS file from GWOSC and plot it on a Mollweide projection using `ligo.skymap` or `healpy`. Overlay the time-delay ring from part (b).

---

### Exercise 7 — Signal Consistency in Frequency Bands

a) Divide the IMRPhenomD GW150914 template into **three equal-contribution frequency sub-bands**: $[20, f_1]$, $[f_1, f_2]$, $[f_2, f_\text{Nyq}/2]$.

b) Run the matched filter using only each sub-band separately on the real GW150914 data. Plot the three partial SNR time series.

c) At the peak trigger time, compare the partial SNRs to the expected value $|\rho|/3$. Compute the chi-squared contribution from each sub-band.

d) Discuss what this test tells you about where in frequency the signal power is concentrated for GW150914.

---

### Exercise 8 *(Challenge)* — Mini Template Bank Search

Using the GW150914 data:

1. Construct a small template bank covering chirp masses $\mathcal{M}_c \in [25, 32]\,M_\odot$ with mass ratio $q \in [0.5, 1.0]$, using a simple grid with spacing $\Delta\mathcal{M}_c = 1\,M_\odot$, $\Delta q = 0.1$.

2. Run the matched filter for each template on both H1 and L1. Record the peak SNR and trigger time for each.

3. Find the **loudest coincident trigger** (the template that maximises the combined H1+L1 SNR with $|\Delta t_{HL}| < 15$ ms).

4. Compare the winning template parameters to the published best-fit masses. What does this tell you about the information content of the matched filter peak?

5. Apply the $\chi^2$ veto and re-weighted SNR. Does the detection survive?
"""))

# ── Cell 23: References ────────────────────────────────────────────────────────
nb["cells"].append(md(
r"""---
## 10. References

### Detection Papers

1. **Abbott et al. (LIGO/Virgo, 2016)** — Observation of Gravitational Waves from a Binary Black Hole Merger.  
   *Phys. Rev. Lett.* 116, 061102. [arXiv:1602.03837](https://arxiv.org/abs/1602.03837)

2. **Abbott et al. (LIGO/Virgo, 2016)** — Properties of the Binary Black Hole Merger GW150914.  
   *Phys. Rev. Lett.* 116, 241102. [arXiv:1602.03840](https://arxiv.org/abs/1602.03840)

3. **Abbott et al. (LIGO/Virgo, 2016)** — GW150914: First results from the search for binary black hole coalescence with Advanced LIGO.  
   *Phys. Rev. D* 93, 122003. [arXiv:1602.03839](https://arxiv.org/abs/1602.03839)

### Matched Filtering Theory

4. **Wainstein & Zubakov (1962)** — *Extraction of Signals from Noise*. Dover Publications.

5. **Allen, B. (2005)** — $\\chi^2$ time-frequency discriminator for gravitational wave detection.  
   *Phys. Rev. D* 71, 062001. [arXiv:gr-qc/0405045](https://arxiv.org/abs/gr-qc/0405045)

6. **Allen, B., Anderson, W. G., Brady, P. R., Brown, D. A., & Creighton, J. D. E. (2012)** — FINDCHIRP: An algorithm for detection of gravitational waves from inspiraling compact binaries.  
   *Phys. Rev. D* 85, 122006. [arXiv:gr-qc/0509116](https://arxiv.org/abs/gr-qc/0509116)

7. **Cutler, C. & Flanagan, É. E. (1994)** — Gravitational waves from merging compact binaries: How accurately can one extract the binary's parameters from the inspiral waveform?  
   *Phys. Rev. D* 49, 2658.

### Search Pipelines

8. **Usman, S. A. et al. (2016)** — The PyCBC search for gravitational waves from compact binary coalescences.  
   *Class. Quantum Grav.* 33, 215004. [arXiv:1508.02357](https://arxiv.org/abs/1508.02357)

9. **Nitz, A. H. et al. (2017)** — Detecting binary compact-object mergers with gravitational waves: Understanding and Improving the sensitivity of the PyCBC search.  
   *Astrophys. J.* 849, 118. [arXiv:1705.01513](https://arxiv.org/abs/1705.01513)

10. **Davies, G. S. et al. (2020)** — Extending the PyCBC search for gravitational waves from compact binary mergers to a global network.  
    *Phys. Rev. D* 102, 022004. [arXiv:2002.08291](https://arxiv.org/abs/2002.08291)

### Template Banks

11. **Owen, B. J. (1996)** — Search templates for gravitational waves from inspiraling binaries: Choice of template spacing.  
    *Phys. Rev. D* 53, 6749. [arXiv:gr-qc/9511032](https://arxiv.org/abs/gr-qc/9511032)

12. **Owen, B. J. & Sathyaprakash, B. S. (1999)** — Matched filtering of gravitational waves from inspiraling compact binaries: Computational cost and template placement.  
    *Phys. Rev. D* 60, 022002. [arXiv:gr-qc/9901023](https://arxiv.org/abs/gr-qc/9901023)

13. **Roy, S., Sengupta, A. S., & Thakor, P. (2017)** — Hybrid geometric-random template-bank generation using a packing algorithm.  
    *Phys. Rev. D* 95, 104045. [arXiv:1702.06771](https://arxiv.org/abs/1702.06771)

### Waveform Models

14. **Bohé, A. et al. (2017)** — Improved effective-one-body model of spinning, nonprecessing binary black holes for the era of gravitational-wave astrophysics with advanced detectors.  
    *Phys. Rev. D* 95, 044028. [arXiv:1611.03703](https://arxiv.org/abs/1611.03703)  
    *(SEOBNRv4)*

15. **Husa, S. et al. (2016)** and **Khan, S. et al. (2016)** — Frequency-domain gravitational waves from non-precessing black-hole binaries. Parts I and II.  
    *Phys. Rev. D* 93, 044006 and 044007. *(IMRPhenomD)*

### Software

16. **Nitz, A. H. et al. (2020)** — PyCBC: A Python toolkit for gravitational wave data analysis.  
    [arXiv:2001.09310](https://arxiv.org/abs/2001.09310) · https://pycbc.org

17. **GWOSC** — Gravitational Wave Open Science Center.  
    https://gwosc.org — hosts all public LIGO/Virgo/KAGRA data.

18. **GW150914 Open Data** — 4 kHz strain, PSDs, and event times.  
    https://gwosc.org/events/GW150914/

### Textbooks

19. **Maggiore, M. (2007)** — *Gravitational Waves: Theory and Experiments.* Oxford University Press.

20. **Jaranowski, P. & Królak, A. (2009)** — *Analysis of Gravitational-Wave Data.* Cambridge University Press.

21. **Creighton, J. D. E. & Anderson, W. G. (2011)** — *Gravitational-Wave Physics and Astronomy.* Wiley-VCH.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# Write notebook
import os
out_path = "/home/runner/work/gw-lvk-python-course/gw-lvk-python-course/Course/Lesson_06_Matched_Filtering_and_Signal_Detection.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written: {out_path}")
print(f"Total cells: {len(nb['cells'])}")
