# 10-Class Plan: LVK Data Analysis with Python

---

## Class 1 — The Universe of Gravitational Waves
Core physics of gravitational waves (GWs): linearized general relativity, strain amplitude $h(t)$,
and polarizations $h_{+}$ and $h_{\times}$. Astrophysical sources: binary black holes (BBH), binary
neutron stars (BNS), and neutron star–black hole (NSBH) mergers. Architecture of the LIGO, Virgo,
and KAGRA interferometers. Overview of observing runs O1–O4 and the LVK/IGWN collaboration.

---

## Class 2 — Working Environment & Data Access
Installing the full Python stack: gwosc, gwpy, pycbc, bilby, and ligo.skymap. Navigating the
Gravitational Wave Open Science Center (gwosc.org). Programmatic data access with the gwosc API:
`find_datasets()`, `event_gps()`, `get_event_urls()`. Gravitational-wave file formats: .gwf and
.hdf5. Data quality flags, bitmasks, and Analysis-Ready (AR) segments.

---

## Class 3 — Time-Series Analysis
Manipulation of data using the `TimeSeries` object in GWpy: cropping, resampling, and band-pass
filtering. Whitening: removing the detector noise floor to reveal embedded signals. Visualisation
of raw vs. processed strain for GW150914 and GW170817. Handling gaps, data artifacts, and GPS
time conventions.

---

## Class 4 — Spectral Analysis & Detector Noise
Discrete Fourier Transform (FFT) and its interpretation in the GW context. Power Spectral Density
(PSD) and Amplitude Spectral Density (ASD): estimation with Welch and median–mean methods.
Detector sensitivity curves across O1–O4. $Q$-Transform spectrograms for time–frequency analysis.
Identifying glitches, spectral lines, and environmental artefacts.

---

## Class 5 — The GWTC Catalog: Exploring Real Events
Programmatic catalog queries with gwosc and pycbc.catalog across GWTC-1 through GWTC-4. Key event
parameters: component masses $m_1$, $m_2$, chirp mass $\mathcal{M}$, mass ratio $q$, luminosity
distance $d_L$, and effective spin $\chi_{\rm eff}$. Population-level distributions: mass spectrum,
redshift, and SNR histograms. Reproducing the "Masses in the Stellar Graveyard" plot. Accessing
and computing from posterior samples in HDF5 format.

---

## Class 6 — Matched Filtering & Signal Detection
Optimal matched filtering in Gaussian noise: derivation of the SNR statistic $\rho$. Generating
waveform templates with `pycbc.waveform` (SEOBNRv4, IMRPhenomD). PSD estimation and
frequency-domain whitening. Computing the complex SNR time series and the $\chi^{2}$ consistency
test. Step-by-step reproduction of the GW150914 detection pipeline from archival LIGO data.

---

## Class 7 — Gravitational-Wave Waveform Models
Hierarchy of waveform approximants: post-Newtonian (PN), effective one-body (EOB), phenomenological
(IMRPhenom), and numerical-relativity surrogates. Physical effects: spin precession, orbital
eccentricity, and tidal deformability $\Lambda$ (BNS). Comparing waveform families by mismatch and
computational cost. Simulated signal injection into real noise frames and recovery efficiency
analysis.

---

## Class 8 — Bayesian Parameter Inference
Foundations of Bayesian inference in GW astronomy: likelihood $\mathcal{L}$, prior $\pi$, and
posterior $p(\theta | d)$ for CBC signals. Setting up a Bilby inference job: waveform model,
likelihood, and prior definitions. Nested sampling with dynesty and nessai; convergence diagnostics
and evidence $\mathcal{Z}$ estimation. Interpreting corner plots for $m_1$, $m_2$, $\chi_{\rm eff}$,
$d_L$, and sky position. Comparison with published LVK posteriors.

---

## Class 9 — Sky Localization & Multi-Messenger Astronomy
Sky localization principles: triangulation, amplitude consistency, and phase coherence across the
detector network. Generating and reading FITS sky maps with ligo.skymap and BAYESTAR. Interpreting
$50\%$ and $90\%$ credible-region areas. Real-time alert infrastructure: GraceDB and GCN notices.
Case study: GW170817, its kilonova counterpart AT2017gfo, GRB 170817A, and the $H_0$ measurement.

---

## Class 10 — Capstone Project: Full Analysis Pipeline
End-to-end pipeline on a student-chosen GWTC event: data download, quality checks, whitening,
matched filter, SNR $\rho$ and $\chi^{2}$, and detection significance. Full Bayesian parameter
estimation with Bilby: corner plots, credible intervals, and model evidence $\mathcal{Z}$. Sky map
generation and comparison with the published LVK localisation. Critical discussion of recovered
parameters against the official LVK paper and their astrophysical implications.
