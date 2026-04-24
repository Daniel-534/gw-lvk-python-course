# Gravitational Wave Ligo-Virgo-Kagra Python Course

A comprehensive educational course on gravitational wave data analysis using Python, covering both foundational signal processing and advanced cosmological parameter inference with real LIGO-Virgo-Kagra data.

## Overview

This repository contains two integrated learning paths:

- **Course**: A 10-class introduction to gravitational wave detection, data analysis, and Bayesian parameter estimation using open science tools from the IGWN (International Gravitational-Wave Network).
- **Cosmology**: A 10-lesson specialized module on using gravitational waves as standard sirens to measure cosmological parameters (H0, Omega_m) using gwcosmo and Bilby.

## Prerequisites

- Python 3.8 or higher
- Basic knowledge of general relativity, signal processing, and Bayesian statistics (helpful but not required)
- Familiarity with Jupyter notebooks and common scientific Python libraries (NumPy, SciPy, Matplotlib)

## Installation

All necessary packages can be installed using:

```bash
pip install gwpy gwosc pycbc bilby ligo-skymap gwcosmo dynesty
```

For detailed environment setup and troubleshooting, see the working environment lesson in the Course module.

## Course Structure

### Course Module - LVK Data Analysis with Python

A 10-class introduction to gravitational wave physics and data analysis:

1. **The Universe of Gravitational Waves** - Fundamental physics, detector architecture, and astrophysical sources
2. **Working Environment & Data Access** - Python stack setup and GWOSC data access
3. **Time-Series Analysis** - Data manipulation and whitening with GWpy
4. **Spectral Analysis & Detector Noise** - FFT, PSD, and noise characterization
5. **The GWTC Catalog** - Exploring real gravitational wave events and populations
6. **Matched Filtering & Signal Detection** - Optimal detection pipeline and SNR computation
7. **Gravitational-Wave Waveform Models** - Post-Newtonian to numerical-relativity approximants
8. **Bayesian Parameter Inference** - Nested sampling and posterior estimation with Bilby
9. **Sky Localization & Multi-Messenger Astronomy** - Triangulation, sky maps, and electromagnetic follow-up
10. **Capstone Project** - Full end-to-end analysis pipeline on a real GWTC event

Each lesson is a Jupyter notebook combining theory, visualizations, and hands-on code examples.

### Cosmology Module - Gravitational Waves as Cosmological Probes

A 10-lesson specialized track on using GW standard sirens:

1. **Gravitational Waves as Standard Sirens** - Concept of bright and dark sirens, luminosity distance, and H0 measurement
2. **Bayesian Inference Foundations with Bilby** - Bayes theorem, priors, likelihoods, and nested sampling
3. **Parameter Estimation on Real Events** - Analyzing GW170817 and BBH events with real GWOSC data
4. **Introduction to gwcosmo** - Statistical framework for cosmological inference
5. **H0 from a Bright Siren** - Using GW170817 with electromagnetic counterpart NGC 4993
6. **H0 from Dark Sirens** - Host-galaxy statistical methods for binary black hole events
7. **Combining Multiple Events** - Hierarchical population-level inference and uncertainty scaling
8. **Selection Effects & Completeness Corrections** - Malmquist bias and catalog incompleteness
9. **Beyond H0: Omega_m and Modified GW Propagation** - Extended cosmological parameters and beyond-GR theories
10. **Capstone** - End-to-end cosmological analysis pipeline with comparison to published results

## Key Tools & Libraries

- **GWpy**: Time-series analysis and data handling
- **gwosc**: Access to Gravitational Wave Open Science Center data
- **PyCBC**: Waveform generation and matched filtering
- **Bilby**: Bayesian inference and parameter estimation
- **ligo.skymap**: Sky localization and FITS map handling
- **gwcosmo**: Cosmological inference from gravitational waves
- **dynesty**: Nested sampling for likelihood exploration

## Learning Outcomes

Upon completion of this course, students will be able to:

- Understand the physics of gravitational wave generation and detection
- Access and analyze open gravitational wave data from LIGO, Virgo, and KAGRA
- Apply signal processing techniques to identify and characterize GW signals
- Implement matched filtering and detect signals in detector noise
- Perform Bayesian parameter estimation on real events
- Generate and interpret sky localization maps
- Use gravitational waves as cosmological probes
- Measure fundamental cosmological parameters (H0, Omega_m)
- Analyze selection effects and systematic uncertainties in cosmological analyses

## Data & References

All course materials use publicly available data from:

- Gravitational Wave Open Science Center (gwosc.org)
- Gravitational-Wave Transient Catalog (GWTC-1 through GWTC-4)
- Published posteriors from LVK collaboration papers

## How to Use This Repository

1. Start with the **Course** module for foundational knowledge
2. Work through lessons sequentially, running all code cells
3. Complete the Course Capstone before proceeding to Cosmology
4. Use the **Cosmology** module for specialized training on standard sirens
5. All notebooks are compatible with Google Colab for cloud-based execution

## Status

Under active development. All lessons are functional; additional tutorials and advanced topics coming soon.

## Contributing

Contributions, bug reports, and suggestions are welcome. Please open an issue to discuss proposed changes.

## License

This course material is provided for educational purposes. Check individual lesson notebooks for specific dataset and software licensing information.

## Acknowledgments

This course is built upon the scientific work of the LIGO Scientific Collaboration, Virgo Collaboration, and KAGRA Collaboration. Course material references published GWTC catalogs and LVK collaboration papers.

## References

- GWTC-1, GWTC-2, GWTC-2.1, GWTC-3, GWTC-4 papers
- GWpy documentation: https://gwpy.github.io
- Bilby documentation: https://bilby.readthedocs.io
- gwcosmo documentation: https://gwcosmo.readthedocs.io
- gwosc API documentation: https://gwosc.org

---

For questions or support, please open an issue in the repository.
