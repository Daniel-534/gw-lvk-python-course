# Class Plan: Cosmology with gwcosmo & Bilby

---

## Lesson 1 — Gravitational Waves as Standard Sirens
Concept of bright and dark sirens. Luminosity distance $d_L$ from GW signals.
The $d_L$ – $z$ relation and its role in measuring $H_0$. Review of key events:
GW170817 (bright siren) and BBH dark sirens.

---

## Lesson 2 — Bayesian Inference Foundations with Bilby
Bayes theorem applied to GW parameter estimation. Setting up priors, likelihoods,
and posteriors in Bilby. Nested sampling with dynesty. Recovering $d_L$, $\iota$,
and sky position from a simulated CBC signal.

---

## Lesson 3 — Parameter Estimation on Real Events
Running Bilby on GWOSC data for GW170817 and a BBH event. Extracting marginal
posteriors on $d_L$, $\mathcal{M}$, and $\chi_{\rm eff}$. Comparing results
against published LVK posteriors from the GWTC catalog.

---

## Lesson 4 — Introduction to gwcosmo
Architecture of the gwcosmo package. Statistical framework for $H_0$ inference
using GW events with and without electromagnetic counterparts. Galaxy catalogs
as redshift proxies: GLADE, DES, and GWENS.

---

## Lesson 5 — $H_0$ from a Bright Siren (GW170817)
Using gwcosmo with the confirmed counterpart NGC 4993. Setting the redshift
likelihood from spectroscopic data. Computing the posterior $p(H_0 | d_{\rm GW},
z)$ and comparing with Planck and SH0ES values.

---

## Lesson 6 — $H_0$ from Dark Sirens
Statistical host-galaxy method applied to BBH events (e.g. GW190814, GW170814).
Marginalising over galaxy catalog incompleteness. Impact of sky localisation
volume on the $H_0$ posterior width.

---

## Lesson 7 — Combining Multiple Events
Hierarchical combination of $H_0$ posteriors across a population of events.
Scaling of uncertainty with catalog size $\sim N^{-1/2}$. Reproducing the
LVK O3 combined $H_0$ result with gwcosmo using public posterior samples.

---

## Lesson 8 — Selection Effects & Completeness Corrections
GW detection selection bias and the Malmquist effect on $d_L$. Catalog
incompleteness corrections in gwcosmo. Sensitivity of $H_0$ to the assumed
mass distribution prior and redshift completeness fraction.

---

## Lesson 9 — Beyond $H_0$: $\Omega_m$ and Modified GW Propagation
Extending gwcosmo to constrain $\Omega_m$ and the dark energy equation of state.
Modified GW luminosity distance in beyond-GR theories: the $(\Xi_0, n)$
parametrisation. Joint inference of $H_0$ and $\Xi_0$ from a simulated catalog.

---

## Lesson 10 — Capstone: End-to-End Cosmological Analysis
Full pipeline: Bilby parameter estimation on a chosen GWTC event → posterior
samples → gwcosmo $H_0$ inference → combination with archival events. Critical
comparison with published results and discussion of future prospects with O5
and LISA.
