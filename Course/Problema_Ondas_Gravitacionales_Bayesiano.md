# Problema aplicado: inferencia bayesiana de masa chirp y distancia en un evento tipo GW150914

## Contexto del problema (vida real)
En la detección de ondas gravitacionales, los equipos LVK estiman parámetros físicos de un sistema binario (masas, distancia, orientación, etc.) ajustando modelos de señal a datos ruidosos. Esto se hace con **estadística bayesiana**: combinamos un *modelo físico* con un *modelo de ruido* y *priors* astrofísicos para obtener un **posterior** que cuantifique incertidumbres reales.  

En este problema se simula un evento compacto similar a **GW150914** (binario de agujeros negros). Se usan relaciones físicas básicas del inspiral newtoniano para construir un conjunto de mediciones sintéticas de:
- **Relación tiempo–frecuencia** de la chirp 
- **Amplitud de la señal** en función de la frecuencia

A partir de estas mediciones y de la estadística del ruido, se pide estimar con Bayes la **masa chirp** \(\mathcal{M}\) y la **distancia luminosidad** \(d_L\).

---

## Marco teórico (resumen)

### 1. Señal inspiral y masa chirp
Para una binaria circular, el tiempo antes de la coalescencia y la frecuencia están relacionados por:

\[
 t(f) = \frac{5}{256}\left(\frac{G\,\mathcal{M}}{c^3}\right)^{-5/3}(\pi f)^{-8/3}
\]

La **masa chirp** \(\mathcal{M} = (m_1 m_2)^{3/5} (m_1+m_2)^{-1/5}\) controla la aceleración de la frecuencia (el *chirp*). Por eso, medir la curva \(t(f)\) permite inferir \(\mathcal{M}\).

### 2. Amplitud y distancia
En el régimen de cuadrupolo (orientación óptima), la amplitud aproximada es:

\[
 h(f) = \frac{4}{d_L}\,\frac{(G\mathcal{M})^{5/3}}{c^4}\,(\pi f)^{2/3}
\]

Así, la amplitud escala con \(\mathcal{M}^{5/3}\) y decrece con la distancia \(d_L\).

### 3. Modelo estadístico
Supondremos mediciones ruidosas con errores gaussianos:

\[
 t_{\text{obs}} = t(f;\mathcal{M}) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,\sigma_t^2)
\]
\[
 h_{\text{obs}} = h(f;\mathcal{M},d_L) + \epsilon_h, \quad \epsilon_h \sim \mathcal{N}(0,\sigma_h^2)
\]

La **verosimilitud** para un conjunto de datos \(D\) es:

\[
 \mathcal{L}(\mathcal{M},d_L) \propto \exp\Big(-\tfrac{1}{2}\chi^2\Big)
\]

con \(\chi^2\) definido por los residuos normalizados.

### 4. Inferencia bayesiana
Aplicamos el teorema de Bayes:

\[
 p(\mathcal{M},d_L|D) \propto \mathcal{L}(\mathcal{M},d_L)\,\pi(\mathcal{M})\,\pi(d_L)
\]

- Prior en masa chirp: uniforme \(\mathcal{M} \in [10,50]~M_\odot\)
- Prior en distancia: volumétrico \(\pi(d_L) \propto d_L^2\) en \([100,1000]~\text{Mpc}\)

---

## Datos sintéticos (a generar en Python)
- Frecuencias de análisis: 15 puntos entre 30 y 150 Hz.
- Parámetros reales (inyección):
  - \(\mathcal{M}_{\text{true}} = 28~M_\odot\)
  - \(d_{L,\text{true}} = 420~\text{Mpc}\)
- Ruido:
  - \(\sigma_t = 0.02~\text{s}\)
  - \(\sigma_h = 0.15\, h\)
- Usar semilla fija para reproducibilidad.

---

## Preguntas (resolubles con Python)

1. **Simulación de datos**: genera \(t_{\text{obs}}\) y \(h_{\text{obs}}\) con el modelo físico y el ruido indicado. Grafica la relación tiempo–frecuencia y la amplitud observada.
2. **Implementación del modelo**: escribe funciones para \(t(f;\mathcal{M})\) y \(h(f;\mathcal{M},d_L)\) en unidades físicas (SI).
3. **Verosimilitud**: construye la función \(\log\mathcal{L}(\mathcal{M},d_L)\) suponiendo errores gaussianos independientes.
4. **Posterior 2D**: calcula el posterior en una malla \((\mathcal{M},d_L)\) y grafica el mapa de densidad y sus contornos de credibilidad.
5. **Marginalización**: obtén \(p(\mathcal{M}|D)\) y \(p(d_L|D)\) y reporta el valor mediano y el intervalo creíble al 90% para cada parámetro.
6. **Probabilidades de interés**: estima
   - \(P(\mathcal{M} > 30~M_\odot)\)
   - \(P(d_L < 500~\text{Mpc})\)
7. **Sensibilidad a priors**: repite la inferencia con un prior uniforme en \(d_L\) y compara los resultados con el prior volumétrico.

---

## Bosquejo de la solución (para el notebook)

1. **Preparación del entorno**: importar librerías y definir constantes físicas.
2. **Simulación**: crear el conjunto de frecuencias, generar \(t_{\text{obs}}\) y \(h_{\text{obs}}\) con ruido.
3. **Modelo físico**: implementar funciones para \(t(f;\mathcal{M})\) y \(h(f;\mathcal{M},d_L)\).
4. **Likelihood + Priors**: construir \(\log\mathcal{L}\) y \(\log\pi\) con los priors dados.
5. **Posterior en malla**: evaluar el posterior en una grilla \((\mathcal{M},d_L)\), normalizar y visualizar.
6. **Marginalización**: calcular distribuciones marginales, medianas e intervalos creíbles.
7. **Probabilidades**: integrar posterior para \(\mathcal{M}>30\) y \(d_L<500\) Mpc.
8. **Comparación de priors**: repetir con prior uniforme en distancia y discutir el cambio.

---

**Resultado esperado:** un notebook reproducible que muestre cómo la estadística bayesiana se aplica directamente en el análisis real de ondas gravitacionales, cuantificando incertidumbres físicas relevantes.
