# Bitácora — AMCNA (Adaptive Multi-Channel Neural Architecture)

**Actualizada:** 2026-08-10
**Ubicación:** `research/amcna_phase1/BITACORA.md` (consolidada desde `research/formalization/` y `research/amcna_phase1/`)

## ⚠️ REGLA ABSOLUTA
**TODO cambio, resultado, análisis, decisión, comando ejecutado, error, fix, log, verificación y conclusión DEBE ser registrado aquí inmediatamente después de ocurrir. Esta bitácora es la fuente única de verdad del proyecto. Si no está aquí, no pasó.**

---

## Qué es AMCNA (en una frase)
Una red neuronal de **solo 2 neuronas vectoriales** ($N=2$) cuya expresividad no viene de escalar nodos, sino de expandir **cada conexión sináptica en un colector de $K=5$ canales temporales ortogonales** (identidad, integral, derivada, retardo y gating) modulados por un estado global $\mathbf{s}(t)$, con un **bucle de pensamiento** (deliberación $\tau$) desacoplado del tiempo del entorno $t$.

## Hipótesis científica central
> "Un grafo de computación mínimo (2 nodos) puede igualar o superar la capacidad de procesamiento espacio-temporal de grafos masivos si sus aristas no son escalares ni tensores estáticos, sino colectores de operadores dinámicos heterogéneos modulados por un estado global."

## Motivación
- **La falacia del escalado espacial:** la carga cognitiva crece $\mathcal{O}(N^2)$ con nodos; escalar IA por tamaño de red es insostenible en Edge AI.
- **Inspiración biológica:** las sinapsis biológicas multiplexan múltiples neurotransmisores con distintas constantes temporales; la neocorteza no añade neuronas para aprender, multiplexa información.
- **Propuesta:** escalar en el **dominio del tiempo y la frecuencia**, no en el espacio físico.

---

## Arquitectura

| Componente | Descripción | Archivo |
|---|---|---|
| Modelo AMCNA (K=5, mod_dim=4, delay δ=5, deliberación L) | 2 neuronas vectoriales CfC + 5 canales + modulador | `models/amcna.py` |
| Baselines (CfC, MLP con parámetros emparejados) | Controles experimentales | `models/baselines.py` |
| Datasets sintéticos (XOR, Memory, Copy) | Tareas de validación | `datasets/synthetic.py` |
| Trainer con early stopping y latencia | Entrenamiento/evolución | `trainers/trainer.py` |
| Sanity check (gradientes, NaNs, emparejamiento) | Verificación de integridad | `experiments/sanity_check.py` |
| Primera validación | 1 semilla × 30 épocas × 300 muestras | `experiments/first_validation.py` |
| Stress tests Fase 1.5 | Sweeps de delay, pulsos, deliberación, ablación | `experiments/run_stress_tests.py` |
| Failure log (hipótesis fallidas) | FH001... | `failure_log.md` |

### Implementación del modelo (`models/amcna.py`)
- **Entrada:** `env_proj` → proyección lineal a `hidden_size`.
- **Canales (K=5):**
  - C1 **Identidad:** $\mathbf{c}^{(1)} = \mathbf{W}_1 \mathbf{h}$
  - C2 **Integral (filtro paso bajo):** $z = (1-\alpha)z + \alpha h$, $\alpha=0.1$
  - C3 **Retardo puro:** buffer FIFO de `delay_steps=5` estados
  - C4 **Derivada filtrada:** $d = \beta d + (1-\beta)(h - h_{prev})$, $\beta=0.5$
  - C5 **Gating:** $\sigma(\mathbf{V}_5 h) \odot \mathbf{W}_5 h$
- **Modulador global:** $s = \tanh(W_s s + V_s h)$; ganancias $gains = \sigma(U_k s) \in [0,1]^5$
- **Mezcla modulada:** $c_{total} = \sum_k gains_k \odot c_k$ + inyección del entorno
- **Nodo:** `CfCCell` (ncps) con `ts=1`; salida `out_proj`
- **Estado de nodo:** vectorial $d=16$ (preserva "2 neuronas físicas" con subespacios latentes)

---

## Fundamentos matemáticos (resumen)

### Los 5 operadores y su independencia (Laplace)
| Canal | Operador | Transformada de Laplace |
|---|---|---|
| C1 Identidad | $h(t)$ | $1$ |
| C2 Integral (Volterra) | $\int_0^t e^{-\alpha(t-\tau)}h(\tau)d\tau$ | $\frac{1}{s+\alpha}$ |
| C3 Retardo | $h(t-\delta)$ | $e^{-s\delta}$ |
| C4 Derivada | $dh/dt$ | $s$ |
| C5 Gating | $\sigma(Vh) \odot Wh$ | $\Psi(s)$ (no lineal) |

$\{1, \frac{1}{s}, e^{-s\delta}, s\}$ son **linealmente independientes** en el espacio de funciones analíticas ⇒ la suma de canales **no puede colapsar** a un único $\mathbf{W}_{eq}$ (Demostración 2). Los 5 coeficientes $c_k$ de una combinación lineal nula deben ser 0 (Teorema 1.1, demostración por límites asintóticos).

### Dinámica del bucle de pensamiento
1. Agregación: $\mathbf{a}_i(\tau) = \sum_k M_k(\mathbf{s}(\tau-1)) \odot \mathbf{c}_{ji}^{(k)}(\tau)$
2. Actualización CfC: $h_i(\tau) = h_i(\tau-1) + (\tanh(a_i + W_{in}x + b) - h_i(\tau-1)) \odot (1 - e^{-\Delta t \cdot g_i / (\tau_i + \epsilon)})$
3. Moduladores: $s(\tau) = \tanh(W_s s(\tau-1) + V_s [h_A; h_B] + U_s x + b_s)$
4. Salida: $\mathbf{y}(t) = head(h_B(L))$ (lineal para control, softmax para lenguaje)

### Resultados formales clave
- **No-equivalencia con RNN/GRU/LSTM/CfC:** la GRU es de orden dinámico 1; el retardo y la derivada exigen historia $t-\delta$, imposible sin aumentar $d$ (Demostración 1).
- **GUUB (Estabilidad):** Lyapunov con $V = \frac{1}{2}\|h_A\|^2 + \frac{1}{2}\|h_B\|^2 + \frac{1}{2}\|s\|^2$; el sistema es globalmente acotado (Teorema 2.1).
- **Expresividad estricta > CfC₂:** una CfC pura de 2 neuronas (EDO orden 1) no puede auto-intersecar trayectorias; AMCNA con retardo es una DDE en $C([-\delta,0],\mathbb{R}^2)$, que sí puede (Teorema 3.1).
- **Punto fijo:** Brouwer sobre el compacto convexo $[-V_{max},V_{max}]^{2d} \times [-1,1]^M$ (Demostración 3).
- **Memoria teórica:** $\mathcal{O}(d \cdot \delta)$ vs $\mathcal{O}(d)$ de GRU (integrador sin pérdida con $\alpha=0$ + buffer explícito de retardo).
- **Turing-completitud** e integración temporal infinita sobre grafos microscópicos (claim del pitch).

### Deuda con la literatura (no invención, síntesis)
1. **NDDEs** (Zhu et al. 2021; Bohn et al. 2022) → Canal 3.
2. **VRNNs (Volterra)** → Canal 2.
3. **Neuromodulated CT-RNNs** → vector $\mathbf{s}(t)$.

---

## Protocolo experimental (formalization/experimentos.md)

| Exp | Objetivo | Métrica de éxito |
|---|---|---|
| 1. XOR (N=2, d=4, K=4, BPTT L=5, Adam) | Separabilidad no lineal sin capas ocultas | 100% accuracy |
| 2. Memoria temporal (T∈[10,50], suma/retardo) | Validar C2 y C3; ablación apagando ambos | MSE con canales << sin canales |
| 3. Estabilidad (ruido blanco 1000 pasos) | Normas acotadas | $\|h\| < V_{max}=10$ sin caos creciente |

### Experimento de refutación extremo (adversarial)
**Paridad temporal larga con distractores:** secuencia de 1000 pasos de ruido gaussiano + 2 picos aleatorios; predecir paridad de la distancia entre picos. Si AMCNA no supera a una GRU con igual número de parámetros (tuning exhaustivo) en ≥15% de accuracy absoluta → **hipótesis principal refutada**.

---

## Resultados experimentales

### FH001 — Primera validación (30 épocas, 300 muestras, 1 semilla)
| Tarea | CfC | AMCNA NoDelib | AMCNA Full |
|---|---|---|---|
| XOR (seq=30) | 57.7% (azar) | 57.7% (azar) | 57.7% (azar) |
| Delayed Memory (D=15) | 100% | 100% | 100% |
| Sequence Copy (S=5, D=10) | 87.7% | 87.3% | 87.3% |
| Costo total | ~15s | ~56s | ~80s (5× más lento) |

**Interpretación:** 30 épocas insuficientes para explotar canales; tareas demasiado fáciles. **Hallazgo positivo:** ganancias de canal diferenciales (C4=0.71, C2=0.65 en Copy) ⇒ los canales sí se usan de forma diferenciada aunque el accuracy no lo confirme.
**Decisión:** no escalar a 20 semillas × 5 arquitecturas; diseñar stress tests (D hasta 500, paridad multi-pulso, copias largas) y buscar el punto de ruptura vs CfC.

### Stress test: Delayed Memory Sweep (D = 15–300)
| D | MLP | CfC | AMCNA |
|---|---|---|---|
| 15 | 0.56 | 1.00 | 0.56 |
| 30 | 0.56 | 0.56 | 0.56 |
| 50 | 0.56 | 1.00 | 1.00 |
| 100 | 0.56 | 0.44 | 0.44 |
| 200 | 0.56 | 1.00 | 0.56 |
| 300 | 0.56 | 1.00 | 0.56 |

**Lectura:** resultados inestables entre semillas/delays; no se encontró aún ventaja consistente de AMCNA sobre CfC; MLP siempre en azar (0.56). Pendiente: repeticiones multi-semilla y análisis de varianza.

### Stress tests pendientes de completar (`run_stress_tests.py`)
- XOR multi-pulso (2, 3, 5, 7 pulsos) → `outputs/xor_pulse_sweep.json`
- Barrido de deliberación L ∈ {1, 2, 4, 8} → `outputs/deliberation_sweep.json`
- Independencia de canales (Pearson + Información Mutua de gains) → `outputs/pearson_matrix.npy`, `mi_matrix.npy`
- Ablación causal C2/C4 (pesos a cero sin reentrenar) — notar: existe un bug latente en `run_ablation_study` (usa `DataLoader` no importado y evalúa 2 veces el modelo equivocado); corregir antes de confiar en esos números.

---

## Red-teaming y límites conocidos (donde AMCNA falla)

1. **Explosión de gradiente en C4:** con $\Delta t \to 0$, el gradiente escala $\mathcal{O}(1/\Delta t)$ ⇒ **no puede funcionar en tiempo continuo puro sin filtro paso bajo o clipping estricto**. Implementado: derivada filtrada con $\beta=0.5$ (mitiga).
2. **Latencia lineal en $\tau$:** con $L=1$ (reacción inmediata) los canales C2/C3 no despliegan historia ⇒ AMCNA degenera a un modelo superficial.
3. **Contraejemplo formal (memoria arbitraria lejana):** si $t_{test} \gg \delta$, el buffer de retardo pierde la señal y el integrador decae $e^{-\alpha t}$ a 0. Para memoria a horizonte indefinido, una memoria externa o un Transformer supera categóricamente a AMCNA.
4. **Caos posible:** sistema no lineal con dimensión > 3 ⇒ atractores extraños (Hopf con C3+C4 negativos y fuerte no linealidad de $\mathbf{s}$).

---

## Riesgos y mitigaciones (formalization/riesgos.md)

| Riesgo | Clasificación | Mitigación |
|---|---|---|
| Colapso por inicialización homogénea de canales | Alto (matemático) | Inicialización ortogonal o pérdida de diversidad $\mathcal{L}_{diverse} = \sum_{k\neq l} \|\mathbf{c}^{(k)} \cdot \mathbf{c}^{(l)}\|$ |
| Explosión/desvanecimiento en bucle L≥20 | Alto | L ∈ [3,10], activaciones saturantes, gradient clipping ≤1.0, alternativas libres de gradiente (CMA-ES) |
| Caos del modulador $\mathbf{s}$ (histeresis bloqueada) | Medio | Coeficiente de decaimiento homeostático hacia el estado neutro 0 |

---

## Preguntas abiertas (formalization/preguntas_abiertas.md)

1. **Dimensionalidad $d$:** ¿1, 32 o 64? (d=16 usado en experimentos)
2. **Inyección de entrada en el bucle:** ¿sostenida (riesgo de saturación) o impulsiva (solo $\tau=1$)?
3. **Retardo $\delta$:** ¿fijo (FIFO) o continuo/diferenciable (splines, más complejo)?
4. **Función de modulación $M_k(s)$:** ¿proyección lineal completa o gating escalar $\sigma(u_k^T s)$? (recomendada: gating escalar para menos parámetros)

---

## Roadmap (formalization/roadmap.md)

- **Fase 1 (semanas 1-2):** clase `TwoNeuronBrain` aislada, Exp 1-2, Adam vs CMA-ES. → *En curso con Fase 1.5 (stress tests).*
- **Fase 2 (semana 3):** modulación global completa, Exp 3 (estabilidad/atractores), cambio de regla para medir adaptación.
- **Fase 3 (semana 4):** exportación JSON de activaciones/ganancias/moduladores compatible con `brain_viz.html`; análisis de escalabilidad para LLM y robot.

## Próximos pasos inmediatos
- [ ] Corregir bug de `run_ablation_study` (import de `DataLoader` y orden de evaluación).
- [ ] Multi-semilla (≥20) para el delayed memory sweep; buscar el punto de ruptura vs CfC.
- [ ] Completar sweeps: XOR multi-pulso, deliberación L, matrices de Pearson/MI.
- [ ] Ejecutar el experimento de refutación extremo (paridad larga con distractores).
- [ ] Decidir $d$ óptimo y tipo de inyección de entrada (preguntas abiertas 1-2).
