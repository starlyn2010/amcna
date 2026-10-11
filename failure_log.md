# Failure Log — AMCNA Fase 1.5

Cada hipótesis que falla aquí se documenta permanentemente.
Los papers fuertes documentan tanto fracasos como éxitos.

Formato:

```
## [FHXXX] Título breve

**Hipótesis:** Qué esperábamos.
**Experimento:** Qué se hizo exactamente.
**Resultado:** Qué ocurrió.
**Interpretación:** Por qué pudo haber pasado.
**Decisión:** Qué cambió en consecuencia.
```

---

## [FH001] Las 3 tareas sintéticas básicas discriminan AMCNA vs CfC a corto plazo

**Hipótesis:** Con 30 épocas y 300 ejemplos, AMCNA mostraría una ventaja medible sobre CfC en al menos una tarea.

**Experimento:** Primera validación rápida. 1 semilla, 30 épocas, 300 muestras. Tareas: XOR (seq=30), Delayed Memory (D=15), Sequence Copy (S=5, D=10).

**Resultado:**
- XOR: todos los modelos ~57.7% (azar)
- Delayed Memory: todos 100%
- Sequence Copy: CfC=87.7%, AMCNA=87.3% (empate técnico)
- Costo: CfC ~15s, AMCNA_Full ~80s (5x más lento)

**Interpretación:**
1. 30 épocas son insuficientes para que AMCNA explote sus canales (mayor complejidad interna => más épocas para optimizar).
2. Las tareas son demasiado fáciles: D=15 es trivial para cualquier RNN. XOR con 2 pulsos tampoco estresa la memoria.
3. Los promedios de ganancia de canal (C4=0.71, C2=0.65 en Copy) indican que los canales se usan diferencialmente, lo cual es un hallazgo positivo aunque el accuracy no lo confirme aún.

**Decisión:**
- No escalar a 20 semillas x 5 arquitecturas todavía.
- Diseñar Fase 1.5: stress tests con tareas mucho más exigentes (D hasta 500, paridad multi-pulso, copias largas).
- Buscar el punto de ruptura donde AMCNA diverge de CfC.

---
<!-- Añadir nuevas entradas aqui a medida que fallen hipotesis -->
