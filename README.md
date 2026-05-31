# Solano-post1-u11 — CUDA Benchmark CPU vs GPU
Arquitectura de Computadores — Unidad 11

## Descripción del Entorno

| Componente | Detalle |
|---|---|
| GPU | NVIDIA Tesla T4 |
| Memoria GPU | 15360 MiB |
| Driver NVIDIA | 580.82.07 |
| CUDA Toolkit | 12.8 |
| Plataforma | Google Colab |

## Resultados — Suma de Vectores (vectorAdd)

| N | CPU (ms) | GPU kernel (ms) | Speedup |
|---|---|---|---|
| 1M | 2.38 | 0.18 | ~13x |
| 4M | 9.79 | 0.19 | ~52x |
| 16M | 39.71 | 0.77 | ~52x |

## Resultados — Multiplicación de Matrices (matMul)

| N | Naive (ms) | Tiled (ms) | Speedup |
|---|---|---|---|
| 512 | 33.50 | 0.47 | ~71x |
| 1024 | 9.21 | 5.81 | ~1.6x |

## Análisis

Para N grande, el kernel GPU supera ampliamente a la CPU porque la GPU ejecuta miles de threads en paralelo, asignando un thread por elemento del vector. Mientras la CPU procesa los 16M elementos secuencialmente en un solo núcleo, la Tesla T4 los distribuye entre sus 2560 CUDA cores y los resuelve en fracciones de milisegundo. Este paralelismo masivo es la ventaja fundamental de la arquitectura GPU para operaciones independientes como la suma de vectores.

En la multiplicación de matrices, el tiling con shared memory demuestra una mejora dramática especialmente en N=512, donde el speedup es de ~71x respecto al kernel naïve. Esto ocurre porque el kernel tiled carga bloques de 16×16 elementos en shared memory (latencia ~5 ciclos) en lugar de acceder repetidamente a memoria global (latencia ~500 ciclos), reduciendo los accesos a memoria global por un factor de 16. Para N=1024 el speedup es menor porque la cantidad de tiles aumenta y el overhead de sincronización entre bloques cobra más peso relativo.

## Capturas de Checkpoints

<img width="1050" height="506" alt="verificacionGPU" src="https://github.com/user-attachments/assets/0eb62d38-2347-4569-a2f7-04dca17eab32" />

<img width="1048" height="197" alt="checkpoint1 png" src="https://github.com/user-attachments/assets/763ce30c-70df-42c6-b246-50527806cf21" />

<img width="1051" height="304" alt="checkpoint2" src="https://github.com/user-attachments/assets/a5b73243-b3d9-4763-bbfe-9481d730906e" />


## Compilación

```bash
# Suma de vectores (N=16M fijo)
nvcc -O2 -o vectorAdd src/vectorAdd.cu
./vectorAdd

# Suma de vectores (N=1M, 4M, 16M comparativo)
nvcc -O2 -o vectorAddSizes src/vectorAddSizes.cu
./vectorAddSizes

# Multiplicación de matrices
nvcc -O2 -o matMul src/matMul.cu
./matMul
```
