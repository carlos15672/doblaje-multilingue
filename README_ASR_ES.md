# Reconocedor de español: primer avance

El objetivo actual es convertir **una grabación de voz en español en texto español**.
No incluye traducción, síntesis de voz, clonación ni sincronización labial.
La inferencia segmenta grabaciones largas en regiones de voz de hasta 25 segundos con solapamiento.

## Qué ya existía y qué se conserva

- `app_interfaz.py`, `app2.py` y los demás scripts originales: referencia del clasificador de vocales.
- 222 WAV de vocales, incluyendo variaciones de velocidad y tono, y un CSV de 25 características.
- Un clasificador Random Forest y un escalador guardados. No se utilizan en el reconocedor nuevo.
- El entorno `venv/` venía de Linux; para Windows se creó `.venv_asr/`.

Las vocales sirven para verificar cálculos, no para entrenar un transcriptor de frases.
No se conocen los identificadores de hablante de ese conjunto: no deben inventarse para
presentar una evaluación de generalización a personas nuevas.

## Estado real de esta entrega

- Implementados lectura/grabación, procesamiento propio, CRNN, importación de Common Voice,
  entrenamiento, guardado/carga, decodificación básica y cálculo de WER/CER.
- El entrenamiento parte de pesos aleatorios. Hay un checkpoint propio en
  `actualizacion/modelo/best.pt`, entrenado sin servicios ni pesos ASR externos.
- Las pruebas sintéticas comprueban el funcionamiento del código, no la calidad del reconocimiento.
- El micrófono se activa únicamente al ejecutar `record`; no se ha grabado al usuario automáticamente.
- La interfaz gráfica original sigue mostrando vocales. El nuevo avance se maneja desde terminal.

## Arquitectura y aportación propia

```text
Audio mono a 16 kHz
  -> quitar componente continua
  -> ventanas Hann de 25 ms, avance de 10 ms
  -> FFT radix-2 propia de 512 puntos
  -> potencia -> 40 filtros Mel triangulares -> logaritmo
  -> normalización de la grabación
  -> Conv2D (16 canales, reducción temporal x2 y frecuencial x2)
  -> Conv2D (32 canales, reducción frecuencial x2)
  -> proyección de 128 características
  -> BiGRU de 2 capas, 128 unidades por dirección
  -> salida de caracteres españoles + espacio + blank
  -> decodificación CTC greedy propia
```

La red tiene **549,731 parámetros** en su configuración inicial. El tamaño es un punto de partida
para experimentar, no una garantía de precisión. La BiGRU usa el contexto anterior y posterior,
por lo que esta versión espera a que termine la grabación.

| Componente | Implementación |
|---|---|
| Captura | `sounddevice` para acceso al micrófono |
| Lectura de formatos | Biblioteca estándar `wave` para PCM; `soundfile` para MP3 y otros formatos |
| Conversión a mono, remuestreo con filtro antialias, ventanas, FFT, filtros Mel, log y normalización | Código propio en `asr_es/features.py` y `audio.py`, con operaciones de NumPy |
| Capas Conv2D, GRU, normalización, gradientes, Adam y pérdida CTC | Implementaciones genéricas de PyTorch, declaradas explícitamente |
| Arquitectura, preparación de datos, ciclo de entrenamiento, decodificación greedy y WER/CER | Código del proyecto |
| Pesos preentrenados / servicios de transcripción | Ninguno |

Usar PyTorch significa que **el entrenamiento es desde cero, pero no todos los algoritmos internos
de aprendizaje están implementados manualmente**. Esta distinción debe figurar en el avance al profesor.
El uso de capas y CTC genéricos sigue la aclaración de que se permiten bibliotecas de apoyo.

## Empezar en esta computadora

En PowerShell, desde `C:\Users\PC\Downloads\vocales`:

```powershell
.\.venv_asr\Scripts\python.exe -m asr_es diagnose
.\.venv_asr\Scripts\python.exe -m asr_es analyze pruebas\A.wav
```

Se detectó PyTorch 2.6.0 con CUDA 12.4, NumPy 2.2.6 y una NVIDIA RTX 4060 de 8 GiB.
El entorno local reutiliza NumPy y PyTorch ya instalados y añade las dependencias de audio.
No es necesario activar el entorno para ejecutar estos comandos.

Grabar cinco segundos, cuando estén listos para hablar:

```powershell
.\.venv_asr\Scripts\python.exe -m asr_es record data_asr\propios\prueba_01.wav --seconds 5
.\.venv_asr\Scripts\python.exe -m asr_es analyze data_asr\propios\prueba_01.wav
```

Si no hay dispositivo predeterminado, seleccionen un micrófono en la configuración de Windows.
`analyze` solo calcula características. Para transcribir se usa `transcribe` con el checkpoint propio.

Para otra computadora, creen un entorno nuevo y utilicen `requirements_asr.txt`; la instalación
de PyTorch debe corresponder al equipo y, cuando proceda, a su GPU. No copien `venv/` ni `.venv_asr/`.

## Datos para entrenar

Common Voice ofrece audio y texto, no pesos de un reconocedor. Opciones revisadas:

- [Español mexicano, voces masculinas, Common Voice 26.0](https://mozilladatacollective.com/datasets/cmr3jjevj0041nt07vb3sm35d): aproximadamente 100 horas, descarga de 2.02 GB.
- [Español mexicano, voces femeninas, Common Voice 26.0](https://mozilladatacollective.com/datasets/cmr3zunmk00flnt07p7ai9p43): aproximadamente 84 horas, descarga de 1.65 GB.

Las fichas indican CC0. Descargar desde Mozilla Data Collective con la cuenta del usuario y respetar
las condiciones de acceso y distribución. No subir el corpus al repositorio. El código no descarga nada
ni envía audio a servidores. Estas grabaciones son de habla leída y no garantizan buen desempeño
en conversación espontánea, ruido o acentos no representados.

Después de extraer ambos archivos, localizar en cada uno la carpeta que contenga:

```text
clips/
train.tsv
dev.tsv
test.tsv
```

Si los tres TSV se descargaron o entregaron por separado, también se pueden importar sin moverlos.
Los archivos contienen metadatos y transcripciones; los MP3 siguen siendo necesarios y deben estar
en la carpeta `clips/` del mismo lanzamiento de Common Voice:

```powershell
.\.venv_asr\Scripts\python.exe -m asr_es prepare-cv `
  --train-tsv C:\ruta\train.tsv `
  --dev-tsv C:\ruta\dev.tsv `
  --test-tsv C:\ruta\test.tsv `
  --clips C:\ruta\clips `
  --output data_asr\manifiestos_completos
```

El importador no confunde los TSV con el audio: si falta `clips/`, se detiene antes de crear
manifiestos inutilizables. Las rutas de la salida apuntan a los MP3 y, por ello, no se debe mover
la carpeta de clips después de preparar el corpus.

Ejemplo, si se colocan respectivamente en `data_asr/cv_mx_male` y `data_asr/cv_mx_female`:

```powershell
.\.venv_asr\Scripts\python.exe -m asr_es prepare-cv data_asr\cv_mx_male data_asr\cv_mx_female --output data_asr\manifiestos_piloto --limit 500
.\.venv_asr\Scripts\python.exe -m asr_es audit --train data_asr\manifiestos_piloto\train.csv --dev data_asr\manifiestos_piloto\dev.csv --test data_asr\manifiestos_piloto\test.csv
```

`--limit 500` selecciona como máximo 500 clips por partición con semilla fija. Es un piloto de
desarrollo, no una cantidad suficiente para afirmar reconocimiento general. Para ampliar, importen
a otra carpeta omitiendo el límite. No mezclen distintas versiones con las mismas grabaciones.
Si detecta hablantes entre particiones, el importador se detiene: revisar y corregir la separación,
sin cambiar arbitrariamente los identificadores para saltar la comprobación.

El importador genera un informe de los ejemplos rechazados. Conserva ñ, tildes y ü; normaliza
mayúsculas, espacios y puntuación. **Rechaza números escritos con dígitos y símbolos desconocidos**:
no elimina números para fabricar una transcripción incorrecta. Una normalización más completa de
números/abreviaturas queda pendiente. Los conjuntos de prueba no se usan para ajustar parámetros.

También pueden usar CSV propios en UTF-8 con columnas `audio,text,speaker`. Las rutas se resuelven
respecto a la carpeta del CSV; `speaker` debe identificar a la persona real mediante un seudónimo
estable. El texto debe corresponder exactamente a lo pronunciado. Separar personas entre train,
dev y test; mantener las variaciones de una grabación en la misma partición. Para comprobar frases
nuevas, revisar además las coincidencias textuales que informa `audit`.

## Entrenamiento piloto, después de obtener los datos

```powershell
.\.venv_asr\Scripts\python.exe -m asr_es train --train data_asr\manifiestos_piloto\train.csv --dev data_asr\manifiestos_piloto\dev.csv --output runs_asr\piloto_01 --epochs 10 --batch-size 8 --device cuda
```

Es una prueba del flujo, no una promesa de convergencia en diez épocas. Ajustar lote y duración
tras medir consumo de memoria. Si falla por memoria, probar `--batch-size 4` en una carpeta nueva.
El programa prepara/cachea características, rechaza secuencias imposibles para CTC, verifica pérdidas
y gradientes finitos y guarda el mejor checkpoint según WER de validación (desempate por pérdida).

Salidas: `config.json`, `history.jsonl`, `best.pt` y `dev_predictions.json`. No sobrescribe una
carpeta de entrenamiento existente. La caché usa espacio en disco proporcional a los datos procesados.
La implementación actual guarda `last.pt` y permite reanudar modelo, optimizador y orden de datos
con `--resume`. También registra hashes de los manifiestos para detectar cambios silenciosos.

Transcribir con VAD, segmentos solapados, prefix beam search, confianza y timestamps:

```powershell
.\.venv_asr\Scripts\python.exe -m asr_es transcribe data_asr\propios\prueba_01.wav --checkpoint actualizacion\modelo\best.pt --beam-width 16
.\.venv_asr\Scripts\python.exe -m asr_es evaluate --checkpoint runs_asr\piloto_01\best.pt --train data_asr\manifiestos_piloto\train.csv --dev data_asr\manifiestos_piloto\dev.csv --test data_asr\manifiestos_piloto\test.csv --output runs_asr\piloto_01\test.json
```

Opcionalmente, construir un modelo de lenguaje bigrama exclusivamente con el texto de entrenamiento:

```powershell
.\.venv_asr\Scripts\python.exe -m asr_es build-lm data_asr\manifiestos_piloto\train.csv --output runs_asr\lm_train.json --order 2
.\.venv_asr\Scripts\python.exe -m asr_es transcribe data_asr\propios\prueba_01.wav --checkpoint actualizacion\modelo\best.pt --lm runs_asr\lm_train.json
.\.venv_asr\Scripts\python.exe -m asr_es evaluate --checkpoint actualizacion\modelo\best.pt --train data_asr\manifiestos_piloto\train.csv --dev data_asr\manifiestos_piloto\dev.csv --test data_asr\manifiestos_piloto\test.csv --output runs_asr\evaluacion_beam.json --beam-width 16 --lm runs_asr\lm_train.json
```

Para nuevos experimentos, `train --augment` activa ruido, ganancia, variación de velocidad y
enmascaramiento temporal/frecuencial propios. Para adaptar el mejor checkpoint a nuevos datos sin
perder sus pesos, se usa `--init-checkpoint`; esto crea una corrida nueva con optimizador nuevo.
Los manifiestos admiten columnas opcionales `accent`, `environment` y `microphone`; `evaluate`
genera WER/CER por esas condiciones y por duración.

### Ruta principal: adaptar el champion a español mexicano

El champion permanece en `actualizacion/modelo/best.pt`. Su línea base congelada sobre el test
mexicano es 51.51% WER con greedy y 50.17% con el beam/LM seleccionado en dev. Una corrida nueva
debe partir de esos pesos y no reemplazar el archivo hasta superar 50.17% en el mismo test:

```powershell
.\.venv_asr\Scripts\python.exe -m asr_es train `
  --train data_asr\manifiestos_mx\train.csv `
  --dev data_asr\manifiestos_mx\dev.csv `
  --output runs_asr\mx_champion_ft_clean `
  --epochs 3 --batch-size 32 --learning-rate 0.0001 `
  --init-checkpoint actualizacion\modelo\best.pt `
  --device cuda --cache data_asr\cache_mx_champion --workers 8
```

`--init-checkpoint` copia únicamente los pesos y registra ruta, hash, época y métricas de origen;
Adam comienza de nuevo porque el checkpoint de inferencia no contiene estado del optimizador.
`--resume` se reserva para continuar posteriormente esa misma carpeta y exige los mismos datos y
parámetros. No se debe consultar test para escoger épocas, tasa de aprendizaje ni decodificador.

WER = (sustituciones + omisiones + inserciones de palabras) / palabras de referencia.
CER usa caracteres y en esta implementación incluye espacios. Se calculan con sumas de todo el
corpus; WER puede superar 100% si hay muchas inserciones. Reportar siempre tamaño del conjunto,
hablantes y condiciones, junto a los errores. El resultado del modelo es una hipótesis, no certeza.

## Plan para dos personas y avances semanales

### Primer avance: martes 13 de octubre de 2026

Meta: demostrar el alcance español -> texto, captura, características temporales, arquitectura,
ausencia de pesos preentrenados y pruebas reproducibles. Explicar qué falta para medir precisión real.

- Persona 1: micrófono, procesamiento, explicación de FFT/Mel y revisión de grabaciones/transcripciones.
- Persona 2: arquitectura, importación/particiones, pruebas de entrenamiento y registro de experimentos.
- Ambos: preparar la demostración, revisar las reglas con la tabla de dependencias y entender el flujo completo.

Si el avance es miércoles 14 o viernes 16, añadir importación del corpus y entrenamiento piloto
si la descarga y preparación están listas. No presentar las vocales ni pruebas sintéticas como ASR general.

### Siguiente avance

Entrenar con frases reales, observar ejemplos y errores, verificar que aprende un conjunto pequeño
y luego medir generalización en validación. Ampliar datos de forma progresiva según los tiempos medidos.

### Avance posterior

Comparar configuraciones CRNN con las mismas particiones y presupuesto; calibrar VAD/confianza,
beam search y aumentos sobre ruido y personas nuevas. Integrar después la inferencia en la interfaz.

## Pruebas

```powershell
.\.venv_asr\Scripts\python.exe -m unittest discover -s tests_asr -v
```

Verifican FFT frente a una referencia independiente, rechazo de aliasing, comportamiento de audio
corto/silencioso, caracteres españoles, colapso CTC, separación de hablantes, métricas y efecto del
relleno en la BiGRU. Incluyen entrenamiento/carga/inferencia con señales sintéticas.
El uso de `np.fft` se limita a la referencia de prueba; el procesamiento de producción usa FFT propia.

La suite actual incluye 22 pruebas para acústica, VAD, aumentos, beam search, n-gram, timestamps,
entrenamiento, inicialización desde checkpoint, reanudación y carga. La prueba CUDA se omite automáticamente
cuando no hay GPU disponible. Las pruebas de infraestructura no sustituyen una evaluación real de calidad.

## Referencias de diseño

- [Deep Speech 2: convoluciones + recurrencia + CTC](https://proceedings.mlr.press/v48/amodei16.pdf).
- [CTC: entrenamiento con secuencias sin alineación temporal](https://www.cs.toronto.edu/~graves/icml_2006.pdf).

La arquitectura es una versión pequeña inspirada en esa familia; no reproduce los resultados publicados.
