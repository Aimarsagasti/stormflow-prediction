# WORKFLOW.md - Cómo trabajar con el proyecto

Guía operativa del flujo de trabajo del proyecto stormflow-prediction. Referencia para cualquier sesión nueva.

---

## Rutas importantes

### Local (Windows)
- Repo local: `C:\Dev\TFM\`
- Código fuente: `C:\Dev\TFM\src\`
- Datos brutos: `C:\Dev\TFM\MC-CL-005\` (NO en GitHub)
- Pesos descargados de Drive: `C:\Dev\TFM\MC-CL-005\Pesos 13-04-2026\`
- Plots locales: `C:\Dev\TFM\outputs\figures\local_eval\`
- Métricas locales: `C:\Dev\TFM\outputs\data_analysis\`

### Colab (Linux)
- Código del proyecto (tras git clone): `/content/stormflow-prediction/`
- Datos brutos en Drive (cuenta con los datos): `/content/drive/.shortcut-targets-by-id/1xGRwVQHSkN11f9PxxArnsSTGyCyXuGym/MC-CL-005/`
- Pesos guardados en Drive (cuenta Colab Pro): `/content/drive/MyDrive/Proyecto de capstone/Archivos del proyecto/Pesos 13 04 2026/` (OJO: con espacios, no guiones).

### GitHub
- URL: https://github.com/Aimarsagasti/stormflow-prediction
- Visibilidad: pública.

### Aviso importante sobre nombres
- Carpeta de pesos en LOCAL: `Pesos 13-04-2026` (con guiones).
- Carpeta de pesos en DRIVE: `Pesos 13 04 2026` (con espacios).
- Los scripts Python manejan la diferencia internamente pero hay que tenerlo presente.

---

## Flujo estándar de una iteración

### 1. Decidir hipótesis
- Un cambio concreto con hipótesis clara (por ejemplo: "probar X feature porque creo que Y").
- Si no hay hipótesis, no hay iteración. Documentar en notas antes de tocar código.

### 2. Modificar código en VS Code local
- Rama `main`.
- Uno o varios archivos en `src/` según la hipótesis.
- Máximo 3 archivos por iteración.

### 3. Commit y push

En Git Bash o terminal de VS Code, dentro de `C:\Dev\TFM\`:

```bash
git status                                        # Ver qué cambió
git add <archivos específicos>                    # Añadir al staging
git commit -m "exp(iterN): descripción breve"     # Commit con mensaje útil
git push                                          # Subir a GitHub
```

**Formato obligatorio de mensajes de commit:**
- `exp(iterN): descripción` → para experimentos nuevos.
- `fix(modulo): descripción` → para correcciones de bugs.
- `refactor(modulo): descripción` → para reorganización sin cambio funcional.
- `docs: descripción` → para cambios en docs/CLAUDE.md/AGENTS.md/README.
- `chore: descripción` → para tareas de mantenimiento.

**Prohibido:** mensajes como "Resultados primer entrenamiento" o "Actualización". Son inútiles para el historial.

### 4. En Colab: git pull

Al inicio de la sesión de Colab:

```python
# Celda 0 del notebook: clonar o actualizar el repo
!cd /content && git clone https://github.com/Aimarsagasti/stormflow-prediction.git 2>/dev/null || cd /content/stormflow-prediction && git pull
```

### 5. Corregir rutas del YAML (celda específica)

Colab Pro tiene una cuenta distinta de la que tiene los datos. La celda 1 (o 3, según el notebook) del notebook debe sobrescribir las rutas del `configs/default.yaml` para apuntar al shortcut de Drive:

```python
# Esta celda es imprescindible al inicio de cada sesión en Colab Pro
import yaml

config_path = '/content/stormflow-prediction/configs/default.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

base = '/content/drive/.shortcut-targets-by-id/1xGRwVQHSkN11f9PxxArnsSTGyCyXuGym/MC-CL-005'
config['data']['base_paths'] = [f'{base}/1parte/', f'{base}/2parte/']
config['data']['temperature_daily_path'] = f'{base}/daily_temperatures_2006_2026.tsf'

with open(config_path, 'w') as f:
    yaml.safe_dump(config, f)
```

### 6. Entrenar en Colab

Ejecutar todas las celdas del notebook hasta la evaluación.

### 7. Guardar pesos en Drive

El notebook debe guardar la tripleta `{weights.pt, norm_params.json, meta.json}` en la carpeta de pesos en Drive. Nombre del modelo debe seguir el patrón: `modelo_H{N}_{conSF|sinSF}_{variante}_{weights.pt|norm_params.json|meta.json}`.

### 8. Descargar pesos al local

Opciones:
- Manual: desde el navegador de Google Drive.
- Automático: sincronización de Google Drive Desktop con la carpeta local `MC-CL-005/Pesos 13-04-2026/`.

### 9. Ejecutar eval local

En terminal local:

```bash
cd C:/Dev/TFM
python evaluate_local.py
```

Este script genera los plots y las métricas en `outputs/figures/local_eval/` y `outputs/data_analysis/local_eval_metrics.json`.

Troubleshooting común:
- Si falla un import de `src/`: asegurarse de estar en `C:\Dev\TFM\` al ejecutar.
- Si va muy lento: BATCH_SIZE=512 en CPU puede ser excesivo. Bajar a 256 en el script.

### 10. Documentar en `EXPERIMENTS.md`

Añadir al final del archivo una nueva entrada con el formato estándar (hipótesis, cambio, resultado, lección).

### 11. Actualizar `STATE.md`

Actualizar:
- Si es el nuevo mejor modelo: la sección "Modelo actual en producción".
- La sección "Resultados del último eval".
- La sección "Siguientes pasos" con lo que toque hacer después.
- La fecha de "Última actualización" al inicio del archivo.

### 12. Commit final de documentación

```bash
git add docs/EXPERIMENTS.md docs/STATE.md
git commit -m "docs: actualizar estado post-iterN"
git push
```

---

## Comandos Git útiles

### Estado y sincronización

```bash
git status                          # Ver archivos modificados y estado de sincronización
git fetch --all                     # Traer info de GitHub sin modificar nada local
git pull                            # Traer cambios de GitHub y aplicarlos
git push                            # Subir commits locales a GitHub
git log --oneline -10               # Ver últimos 10 commits en formato compacto
```

### Modificaciones

```bash
git add <archivo>                   # Marcar un archivo específico para el próximo commit
git add .                           # Marcar TODOS los archivos modificados (cuidado)
git commit -m "mensaje"             # Crear commit con mensaje
git commit --amend                  # Modificar el último commit (antes de push)
```

### Ver diferencias

```bash
git diff                            # Ver cambios no commiteados
git diff --staged                   # Ver cambios marcados para el próximo commit
git show HEAD                       # Ver el último commit completo
git show <hash>                     # Ver un commit específico
```

### Recuperar archivos

```bash
git checkout <archivo>              # Descartar cambios locales en un archivo
git restore <archivo>               # Equivalente moderno de checkout
git reset HEAD <archivo>            # Quitar un archivo del staging sin perder cambios
```

---

## Errores conocidos y soluciones

### Error de codificación en scripts de Python en Windows

Síntoma: `UnicodeEncodeError: 'charmap' codec can't encode characters`.

Solución: al inicio del script añadir:

```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

### `py: command not found` en Git Bash

Git Bash no conoce el launcher `py` de Windows. Usar `python` directamente.

Si `python` tampoco funciona, usar `python.exe` o abrir una terminal PowerShell.

### Colab: `FileNotFoundError` al leer YAML

Se olvidó ejecutar la celda de corrección de rutas. Las rutas del YAML apuntan a la cuenta de datos pero Colab Pro es otra cuenta. Ejecutar la celda 1 (ver sección 5 arriba).

### Bug de doble normalización de stormflow

Si `stormflow_mgd` aparece TANTO en FEATURES como en TARGET_COL, `normalize_splits` lo normaliza dos veces. Parche obligatorio:

```python
FEATURES_FOR_NORM = [f for f in FEATURES if f != 'stormflow_mgd']
df_tn, df_vn, df_tsn, npar = normalize_splits(
    df_tr, df_va, df_te, FEATURES_FOR_NORM, TARGET_COL
)
tl, vl, tstl = create_dataloaders(
    df_tn, df_vn, df_tsn, FEATURES, TARGET_COL, AUX_COL, ...
)
```

Sin este parche, el target desnormalizado da valores absurdos (52,466 MGD en vez de 135 MGD).

---

## Limitaciones de Colab Pro

- RAM: 12.7 GB. El dataset (1.1M filas × 23 cols) + DataLoaders consume mucho.
- Estrategias implementadas para controlarlo:
  - `float32` en vez de `float64`.
  - Eliminar DataFrames intermedios después de uso.
  - Para múltiples entrenamientos consecutivos: reiniciar runtime entre cada uno si la RAM se satura.
- GPU: T4. Suficiente para TCN con seq_length=72 y batch=256.
- Tiempo máximo de sesión: ~12 horas de cómputo con Pro. Guardar pesos en Drive periódicamente.

---

## Checklist rápido al final de cada sesión

- [ ] Commit de todos los cambios de código.
- [ ] Push a GitHub.
- [ ] Pesos guardados en Drive (si se entrenó).
- [ ] Pesos descargados a local (si se entrenó).
- [ ] `evaluate_local.py` ejecutado (si se entrenó).
- [ ] Entrada nueva en `docs/EXPERIMENTS.md` (si se iteró).
- [ ] `docs/STATE.md` actualizado con el nuevo estado.
- [ ] Commit y push final con `docs: ...`.