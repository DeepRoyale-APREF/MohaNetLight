# MohaNetLight

Red neuronal ligera inspirada en AlphaStar, especializada para Clash Royale Arena 1.
Entrenamiento por PPO con cabezas jerárquicas autoregresivas y evaluación continua mediante liga contra bots heurísticos.

**~1.6M parámetros** — entrenable en GPU doméstica o Google Colab gratuito.

> **Nota:** Este proyecto depende de otros dos repositorios del ecosistema DeepRoyale-APREF — **cr-engine** y **cr-gym** — que **no están publicados en PyPI** y deben clonarse manualmente desde GitHub. El proceso completo se describe en la sección de instalación a continuación.

---

## Ecosistema de paquetes

El proyecto está compuesto por tres repositorios públicos que deben instalarse en orden:

| Paquete | Repositorio | Descripción |
|---------|-------------|-------------|
| `clash-royale-engine` | [DeepRoyale-APREF/cr-engine](https://github.com/DeepRoyale-APREF/cr-engine) | Simulador headless de Clash Royale Arena 1 — física, combate, entidades, GUI |
| `clash-royale-gymnasium` | [DeepRoyale-APREF/cr-gym](https://github.com/DeepRoyale-APREF/cr-gym) | Wrapper Gymnasium con recompensas, máscaras de acción y sistema de liga |
| `mohanetlight` | [DeepRoyale-APREF/MohaNetlight](https://github.com/DeepRoyale-APREF/MohaNetlight) | Red neuronal + PPO + bots heurísticos (este repositorio) |

---

## Instalación

### Opción A — Entorno local (recomendado para desarrollo)

#### 1. Crear entorno Conda

```bash
conda create -n cr-engine python=3.12 -y
conda activate cr-engine
```

> Puedes usar `venv` si prefieres, pero Conda facilita aislar versiones de CUDA.

#### 2. Instalar PyTorch

Instala **primero** PyTorch con el backend correcto para tu hardware.
Si lo instalas después, pip puede descargar la versión CPU-only automáticamente.

```bash
# CUDA 12.4 (Linux / Windows con NVIDIA — recomendado para entrenamiento)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Solo CPU (para pruebas rápidas sin GPU)
pip install torch

# macOS con Apple Silicon (MPS ya incluido)
pip install torch
```

Verifica que CUDA esté disponible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

#### 3. Clonar e instalar los tres repositorios

Los tres repos deben quedar en el mismo directorio padre o donde prefieras.
El orden importa: primero el motor, luego el entorno, luego la red.

```bash
# Clonar los tres repos
git clone https://github.com/DeepRoyale-APREF/cr-engine.git
git clone https://github.com/DeepRoyale-APREF/cr-gym.git
git clone https://github.com/DeepRoyale-APREF/MohaNetlight.git

# Instalar en modo editable (los cambios en el código se reflejan sin reinstalar)
pip install -e cr-engine/
pip install -e "cr-gym/[dev]"
pip install -e "MohaNetlight/[dev]"
```

El flag `-e` ("editable") significa que Python carga el código directamente desde el
directorio clonado. Cualquier cambio que hagas en el código fuente tiene efecto
inmediato sin necesidad de reinstalar.

#### 4. Verificar la instalación

```bash
python -c "
import clash_royale_engine, clash_royale_gymnasium, mohanetlight
from mohanetlight import MohaNetLight, ModelConfig
model = MohaNetLight(ModelConfig())
print(f'Todo OK — Parámetros del modelo: {model.count_parameters():,}')
"
# Salida esperada: Todo OK — Parámetros del modelo: 1,615,355
```

---

### Opción B — Google Colab

Abre el notebook incluido que automatiza toda la configuración:

```
notebooks/colab_training.ipynb
```

Colab ya incluye PyTorch con CUDA. El notebook ejecuta automáticamente:

```python
# Dentro del notebook (ya incluido)
!git clone https://github.com/DeepRoyale-APREF/cr-engine.git
!git clone https://github.com/DeepRoyale-APREF/cr-gym.git
!git clone https://github.com/DeepRoyale-APREF/MohaNetlight.git

!pip install -e cr-engine/
!pip install -e "cr-gym/[dev]"
!pip install -e "MohaNetlight/[dev]"
```

> En Colab, una celda de instalación suele tardar 60–90 segundos.
> Tras instalar, reinicia el runtime (*Runtime → Restart session*) y ejecuta
> las celdas de entrenamiento.

---

### Actualizar los repos

Si los repos ya están clonados y quieres obtener los últimos cambios:

```bash
cd cr-engine    && git pull && cd ..
cd cr-gym       && git pull && cd ..
cd MohaNetlight && git pull && cd ..
```

No es necesario reinstalar si usaste `-e`: los cambios en el código son inmediatos.

---

## Estructura del proyecto

```
MohaNetlight/
├── mohanetlight/
│   ├── __init__.py              # Exporta MohaNetLight, ModelConfig, TrainingConfig
│   ├── config.py                # ModelConfig (arquitectura) + TrainingConfig (PPO)
│   ├── network/
│   │   ├── encoders.py          # ScalarEncoder, EntityEncoder, CardEncoder
│   │   ├── core.py              # LSTMCore (384→256, 2 capas)
│   │   ├── heads.py             # Cabezas jerárquicas + ValueHead
│   │   └── mohanet.py           # MohaNetLight (modelo ensamblado)
│   ├── inference/
│   │   └── agent.py             # MohaNetAgent (PlayerSlot para liga)
│   ├── bots/
│   │   └── strategies.py        # 5 bots heurísticos parametrizados + default_bot_roster
│   ├── training/
│   │   ├── rollout.py           # RolloutBuffer con estados LSTM
│   │   ├── ppo.py               # PPOTrainer (clipped surrogate + BPTT truncado)
│   │   ├── trainer.py           # LeagueTrainer (orquestador completo)
│   │   └── reward_debugger.py   # RewardDebugger — diagnóstico de señales de recompensa
│   └── utils/
│       └── tensor_utils.py      # Conversión obs→tensores
├── scripts/
│   ├── train_league.py          # Entrenamiento PPO por CLI
│   ├── watch_agent.py           # Depurador visual — GUI + overlay de recompensas
│   └── diagnose_rewards.py      # Diagnóstico headless de señales de recompensa
├── notebooks/
│   └── colab_training.ipynb     # Configuración + entrenamiento en Google Colab
├── tests/
│   └── test_network.py          # 35 tests (shapes, forward pass, buffer, bots)
├── pyproject.toml
└── README.md
```

---

## Arquitectura

```
┌─────────────┐  ┌──────────────────┐  ┌─────────────┐
│   Scalars   │  │     Entities     │  │    Cards    │
│  MLP 16→128 │  │ Transformer L=2  │  │  MLP 5→128  │
│             │  │  H=4, d=64       │  │  (×8 cards) │
│             │  │ + Pos. Encoding   │  │             │
└──────┬──────┘  └────────┬─────────┘  └──────┬──────┘
       │                  │                    │
       └──────────┬───────┴────────────────────┘
                  │ concat → 384
          ┌───────▼───────┐
          │   LSTM Core   │
          │  384→256, L=2 │
          └───────┬───────┘
                  │ 256
    ┌─────────────┼─────────────────────────┐
    │             │                         │
    │   ┌─────────▼──────────┐              │
    │   │  Card Head         │←── entity ctx│
    │   │  384→128→9         │              │
    │   └────────┬───────────┘              │
    │            │ embed(9→64)              │
    │   ┌────────▼──────────┐               │
    │   │  Tile X Head      │←── entity ctx │
    │   │  448→128→18       │               │
    │   └────────┬──────────┘               │
    │            │ embed(18→64)             │
    │   ┌────────▼──────────┐               │
    │   │  Tile Y Head      │←── entity ctx │
    │   │  448→128→32       │               │
    │   └────────┬──────────┘               │
    │            │                          │
    │   ┌────────▼──────────┐    ┌──────────▼───┐
    │   │  Acciones (π)     │    │  Value Head  │
    │   │  {card, x, y}     │    │  256→128→1   │
    │   └───────────────────┘    └──────────────┘
```

### Componentes

| Módulo | Descripción | Parámetros |
|--------|-------------|------------|
| `ScalarEncoder` | MLP 16→64→128 (elixir, torres, tiempo, flags) | ~9K |
| `EntityEncoder` | Transformer 2 capas, 4 cabezas, d=64 + codificación posicional | ~112K |
| `CardEncoder` | MLP compartido por carta 5→32, agregado 256→128 | ~17K |
| `LSTMCore` | LSTM(384→256, 2 capas, dropout=0.1) | ~1.18M |
| Cabezas jerárquicas | Card→TileX→TileY con embeddings autorregresivos | ~200K |
| `ValueHead` | MLP 256→128→64→1 (crítico) | ~41K |
| **Total** | | **~1.56M** |

### Cabezas autorregresivas

Cada cabeza recibe como contexto adicional el **embedding de la acción muestreada** de la cabeza anterior, más el **contexto de entidades** (skip connection del encoder):

1. **Card** ← salida LSTM + entity_ctx
2. **Tile X** ← LSTM + embed(card) + entity_ctx  
3. **Tile Y** ← LSTM + embed(tile_x) + entity_ctx

---

## Uso rápido

### Detección automática de dispositivo

Al instanciar `TrainingConfig()`, el dispositivo se auto-detecta (CUDA > MPS > CPU):

```python
from mohanetlight.config import TrainingConfig

tc = TrainingConfig()
# [MohaNetLight] Using CUDA device: NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

Para forzar un dispositivo:

```python
tc = TrainingConfig(device="cpu")
```

### Inferencia con el modelo

```python
import torch
from mohanetlight import MohaNetLight, ModelConfig

model = MohaNetLight(ModelConfig())
model.eval()

B = 1
scalars    = torch.randn(B, 16)
troops     = torch.randn(B, 100, 14).abs()
troop_mask = torch.zeros(B, 100, dtype=torch.bool)
troop_mask[0, :5] = True
cards      = torch.randn(B, 4, 4).abs()
action_masks = {
    "card":     torch.ones(B, 9,  dtype=torch.bool),
    "tile_x_per_card": torch.ones(B, 9, 18, dtype=torch.bool),
    "tile_y_per_card": torch.ones(B, 9, 32, dtype=torch.bool),
}

hidden = model.init_hidden(B)
output = model.act(scalars, troops, troop_mask, cards, action_masks, hidden)

print(output.actions)   # {card, tile_x, tile_y} — cada uno Tensor(B,)
print(output.value)     # V(s) — estimación del crítico
print(output.log_prob)  # log π(a|s)
```

### Entrenamiento PPO con liga

```python
from mohanetlight.config import ModelConfig, TrainingConfig
from mohanetlight.training.trainer import LeagueTrainer

trainer = LeagueTrainer(
    model_cfg=ModelConfig(),
    train_cfg=TrainingConfig(
        total_timesteps=1_000_000,
        log_dir="./logs/mi_experimento",
    ),
)
trainer.train()
```

### Desde línea de comandos

```bash
# Entrenamiento completo
python scripts/train_league.py --total-steps 1000000 --device cuda

# Entrenamiento rápido de prueba
python scripts/train_league.py --total-steps 50000 --eval-interval 5

# Ver todas las opciones
python scripts/train_league.py --help
```

### Agente para liga cr-gym

```python
from mohanetlight.inference.agent import MohaNetAgent
from clash_royale_gymnasium.league.tournament import LeagueTournament
from clash_royale_gymnasium.league.player_slot import HeuristicSlot

# Cargar desde checkpoint
agent = MohaNetAgent.from_checkpoint(
    "logs/mohanet/model_final.pt",
    name="MohaNet-v1",
    device="cuda",
)

# Torneo contra un bot heurístico
league = LeagueTournament(
    players=[agent, HeuristicSlot("Bot", aggression=0.5)],
    matches_per_pair=10,
)
results = league.run()
```

---

## Herramientas de depuración

El proyecto incluye dos scripts para diagnosticar el entrenamiento y validar
visualmente que la simulación funciona correctamente.

### Depurador visual — `watch_agent.py`

Ejecuta una partida completa con la GUI de Clash Royale y muestra un panel
lateral con información en tiempo real: recompensas por componente, acción
del agente, estimación del crítico V(s), sparkline de historial de recompensa
y señales de depuración del motor.

```bash
# Ver modelo aleatorio (sin checkpoint) vs bot Balanced
python scripts/watch_agent.py

# Ver modelo entrenado desde checkpoint
python scripts/watch_agent.py --checkpoint logs/mohanet/model_final.pt

# Elegir oponente
python scripts/watch_agent.py --opponent GiantPush
# Opciones: GiantPush, BridgeSpam, SpellCycle, DefCounter, Balanced

# Modo determinista (argmax en vez de muestreo)
python scripts/watch_agent.py --checkpoint model.pt --deterministic

# Sin música de fondo
python scripts/watch_agent.py --no-music
```

> **¿Por qué `frame_skip` en vez de reducir `fps`?**
> El motor físico está calibrado a 30 fps — las velocidades de tropas,
> los ticks de daño y las colisiones asumen ~33 ms por frame.
> Reducir `fps` a 10 haría que las tropas se "teleportasen" distancias más
> grandes y el daño se aplicara de forma brusca.
> `frame_skip=3` mantiene la física a 30 fps (simulación precisa) y solo
> pide una decisión RL cada 3 frames (10 decisiones/segundo).
> El script visual usa `frame_skip=1` para mostrar cada frame a velocidad real.

El panel lateral muestra:

| Sección | Información |
|---------|-------------|
| **MATCH INFO** | Paso, tiempo transcurrido, fase del juego, elixir |
| **AGENT ACTION** | Estrategia, carta, tile destino, si la acción fue válida, V(s) |
| **REWARD** | Recompensa del paso, acumulada, episodios completados |
| **Componentes** | Barras de cada componente (Damage, Elixir, Terminal, Strategy) |
| **REWARD HISTORY** | Sparkline de los últimos 200 pasos |
| **TOWER HP** | HP de las 6 torres con colores por ratio |
| **ENGINE SIGNALS** | Daño infligido/recibido, tropas en campo |
| **HAND** | Cartas en mano (verdes = pagables) |

### Diagnóstico headless — `diagnose_rewards.py`

Ejecuta 2560 pasos sin GUI e imprime estadísticas detalladas del pipeline
de recompensas. Útil para verificar señales de recompensa sin necesitar pantalla.

```bash
python scripts/diagnose_rewards.py
```

Salida esperada (modelo vs `BalancedBot`, 2560 pasos con `frame_skip=3`):

```
Episode ended at step 1205
Episode ended at step 2218
============================================================
  REWARD DIAGNOSTIC — 2560 steps
============================================================
  Sum:      243.199000     Mean:  0.095000
  Nonzero:  1799/2560 (70.3%)
  Per-component sums over 2560 steps:
    DamageComponent    sum=+0.56   nonzero=121
    ElixirComponent    sum=-28.30  nonzero=283
    TerminalComponent  sum=+0.00   nonzero=6
    StrategyComponent  sum=+270.94 nonzero=1690
```

---

## Bots heurísticos

MohaNetLight incluye 5 estrategias parametrizadas como oponentes de entrenamiento.
Todos implementan `PlayerSlot` (interfaz de cr-gym) y son directamente utilizables
como oponentes en el entorno `ClashRoyaleGymEnv`.

| Bot | Estrategia | Parámetro clave |
|-----|-----------|-----------------|
| `GiantPushBot` | Push lento: Giant atrás + soporte ranged | `elixir_threshold` (5–8) |
| `BridgeSpamBot` | Presión constante con tropas rápidas al puente | `elixir_threshold` (3–5) |
| `SpellCycleBot` | Defensa + daño chip con hechizos a torres | `spell_threshold` (5–7) |
| `DefensiveCounterBot` | Espera ataque enemigo, contraataca eficientemente | `counter_elixir` (6–8) |
| `BalancedBot` | Adapta agresividad según ventaja de HP | `base_threshold` (3–6) |

```python
from mohanetlight.bots.strategies import default_bot_roster

# Roster completo de 10 bots con variaciones de parámetros
bots = default_bot_roster(base_seed=42)
for bot in bots:
    print(f"{bot.name}: {bot.metadata()['type']}")
```

---

## Estructura del proyecto

```
MohaNetlight/
├── mohanetlight/
│   ├── __init__.py              # Exporta MohaNetLight, ModelConfig, TrainingConfig
│   ├── config.py                # ModelConfig (arquitectura) + TrainingConfig (PPO)
│   ├── network/
│   │   ├── encoders.py          # ScalarEncoder, EntityEncoder, CardEncoder
│   │   ├── core.py              # LSTMCore (384→256, 2 capas)
│   │   ├── heads.py             # Cabezas jerárquicas + ValueHead
│   │   └── mohanet.py           # MohaNetLight (modelo ensamblado)
│   ├── inference/
│   │   └── agent.py             # MohaNetAgent (PlayerSlot para liga)
│   ├── bots/
│   │   └── strategies.py        # 5 bots heurísticos parametrizados
│   ├── training/
│   │   ├── rollout.py           # RolloutBuffer con estados LSTM
│   │   ├── ppo.py               # PPOTrainer (clipped surrogate + BPTT truncado)
│   │   └── trainer.py           # LeagueTrainer (orquestador completo)
│   └── utils/
│       └── tensor_utils.py      # Conversión obs→tensores
├── scripts/
│   └── train_league.py          # Punto de entrada CLI
├── notebooks/
│   └── colab_training.ipynb     # Configuración y entrenamiento en Colab
├── tests/
│   └── test_network.py          # 35 tests (shapes, forward pass, buffer, bots)
├── pyproject.toml
└── README.md
```

---

## Hiperparámetros PPO

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `total_timesteps` | 1,000,000 | Pasos totales de entrenamiento |
| `n_steps` | 2,560 | Pasos por rollout (≈ 1 partida completa a 10 decisiones/s) |
| `frame_skip` | 3 | Frames de motor por decisión RL (30 fps ÷ 3 = 10 Hz) |
| `n_epochs` | 4 | Épocas PPO por actualización |
| `batch_chunk_len` | 32 | Longitud de secuencia para BPTT truncado |
| `gamma` | 0.99 | Factor de descuento |
| `gae_lambda` | 0.95 | Lambda para GAE |
| `clip_eps` | 0.2 | Epsilon de clipping PPO |
| `vf_coef` | 0.5 | Coeficiente de pérdida del valor |
| `ent_coef` | 0.01 | Coeficiente de bonus de entropía |
| `max_grad_norm` | 0.5 | Norma máxima del gradiente |
| `lr` | 3e-4 | Tasa de aprendizaje (Adam) |

> **¿Por qué `n_steps=2560`?**
> Una partida completa dura hasta 240 s (180 s regulares + 60 s de tiempo extra).
> A 30 fps con `frame_skip=3` eso equivale a `240 × 30 / 3 = 2400` pasos RL.
> Los 2560 proveen margen para al menos una partida completa por rollout.

---

## Espacios de observación y acción

### Observación (desde cr-gym)

| Clave | Shape | Descripción |
|-------|-------|-------------|
| `troops` | (100, 14) | Entidades: nombre, categoría, posición, HP, stats |
| `troop_mask` | (100,) | `True` donde hay tropa real (padding = `False`) |
| `scalars` | (16,) | Elixir, HP torres, tiempo, flags (sin elixir enemigo) |
| `cards` | (4, 4) | Mano propia: nombre, costo, es_hechizo, es_pagable |
| `action_mask` | dict | Máscaras por cabeza: strategy(3), card(5), tile_x(18), tile_y(32) |

### Acción

| Cabeza | Opciones | Significado |
|--------|----------|-------------|
| `strategy` | 3 | AGRESIVO / DEFENSIVO / FARMEAR |
| `card` | 5 | Slot de mano 0–3 o NOOP (4) |
| `tile_x` | 18 | Columna del tile (0–17) |
| `tile_y` | 32 | Fila del tile (0–31) |

---

## Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Solo tests de la red neuronal
pytest tests/test_network.py::TestMohaNetLight -v

# Solo tests de los bots heurísticos
pytest tests/test_network.py::TestBots -v
```

---

## Solución de problemas frecuentes

### `ModuleNotFoundError: No module named 'clash_royale_engine'`

cr-engine no está instalado o no se instaló con `-e`. Solución:

```bash
pip install -e /ruta/a/cr-engine
```

### `ModuleNotFoundError: No module named 'clash_royale_gymnasium'`

cr-gym no está instalado. Solución:

```bash
pip install -e /ruta/a/cr-gym
```

### `torch.cuda.is_available()` devuelve `False`

Instalaste la versión CPU-only de PyTorch. Solución:

```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### La GUI no abre (`pygame.error: No video mode`)

Estás ejecutando en un servidor sin pantalla. Usa el diagnóstico headless:

```bash
python scripts/diagnose_rewards.py
```

O exporta una pantalla virtual si tienes Xvfb:

```bash
Xvfb :99 -screen 0 1280x720x24 &
DISPLAY=:99 python scripts/watch_agent.py --no-music
```

### `UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False`

Advertencia inofensiva de PyTorch relacionada con el Transformer interno.
Puedes ignorarla con seguridad — no afecta el entrenamiento ni la inferencia.

---

## Licencia

MIT
