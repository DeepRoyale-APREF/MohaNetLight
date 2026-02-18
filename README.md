# MohaNetLight

Red neuronal ligera inspirada en AlphaStar, especializada para Clash Royale Arena 1.
Entrenamiento por PPO con cabezas jerárquicas y evaluación mediante liga contra bots heurísticos.

**~1.6M parámetros** — entrenable en GPU doméstica o Google Colab gratuito.

---

## Arquitectura

```
┌─────────────┐  ┌──────────────────┐  ┌─────────────┐
│   Scalars   │  │     Entities     │  │    Cards    │
│  MLP 16→128 │  │ Transformer L=2  │  │  MLP 4→128  │
│             │  │  H=4, d=64       │  │  (×4 cards) │
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
    │   ┌─────────▼─────────┐               │
    │   │  Strategy Head    │               │
    │   │  256→128→3        │               │
    │   └────────┬──────────┘               │
    │            │ embed(3→64)              │
    │   ┌────────▼──────────┐               │
    │   │  Card Head        │←── entity ctx │
    │   │  448→128→5        │               │
    │   └────────┬──────────┘               │
    │            │ embed(5→64)              │
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
    │   │  {strat,card,x,y} │    │  256→128→1   │
    │   └───────────────────┘    └──────────────┘
```

### Componentes

| Módulo | Descripción | Parámetros |
|--------|-------------|------------|
| `ScalarEncoder` | MLP 16→64→128 (elixir, torres, tiempo, flags) | ~9K |
| `EntityEncoder` | Transformer 2 capas, 4 cabezas, d=64 + codificación posicional | ~112K |
| `CardEncoder` | MLP compartido por carta 4→32, agregado 128→128 | ~17K |
| `LSTMCore` | LSTM(384→256, 2 capas, dropout=0.1) | ~1.18M |
| Cabezas jerárquicas | Strategy→Card→TileX→TileY con embeddings autoregresivos | ~253K |
| `ValueHead` | MLP 256→128→64→1 (crítico) | ~41K |
| **Total** | | **1,615,355** |

### Cabezas autoregresivas

Cada cabeza recibe como contexto adicional el **embedding de la acción muestreada** de la cabeza anterior, más el **contexto de entidades** (skip connection del encoder):

1. **Strategy** ← salida LSTM
2. **Card** ← LSTM + embed(strategy) + entity_ctx
3. **Tile X** ← LSTM + embed(card) + entity_ctx  
4. **Tile Y** ← LSTM + embed(tile_x) + entity_ctx

---

## Instalación

### Requisitos previos

- Python ≥ 3.10
- [cr-engine](https://github.com/DeepRoyale-APREF/cr-engine) instalado
- [cr-gym](https://github.com/DeepRoyale-APREF/cr-gym) instalado

### Paso 1: Instalar PyTorch con soporte GPU

```bash
# CUDA 12.4 (Linux / Windows con NVIDIA)
pip install torch>=2.1 --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch>=2.1 --index-url https://download.pytorch.org/whl/cu121

# Solo CPU
pip install torch>=2.1

# macOS (MPS incluido por defecto)
pip install torch>=2.1
```

### Paso 2: Instalar dependencias

```bash
# Clonar e instalar cr-engine
git clone https://github.com/DeepRoyale-APREF/cr-engine.git
pip install -e cr-engine/

# Clonar e instalar cr-gym
git clone https://github.com/DeepRoyale-APREF/cr-gym.git
pip install -e cr-gym/

# Clonar e instalar MohaNetLight
git clone https://github.com/DeepRoyale-APREF/MohaNetlight.git
pip install -e "MohaNetlight/[dev]"
```

### Google Colab

Abrir el notebook incluido que automatiza toda la configuración:

```
notebooks/colab_training.ipynb
```

> Colab ya trae PyTorch con CUDA preinstalado. Solo necesitas clonar los repos e instalar.

---

## Detección automática de dispositivo

Al crear `TrainingConfig()`, el dispositivo se auto-detecta:

```python
from mohanetlight.config import TrainingConfig

tc = TrainingConfig()
# [MohaNetLight] Using CUDA device: NVIDIA GeForce RTX 3050 6GB Laptop GPU
# Device: cuda
```

Prioridad: `cuda` > `mps` > `cpu`. Si no hay GPU, imprime un aviso y usa CPU.

Para forzar un dispositivo específico:

```python
tc = TrainingConfig(device="cpu")  # ignorar GPU
```

---

## Uso rápido

### Inferencia con el modelo

```python
import torch
from mohanetlight import MohaNetLight, ModelConfig

model = MohaNetLight(ModelConfig())
model.eval()

# Observación dummy (en producción viene del entorno cr-gym)
B = 1
scalars = torch.randn(B, 16)
troops = torch.randn(B, 100, 14).abs()
troop_mask = torch.zeros(B, 100, dtype=torch.bool)
troop_mask[0, :5] = True
cards = torch.randn(B, 4, 4).abs()
action_masks = {
    "strategy": torch.ones(B, 3, dtype=torch.bool),
    "card": torch.ones(B, 5, dtype=torch.bool),
    "tile_x": torch.ones(B, 18, dtype=torch.bool),
    "tile_y": torch.ones(B, 32, dtype=torch.bool),
}

hidden = model.init_hidden(B)
output = model.act(scalars, troops, troop_mask, cards, action_masks, hidden)

print(output.actions)   # {strategy: tensor, card: tensor, tile_x: tensor, tile_y: tensor}
print(output.value)     # V(s)
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
python scripts/train_league.py --total-steps 1000000 --device cuda
python scripts/train_league.py --total-steps 500000 --eval-interval 10
python scripts/train_league.py --help
```

### Agente para liga

```python
from mohanetlight.inference.agent import MohaNetAgent

# Cargar desde checkpoint
agent = MohaNetAgent.from_checkpoint(
    "logs/mohanet/mohanet_u100.pt",
    name="MohaNet-v1",
    device="cuda",
)

# Usar en torneo de cr-gym
from clash_royale_gymnasium.league.tournament import LeagueTournament
from clash_royale_gymnasium.league.player_slot import HeuristicSlot

league = LeagueTournament(
    players=[agent, HeuristicSlot("Bot", aggression=0.5)],
    matches_per_pair=10,
)
results = league.run()
```

---

## Bots heurísticos

MohaNetLight incluye 5 estrategias parametrizadas como oponentes de entrenamiento:

| Bot | Estrategia | Parámetro clave |
|-----|-----------|-----------------|
| `GiantPushBot` | Push lento: Giant atrás + soporte ranged | `elixir_threshold` (5-8) |
| `BridgeSpamBot` | Presión constante con tropas rápidas al puente | `elixir_threshold` (3-5) |
| `SpellCycleBot` | Defensa + daño chip con hechizos a torres | `spell_threshold` (5-7) |
| `DefensiveCounterBot` | Espera ataque enemigo, contraataca eficientemente | `counter_elixir` (6-8) |
| `BalancedBot` | Adapta agresividad según ventaja de HP | `base_threshold` (3-6) |

```python
from mohanetlight.bots import default_bot_roster

# 10 bots con variaciones de parámetros
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
| `n_steps` | 512 | Pasos por rollout |
| `n_epochs` | 4 | Épocas PPO por actualización |
| `batch_chunk_len` | 32 | Longitud de secuencia para BPTT truncado |
| `gamma` | 0.99 | Factor de descuento |
| `gae_lambda` | 0.95 | Lambda para GAE |
| `clip_eps` | 0.2 | Epsilon de clipping PPO |
| `vf_coef` | 0.5 | Coeficiente de pérdida del valor |
| `ent_coef` | 0.01 | Coeficiente de bonus de entropía |
| `max_grad_norm` | 0.5 | Norma máxima del gradiente |
| `lr` | 3e-4 | Tasa de aprendizaje (Adam) |

---

## Tests

```bash
# Ejecutar todos los tests (35)
pytest tests/ -v

# Solo tests de la red
pytest tests/test_network.py::TestMohaNetLight -v

# Solo tests de bots
pytest tests/test_network.py::TestBots -v
```

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
| `card` | 5 | Slot de mano 0-3 o NOOP (4) |
| `tile_x` | 18 | Columna del tile |
| `tile_y` | 32 | Fila del tile |

---

## Licencia

MIT
