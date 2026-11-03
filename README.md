# Drone Commander, Powered by Jev

This repository is a field test of [Drone Commander MK2](https://github.com/Ramolisdenneyous/DRONE-COMMANDER-MK2) with unit tactics driven by [Jev](https://typesafe.ai/), TypeSafe’s typed decision model.

It is an experiment, not the live playable build. The deployed MK2 line stays on its own repository. This one exists so the Jev trial can be published, played, and judged on its own before it ever replaces that version. The frozen MK1 proof of concept remains at [DRONE-COMMANDER-MK1](https://github.com/Ramolisdenneyous/DRONE-COMMANDER-MK1).

## Why Jev

Drone Commander does not need a chat model to invent the game. The rules engine already knows every legal move, shot, mine, and detonation. What it needs is a fast judge: look at this unit, this order, and this short menu, then pick one option.

Jev is built for that kind of call. The backend sends compact battlefield state and the legal option IDs. Jev returns one choice, a confidence score, and a probability for each option. The server checks the ID again, then resolves movement, fire, damage, and objectives itself. Jev cannot invent hexes, targets, dice, or damage.

If a reply is late, malformed, unoffered, or below the confidence and margin gates, the existing deterministic fallback takes the turn. The match keeps moving. Friendly radio stays on a short template during this test, because Jev selects actions and does not write prose.

## First results

The first batch of local matches came back successful.

Jev made the unit choices. Gameplay got faster, because a bounded decision returns in a fraction of the time a full tactical chat completion needs. Token cost for those decisions fell by about **75 percent**. Across those games there was no obvious loss of tactical ability. Squads still pushed the objective, took the shot when it mattered, and used the tools the rules actually gave them.

These are early playtest results, not a finished benchmark. The confidence gate is part of the experiment: Jev reports how sure it is, and the backend can refuse a weak call instead of playing it. The first matches ran with that gate open so every legal Jev choice could be seen in the open. Diagnostics on the battle screen record the provider, model, confidence, margin, latency, and whether the fallback stepped in.

## Run the Jev test locally

Docker is the supported setup. From this repository:

```bash
cp .env.example .env
```

Put your TypeSafe key in `.env`, then turn the tactical provider over to Jev:

```bash
TACTICAL_PROVIDER=jev
TYPESAFE_API_KEY=your-key-here
RADIO_PROVIDER=template
```

The example file keeps the confidence gate at `0.60` and the margin gate at `0.10`. A choice under that bar falls back. The first playtests lowered both local gates to `0` so low-confidence legal choices still played. Do not commit a real key. `.env` is gitignored.

```bash
docker compose up --build
```

| Service | Port |
|---|---|
| Frontend | http://localhost:5179 |
| API health | http://localhost:8006/health |
| Postgres | 5438 |

Health should report `tactical_provider: jev` once the key is set. Hard-refresh the page after a frontend rebuild.

`TACTICAL_SHADOW=true` asks Jev what it would pick, logs the confidence, and still commits the deterministic fallback. Use that when you want a comparison without letting Jev move the pieces.

## What this build still is

MK2’s game is intact underneath the provider swap: backend-authoritative hex rules, Freestyle army building, PixiJS battlefield, content catalogs, and the pytest suite in `backend/tests/`. The Jev adapter tests live beside that suite and cover legal selections, bad IDs, timeouts, low confidence, and shadow mode.

See [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md) for the wider MK2 gate notes. Those notes describe the game, not this provider trial.

## License

[MIT License](./LICENSE) — Copyright (c) 2026 Ramolisdenneyous.
