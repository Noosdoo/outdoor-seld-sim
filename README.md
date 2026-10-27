# outdoor-seld-sim

[![tests](https://github.com/Noosdoo/outdoor-seld-sim/actions/workflows/tests.yml/badge.svg)](https://github.com/Noosdoo/outdoor-seld-sim/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Physics-based outdoor SELD: simulate → label → train → evaluate → notify.**

An open, end-to-end pipeline that renders outdoor **moving** sound sources with real
acoustics into 4-channel Ambisonics, emits DCASE-format labels **from the same geometry
that produced the audio**, fine-tunes a SELD model, and scores a notification layer built
for deaf and hard-of-hearing pedestrians.

```
moving source (car / siren / horn / bell / crossing alarm)
  │  propagation delay · Doppler · 1/r spreading · atmospheric absorption · ground reflection
  ▼
4-ch FOA (ACN: W,Y,Z,X / SN3D, 24 kHz FLAC)
  │  same geometry, same emission-time solver
  ▼
DCASE metadata CSV (frame, class, track, azimuth, elevation @ 0.1 s)
  │
  ▼
SELD model (HTS-AT + multi-ACCDOA, via PSELDNets)
  │
  ▼
notification layer  →  "car approaching, 4 o'clock, 5 seconds"
```

## Why this exists

Sirens, horns, backup beeps, bicycle bells and level-crossing alarms are the channel
pedestrians use to stay safe. For deaf and hard-of-hearing people, that channel is simply
absent. A wearable that reports **what** sound and **from which direction** can carry it —
which is exactly the SELD (sound event localization and detection) task.

The blocker is data. Open spatial-audio generators today (SpatialScaper, the DCASE
generators) are built on **indoor** room impulse responses. Outdoor traffic is a different
physical problem: sources move fast, so propagation delay, Doppler shift, distance
attenuation, air absorption and ground reflection all matter, and the ground-truth
direction is the direction the sound *appeared to come from* at reception time — not the
source's position at that instant. There is no shared, open recipe for generating that.

This repository is that recipe, plus the evaluation harness around it.

## What is in the box

| Path | What it does |
|---|---|
| `src/outdoor_seld/geometry.py` | Coordinate transforms and the **emission-time solver** — the single path shared by audio and labels |
| `src/outdoor_seld/foa.py` | Analytic FOA encoder (W,Y,Z,X / SN3D) and intensity-vector DOA estimation |
| `src/outdoor_seld/labels.py` | DCASE 5-column CSV writer (emission-time corrected DOA) |
| `src/outdoor_seld/scene.py` | Scene definition, simulator wrapper, decimation |
| `src/outdoor_seld/siren_v26.py`, `alert_sounds_v23.py`, `bike_bell_v2.py` | Warning sounds: siren, horn, backup beeper, bicycle bell |
| `src/outdoor_seld/engine.py`, `engine_ev.py`, `avas_v26.py`, `motorcycle.py`, `kickboard.py`, `bicycle_v2.py`, `train.py` | Vehicle sources: cars (incl. EV/HV approach sound), motorcycles, e-scooters, bicycles, trains |
| `src/outdoor_seld/noise_v30.py`, `rain.py` | Urban background with per-clip spectral shape, rain |
| `src/outdoor_seld/calibration.py`, `level_v24.py`, `band_gate.py` | Absolute level calibration (dB SPL), level design, in-band audibility gate for labels |
| `src/outdoor_seld/fastsim.py` | Fast renderer, waveform-matched to the reference simulator (rel. RMS 3.4e-05) |
| `scripts/step10_*`, `step11_*` | Scene planning and rendering, one script per dataset version |
| `scripts/step12_*`, `step20_*` | Notification layer and scoring |
| `scripts/step19*` | Field-recording tools: conversion, annotation, video-based ground truth |
| `server_sde/*.sbatch` | SLURM jobs for cluster runs |
| `tests/` | Physics and convention checks (see below) |

## Correctness, not vibes

The hard part of synthetic SELD data is that the labels can silently disagree with the
audio. Three guards:

1. **One geometry path.** The FOA gains and the label CSV are computed from the same
   emission-time solution `‖p_r − p_s(t_e)‖ = c(t_r − t_e)`. There is no second
   implementation to drift out of sync.
2. **Cross-checked solver.** The solver agrees with `dynamic-sound`'s internal
   `_compute_emission` to **zero error** (unit test).
3. **End-to-end audit.** Running intensity-vector DOA estimation on the rendered FOA and
   comparing against the written labels gives a **median azimuth error of 0.31°**
   (max 0.96°) and **0.26°** in elevation. The audio really does come from where the label
   says it does.

FOA convention is verified end-to-end too: a static source at az = +90° yields
`Y/W = 1.000000`, `X = Z = 0`.

Simulation runs at 48 kHz and is decimated to 24 kHz, because interpolating a delay line
directly at 24 kHz leaves audible Doppler stretch artifacts; residual ghosting measures
below −60 dB.

## Tests

```bash
python tests/test_geometry.py   # 12 checks — emission time, Doppler, az/el round-trip
python tests/test_foa.py        # 13 checks — ACN/SN3D gains, IV-DOA recovery
python tests/test_noise.py      #  8 checks — pink/brown spectra, diffuse FOA noise
python -m pytest tests/test_doppler_direction.py   # 2 tests — pitch drops after the pass, walking observer
python tests/test_fastsim.py    #  6 checks — fast renderer vs reference simulator
```

The first four run in CI on Linux and Windows on every push (33 checks + 2 tests).
`test_fastsim.py` compares the fast renderer against a previously generated clip, so it
needs a clip rendered by the pipeline first and is local-only.

## Results so far

These numbers are from **v0.1.0 (August 2026)**. The model is being retrained on the
current data recipe and the evaluation will be redone once, on a sealed test set; the
final numbers will be published with the thesis.

Synthetic held-out evaluation (development evaluation, 3,342 clips). Detection rate at
SNR ≥ 0 dB, and median direction-of-arrival error:

| Class | Detection | Median DOA error |
|---|---|---|
| Siren | 99.0 % | 3.2° |
| Horn | 98.8 % | 2.2° |
| Backup beep | 94.9 % | 5.1° |
| Bicycle bell | 98.4 % | 5.1° |
| Car | 99.4 % | 2.0° |
| Level-crossing alarm | 99.8 % | 4.1° |

Notification layer, scored against complete ground truth:

- 97.4 % of warning sounds notified, median direction error 4.2°, median firing delay 0.32 s
- Cars on a collision course: 97.3 % notified, median lead time **4.98 s**, 79.9 % at least
  2.5 s ahead — against an oracle ceiling of 85.4 %
- **0.53 false alerts per hour** over 9.0 hours of audio; zero false alerts for dangerous cars

## Known limitations

Stated plainly, because they matter more than the good numbers:

- All results above are **synthetic-domain**. Real-world recordings are in progress.
- Multiple simultaneous cars degrade: both of 2 cars detected 69.7 %, all of 3 cars 34.8 %.
- **Quiet EVs**: 20.0 % — when a car is near-silent, there is nothing to hear. The same wall
  human listeners hit in NHTSA's quiet-vehicle studies.
- Sudden appearance from behind an occluder: 84.7 %, median lead 1.47 s.
- Comments in the code are mostly Japanese.

## Built on

- [dynamic-sound](https://pypi.org/project/dynamic-sound/) — outdoor moving-source acoustic
  simulation (Apache-2.0)
- [PSELDNets](https://github.com/Jinbo-Hu/PSELDNets) — pretrained SELD models,
  HTS-AT + multi-ACCDOA (MIT)

## Related work

PAWS (IPSN 2018) detects approaching vehicles for pedestrians but reports coarse sectors
and states that trajectory tracking and multiple vehicles are out of scope. Siren-recognition
wearables for the hearing impaired (Sensors 2023) classify sounds without direction.
6DoF SELD (ICASSP 2024) motivates walking-microphone SELD but trains on recordings, indoors.
This project targets the gap: an open, physically controllable outdoor generator, multi-class
and multi-source, with a notification layer that is evaluated rather than demonstrated.

## Upstream contributions

- [Jinbo-Hu/PSELDNets#6](https://github.com/Jinbo-Hu/PSELDNets/pull/6) — fix dataset name
  parsing on Windows (`path.split('/')` → `Path(path).parts`). Merged 2026-10-27.

## What changed in v0.2.0

- New and revised sound sources: warning sounds (3rd revision), siren (6th), bicycle
  rolling sound and bell modelled on measurements, EV/HV approach sound (AVAS),
  rain, and an urban background whose spectral shape varies per clip
- Absolute level design in dB SPL and an in-band audibility gate, so a label is only
  emitted when the source is actually audible above the background
- Field-recording toolchain for the real-world evaluation: conversion, annotation and
  video-based ground truth (`scripts/step19*`)
- Stable clip identifiers (`src/outdoor_seld/clip_ids.py`)
- CI now also checks the direction of the Doppler shift

## Documentation

Internal design notes, audit records and field-recording logs are not published, because
they include details about recording participants and locations. The code is complete.

## License

MIT — see [LICENSE](LICENSE).
