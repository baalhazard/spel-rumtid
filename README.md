# S³ Lab

S³ Lab är ett litet Python-laboratorium för att utforska ett krökt, slutet och gränslöst 3D-rum.

Just nu modellerar projektet rummet som S³: 3-sfären, representerad som punkter på en 4D-enhetssfär. Kameran ligger på S³, har en lokal ortonormal tangentbas, rör sig längs geodesier och renderar scenen med enkel geodesisk ray-casting. De första objekten är geodesiska sfärer.

Det här är inte tänkt som en vanlig spelmotor. Fokus i nuläget är att hålla geometrin korrekt och begriplig, utan portalrendering, teleport-hack, mesh-assets eller fysikmotor.

## Vad som finns nu

- S³-geometri i 4D med normalisering, tangentprojektion och geodesier.
- Kamerarörelse i lokal tangentbas med parallelltransporterad orientering.
- Analytisk ray-casting mot geodesiska sfärer.
- Progressiv renderer med overlay som visar UI-Hz, render-fps och frame-counter.
- Vektoriserad radblocksrendering med NumPy för mycket bättre prestanda än den första pixel-för-pixel-prototypen.
- En liten testsvit för S³-invarianter och jämförelse mellan scalar och vektoriserad rendering.

## Installera från GitHub

Kräver Python 3.10 eller nyare.

```powershell
git clone https://github.com/baalhazard/spel-rumtid.git
cd spel-rumtid
python -m pip install -e .
```

För att även installera testverktyg:

```powershell
python -m pip install -e ".[dev]"
```

## Kör

```powershell
python -m s3lab.app
```

## Kontroller

```text
W/S      framåt/bakåt
A/D      höger/vänster
Space    upp
Shift    ned
←/→      yaw
↑/↓      pitch
Q/E      roll
Esc      avsluta
```

## Tester

Om du installerat med `.[dev]`:

```powershell
python -m pytest
```

## Upplösning och prestanda

Standardläget renderar just nu i låg intern upplösning och skalar upp bilden i Pygame-fönstret. Efter vektoriseringssteget finns det utrymme att experimentera med högre intern upplösning i `s3lab/app.py`, till exempel genom att höja `render_width` och `render_height`.

Öka helst stegvis, eftersom varje höjning multiplicerar antalet geodesiska strålar som behöver kastas.
