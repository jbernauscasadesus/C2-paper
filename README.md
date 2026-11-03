# Flux òptic TV-L1 en temps real

Rèplica en PyTorch del mètode de Zach, Pock i Bischof (2007), amb una comparació amb Horn-Schunck i dues demos amb webcam.

## Instal·lació

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Per fer servir la GPU, instal·la abans la versió de PyTorch adequada des de [pytorch.org](https://pytorch.org/get-started/locally/). En executar els scripts, han d'indicar `Dispositiu: cuda` o `Dispositiu: mps`.

## Ús

Demos amb webcam:

```bash
python tvl1_demo.py mirror
python tvl1_demo.py pong
```

Tecles: `q` o `Esc` per sortir i `r` per reiniciar el Pong. Si va lent, prova `--width 240 --iters 20`.

Vídeo comparatiu amb tres panells (entrada, Horn-Schunck i TV-L1):

```bash
python compare_flow.py video.mp4 --output comparacio.mp4
```

Proves amb dades sintètiques i comparació amb OpenCV:

```bash
python check_tvl1.py
```

## Fitxers

| Fitxer | Contingut |
|---|---|
| `tvl1_demo.py` | TV-L1 i les demos del mirall i el Pong |
| `compare_flow.py` | Horn-Schunck i generació del vídeo comparatiu |
| `check_tvl1.py` | Proves amb flux sintètic conegut |

## Validació

Error mitjà en píxels (EPE) amb dades sintètiques, 1 warp i 4 nivells. La rèplica fa 50 iteracions per nivell.

| Moviment | Rèplica | OpenCV |
|---|---|---|
| Translació | 0,058 | 0,012 |
| Flux suau | 0,065 | 0,036 |
| Objecte amb vores | 0,057 | 0,074 |

Aquests valors no són una reproducció dels resultats del paper. El codi només s'ha provat en CPU, i no s'ha avaluat amb Yosemite ni Middlebury.

## Referències

- C. Zach, T. Pock i H. Bischof. *A Duality Based Approach for Realtime TV-L1 Optical Flow*. DAGM 2007.
- B. K. P. Horn i B. G. Schunck. *Determining Optical Flow*. Artificial Intelligence, 1981.

## Llicència

MIT. Consulta el fitxer `LICENSE`.
