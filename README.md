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
| `exp1_solvers.py` | Exp. 1: la mateixa energia TV-L1 amb dos algorismes (el del paper i |x| arrodonit + descens de gradient) |
| `exp2_yosemite.py` | Exp. 2: Horn-Schunck, TV-L1 i RAFT a Yosemite 8-9, amb la mètrica del paper (AAE) i proves de robustesa |
| `comparison.ipynb` | Comparació a KITTI 2015 i Middlebury RubberWhale |

## Experiments per a la presentació

Tots dos scripts descarreguen Yosemite (sense núvols, M. Black) a `data/` i desen figures i taules a `results/`. Funcionen en CPU en uns minuts.

```bash
python exp1_solvers.py
python exp2_yosemite.py
```

Resultats en CPU (Apple M, PyTorch 2.7):

- **Exp. 1.** Amb una sola escala i la mateixa energia, l'algorisme del paper arriba a menys d'un 5 % de l'energia mínima en 0,04 s. El descens de gradient amb |x| arrodonit hi arriba en 10,4 s amb ε = 0,1. Amb ε = 1 s'estanca un 17 % per sobre i amb ε = 0,01 és massa lent.
- **Exp. 2.** A Yosemite, amb 50 iteracions per nivell, el nostre TV-L1 dona un AAE de 2,82° sense cel (paper: 2,85°), Horn-Schunck 5,46° i RAFT 3,09°. Amb cel el nostre error és més alt que al paper (10,5° davant 5,1°). Amb un 20 % de píxels corromputs, l'EPE de TV-L1 és 1,01 px i el de Horn-Schunck 2,13 px.

## Validació

Error mitjà en píxels (EPE) amb dades sintètiques, 1 warp i 4 nivells. La rèplica fa 50 iteracions per nivell.

| Moviment | Rèplica | OpenCV |
|---|---|---|
| Translació | 0,058 | 0,012 |
| Flux suau | 0,065 | 0,036 |
| Objecte amb vores | 0,057 | 0,074 |

Aquests valors no són una reproducció dels resultats del paper. La comparació amb el paper a Yosemite és a `exp2_yosemite.py`.

## Referències

- C. Zach, T. Pock i H. Bischof. *A Duality Based Approach for Realtime TV-L1 Optical Flow*. DAGM 2007.
- B. K. P. Horn i B. G. Schunck. *Determining Optical Flow*. Artificial Intelligence, 1981.

## Llicència

MIT. Consulta el fitxer `LICENSE`.
