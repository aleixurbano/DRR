# Project page (`web/`)

Source for the DRR GitHub Pages site, deployed by
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml).

- `index.html` — single-page academic landing page.
- `style.css` — styling (no build step, no framework).
- `assets/fig/` — figures generated from the repo (owned artifacts only):
  - `calibration_std_vs_target.png` ← `analysis/scripts/make_web_calibration_figure.py`,
    built from `results_all_models.csv` of the standard-entropy and
    target-distribution calibration experiments
- `assets/media/` — recovery video for the media panel (see
  `assets/media/README.md`). The recovery trace next to it is real text in
  `index.html`, not an image.
- `assets/lectoraat-logo.svg` — lectoraat logo, used as the page icon (favicon).
- `assets/htes_logo_long.png` — Fontys HTES logo, used in the footer. If it needs
  updating, drop the new file here and update the `<img src>` in `index.html` if
  the filename changes.

## Local preview

Open `index.html` directly in a browser, or serve the folder:

```bash
python -m http.server -d web 8000   # then open http://localhost:8000
```

## Enable Pages (one-time)

Repo **Settings → Pages → Build and deployment → Source: GitHub Actions**.
The workflow then runs on every push that touches `web/`. Live URL:
`https://aleixurbano.github.io/DRR/`.

## Refresh figures

If the underlying calibration results change, regenerate the page figure
(published ECE values in the panel headers are set in the script — update them
from `metrics_by_model.csv` if they change):

```bash
python analysis/scripts/make_web_calibration_figure.py
```
