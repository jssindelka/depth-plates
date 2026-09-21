# DEPTH//PLATES

Scroll-driven site where photographs extrude into a 3D space you can zoom into, plus a lab for testing image effects.

## Run it

Double-click **`start.command`**. It starts a local server and opens the site in your browser.
(Or in a terminal: `cd ~/Desktop/depth-plates && python3 -m http.server 4173`.)

Opening the `.html` files directly by double-click will not work — browsers block images from reaching the graphics card over `file://`.

| Page | Address | What it is |
|---|---|---|
| Site (dither) | `/` | Main brutalist site. Colour images as a dithered pixel field, red scan sweep, scanlines. |
| Site (clean) | `/clean.html` | Same site, photo untouched: full-resolution textured mesh, no effects over the image. |
| Playground | `/lab/` | Upload any image, switch effects on, stack them, read what each one does. |
| Recipes | `/lab/recipes.html` | Recommended effect combinations as short directed sequences. |

## Folder map

```
depth-plates/
├── start.command        double-click launcher
├── index.html           the site — dither edition (self-contained)
├── clean.html           the site — clean edition (self-contained)
├── images/
│   ├── 01.jpg           photos, numbered in plate order
│   ├── 01-depth.png     matching depth map (white = near), same number
│   └── …
├── lab/
│   ├── index.html       playground page
│   ├── recipes.html     recipes page
│   ├── engine.js        the WebGL pipeline: every effect lives here
│   ├── fx.js            effect descriptions, image library, recipes
│   ├── shared.js        image / video loading helpers
│   └── lab.css          shared styling
└── archive/
    └── minimal.html     parked minimalist version (still runs at /archive/minimal.html)
```

## Add a photo

**To the site** — put the file in `images/` as the next number, then add an entry to the `PLATES` array near the bottom of `index.html` and `clean.html`:

```js
{ id:'05', title:'MY PLATE', src:'images/05.jpg', depth:'images/05-depth.png', focus:[0.5,0.5], … }
```

`focus` is where the scroll camera travels to (0–1 from the top-left). Optional per plate: `zoom` (end of the dolly, default 4.2), `relief` (depth strength), `swing` (camera yaw; negative mirrors it), `edge` (clean edition: raise for subjects with thin gaps like torn leaves or hair).

**Depth map** — leave `depth:null` and the page estimates one in the browser on load (nothing is uploaded; ~20 s the first time). Press `[ SAVE DEPTH ]` inside the plate's zoom view, drop the PNG into `images/`, and reference it so visitors never wait for the model.

**To the lab** — drop the file onto the playground stage for a quick test, or add it to `LIBRARY` at the top of `lab/fx.js` to keep it in the strip.

## Keep a look from the playground

The **STATE** box at the bottom of the playground panel lists every setting that differs from the defaults. Copy it; it can become a recipe in `lab/fx.js` (`RECIPES`) or be ported to the site.

## Notes

- Everything is plain HTML/JS, no build step, no dependencies to install. Fonts come from Google Fonts; the depth model (only when a depth map is missing) from jsDelivr / Hugging Face.
- Gaussian splats are not part of this: they need 30–100 frames or an orbit video of the scene, not a single photo.
