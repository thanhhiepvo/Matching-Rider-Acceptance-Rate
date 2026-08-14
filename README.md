# Matching Rider Acceptance Rate

Rider/customer cancellation prediction project. This README covers deploying and updating the **FT-Transformer math demo page** (`docs/`) via GitHub Pages.

## Live site

Once GitHub Pages is enabled (see below), the site is served at:

```
https://thanhhiepvo.github.io/Matching-Rider-Acceptance-Rate/
```

## One-time setup: enable GitHub Pages

1. Go to the repo on GitHub → **Settings** → **Pages**.
2. Under **Build and deployment** → **Source**, choose **Deploy from a branch**.
3. **Branch**: `main`, folder: **`/docs`**.
4. Click **Save**.
5. Wait ~1 minute, then visit the URL above.

No build step is required — `docs/` is a static site (plain HTML/CSS/JS + KaTeX and Google Fonts loaded from CDN).

## Updating the site

Edit `docs/index.html` (and `docs/ft-transformer-architecture.png` if the diagram changes), then:

```bash
git add docs/
git commit -m "Update FT-Transformer demo page"
git push
```

GitHub Pages redeploys automatically on every push to `main` — no extra action needed.

## Previewing locally before pushing

```bash
python3 -m http.server 8631 --directory docs
```

Then open `http://localhost:8631` in a browser.

## Site contents

- `docs/index.html` — single-page math report: input features, Feature Tokenizer formulas, `[CLS]` token, Transformer Encoder, prediction head, and the real tuned hyperparameters.
- `docs/ft-transformer-architecture.png` — architecture diagram (hand-drawn in draw.io, exported as PNG).
