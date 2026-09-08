# ANN-DL Course Repo — Structure Reference

Internal reference doc, not part of the published site (lives at repo root, outside `docs/`,
so mkdocs never builds it). Sourced from two places that serve different purposes:

- **Canonical structural template:** https://repo-classes.github.io/ann-dl/ (repo:
  `github.com/repo-classes/ann-dl`) — defines the required folder layout, front matter,
  and deployment mechanics. Exercise pages here are empty skeletons ("Espelhe a estrutura
  do enunciado" — mirror the structure of the statement).
- **Actual graded assignment specs (this semester):** https://insper.github.io/ann-dl/2026.2/ —
  has the real, concrete task statements, parameters, and grading rubrics per exercise. Always
  pull real requirements from here, not from the template's placeholder pages.

Note: this repo (`luigilopesz/Neural-Networks-Deep-Learning`) was originally forked from
`hsandmann/documentation.template` (a generic doc template), not from `repo-classes/ann-dl`
directly — but the required output structure is identical, and everything below is what we
actually converged to for the Data exercise.

## Required repository layout

```
docs/
├── index.md                          # cover page — group/author info
├── exercises/
│   ├── data/
│   │   ├── index.md                  # report (required front matter, see below)
│   │   ├── code/                     # standalone .py scripts, one per exercise part
│   │   └── figures/                  # fig1.png, fig2.png, ... referenced by index.md
│   ├── perceptron/                   # same shape, not yet started
│   ├── mlp/                          # same shape, not yet started
│   └── vae/                          # same shape, not yet started
├── projects/
│   ├── index.md                      # team overview & dataset spec (not started)
│   ├── eda/                          # phase 1
│   ├── classification/  OR  regression/   # phase 2 — pick ONE, delete the other's
│   │                                        folder + nav entry entirely
│   └── generative/                   # phase 3
├── roteiro1..4/                      # lab worksheets, filled in as assignments land
mkdocs.yml
requirements.txt
```

Each exercise/project submission folder is `index.md` (the report) + `code/` (scripts) +
`figures/` (images) — never notebooks embedded via mkdocs-jupyter for graded work, even
though the template's own `/examples/` section demonstrates that pattern (nav item pointing
straight at an `.ipynb`, plus an "Open in Colab" link). That notebook-embed pattern is a
*demo feature*, not the required submission format — don't use it for graded exercises.

## Front matter (mandatory, exact keys)

**Exercises** (`docs/exercises/<slug>/index.md`):
```yaml
---
exercise: [slug]              # data | perceptron | mlp | vae
ai_use: "[description, or 'none']"
---
```

**Projects** (`docs/projects/<phase>/index.md`):
```yaml
---
project: [eda|classification|regression|generative]
ai_use: "[description, or 'none']"
---
```

`ai_use` is mandatory in both — must honestly disclose AI collaboration. The course allows
AI use but requires the student to understand and be able to defend every part of the
submitted code/analysis; oral exams may be conducted.

## Report structure convention

- Headings mirror the exercise statement exactly: `## Exercise N`, `### A`/`### B`/... for
  lettered subsections, ending in `## Results summary` with a required table.
- Code lives in `code/*.py`, embedded into the report via the snippet-include syntax
  (`pymdownx.snippets`, already enabled in `mkdocs.yml`):
  ```
  ```python
  --8<-- "docs/exercises/<slug>/code/<file>.py"
  ```
  ```
- Figures are committed PNGs under `figures/`, embedded as standard markdown images
  (`![Figure N](figures/figN.png)`), not generated at build time — mkdocs never executes
  the scripts; they're run locally/offline and their output committed.
- Every requested numeric result must appear in the report's prose text, not only as code
  output or only inside the summary table.

## Deployment mechanics (things that silently break on a fresh/dormant fork)

Confirmed the hard way on 2026-09-08 — the template's own setup guide lists the same two
gotchas:

1. **Settings → Actions → General → Workflow permissions** must be **"Read and write
   permissions"** (default is read-only, which makes `mkdocs gh-deploy` fail with
   `403: Permission ... denied to github-actions[bot]` when it tries to push `gh-pages`).
2. **Settings → Pages → Source** must point at the `gh-pages` branch, root — this doesn't
   happen automatically just because the branch exists; it's a separate one-time toggle.
3. Local validation before pushing: `mkdocs build --strict` (stricter than the plain
   `mkdocs build` we've been using — worth switching to before future deploys) or
   `mkdocs serve -o` to preview.
4. `site_url` / `repo_url` in `mkdocs.yml` must match the actual repo
   (`https://luigilopesz.github.io/Neural-Networks-Deep-Learning`,
   `https://github.com/luigilopesz/Neural-Networks-Deep-Learning`) — already set correctly.

Live site: https://luigilopesz.github.io/Neural-Networks-Deep-Learning/

## Current status (as of 2026-09-08)

- ✅ **Data exercise** — done, matches the exact 2026.2 spec and required layout, deployed live.
- ⬜ **Perceptron / MLP / VAE** — not started; no assignment statement published yet at the
  semester spec site as of this writing. When one lands, pull the real spec from
  `insper.github.io/ann-dl/2026.2/exercises/<slug>/`, not from the template's empty page.
- ⬜ **Roteiro 1-4** — cleared to blank worksheet stubs, pending real lab assignments.
  Original template demo content preserved in `examples/` for reference.
- ⬜ **Projeto** — still the original one-line placeholder. Per the template: team-based
  (2-3 people, all listed in `mkdocs.yml`), one shared dataset across all phases, 3 graded
  phases (EDA → Classification-or-Regression → Generative), needs a Decision Log and a
  dataset-spec table (name/source/license/sample count/feature count/target/justification).
  **Open question:** confirm solo vs. team for this course instance before starting — the
  template assumes teams.

## Known open issue

Push-triggered CI intermittently failed to register runs at all on this repo for its first
~10 minutes of use post-dormancy (separate from the permissions issue above — `workflow_dispatch`
was added to `.github/workflows/main.yaml` as a manual-trigger fallback). It self-resolved and
push-triggering has worked since, but if a future push doesn't appear under the repo's Actions
tab within a minute or two, run `gh workflow run ci --repo luigilopesz/Neural-Networks-Deep-Learning --ref main`
to deploy manually rather than waiting indefinitely.
