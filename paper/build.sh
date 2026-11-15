#!/usr/bin/env bash
# Build main.pdf and the arXiv upload bundle.
#
#   ./build.sh            # PDF via local latexmk, or the TeX Live container if latexmk is absent
#   ./build.sh bundle     # also writes arxiv-upload.tar.gz (main.tex + main.bbl)
#
# arXiv does not run BibTeX, so the bundle ships the generated main.bbl.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${TEXLIVE_IMAGE:-texlive/texlive:latest-medium}"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
else
  docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/work -w /work "$IMAGE" \
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
fi

if [[ "${1:-}" == "bundle" ]]; then
  tar czf arxiv-upload.tar.gz --exclude='__pycache__' \
    --transform 's,^experiments,anc/experiments,' \
    main.tex main.bbl figures/*.pdf experiments
  echo "arxiv-upload.tar.gz:"
  tar tzf arxiv-upload.tar.gz
fi
