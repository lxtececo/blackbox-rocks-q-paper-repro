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

# An overfull page cuts content off at the page bottom without an error.
if grep -q 'Overfull \\vbox' main.log; then
  echo "main.log reports an overfull page: content runs past the bottom of a page" >&2
  exit 1
fi

if [[ "${1:-}" == "bundle" ]]; then
  # Ancillary files: a redacted, scanned copy of experiments/ where the builder exists (the platform
  # repository); in the public package experiments/ is that copy already.
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  builder="../../../scripts/paper/build_repro_package.py"
  if [[ -f "$builder" ]]; then
    python3 "$builder" --ancillary "$tmp/anc/experiments"
  else
    mkdir -p "$tmp/anc" && cp -r experiments "$tmp/anc/experiments"
    find "$tmp/anc" -name '__pycache__' -prune -exec rm -rf {} +
  fi
  tar czf arxiv-upload.tar.gz main.tex main.bbl figures/*.pdf -C "$tmp" anc
  echo "arxiv-upload.tar.gz:"
  tar tzf arxiv-upload.tar.gz
fi
