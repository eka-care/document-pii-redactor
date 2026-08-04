#!/usr/bin/env bash
# Deploy the current commit to the ekacare/pii-redactor-demo HF Space.
#
# The Space's README.md needs YAML front matter (title/sdk/app_port/...) at
# the very top so HF renders the Space card correctly. That front matter
# would also leak into the GitHub README and the PyPI project description if
# it lived in the tracked README.md (see .space-metadata.yaml for why it
# doesn't). So this script builds a throwaway tree — the repo's current
# content with the front matter prepended to README.md — and force-pushes
# just that tree to the Space, without touching origin's history.
#
# Usage: scripts/push_space.sh
# Auth: uses HF_TOKEN if set, else falls back to the huggingface_hub CLI's
# cached login token.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SPACE_URL="https://huggingface.co/spaces/ekacare/pii-redactor-demo"
TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || true)}"
if [ -z "$TOKEN" ]; then
  echo "No HF token found. Set HF_TOKEN or run 'hf auth login' first." >&2
  exit 1
fi

SRC_SHA=$(git rev-parse --short HEAD)
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

git archive HEAD | tar -x -C "$TMPDIR"
cat .space-metadata.yaml "$TMPDIR/README.md" > "$TMPDIR/README.md.new"
mv "$TMPDIR/README.md.new" "$TMPDIR/README.md"

(
  cd "$TMPDIR"
  git init -q
  git add -A
  git -c user.name="deploy" -c user.email="deploy@local" \
    commit -q -m "Deploy from $SRC_SHA" --allow-empty
  git push -f "https://ds-EkaCare:${TOKEN}@${SPACE_URL#https://}" HEAD:main
)

echo "Pushed to $SPACE_URL"
