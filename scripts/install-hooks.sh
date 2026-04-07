#!/usr/bin/env bash
# Install project git hooks. Run once after cloning.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$REPO_ROOT/.git-hooks"

if [ ! -d "$HOOKS_DIR" ]; then
    echo "ERROR: .git-hooks/ directory not found at $REPO_ROOT"
    exit 1
fi

git config core.hooksPath "$HOOKS_DIR"
chmod +x "$HOOKS_DIR/pre-commit"

echo "Git hooks installed from $HOOKS_DIR"
echo "  pre-commit: blocks API keys and secret patterns"
