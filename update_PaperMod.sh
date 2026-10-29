#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
PINNED=$(git rev-parse HEAD:themes/PaperMod)
git submodule update --init --remote themes/PaperMod
LATEST=$(git -C themes/PaperMod rev-parse HEAD)

if [[ "$LATEST" == "$PINNED" ]]; then
	echo "PaperMod is already up to date"
	exit 0
fi

echo "New PaperMod commits:"
git -C themes/PaperMod log --oneline "$PINNED..$LATEST"
echo "Preview with 'hugo server', then pin the new version:"
echo "  git add themes/PaperMod && git commit -m '[update] PaperMod'"
