#!/bin/bash
set -euo pipefail

# Fetch the latest Libron webfonts into static/fonts/libron.
# The OFL licence requires its text to ship alongside the font files.

REPO="nicoverbruggen/libron"
DEST="$(dirname "$0")/static/fonts/libron"
FONTS="Libron-Regular.woff2 Libron-Italic.woff2 Libron-Bold.woff2 Libron-BoldItalic.woff2"

TAG=$(curl -sf "https://api.github.com/repos/$REPO/releases/latest" | grep -Po '"tag_name": "\K.*?(?=")')
echo "Latest Libron release: $TAG"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

curl -sfL -o "$TMP/Libron_Web.zip" "https://github.com/$REPO/releases/download/$TAG/Libron_Web.zip"
curl -sfL -o "$TMP/LICENSE" "https://raw.githubusercontent.com/$REPO/$TAG/LICENSE"
unzip -q -j "$TMP/Libron_Web.zip" -d "$TMP/fonts"

# assets/css/extended/fonts.css references these exact file names
for FONT in $FONTS; do
	if [[ ! -f "$TMP/fonts/$FONT" ]]; then
		echo "Error: $FONT missing from Libron_Web.zip ($TAG); check fonts.css"
		exit 1
	fi
done

mkdir -p "$DEST"
for FONT in $FONTS; do
	cp "$TMP/fonts/$FONT" "$DEST/"
done
cp "$TMP/LICENSE" "$DEST/LICENSE"

git -C "$(dirname "$0")" status --short -- static/fonts/libron
