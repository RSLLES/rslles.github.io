#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <path/to/image>"
	exit 1
fi

INPUT="$1"

if [[ ! -f "$INPUT" ]]; then
	echo "Error: file '$INPUT' not found"
	exit 1
fi

DIR=$(dirname "$INPUT")
BASENAME=$(basename "$INPUT")
NAME="${BASENAME%.*}"

OUTPUT_AVIF="$DIR/$NAME.avif"
# OUTPUT_WEBP="$DIR/$NAME.webp"

echo "Converting '$INPUT'..."
convert "$INPUT" -quality 85 "$OUTPUT_AVIF"
echo "  → $OUTPUT_AVIF"
# magick "$INPUT" -quality 85 "$OUTPUT_WEBP"
# echo "  → $OUTPUT_WEBP"
