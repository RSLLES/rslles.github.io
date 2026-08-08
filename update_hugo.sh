#!/bin/bash

# Define the repo and get system architecture
REPO="gohugoio/hugo"
ARCH=$(dpkg --print-architecture)

echo "--- Checking for Hugo updates ---"

# 1. Get the latest release JSON from GitHub API
LATEST_JSON=$(curl -s "https://api.github.com/repos/$REPO/releases/latest")
LATEST_VERSION=$(echo "$LATEST_JSON" | grep -Po '"tag_name": "v\K.*?(?=")')

# 2. Check current version
if command -v hugo >/dev/null 2>&1; then
	CURRENT_VERSION=$(hugo version | grep -oP 'v\K[0-9]+\.[0-9]+\.[0-9]+')
	echo "Current version: $CURRENT_VERSION"
else
	CURRENT_VERSION="0.0.0"
	echo "Hugo is not currently installed."
fi

echo "Latest version available: $LATEST_VERSION"

# 3. Compare and Install
if [ "$CURRENT_VERSION" != "$LATEST_VERSION" ]; then
	echo "Update available! Downloading version $LATEST_VERSION..."

	# Extract the download URL for the extended .deb package matching your arch
	# We look for "hugo_extended" to ensure you get Sass support
	DOWNLOAD_URL=$(echo "$LATEST_JSON" | grep -Po "\"browser_download_url\": \"\K.*hugo_extended_${LATEST_VERSION}_linux-${ARCH}\.deb(?=\")")

	if [ -z "$DOWNLOAD_URL" ]; then
		echo "Error: Could not find a .deb for architecture $ARCH. Checking for non-extended..."
		DOWNLOAD_URL=$(echo "$LATEST_JSON" | grep -Po "\"browser_download_url\": \"\K.*hugo_${LATEST_VERSION}_linux-${ARCH}\.deb(?=\")")
	fi

	# Download and Install
	TEMP_DEB="/tmp/hugo_update.deb"
	wget -O "$TEMP_DEB" "$DOWNLOAD_URL"

	echo "Installing..."
	sudo dpkg -i "$TEMP_DEB"
	rm "$TEMP_DEB"

	echo "--- Successfully updated to $(hugo version) ---"
else
	echo "You are already up to date."
fi

# 4. Keep the GitHub Actions workflow pinned to the locally installed version
WORKFLOW="$(dirname "$0")/.github/workflows/hugo.yaml"
INSTALLED_VERSION=$(hugo version 2>/dev/null | grep -oP 'v\K[0-9]+\.[0-9]+\.[0-9]+')
WORKFLOW_VERSION=$(grep -oP '^\s*HUGO_VERSION:\s*\K[0-9]+\.[0-9]+\.[0-9]+' "$WORKFLOW" 2>/dev/null)

if [ -n "$INSTALLED_VERSION" ] && [ -n "$WORKFLOW_VERSION" ] && [ "$INSTALLED_VERSION" != "$WORKFLOW_VERSION" ]; then
	echo
	echo "Workflow pins HUGO_VERSION $WORKFLOW_VERSION, but Hugo $INSTALLED_VERSION is installed."
	read -rp "Update the workflow to $INSTALLED_VERSION? [y/N] " REPLY
	if [[ "$REPLY" =~ ^[Yy]$ ]]; then
		sed -i "s/^\(\s*HUGO_VERSION:\s*\).*/\1$INSTALLED_VERSION/" "$WORKFLOW"
		echo "Updated $WORKFLOW."
	else
		echo "Left unchanged."
	fi
fi
