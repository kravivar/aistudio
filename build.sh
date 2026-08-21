#!/bin/bash
set -e

echo "🚀 Building AI Studio macOS Desktop App with Nuitka..."

# Ensure we are in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  No virtual environment detected. Please run: source .venv/bin/activate"
    exit 1
fi

# Detect python version path for the virtual environment
PYTHON_VERSION=$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
SITE_PACKAGES="$VIRTUAL_ENV/lib/$PYTHON_VERSION/site-packages"

# Open WebUI bundles a massive SvelteKit frontend that Nuitka will completely ignore by default.
# We must explicitly bundle it into the binary's data folder.
if [ ! -d "$SITE_PACKAGES/open_webui/frontend" ]; then
    echo "❌ Error: Could not find open_webui frontend at $SITE_PACKAGES/open_webui/frontend"
    exit 1
fi

echo "📦 Bundling Open WebUI Frontend and MLX dependencies..."

# Leave 1 CPU core free to prevent the system from completely freezing during compilation
JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
# JOBS=1
echo "⚡ Forcing Nuitka to use $JOBS parallel jobs..."
# ==============================================================================
# 🔏 MOCK CODESIGN (BYPASS NUITKA BUFFER OVERFLOW BUG)
# ==============================================================================
# Nuitka crashes when building large ML projects on macOS because it passes
# too many .so files to the codesign command, exceeding the kernel's ARG_MAX limit.
# We bypass this by intercepting codesign during the build, letting Nuitka succeed,
# and then applying the signature manually using --deep at the end!
mkdir -p .mock_bin
cat > .mock_bin/codesign << 'MOCK'
#!/bin/bash
# Silently intercept and ignore Nuitka's codesign blasts to prevent ARG_MAX crash!
exit 0
MOCK
chmod +x .mock_bin/codesign
export PATH="$PWD/.mock_bin:$PATH"

echo "🔨 Building AI Studio with Nuitka..."
yes | nuitka \
    --jobs=$JOBS \
    --assume-yes-for-downloads \
    --standalone \
    --macos-create-app-bundle \
    --macos-app-name="AI Studio" \
    --output-folder-name="AI Studio" \
    --macos-app-version="0.1.0" \
    --macos-app-protected-resource="NSMicrophoneUsageDescription:AI Studio requires microphone access for Voice Mode." \
    --include-package=open_webui \
    --include-package=mlx \
    --include-package=mlx_lm \
    --include-package=mlx_whisper \
    --include-package=diffusers \
    --include-package=mflux \
    --include-package=fastapi \
    --include-package=uvicorn \
    --include-package=langchain_community \
    --include-package=langchain_core \
    --include-package=langchain_text_splitters \
    --include-package=langchain \
    --include-package=chromadb \
    --include-package-data=open_webui \
    --include-data-dir=src/aistudio/webui_tools=src/aistudio/webui_tools \
    --noinclude-data-files="*/open_webui/frontend/*" \
    main.py

echo "📦 Injecting Open WebUI Frontend assets and Alembic Migrations..."
mkdir -p "AI Studio.app/Contents/Resources/open_webui/frontend"
cp -r "$SITE_PACKAGES/open_webui/frontend/" "AI Studio.app/Contents/Resources/open_webui/frontend/"

# Open WebUI relies on Alembic, which needs raw .py files for migrations on disk
mkdir -p "AI Studio.app/Contents/MacOS/open_webui/migrations"
cp -r "$SITE_PACKAGES/open_webui/migrations/" "AI Studio.app/Contents/MacOS/open_webui/migrations/"
cp "$SITE_PACKAGES/open_webui/alembic.ini" "AI Studio.app/Contents/MacOS/open_webui/"

# Remove the mock codesign from PATH so we can use the real one
export PATH=$(echo $PATH | sed -e "s|$PWD/.mock_bin:||")
rm -rf .mock_bin

echo "🔏 Applying Apple Silicon ad-hoc codesignature to the App Bundle..."
xattr -cr "AI Studio.app"
codesign --force --deep -s - "AI Studio.app"

echo "✅ Build complete! You can find AI Studio.app in the current directory."

# Calculate and display total build time
BUILD_MINUTES=$((SECONDS / 60))
BUILD_SECONDS=$((SECONDS % 60))
echo "⏱️  Total build time: ${BUILD_MINUTES}m ${BUILD_SECONDS}s"
