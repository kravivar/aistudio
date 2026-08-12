#!/usr/bin/env bash
# ==============================================================================
# AI Studio - Third-Party Dependency Installer & Bootstrapper
# Clones pinned LTX-2-MLX repository into third_party/, patches package metadata,
# and synchronizes the project environment with uv.
# ==============================================================================

set -euo pipefail

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

LTX_REPO_URL="https://github.com/dgrauet/ltx-2-mlx.git"
LTX_PINNED_SHA="e1838a855bfd1640135c424c96cb27a0c0ad150e"
TARGET_DIR="third_party/ltx-2-mlx"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}⚡ AI Studio - Installing Third-Party Dependencies${NC}"
echo -e "${BLUE}======================================================${NC}"

# Check for required tools
command -v git >/dev/null 2>&1 || { echo -e "${RED}❌ git is required but not installed.${NC}" >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo -e "${RED}❌ uv is required but not installed.${NC}" >&2; exit 1; }

# Step 1: Clone or update the repository
mkdir -p third_party

if [ -d "${TARGET_DIR}/.git" ]; then
    echo -e "${YELLOW}🔄 Existing git clone detected in ${TARGET_DIR}. Updating...${NC}"
    cd "${TARGET_DIR}"
    git fetch origin
    git checkout "${LTX_PINNED_SHA}"
    cd - > /dev/null
elif [ -d "${TARGET_DIR}" ]; then
    echo -e "${YELLOW}📁 ${TARGET_DIR} exists without .git. Re-cloning fresh pinned copy...${NC}"
    rm -rf "${TARGET_DIR}"
    git clone "${LTX_REPO_URL}" "${TARGET_DIR}"
    cd "${TARGET_DIR}"
    git checkout "${LTX_PINNED_SHA}"
    cd - > /dev/null
else
    echo -e "${GREEN}📥 Cloning ltx-2-mlx (${LTX_PINNED_SHA:0:8})...${NC}"
    git clone "${LTX_REPO_URL}" "${TARGET_DIR}"
    cd "${TARGET_DIR}"
    git checkout "${LTX_PINNED_SHA}"
    cd - > /dev/null
fi

# Step 2: Patch invalid external readme references from hatchling pyproject.toml files
echo -e "${GREEN}🔧 Patching subpackage pyproject.toml files (removing external readme)...${NC}"
for pyproject in "${TARGET_DIR}"/packages/*/pyproject.toml; do
    if [ -f "${pyproject}" ]; then
        # Cross-platform in-place sed (macOS BSD sed vs GNU sed)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' '/readme =/d' "${pyproject}"
        else
            sed -i '/readme =/d' "${pyproject}"
        fi
        echo -e "   ✔ Patched: ${pyproject}"
    fi
done

# Step 3: Run uv sync
echo -e "${GREEN}🚀 Running uv sync --extra dev...${NC}"
uv sync --extra dev

echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}✅ All dependencies successfully installed and synced!${NC}"
echo -e "${GREEN}======================================================${NC}"
