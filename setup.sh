#!/usr/bin/env bash
# setup.sh — bootstrap the RAG_RUN virtual environment
#
# Usage:
#   bash setup.sh          # sets everything up
#   source setup.sh        # sets up AND activates the env in your current shell

set -euo pipefail

ENV_NAME="RAG_RUN"
REQUIRED_PYTHON_MINOR=12   # 3.12+
ENV_DIR="$(pwd)/$ENV_NAME"

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn] ${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*" >&2; }

# ── 1. Find a suitable Python (3.12+) ────────────────────────────────────────
find_python() {
    # Prefer explicit version binaries first
    for candidate in python3.12 python3.13 python3.14; do
        if command -v "$candidate" &>/dev/null; then
            echo "$candidate"; return
        fi
    done

    # pyenv — install 3.12 if not already present
    if command -v pyenv &>/dev/null; then
        local ver
        ver=$(pyenv versions --bare | grep -E '^3\.1[2-9]' | sort -V | tail -1 || true)
        if [[ -z "$ver" ]]; then
            info "pyenv found — installing Python 3.12.9 ..."
            pyenv install -s 3.12.9
            ver="3.12.9"
        fi
        echo "$(pyenv root)/versions/$ver/bin/python3"
        return
    fi

    # Last resort: plain python3 — check its version
    if command -v python3 &>/dev/null; then
        local minor
        minor=$(python3 -c "import sys; print(sys.version_info.minor)")
        if [[ "$minor" -ge "$REQUIRED_PYTHON_MINOR" ]]; then
            echo "python3"; return
        fi
    fi

    error "Python 3.$REQUIRED_PYTHON_MINOR+ not found."
    error "Install it via:  brew install python@3.12  OR  pyenv install 3.12.9"
    exit 1
}

PYTHON=$(find_python)
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
info "Using Python $PY_VER  ($PYTHON)"

# ── 2. Create virtual environment ────────────────────────────────────────────
if [[ -d "$ENV_DIR" ]]; then
    warn "Environment '$ENV_NAME' already exists at $ENV_DIR — skipping creation."
else
    info "Creating virtual environment '$ENV_NAME' ..."
    "$PYTHON" -m venv "$ENV_DIR"
    info "Environment created."
fi

# ── 3. Activate (works when sourced; subshell-only otherwise) ────────────────
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
info "Activated $ENV_NAME  (Python: $(python --version))"

# ── 4. Install requirements ──────────────────────────────────────────────────
if [[ -f "requirements.txt" ]]; then
    info "Installing requirements.txt ..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt
    info "Requirements installed."
else
    warn "requirements.txt not found — skipping package install."
fi

# ── 5. Load .env if present ──────────────────────────────────────────────────
if [[ -f ".env" ]]; then
    info "Loading environment variables from .env ..."
    # Export only KEY=VALUE lines; skip comments and blanks
    set -a
    # shellcheck disable=SC2046
    eval $(grep -E '^[A-Z_][A-Z0-9_]*=' .env | sed 's/[[:space:]]*#.*//')
    set +a
    info "API keys loaded:"
    grep -E '^[A-Z_][A-Z0-9_]*=' .env | sed 's/=.*//' | while read -r key; do
        echo "    $key"
    done
else
    warn ".env file not found — API keys not loaded."
    warn "Create one from .env.example:  cp .env.example .env"
fi

# ── 6. Done ──────────────────────────────────────────────────────────────────
echo ""
info "Setup complete."
echo ""
echo "  Environment : $ENV_DIR"
echo "  Python      : $(python --version)"
echo ""
echo "  To activate in a new shell:"
echo "    source $ENV_DIR/bin/activate"
echo ""
echo "  To deactivate:"
echo "    deactivate"
