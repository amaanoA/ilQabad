#!/bin/bash
#
# Deploy ilQabad to Sipeed MaixCam device
#
# This script copies the necessary files to run the attendance system
# on a MaixCam device over SSH/SCP.
#
# Usage:
#   ./scripts/deploy_to_maixcam.sh <maixcam_ip> [username]
#
# Example:
#   ./scripts/deploy_to_maixcam.sh 192.168.1.100
#   ./scripts/deploy_to_maixcam.sh 192.168.1.100 root
#
# Prerequisites:
#   - MaixCam connected to same network
#   - SSH access to MaixCam (default user: root)
#   - Models downloaded locally (run download_models.py first)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
DEFAULT_USER="root"
REMOTE_PATH="/root/ilqabad"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <maixcam_ip> [username]"
    echo ""
    echo "Example:"
    echo "  $0 192.168.1.100"
    echo "  $0 192.168.1.100 root"
    exit 1
fi

MAIXCAM_IP="$1"
MAIXCAM_USER="${2:-$DEFAULT_USER}"

echo "=================================================="
echo "  ilQabad MaixCam Deployment"
echo "=================================================="
echo ""
echo "Target: ${MAIXCAM_USER}@${MAIXCAM_IP}:${REMOTE_PATH}"
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check for required files
echo "[1/5] Checking local files..."

check_file() {
    if [ -f "$1" ]; then
        echo -e "  ${GREEN}[OK]${NC} $1"
        return 0
    else
        echo -e "  ${RED}[MISSING]${NC} $1"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "  ${GREEN}[OK]${NC} $1/"
        return 0
    else
        echo -e "  ${YELLOW}[MISSING]${NC} $1/"
        return 1
    fi
}

MISSING=0

check_file "scripts/maixcam_demo.py" || MISSING=$((MISSING + 1))
check_dir "src" || MISSING=$((MISSING + 1))
check_dir "models/detection" || MISSING=$((MISSING + 1))
check_dir "models/recognition" || MISSING=$((MISSING + 1))
check_dir "models/liveness" || MISSING=$((MISSING + 1))

# Check for model files
if [ -d "models/detection" ]; then
    if [ -f "models/detection/yunet.onnx" ]; then
        SIZE=$(du -h "models/detection/yunet.onnx" | cut -f1)
        echo -e "  ${GREEN}[OK]${NC} models/detection/yunet.onnx ($SIZE)"
    else
        echo -e "  ${RED}[MISSING]${NC} models/detection/yunet.onnx"
        MISSING=$((MISSING + 1))
    fi
fi

if [ -d "models/recognition" ]; then
    if [ -f "models/recognition/mobilefacenet.onnx" ]; then
        SIZE=$(du -h "models/recognition/mobilefacenet.onnx" | cut -f1)
        echo -e "  ${GREEN}[OK]${NC} models/recognition/mobilefacenet.onnx ($SIZE)"
    else
        echo -e "  ${RED}[MISSING]${NC} models/recognition/mobilefacenet.onnx"
        MISSING=$((MISSING + 1))
    fi
fi

if [ -d "models/liveness" ]; then
    if [ -f "models/liveness/deeppixbis.onnx" ]; then
        SIZE=$(du -h "models/liveness/deeppixbis.onnx" | cut -f1)
        echo -e "  ${GREEN}[OK]${NC} models/liveness/deeppixbis.onnx ($SIZE)"
    else
        echo -e "  ${RED}[MISSING]${NC} models/liveness/deeppixbis.onnx"
        MISSING=$((MISSING + 1))
    fi
fi

if [ $MISSING -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Warning: $MISSING required file(s) missing.${NC}"
    echo "Run 'python scripts/download_models.py' to download models."
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Test SSH connection
echo ""
echo "[2/5] Testing SSH connection..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes "${MAIXCAM_USER}@${MAIXCAM_IP}" "echo connected" 2>/dev/null; then
    echo -e "  ${GREEN}[OK]${NC} SSH connection successful"
else
    echo -e "  ${RED}[FAIL]${NC} Cannot connect to ${MAIXCAM_USER}@${MAIXCAM_IP}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check MaixCam is powered on and connected to network"
    echo "  2. Verify IP address: $MAIXCAM_IP"
    echo "  3. Try: ssh ${MAIXCAM_USER}@${MAIXCAM_IP}"
    exit 1
fi

# Create remote directory structure
echo ""
echo "[3/5] Creating remote directories..."
ssh "${MAIXCAM_USER}@${MAIXCAM_IP}" "mkdir -p ${REMOTE_PATH}/{src,scripts,models/{detection,recognition,liveness},data}"
echo -e "  ${GREEN}[OK]${NC} Remote directories created"

# Copy files
echo ""
echo "[4/5] Copying files to MaixCam..."

# Function to copy with progress
copy_with_status() {
    local src="$1"
    local dst="$2"
    local name="$3"

    if [ -e "$src" ]; then
        scp -r "$src" "${MAIXCAM_USER}@${MAIXCAM_IP}:${dst}" 2>/dev/null
        echo -e "  ${GREEN}[OK]${NC} $name"
    else
        echo -e "  ${YELLOW}[SKIP]${NC} $name (not found)"
    fi
}

# Copy source code
echo "  Copying source code..."
scp -r src/* "${MAIXCAM_USER}@${MAIXCAM_IP}:${REMOTE_PATH}/src/" 2>/dev/null
echo -e "  ${GREEN}[OK]${NC} src/"

# Copy scripts
echo "  Copying scripts..."
scp scripts/maixcam_demo.py "${MAIXCAM_USER}@${MAIXCAM_IP}:${REMOTE_PATH}/scripts/" 2>/dev/null
echo -e "  ${GREEN}[OK]${NC} scripts/maixcam_demo.py"

# Copy models (these are large, show progress)
echo "  Copying models (this may take a while)..."

if [ -f "models/detection/yunet.onnx" ]; then
    scp "models/detection/yunet.onnx" "${MAIXCAM_USER}@${MAIXCAM_IP}:${REMOTE_PATH}/models/detection/"
    echo -e "  ${GREEN}[OK]${NC} models/detection/yunet.onnx"
fi

if [ -f "models/recognition/mobilefacenet.onnx" ]; then
    scp "models/recognition/mobilefacenet.onnx" "${MAIXCAM_USER}@${MAIXCAM_IP}:${REMOTE_PATH}/models/recognition/"
    echo -e "  ${GREEN}[OK]${NC} models/recognition/mobilefacenet.onnx"
fi

if [ -f "models/liveness/deeppixbis.onnx" ]; then
    scp "models/liveness/deeppixbis.onnx" "${MAIXCAM_USER}@${MAIXCAM_IP}:${REMOTE_PATH}/models/liveness/"
    echo -e "  ${GREEN}[OK]${NC} models/liveness/deeppixbis.onnx"
fi

# Copy enrollment data if exists
if [ -d "data/sample_faces" ]; then
    echo "  Copying enrollment data..."
    scp -r data/sample_faces "${MAIXCAM_USER}@${MAIXCAM_IP}:${REMOTE_PATH}/data/" 2>/dev/null
    echo -e "  ${GREEN}[OK]${NC} data/sample_faces/"
fi

# Verify deployment
echo ""
echo "[5/5] Verifying deployment..."
REMOTE_FILES=$(ssh "${MAIXCAM_USER}@${MAIXCAM_IP}" "ls -la ${REMOTE_PATH}/ 2>/dev/null | wc -l")
echo -e "  ${GREEN}[OK]${NC} Deployment verified (${REMOTE_FILES} items in remote directory)"

# Print usage instructions
echo ""
echo "=================================================="
echo "  Deployment Complete!"
echo "=================================================="
echo ""
echo "To run on MaixCam:"
echo ""
echo "  1. SSH into MaixCam:"
echo "     ssh ${MAIXCAM_USER}@${MAIXCAM_IP}"
echo ""
echo "  2. Navigate to project:"
echo "     cd ${REMOTE_PATH}"
echo ""
echo "  3. Run the demo:"
echo "     python scripts/maixcam_demo.py"
echo ""
echo "  Or with options:"
echo "     python scripts/maixcam_demo.py --skip-liveness"
echo "     python scripts/maixcam_demo.py --enroll-dir data/sample_faces"
echo ""
echo "  One-liner:"
echo "     ssh ${MAIXCAM_USER}@${MAIXCAM_IP} 'cd ${REMOTE_PATH} && python scripts/maixcam_demo.py'"
echo ""
