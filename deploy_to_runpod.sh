#!/bin/bash

# Deployment Script for Analyze API Fix
# Target: workspace/hybrid_vto_v1_latest_v1 on RunPod

set -e  # Exit on error

echo "=========================================="
echo "Analyze API Fix - Deployment Script"
echo "=========================================="
echo ""

# Configuration
REMOTE_HOST="root@157.157.221.30"
REMOTE_PORT="50423"
REMOTE_PATH="/workspace/hybrid_vto_v1_latest_v1"
LOCAL_PATH="."

echo "Target: $REMOTE_HOST:$REMOTE_PORT"
echo "Remote path: $REMOTE_PATH"
echo ""

# Files to deploy
FILES_TO_DEPLOY=(
    "core/flux2_cvton_runner.py"
    "services/analyze_service.py"
    "main.py"
    ".env"
)

echo "Step 1: Backing up current files on remote..."
ssh -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /workspace/hybrid_vto_v1_latest_v1
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
cp core/flux2_cvton_runner.py backups/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
cp services/analyze_service.py backups/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
cp main.py backups/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
cp .env backups/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
echo "✓ Backup created"
EOF

echo ""
echo "Step 2: Uploading modified files..."
for file in "${FILES_TO_DEPLOY[@]}"; do
    echo "  Uploading $file..."
    scp -P $REMOTE_PORT "$file" "$REMOTE_HOST:$REMOTE_PATH/$file"
done
echo "✓ Files uploaded"

echo ""
echo "Step 3: Verifying deployment..."
ssh -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /workspace/hybrid_vto_v1_latest_v1

echo "Checking Python syntax..."
python3 -m py_compile core/flux2_cvton_runner.py
python3 -m py_compile services/analyze_service.py
python3 -m py_compile main.py
echo "✓ Syntax check passed"

echo ""
echo "Checking imports..."
python3 -c "from services.analyze_service import AnalyzeService; print('✓ AnalyzeService imports successfully')"
python3 -c "from core.flux2_cvton_runner import Flux2CVTONRunner; print('✓ Flux2CVTONRunner imports successfully')"

echo ""
echo "Verifying .env changes..."
if grep -q "ANALYZE_VTON_CLOTH_ONLY_ENDPOINT" .env; then
    echo "⚠ WARNING: Old HTTP endpoint still in .env"
    echo "  Please remove: ANALYZE_VTON_CLOTH_ONLY_ENDPOINT"
else
    echo "✓ .env cleaned up correctly"
fi
EOF

echo ""
echo "Step 4: Restarting application..."
ssh -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /workspace/hybrid_vto_v1_latest_v1

# Find and kill existing process
echo "Stopping existing application..."
pkill -f "uvicorn main:app" || echo "No existing process found"
sleep 2

# Start application in background
echo "Starting application..."
FLUX2_ENABLE_LORA=0 FLUX2_REQUIRE_LORA=0 FLUX2_FUSE_LORA=0 ANALYZE_FLUX_DISABLE_LORA=1 \
nohup .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/app.log 2>&1 &
sleep 5

# Check if started successfully
if pgrep -f "../.venv/bin/uvicorn main:app" > /dev/null || pgrep -f "uvicorn main:app" > /dev/null; then
    echo "✓ Application started successfully"
    echo "  PID: $(pgrep -f '../.venv/bin/uvicorn main:app\|uvicorn main:app' | head -n 1)"
else
    echo "✗ Application failed to start"
    echo "  Check logs: tail -100 /tmp/app.log"
    exit 1
fi
EOF

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Test /analyze endpoint"
echo "2. Test /v1/flux2/tryon endpoint"
echo "3. Monitor logs for any errors"
echo ""
echo "To check logs:"
echo "  ssh -p $REMOTE_PORT $REMOTE_HOST 'tail -f /tmp/app.log'"
echo ""
