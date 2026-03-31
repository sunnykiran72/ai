#!/bin/bash

# Automatic Deployment Script
# This script does EVERYTHING for you!

set -e

echo "=========================================="
echo "🚀 Automatic Deployment to RunPod"
echo "=========================================="
echo ""

# Configuration
REMOTE_HOST="root@157.157.221.30"
REMOTE_PORT="50423"
REMOTE_PATH="/workspace/hybrid_vto_v1_latest_v1"

echo "📡 Connecting to RunPod..."
echo "Host: $REMOTE_HOST"
echo "Port: $REMOTE_PORT"
echo ""

# Execute all commands on remote server
ssh -p $REMOTE_PORT $REMOTE_HOST << 'ENDSSH'

echo "=========================================="
echo "Step 1: Going to project directory..."
echo "=========================================="
cd /workspace/hybrid_vto_v1_latest_v1
pwd
echo ""

echo "=========================================="
echo "Step 2: Pulling latest changes from GitHub..."
echo "=========================================="
git pull origin wardobe_and_tryon
echo ""

echo "=========================================="
echo "Step 3: Stopping existing application..."
echo "=========================================="
if pgrep -f "uvicorn main:app" > /dev/null; then
    echo "Found running application, stopping it..."
    pkill -f "uvicorn main:app"
    sleep 2
    echo "✓ Application stopped"
else
    echo "No running application found"
fi
echo ""

echo "=========================================="
echo "Step 4: Starting application..."
echo "=========================================="
FLUX2_ENABLE_LORA=0 FLUX2_REQUIRE_LORA=0 FLUX2_FUSE_LORA=0 ANALYZE_FLUX_DISABLE_LORA=1 \
nohup .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/app.log 2>&1 &
sleep 5
echo ""

echo "=========================================="
echo "Step 5: Checking if application started..."
echo "=========================================="
if pgrep -f "../.venv/bin/uvicorn main:app" > /dev/null || pgrep -f "uvicorn main:app" > /dev/null; then
    echo "✅ Application is running!"
    echo "PID: $(pgrep -f '../.venv/bin/uvicorn main:app\|uvicorn main:app' | head -n 1)"
else
    echo "❌ Application failed to start!"
    echo "Showing error logs:"
    tail -50 /tmp/app.log
    exit 1
fi
echo ""

echo "=========================================="
echo "Step 6: Showing recent logs..."
echo "=========================================="
tail -30 /tmp/app.log
echo ""

echo "=========================================="
echo "Step 7: Verifying changes..."
echo "=========================================="
echo "Checking if old HTTP endpoint is removed..."
if grep -q "ANALYZE_VTON_CLOTH_ONLY_ENDPOINT" .env; then
    echo "⚠️  WARNING: Old HTTP endpoint still in .env"
else
    echo "✅ Old HTTP endpoint removed"
fi
echo ""

echo "Checking Python imports..."
python3 -c "from services.analyze_service import AnalyzeService; print('✅ AnalyzeService imports successfully')" 2>&1
python3 -c "from core.flux2_cvton_runner import Flux2CVTONRunner; print('✅ Flux2CVTONRunner imports successfully')" 2>&1
echo ""

echo "=========================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "=========================================="
echo ""
echo "API is now running at: http://157.157.221.30:8000"
echo "API Documentation: http://157.157.221.30:8000/docs"
echo ""
echo "To monitor logs in real-time:"
echo "  ssh -p 53061 root@157.157.221.30 'tail -f /tmp/app.log'"
echo ""

ENDSSH

echo ""
echo "=========================================="
echo "🎉 All Done!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Test the /analyze endpoint"
echo "2. Test the /v1/flux2/tryon endpoint"
echo "3. Monitor for any errors"
echo ""
