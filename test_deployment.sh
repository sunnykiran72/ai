#!/bin/bash

# Test Script for Deployed Analyze API Fix
# Run this after deployment to verify everything works

set -e

echo "=========================================="
echo "Testing Deployed Analyze API Fix"
echo "=========================================="
echo ""

# Configuration
REMOTE_HOST="root@157.157.221.30"
REMOTE_PORT="53061"
API_BASE_URL="http://157.157.221.30:8000"

# Test 1: Check if application is running
echo "Test 1: Checking if application is running..."
ssh -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
if pgrep -f "uvicorn main:app" > /dev/null; then
    echo "✓ Application is running"
    echo "  PID: $(pgrep -f 'uvicorn main:app')"
else
    echo "✗ Application is NOT running"
    exit 1
fi
EOF
echo ""

# Test 2: Check application logs
echo "Test 2: Checking application logs for errors..."
ssh -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /workspace/hybrid_vto_v1_latest_v1/ai
if [ -f /tmp/app.log ]; then
    echo "Last 20 lines of log:"
    tail -20 /tmp/app.log
    echo ""
    if grep -i "error\|exception\|failed" /tmp/app.log | tail -5; then
        echo "⚠ Found errors in logs (check above)"
    else
        echo "✓ No recent errors in logs"
    fi
else
    echo "⚠ Log file not found"
fi
EOF
echo ""

# Test 3: Test /analyze endpoint (health check)
echo "Test 3: Testing /analyze endpoint..."
echo "Note: This requires a test image. Skipping actual upload test."
echo "Manual test command:"
echo "  curl -X POST $API_BASE_URL/analyze -F 'file=@test_garment.jpg' -F 'type=top'"
echo ""

# Test 4: Test /v1/flux2/tryon endpoint (health check)
echo "Test 4: Testing /v1/flux2/tryon endpoint..."
echo "Note: This requires test images. Skipping actual test."
echo "Manual test command:"
echo "  curl -X POST $API_BASE_URL/v1/flux2/tryon -H 'Content-Type: application/json' -d '{...}'"
echo ""

# Test 5: Verify Flux2 model is loaded
echo "Test 5: Verifying Flux2 model..."
ssh -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /workspace/hybrid_vto_v1_latest_v1/ai
if [ -d "./flux2-klein" ]; then
    echo "✓ Flux2 model directory exists"
    ls -lh ./flux2-klein/ | head -5
else
    echo "✗ Flux2 model directory not found"
fi
EOF
echo ""

# Test 6: Check environment configuration
echo "Test 6: Checking environment configuration..."
ssh -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
cd /workspace/hybrid_vto_v1_latest_v1/ai
echo "Checking critical environment variables..."
if grep -q "FLUX2_MODEL_PATH" .env; then
    echo "✓ FLUX2_MODEL_PATH configured"
else
    echo "✗ FLUX2_MODEL_PATH missing"
fi

if grep -q "FLUX2_LORA_PATH" .env; then
    echo "✓ FLUX2_LORA_PATH configured"
else
    echo "✗ FLUX2_LORA_PATH missing"
fi

if grep -q "ANALYZE_VTON_CLOTH_ONLY_ENDPOINT" .env; then
    echo "✗ Old HTTP endpoint still present (should be removed)"
else
    echo "✓ Old HTTP endpoint removed"
fi
EOF
echo ""

# Test 7: Check GPU availability
echo "Test 7: Checking GPU availability..."
ssh -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
    echo "✓ GPU available"
else
    echo "⚠ nvidia-smi not found"
fi
EOF
echo ""

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Automated tests completed."
echo ""
echo "Manual testing required:"
echo "1. Upload a test garment image to /analyze"
echo "2. Test /v1/flux2/tryon with sample data"
echo "3. Monitor performance and latency"
echo ""
echo "To monitor logs in real-time:"
echo "  ssh -p $REMOTE_PORT $REMOTE_HOST 'tail -f /tmp/app.log'"
echo ""
