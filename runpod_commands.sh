#!/bin/bash
# Copy and paste these commands one by one into your RunPod terminal

# Step 1: Go to project directory
cd /workspace/hybrid_vto_v1_latest_v1

# Step 2: Pull latest changes
git pull origin wardobe_and_tryon

# Step 3: Stop existing application
pkill -f "uvicorn main:app"
sleep 2

# Step 4: Start application
FLUX2_ENABLE_LORA=0 FLUX2_REQUIRE_LORA=0 FLUX2_FUSE_LORA=0 ANALYZE_FLUX_DISABLE_LORA=1 \
nohup .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/app.log 2>&1 &
sleep 3

# Step 5: Check if running
ps aux | grep uvicorn

# Step 6: Show recent logs
echo "========== Recent Logs =========="
tail -30 /tmp/app.log

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "To monitor logs: tail -f /tmp/app.log"
echo "To test API: curl http://157.157.221.30:8000/docs"
echo ""
