#!/bin/bash
# Copy and paste these commands one by one into your RunPod terminal

# Step 1: Go to project directory
cd /workspace/hybrid_vto_v1_latest_v1/ai

# Step 2: Pull latest changes
git pull origin kiran/latest_v1

# Step 3: Stop existing application
pkill -f "uvicorn main:app"
sleep 2

# Step 4: Start application
nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/app.log 2>&1 &
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
