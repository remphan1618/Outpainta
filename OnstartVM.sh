env >> /etc/environment

# Original init script
/opt/ai-dock/bin/init.sh

# Wait for system to fully initialize
sleep 10

# ===== HARDWARE ENCODING FIX =====
echo "Starting hardware encoding fix..." > /var/log/hw-encoding-fix.log

# Create fix script in a persistent location
cat > /opt/ai-dock/bin/fix-hw-encoding.sh << 'EOL'
#!/bin/bash

echo "=== NVIDIA Hardware Encoding Fix for AI-Dock ==="

# Set proper environment
export GST_GL_API=gles2
export GST_GL_PLATFORM=egl
export SELKIES_ENCODER=nvh264enc
export SELKIES_BUFFER_TIME=30
export SELKIES_LATENCY=30
export SELKIES_MAX_LATENCY=150
export NVIDIA_DRIVER_CAPABILITIES=all
export NVIDIA_VISIBLE_DEVICES=all

# Fix permissions for config files
mkdir -p ~/.config/selkies
chmod 1777 /tmp 2>/dev/null || true
touch /tmp/selkies_config.json
chmod 777 /tmp/selkies_config.json 2>/dev/null || true

# Create optimized config
cat > ~/.config/selkies/config.json << EOF
{
  "encoder": "nvh264enc",
  "encoder_params": "preset=low-latency-hp zerolatency=true rc-mode=cbr-ld-hq bitrate=8000",
  "framerate": 60,
  "video_bitrate": 8000,
  "resize_width": 1920,
  "resize_height": 1080,
  "enable_resize": true,
  "enable_audio": true,
  "turn_protocol": "tcp"
}
EOF

# Fix device nodes if needed
if [ ! -e /dev/nvidia0 ]; then
    echo "Fixing NVIDIA device nodes..."
    mknod -m 666 /dev/nvidia0 c 195 0 2>/dev/null || echo "Could not create /dev/nvidia0"
    mknod -m 666 /dev/nvidiactl c 195 255 2>/dev/null || echo "Could not create /dev/nvidiactl"
    mknod -m 666 /dev/nvidia-uvm c 249 0 2>/dev/null || echo "Could not create /dev/nvidia-uvm"
fi

# Test hardware encoder directly
echo "Testing hardware encoder capability..."
if gst-launch-1.0 videotestsrc num-buffers=10 ! video/x-raw,width=320,height=240 ! nvh264enc ! fakesink -v &>/dev/null; then
    echo "✅ Hardware encoder is working!"
    ENCODER="nvh264enc"
    PARAMS="preset=low-latency-hp zerolatency=true rc-mode=cbr-ld-hq bitrate=8000"
else
    echo "❌ Hardware encoder test failed, using optimized software encoder"
    ENCODER="x264enc"
    PARAMS="tune=zerolatency speed-preset=ultrafast bitrate=6000 key-int-max=30 threads=8"
    
    # Update config for software encoding
    sed -i 's/"encoder": "nvh264enc"/"encoder": "x264enc"/g' ~/.config/selkies/config.json
    sed -i 's/"encoder_params": ".*"/"encoder_params": "tune=zerolatency speed-preset=ultrafast bitrate=6000 key-int-max=30 threads=8"/g' ~/.config/selkies/config.json
fi

# Stop unnecessary services
supervisorctl stop kasmvnc kasmxproxy kde-plasma 2>/dev/null || true

# Restart Selkies service
supervisorctl restart selkies-gstreamer || true

echo "Hardware encoding setup complete with encoder: $ENCODER"
echo "Please wait a few seconds before connecting for best experience"
EOL

# Make fix script executable
chmod +x /opt/ai-dock/bin/fix-hw-encoding.sh

# Run the fix script
/opt/ai-dock/bin/fix-hw-encoding.sh >> /var/log/hw-encoding-fix.log 2>&1

# Create a convenient restart script for the user
cat > /opt/ai-dock/bin/restart-desktop.sh << 'EOL'
#!/bin/bash
echo "Restarting optimized remote desktop..."
/opt/ai-dock/bin/fix-hw-encoding.sh
echo "Done! Please refresh your browser connection."
EOL

chmod +x /opt/ai-dock/bin/restart-desktop.sh

# Log completion
echo "Hardware encoding fix completed at $(date)" >> /var/log/hw-encoding-fix.log