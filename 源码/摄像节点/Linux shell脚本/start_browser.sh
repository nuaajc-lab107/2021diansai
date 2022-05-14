#!/bin/bash
sleep 30
chromium-browser  --disable-popup-blocking --no-first-run --disable-desktop-notifications  --kiosk "http://169.254.177.18:8080/?action=stream"
