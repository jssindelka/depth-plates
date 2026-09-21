#!/bin/bash
# Double-click to run DEPTH//PLATES locally. Close this window (or press Ctrl+C) to stop.
cd "$(dirname "$0")"
PORT=4173
while lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; do PORT=$((PORT+1)); done
echo "DEPTH//PLATES  ->  http://localhost:$PORT"
echo "  site        http://localhost:$PORT/"
echo "  clean       http://localhost:$PORT/clean.html"
echo "  playground  http://localhost:$PORT/lab/"
echo "  recipes     http://localhost:$PORT/lab/recipes.html"
(sleep 1; open "http://localhost:$PORT/") &
python3 -m http.server $PORT
