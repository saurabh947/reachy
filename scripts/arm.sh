#!/usr/bin/env bash
# Teach the SO-101 follower its moves (runs the arm server in .venv-lerobot).
#   scripts/arm.sh teach rest              # pose: move the arm by hand, press Enter
#   scripts/arm.sh teach gripper_open      # likewise: gripper_closed
#   scripts/arm.sh record wave --seconds 5 # gesture: perform it on the leader arm
# Needs ARM_PORT (and LEADER_PORT for record) in .env; find ports with lerobot-find-port.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/.venv-lerobot/bin/python" "$ROOT/src/reachy_emotion/arm_server.py" "$@"
