#!/usr/bin/env bash
# Teach the SO-101 follower its moves, and check arm tasks (runs the arm server in .venv-lerobot).
#   scripts/arm.sh teach rest              # pose: move the arm by hand, press Enter
#   scripts/arm.sh teach gripper_open      # likewise: gripper_closed, and ready (arm raised over
#                                          # its workspace, gripper pointing down) for arm tasks
#   scripts/arm.sh record wave --seconds 5 # gesture: perform it on the leader arm
#   scripts/arm.sh fk [--registers]        # read-only: where the gripper is as you move the arm by hand
#   scripts/arm.sh ik-check                # no arm needed: the IK, the taught poses, the reach map
#   scripts/arm.sh try grasp --x 23 --y 22 # an arm task on the arm, step by step (hover, close, grasp, place)
# Needs ARM_PORT (and LEADER_PORT for record) in .env; find ports with lerobot-find-port.
# Arm tasks also need the ARM_BASE_* mount keys (see .env.example) and the URDF in third_party/so101.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/.venv-lerobot/bin/python" "$ROOT/src/reachy_emotion/arm_server.py" "$@"
