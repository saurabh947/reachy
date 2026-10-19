# Architecture

Both diagrams show only what runs today.

**At a glance:**

![Reachy + SO-101 architecture at a glance](architecture-simple.svg)

**In detail:**

![Reachy + SO-101 system architecture](architecture.svg)

The robot runs on three layers:

- **Gemini 3.8 Live** is the voice and personality. It decides which tool to call.
- **Gemini Robotics-ER 2** finds things in the room, points at them, plans steps and, with arm tasks on, runs them step by step.
- **Executors** are the things that actually move:
  - the Reachy head, through the reachy_mini daemon;
  - the SO-101 arm, through the arm server.

The local emotion model is a tool that Gemini calls. It is the only way the robot judges emotion: Gemini 3.8 Live dropped affective dialogue, and ER 2 has none.

Reachy's memory, `reachy_memory.md`, is loaded into Gemini's prompt when a conversation starts.

## Which model does what

| Job | Model / part | Where it runs |
|---|---|---|
| Conversation: hearing, speaking, persona, choosing tools | Gemini 3.8 Live (`gemini-3.8-live`) | Google cloud |
| Finding things, pointing, planning steps | Gemini Robotics-ER 2 (`gemini-robotics-er-2-preview`, a preview, so `GEMINI_ER_MODEL` stays configurable) | Google cloud |
| Reading the user's emotion | emotion-detection-action SDK (`phase2_best_calibrated.pt`) | Mac GPU (MPS) |
| Head: gaze, room search, facing people | Reachy SDK, called by the app | Mac → daemon → head |
| Arm gestures (wave, gripper, rest) | Taught poses and recorded gestures, played by the arm server | Mac → SO-101 |
| Arm tasks (grasp, place) | ER 2 picks the points; the arm server plans the motion with LeRobot IK (placo + the SO-101 URDF) | Google cloud + Mac → SO-101 |
| Remembering across conversations | `reachy_memory.md`, read and written by the app | Mac |

Live runs on Google's Live API. ER 2 is called through the Interactions API with `store=False`, so camera frames aren't kept on Google's side.

## How a request flows

**"Where's the fridge?"**
If memory already has the fridge, Gemini answers from memory straight away, for example "where I last saw it: behind me, on my left". It doesn't search. People and pets move about, so for them it always searches. Otherwise:
1. Gemini hears the question through Reachy's mic, the app and Live. It calls `look_at_scene("fridge")` and says it's looking.
2. The app sends camera frames to ER 2, one heading at a time, turning Reachy's body until ER 2 finds the fridge.
3. The answer goes back to Gemini, which says it once it finishes its current sentence. The position is given from Reachy's default facing, for example "behind me on my left". Reachy then glances at the fridge and turns back. The location is saved to memory. If a search that checks every heading finds nothing, the location Reachy saw before is removed. A place you told it is kept, because it may just be out of sight, like keys in a drawer.

**"Wave at me."** Gemini calls `move_arm("wave")` and keeps talking. The arm server queues the move, plays it and parks the arm. The result goes into Gemini's context without a reply. If the move fails, Gemini is told so it can tell the user.

**A motor overloads while the arm is holding a pose.** The arm server freezes the arm, cancels its queued moves and announces the fault. The app waits for Gemini to finish its sentence, then tells it, and Gemini tells the user.

## Who waits for a tool's result

The rule: Gemini waits only when the user would otherwise hear a guess. A slow tool's answer arrives later, once Gemini pauses.

| Tool | Gemini waits? | How the result is delivered |
|---|---|---|
| `detect_emotion` | Yes. It's instant, and waiting means Gemini can't guess the mood | Straight away |
| `remember` | Yes. It's instant, and Gemini then knows it's saved | Straight away |
| `look_at_scene`, `plan_task` | No. They take seconds, and the prompt says the answer comes later | Spoken once Gemini finishes its sentence |
| `move_arm` | No | Added to context silently if it worked, spoken if it failed |
| Arm fault while holding | — | A notice sent once Gemini finishes its sentence |
| A call the user talked over | — | Dropped. A room search stops; a move keeps going, and only `stop` stops the arm |

## Memory

`reachy_memory.md` is in the repo folder; set `REACHY_MEMORY` to put it elsewhere. It's plain Markdown with four sections:
- **Where things are:** what room searches find (one thing, not a person or a pet), and places you tell it.
- **About the user:** name, preferences, and guidelines for how Reachy should behave.
- **About me (Reachy):** settled details of its personality.
- **Other notes.**

Gemini saves to it with the `remember` tool, on its own when it learns something lasting and whenever you ask. Each fact is one `- item: detail` line, and saving the same item again replaces it.

Each section holds up to 40 facts. When **Where things are** is full, the place Reachy saw longest ago makes room. Other sections refuse a new fact, and Gemini asks you what to forget. Nothing you told it or wrote yourself is dropped.

You can edit the file by hand, even while the app runs. Your lines, notes and sections are kept as written. If the file can't be read or is read-only, Reachy uses what it can and never writes over it. The file is gitignored because it's personal.

**"Pick up the eraser and put it in the cup"** (arm tasks on). Gemini calls `arm_task`, and the app:
1. Turns Reachy's head toward the arm's workspace, tilted down, and shows ER 2 the frame.
2. Runs ER 2's function-calling loop. ER points at the eraser and calls `grasp`; the app turns the point into a spot on the table (the camera ray meets the table) and the arm server plans and runs it: open, hover, come down carefully, close gently, lift.
3. Shows ER 2 a fresh frame after every action, so ER checks it really worked before calling `place` on the cup.
4. Sends Gemini ER's one-line summary, for example "Done: the eraser is in the cup."

One task runs at a time, with the head held still. "Stop" freezes the arm and keeps its grip.

## Arm tasks

They are off until the hardware checks pass. Set `ARM_TASKS` in `.env`: `off` (Live gets `plan_task`, which only plans), `hover` (the arm only hovers over each target, for testing) or `on`. They also need:
- **The arm's mount** in `.env`:
  - `ARM_BASE_X_CM` and `ARM_BASE_Y_CM`: where the arm's turning axis is, ahead of and to the left of Reachy's foot centre;
  - `ARM_BASE_YAW_DEG`: which way the arm faces.
- **The `ready` pose:** the arm raised over its workspace, gripper down.

Before any motion the arm server checks the whole plan:
- joint and wrist-roll limits;
- jumps between IK solutions;
- a 16 cm keep-out around Reachy;
- clearance above the table;
- reach.

A refused plan moves nothing.

To check the setup, step by step:

```bash
scripts/arm.sh ik-check                  # no hardware: what the arm can reach with this mount
scripts/arm.sh fk                        # read-only: where the gripper is as you move it by hand
scripts/arm.sh try hover --x 22 --y 29   # prints the plan, waits for Enter, hovers 10 cm above
```

## Teaching the arm

Poses are taught by hand on the follower:

```bash
scripts/arm.sh teach rest
scripts/arm.sh teach ready    # for arm tasks: raised over the workspace, gripper pointing down
```

Gestures are recorded with the SE leader arm:

```bash
scripts/arm.sh record wave
```

When recording, you move the leader and the follower mirrors it. Both are saved to `arm_motions.json`, and the arm server plays them back when Gemini asks.

## Not built yet

None of this is in the diagrams. The reasoning is in [the research report](../reports/Gemini%20robotics%20VLA%20for%20Reachy.md).

- **A second SO-101 arm.** It gets its own arm server, and `move_arm` gains an `arm` choice. The code already supports this.
- **Learned arm skills behind a new `run_arm_skill` tool:**
  - SmolVLA, trained on 50–100 demos per skill and run on the Mac;
  - or Gemini Robotics On-Device 2, which supports the SO-101 but is open to trusted testers only and runs on a Linux NVIDIA box.
- **ER 2 running learned skills** too (today it grasps and places with planned motion).
