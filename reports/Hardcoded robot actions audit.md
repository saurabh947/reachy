# Hardcoded arm and Reachy behaviour, and what an LLM could decide instead

## 1. Summary

I checked the 157-item inventory against the code at commit `638ff73` and merged duplicates. A second pass against the code found 8 more and corrected 9 rows. That makes **124 distinct hardcoded behaviours**, in seven groups:

- **A. Arm poses and gestures** (10)
- **B. Arm task sequences and parameters** (35)
- **C. Arm safety invariants and calibration** (25)
- **D. Head and gaze** (19)
- **E. Search and perception** (17)
- **F. Conversation and prompt-scripted behaviour** (13)
- **G. Text mode and other** (5)

How they split:
- **29 can be fully chosen by a model** (Live, ER 2, or both).
- **49 can partly be chosen by a model.** The model sets the parameters; code keeps the skill and the checks.
- **44 should stay fixed**, or be added as fixed safety behaviour. These are safety invariants, hardware calibration, contact loops that must react in tens of milliseconds, lifecycle steps and dev tools.
- **2 aren't counted:** F9 (Live already decides it) and G5 (connection config, not behaviour).

**What can and can't change:**
- **Keep the current split.** ER picks *what* and *where on the image*. Code does the metric geometry, IK, collision checks and the 50 Hz contact loops.
- **The hardcoding that blocks dynamic production use is above that split:**
  - **Arm motion.** It is built from one recorded wave, four taught poses and two fixed skills (top-down grasp, place). Heights, gripper openings, approach angles and the jaw direction for compact objects are fixed. Tasks go out and back through the taught `ready` pose; gestures go rest → first frame → rest; a successful grasp stays out at the hover, holding.
  - **Head search and gaze.** Search headings, tilts and gaze tours are fixed, and the head always faces the same way as the body.
  - **Prompts.** Prompt rules script the social behaviour.
- **These safety gaps must close before you add autonomy:**
  - Stopping the arm depends on Live hearing "stop" (C17).
  - Nothing stops the head or body mid-move (C24).
  - The "hand or person in the workspace" check (C16) and "only when the user asks" (F1) exist only in prompts.
  - `gripper_close` and `rest` close the gripper at full force, as gestures Live may make on its own (A4, A9).
  - Gestures skip the geometric checks (A6).
  - After a task, the arm can hold an object out indefinitely, with no way to let go except a new task (B33).

All `.py` paths below are under `src/reachy_emotion/` unless shown otherwise.
**Yes** = a model can decide it. **Partly** = a model sets the parameters and code keeps the skill and the checks. **Keep** = must stay fixed.

---

## 2. The full list

### A. Arm: poses and gestures

| # | Item | Where | Why it's fixed | Can an LLM decide it? How |
|---|---|---|---|---|
| A1 | Taught `rest` pose (folded park pose, motors off) | arm_motions.json:3-10; arm_server.py:836-847, 908, 1014, 1051, 1440-1441 | Safety: the only pose known to hold itself up with torque off. Its shoulder_lift of −107° is outside the URDF's ±100°, so IK can't produce it. | **Keep.** Live already decides *when* to rest. |
| A2 | Taught `ready` pose as a required via-point | arm_motions.json:27-34; arm_server.py:141, 933-944, 976-979, 995, 1013, 1050, 858-861 | Simplicity and safety: one known bridge from rest into the IK workspace, so only the ready↔hover move needs a path check | **Partly.** Keep one taught "unfold" pose for leaving rest. After that, compute via-points with IK at a clearance ER picks, and check them with `Reach.check` (see B10). |
| A3 | Fixed gripper ends: `gripper_open` 38.36, `gripper_closed` 0.27. Every grasp opens fully. | arm_motions.json:11-26; arm_server.py:991, 1011, 1046, 1140 | Hardware calibration. A full opening forgives aim error. | **Partly.** Keep the two values as calibration endpoints. ER points to two edges of the object; code turns that into an opening (width plus a margin), clamped between them. |
| A4 | Open/close-only `gripper_open` / `gripper_close` actions. `gripper_close` is a plain position move to 0.27, not the compliant close that grasps use. | arm_server.py:885-888, 917-920; arm_client.py:31 | Expressive vocabulary kept minimal | **Partly.** `gripper{percent, speed}` chosen by Live, but only after the close is force-limited (reuse `_close_on_object`): today a gesture close can pinch a finger at full force. Keep the refusal while holding (arm_server.py:913-914). |
| A5 | One recorded `wave`, replayed exactly. Only wrist_flex and wrist_roll move; the gripper is sent to 0.0. | arm_motions.json:36-1690 (150 frames, 4.97 s); arm_server.py:890-908 | Model capability: LLMs can't produce smooth 30 Hz joint streams. Leader-arm recordings look natural. | **Partly.** (1) A library of recorded gestures. (2) Bounded parameters: speed 0.5-1.5, repeats, amplitude, and a target bearing from ER. (3) 3-6 keyframes from Live or ER that the server smooths and checks. Never let a model stream raw joint positions. |
| A6 | Gestures and taught-pose moves skip every geometric check (no keep-out, table floor or roll limit; only LeRobot's 10° clamp) | arm_server.py:824-834, 885-908, 1380 | Recorded near rest by a human, so trusted | **Must change before A5 goes dynamic.** Interpolate every gesture finely and run `Reach.check` plus the roll limit on it. Refuse before moving if it fails. |
| A7 | Motions come only from demonstration: `teach` for poses; `record` at 30 Hz, 5 s by default, after a 3-2-1 countdown | arm_server.py:74, 1467-1537, 1869 | Natural and known to be safe | **Partly.** Keep it for the recorded library. Add model keyframe gestures behind A6's checks. |
| A8 | Live's `move_arm` list is a fixed tuple (wave, gripper_open, gripper_close, rest, stop), not the server's reported `actions` | arm_client.py:31; live_conversation.py:961 (`ArmClient.actions` is only logged at arm_client.py:216 and checked for `grasp` at live_conversation.py:581) | Simplicity: the tool list is set once at connect | **Yes.** Build the list at session start from the server's actions (minus hover, grasp and place), plus a description of each gesture in arm_motions.json. Today a newly recorded gesture can't be called, and `wave` is offered even when it isn't recorded. |
| A9 | Prompt triggers: "wave when greeting someone or saying goodbye, open or close the gripper as a small gesture, rest it when the conversation winds down" | live_conversation.py:193-200 | Prompt steering: Live had no other cue for when to move the arm | **Partly.** `rest` also closes the gripper (0.68), and the prompt tells Live to close it on its own, so A4's force limit comes first. Then describe what each gesture means and let Live choose from the conversation, detect_emotion and preferences saved in memory. Move "rest when idle" into a timer in code, so motor heat doesn't depend on Live remembering. |
| A10 | "Call move_arm once for each move, in order" | live_conversation.py:194-197 | Live used to batch, reorder or skip moves | **Partly.** Accept a `sequence` argument. The server queue still runs it in order, and `stop` still jumps the queue. |

### B. Arm: task sequences and parameters

| # | Item | Where | Why it's fixed | Can an LLM decide it? How |
|---|---|---|---|---|
| B1 | Fixed task vocabulary. The server runs rest, gripper moves, gestures, hover, grasp, place and stop, and treats any unknown action as a gesture name. ER gets only look, grasp, place and finish. | arm_server.py:738-743, 910-929; arm_task.py:66, 135-186, 383 | Each skill is planned and checked in full (IK, keep-out, guarded descent) before anything moves. ER outputs 2D points, not trajectories. | **Partly.** Add primitives the server checks and ER can call: `move_to`, `point_at`, `set_gripper`, `push/slide`, `rotate_wrist`, `handover`, `release_here`, `wait`. Each needs its own contact logic and tests; tests/test_arm_task.py and tests/test_arm_integration.py pin today's schema. |
| B2 | Grasp is one fixed sequence: open fully → ready → hover with the jaw tips 6 cm up → 0.3 s settle → guarded descent to 4 mm → back off 3 mm → close → lift to the hover. ER gives only a point, plus `ends` for long objects. | arm_server.py:982-1015, 596-613; arm_task.py:459-477 | A top-down pick of small objects needs no 3D model, and the whole path is planned before moving | **Partly.** Keep the order as a local skill (it contains 50 Hz loops). Add per-call parameters that ER sets: grasp depth, clearance, approach, gripper opening, grip profile, lift height. |
| B3 | `_GRASP_Z` 4 mm: the descent always aims at the table surface | arm_server.py:104, 606 | Built for small flat objects; the guard stops early on taller ones | **Yes.** ER points to the object's base and top, `height_above` gives its height h, and the grasp depth becomes about 0.4 h, clamped between 3 mm and h − 1 cm. |
| B4 | `_HOVER_Z` 6 cm: the pre-grasp hover and the minimum carry height | arm_server.py:103, 602-606, 623 | Clears small objects. Higher hovers lose reach (top-down reaches only 26 cm). | **Yes.** Clearance = max(6 cm, tallest nearby object + 3 cm), from ER. The planner still falls back or refuses. |
| B5 | Approach pitches 90/75/60° with maximum radii of 26/30/31 cm and a 10 cm minimum. The first one that plans wins, always in the arm's own vertical plane. | arm_server.py:100-102, 574-594 | SO-101 kinematics, taken from the reach map | **Partly.** Keep the radii. ER passes `approach: top/angled` to change the order tried. |
| B6 | Every descent and lift is a straight vertical line in 5 mm IK steps | arm_server.py:105, 507-526 | Easy to check against the table and keep-out, and easy to guard | **Partly.** Add an approach direction (along the tool axis, or horizontal for a shelf). The guard would need re-tuning for non-vertical moves. |
| B7 | ER's grasp point is projected onto a plane 5 mm above the table. ER is told objects must be "under 3 cm tall". | arm_task.py:57, 193-194, 464, 472 | One camera, no depth: object height is unknown | **Yes.** ER points to base and top to get h. Project the grasp point at h/2 and send h to the server. |
| B8 | Every target sits on one flat plane at the level of Reachy's foot. Nothing can be picked from or placed on a shelf, box or book. | arm_task.py:449-457; scene_brain.py:195, 219-233 | A known plane is the only way to turn a 2D point into a 3D one | **Partly.** ER points to where a raised surface meets the table and to its top edge (as it already does for containers), giving the surface height. That raises the floor locally and never lowers it. |
| B9 | Place is a fixed sequence: lift to at least 6 cm → move over the spot, 3 cm above the release → descend (guarded only for releases at 3 cm or lower) → open fully → back up → ready → rest | arm_server.py:1017-1051, 615-652 | Simplicity. Ending limp at rest is the safe default. | **Partly.** ER picks the release style (set down until contact, or drop from a height), the clearance and the end state. The server clamps them. |
| B10 | The arm always parks: after every place, and after a grasp that closed on nothing (open → ready → rest). No retry. | arm_server.py:1008-1015, 1049-1051; arm_task.py:161-165 | Known state, torque off, arm out of the camera's view | **Yes.** Add `end: stay/ready/rest`. An empty grasp ends open at the hover so ER can re-aim from a fresh frame; a place can end at ready so ER can chain objects. The server still parks on finish, timeout, stop, fault and session end. |
| B11 | Every place on the table lets go with the jaw tips 2 cm up | arm_task.py:58, 490; arm_server.py:111, 1035-1039 | Object height and what's underneath are unknown | **Partly.** Release height = surface height + object height + clearance (from B7/B8). The server keeps its 0.3-15 cm range. |
| B12 | Dropping into a container: rim height 2-9 cm (the 9 cm cap comes from the IK reach map), drop at the midpoint of the near and far rim points, release 3 cm above the rim | arm_task.py:59-62, 492-516; scene_brain.py:196, 236-252 | IK reach result; assumes a round opening | **Partly.** Replace the fixed cap with a dry-run reach check on the server. Let ER point at the centre of the opening. |
| B13 | Place has no yaw: the object lands in whatever orientation it was grasped | arm_server.py:638 | Only grasp needed a jaw direction | **Yes.** `yaw_deg` from two ER points. Reuse `solve()`'s yaw search, `_YAW_SLACK` and the roll limit. |
| B14 | Place constants: come down from 3 cm above (`_PLACE_ABOVE`); guard only releases at 3 cm or lower (`_GUARDED_BELOW`); refuse releases above 15 cm (`_MAX_RELEASE`) | arm_server.py:110-112, 216-220, 1035-1037 | Reach limits, plus the guard was thought necessary only near the table | **Partly.** Keep `_MAX_RELEASE`. Turn the guard on whenever ER says it is setting down onto any surface. Without it, the arm pushes into a box 5 cm up until a servo overloads. |
| B15 | The carry path knows no obstacles. It is checked only against the keep-out, the 3 cm floor and the lower of the two end heights. | arm_server.py:530-556, 623-650 | No scene geometry reaches the server; ER returns only points | **Yes.** ER lists obstacles (points plus estimated heights). They become table-frame cylinders added to `check()`, and the carry height rises above them. Obstacles can only add constraints. |
| B16 | One grip for every object: 0.5-unit close steps, contact at lag 2.5 and load 120, squeeze 1.5 past contact, hold load 40-120, 3 s limit | arm_server.py:127-136, 1135-1169 | Hardware calibration far below the overload trip, with no servo register writes; a 50 Hz loop | **Partly.** ER or Live classify the object (fragile, soft, rigid, heavy), which picks one of 3-4 grip profiles validated on the hardware. |
| B17 | Holding and drop thresholds: held means position more than 1 above closed and load above 30; a drop is two slack checks in a row at 1 Hz; ease the grip if load goes above 200 | arm_server.py:80, 137-140, 1171-1189 | The gripper's sensors are the only signal | **Partly.** Keep them. Let the grip profile set the held-load threshold (very light items read as "dropped") and let ER confirm by vision. |
| B18 | Speeds: 45°/s for poses and gestures, 30°/s in tasks, at most 1° per tick, descent 2 cm/s, lift 4 cm/s, at least 0.4 s per move, eased in and out | arm_server.py:67-69, 114-117, 824-834, 1069-1099 | Smooth moves under the 10° clamp. The guard is calibrated at 2 cm/s. | **Partly.** Keep these as caps. Live or ER pick a `speed_factor` of 0.3-1.0 (near hands, fragile items, a calm gesture). Descent speed stays fixed unless the guard is re-tuned. |
| B19 | Hover test: gripper 10 cm above the spot, held 10 s, then park | arm_server.py:109, 119, 959-980 (the docstring at :39 still says 2 s) | Commissioning: check the aim by eye | **Partly.** Turn it into a `point_at {x, y, height, hold_s}` gesture Live or ER can use ("this one?"). |
| B20 | Every arm task starts by looking at the table 18 cm in front of the arm's base, tilted 30° down | arm_task.py:63-64, 116-122, 324 | Latency (saves an ER turn), and ER doesn't know the mount | **Yes.** Live passes a hint (the place stored in memory, or the last look_at_scene result), or ER's first turn chooses the view. The work view stays as the fallback. |
| B21 | Task limits: 8 ER turns, 150 s | arm_task.py:53-54, 331-335, 400 | Cost, latency, runaway loops | **Partly.** Keep a hard ceiling. Live passes `max_steps` for multi-object goals, and each frame caption shows ER its remaining turns and time. |
| B22 | Retry and search scripted in ER's prompt: "look (down at the table first)", "finish … after a second failed try" | arm_task.py:207-213 | Prompt steering to keep the loop short | **Yes.** Remove these rules. Give ER its failure history, the server's error reasons and the remaining budget, and let it decide to retry, re-aim, search or give up. Keep the hard turn cap. |
| B23 | Arm limits copied by hand into ER's prompt (reach 10-26 cm, rim up to 9 cm, objects under 3 cm), although the server actually reaches 31 cm at a 60° approach | arm_task.py:193-196, 228 vs arm_server.py:101-102 | The app and the arm server run in separate venvs, so the app can't import the server's constants | **Partly.** The server reports its limits when it starts up. Better: a `can_reach` tool that runs `plan_grasp`/`plan_place` without moving. |
| B24 | How ER is told where the arm stands (a fixed sentence built from the `.env` mount) | arm_task.py:92-113, 217-228 | Hand-measured calibration | **Partly.** Keep `.env` as the source of truth. ER may only flag a mount that disagrees with what it sees, never override it. |
| B25 | ER always plans arm steps with low thinking (`thinking_level` "low") | arm_task.py:340 | Latency | **Partly.** Raise it after a failure, a refused point or a cluttered scene, or when Live passes a `careful` flag. |
| B26 | Fixed verification text, frame captions, fallback speech and outcome list | arm_task.py:67, 549-569, 589-592; fallback `say` strings 326, 335, 344, 352, 356, 394, 400, 616 | Grounding: stops ER from claiming success without visual proof | **Partly.** Keep the outcome list and "Claim success only if the new frame shows …". Let Live word failures from `did` and `outcome`. |
| B27 | Only one arm does tasks (the first one whose server offers `grasp`), and the config supports one arm | live_conversation.py:581; arm_client.py:187-191 | Only one follower arm exists | **Yes, once there are two.** Give ER each arm's limits and an `arm` argument, or pick the arm by dry-run reach. Needs a keep-out zone between the arms. |
| B28 | `arm_task` takes only a free-text `goal` | live_conversation.py:1005-1029 | ER does the spatial grounding; Live is weak at it | **Yes.** Add optional `hint`, `constraints`, `max_steps` and `confirm_before_contact`. |
| B29 | `plan_task` (plan only, no movement) replaces arm tasks when they're off, decided once per session | live_conversation.py:156-168, 202-205, 570-591, 1030-1054 | No mount, IK or `ready` pose | **Partly.** Re-check the server's actions per call, so tasks switch on without a restart. |
| B30 | `plan_task`'s prompt says the arm is "mounted on its left side" whatever `.env` says | scene_brain.py:90-97 | Stale: written before the mount setting existed | **Yes (a bug).** Build side, distance and limits from `arm_mount_from_env()`, as `arm_instruction` does. |
| B31 | Plan steps limited to locate/grasp/place/move, at most 8 steps. Nothing executes `move`, and `last_plan` is never read. | scene_brain.py:44, 51, 445, 507 | Placeholder for a skill router that was never built | **Partly.** Build the list from the server's actions, or retire `plan_task` in favour of the arm_task loop. |
| B32 | For a compact object (no `ends`), the jaws close at whatever angle the wrist roll already had, usually rest's ~7°. ER is told to give `ends` only for long objects. | arm_server.py:411-417 (seed copies `wrist_roll`), 431-433 (`jaw_yaw_deg` None keeps it); arm_task.py:201-202 | Simplicity: only long objects seemed to need a jaw direction | **Yes.** ER gives a jaw axis (two edge points) for anything that isn't round. The yaw search and `_YAW_SLACK` already exist. Today a rectangular eraser lying crosswise is gripped along its long side. |
| B33 | After a task ends holding something (a "pick up X" goal, or a failed or stopped task), the arm stays out at the hover, powered, indefinitely. `move_arm` can't release it: everything but `place` is refused while holding. | arm_server.py:913-914, 982-1007; arm_task.py:615-616 | The holding interlock (C8); no release policy was designed | **Partly.** A hold timeout in code (put back with a guarded descent), plus a `put_down` Live can call. Motor heat, and the arm stays out near people. |
| B34 | Arm tasks never update memory. After "put the eraser in the cup", memory still has the old spot, and E16's rule makes Live repeat it. | live_conversation.py:594-627 (no `memory` passed); arm_task.py (no place writes) | Not built | **Yes.** On `done`, write the place target, or forget the object's place. |
| B35 | "If several things match, don't move: ask" | arm_task.py:210-212 | Prompt steering | **Partly.** Give ER context (memory, what the user said before) and let it pick, behind `confirm_before_contact` (B28). |

### C. Arm: safety invariants and calibration

| # | Item | Where | Why it's fixed | Can an LLM decide it? How |
|---|---|---|---|---|
| C1 | Keep-out cylinder of 16 cm around Reachy's foot (the arm's shells, about 2.5 cm, are not modelled) | .env.example:59-62; arm_server.py:270, 548-552, 577-579 | Never hit Reachy | **Keep.** ER may only *add* temporary zones (a hand, a glass, a laptop). |
| C2 | Table floors: every checked arm point at least 3 cm up; only the jaws may go to 3 mm, and only on contact moves | arm_server.py:107-108, 553-555 | No surface map | **Keep** as minimums. ER may raise them locally (B8). |
| C3 | Joint limits (URDF, widened only for the taught rest and ready poses) and wrist roll ±90° | .env.example:66-67; arm_server.py:272, 365-379, 538-543 | Mechanical limits; the roll once overloaded against its cable | **Keep.** |
| C4 | `max_relative_target` 10° | arm_server.py:70, 1380 | Last line of defence against jumps | **Keep.** |
| C5 | Guarded descent: contact is a load change over 150 or a lag change over 3° for 3 ticks; then freeze and back off 3 mm | arm_server.py:120-126, 1101-1133 | Must react in about 60 ms; an LLM takes seconds | **Keep.** A model may only supply an expected contact height. |
| C6 | Motor faults: status polled every 0.25 s while moving and every 1 s while idle and powered. Overload, over-current or overheat cancels the queue, freezes the arm and is announced once; a fault or a lost arm ends the ER task. | arm_server.py:79-85, 167-181, 800-816, 1303-1348; arm_task.py:385-388, 537-545; live_conversation.py:812-836 | Hardware protection | **Keep.** Live decides what to tell the user. A new task starts only after the user approves; never retry through a fault. |
| C7 | Stop: jumps the queue, cancels waiting moves, freezes. While holding, the gripper keeps its squeeze, and joints within 3° keep their goal. | arm_server.py:106, 782-798, 1223-1238; arm_client.py:105-109 | A predictable emergency behaviour | **Keep.** Models may trigger it, never shape it. |
| C8 | Holding interlock: while holding, only place, stop, status and shutdown run; no grasp while holding; no place with an empty gripper | arm_server.py:894-897, 913-914, 971-972; arm_task.py:461-462, 481-482 | Gesture frames would close the gripper at full force | **Keep.** |
| C9 | rest(), park() and settle(): reach rest within 4° in 1.5 s, or freeze and raise; open the jaws first; go via ready if the arm is out | arm_server.py:72-73, 836-880 | Never drop the arm short of rest, never squeeze while parking | **Keep.** |
| C10 | Bring-up, then move to rest at startup | arm_server.py:1392-1404, 1438-1447 | Known, relaxed starting state | **Keep.** |
| C11 | Shutdown: put a held object back where it was picked up (1 cm, guarded); if that's refused, open where the arm is; then park. Also runs when the app's connection closes. | arm_server.py:113, 1053-1067, 1461-1462; arm_client.py:140-152; live_conversation.py:1456-1460 | No model is available at shutdown | **Keep as the fallback.** While the app is alive, Live or ER may choose a put-down spot first. The fallback could descend guarded before opening; today it can drop the object from height. |
| C12 | `ARM_TASKS` gate (off / hover / on) | .env.example:33-38; arm_task.py:51, 80-89; live_conversation.py:570-591 | Operator safety gate | **Keep.** Never under model control. |
| C13 | One physical action per ER turn; `finish` refused in a turn that moved | arm_task.py:69-73, 360-383 | Each step must be checked in a fresh frame | **Keep** for contact actions. Several looks or zooms per turn could be allowed. |
| C14 | ER returns only [y,x] points on the latest frame. Points on an old frame, a frame without a head pose, or rays less than 10° below level are refused. | arm_task.py:199-201, 434-457; scene_brain.py:219-233 | ER's 2D pointing is reliable; its metric 3D isn't | **Keep.** This is the right split between model and code. |
| C15 | Timeouts: 90 s per grasp or place, then stop; status 5 s; server start 30 s; shutdown 60 s | arm_task.py:55-56, 528-535; arm_client.py:50, 140 | Watchdogs | **Keep.** They could scale with the planned path length, computed in code. |
| C16 | Refusals for hands, people and hazardous objects exist only in ER's prompt | arm_task.py:196-198, 208-210 | Safety policy | **Keep the policy, and add a code check:** a local person or hand detector on the latest frame before every contact move. |
| C17 | Stop depends on Live hearing "stop" and calling `move_arm(stop)`. Talking over Reachy deliberately doesn't cancel arm moves. | live_conversation.py:206-208, 704-708, 1264-1272, 1294, 1306 | A laugh shouldn't freeze a wave | **Keep the decoupling, but add a stop path that doesn't go through an LLM:** a keyword check on the input transcription (1342-1345) or a local spotter, plus a physical e-stop, both calling `submit('stop')` directly. **This is a production gap.** |
| C18 | One task at a time; looks refused during a task or in the same turn; `move_arm` takes only stop during a task | live_conversation.py:704-713, 1280-1304 | One owner of the head and the arm | **Keep.** Add a `tell_task` path for corrections mid-task, and a queue of goals. |
| C19 | Queue of at most 8 moves, run in order | arm_server.py:75, 1243-1250 | Bounded backlog | **Keep.** |
| C20 | 50 Hz commands, 3 bus retries, 0.3 s settle before a guarded descent | arm_server.py:67, 71, 118 | Servo bus timing; the guard needs a settled baseline | **Keep.** |
| C21 | Jaw geometry: the jaws reach 7.5 mm past the gripper frame; the grasp centre is 2 cm toward the moving jaw, marked *[unverified]* in the code | arm_server.py:89-93 | Fixed geometry | **Keep.** Calibrate `_JAW_CENTRE_M` once: hover over a mark and have ER point at the jaws and the mark. |
| C22 | Mount and joint calibration (ARM_BASE_X/Y/YAW/Z, joint offsets, URDF). Z is corrected by hand for the loaded arm sitting lower. | .env.example:39-58, 63-70; arm_server.py:241-274, 311-326 | Measured once with a ruler | **Partly.** A periodic camera check of where the gripper is, in which ER points at the gripper in several IK poses and the mount is refit. Apply it only within bounds and after confirmation. |
| C23 | IK numerics: 60 steps, 2 mm / 3° tolerance, orientation weight 0.1, 5° branch-flip limit, fixed elbow-up seed, 10° yaw slack | arm_server.py:94-99, 411-417, 483-505 (IK loop), 507+ (`line()`, branch-flip check) | Solver accuracy and speed | **Keep.** The yaw slack could become a per-object tolerance from ER (a pen vs a ball). |
| C24 | Head and body motion has no stop. `move_arm(stop)` reaches only the arms. A gaze tour (up to 8 × 1.3 s), `face_toward`, `go_neutral` and the room search's turns check no stop flag; a search stops only between headings, and only when Live cancels the call. | live_conversation.py:704-715, 839-852; scene_brain.py:369-396 | Not built | **Must add.** A stop that reaches the head and body, beside C17. The body turns up to ±150° next to the arm. |
| C25 | With two arms, `move_arm` needs an `arm` name; there is no "stop everything" | live_conversation.py:702, 961-963 | One arm today | **Must add** before a second arm: one stop for all arms (and the head, C24). |

### D. Head and gaze

| # | Item | Where | Why it's fixed | Can an LLM decide it? How |
|---|---|---|---|---|
| D1 | Facing words map to 8 headings: 0, ±45, ±90, ±135, and 150 for "behind me". Used by look_at_scene and by ER's look tool. | live_conversation.py:252-255, 1094-1100; arm_task.py:137-141, 419-432 | Live handles words reliably; free numbers invite wild angles. 150 respects the body limit. | **Partly.** Add a numeric `facing_deg` (clamped to ±150) next to the words. Give ER `look_at: [y,x]` on the latest frame, turned into a head pose by `_gaze_targets`/`look_at_image`. |
| D2 | Three tilts only: level 0°, down 30°, up −25° | live_conversation.py:256, 1102-1109; arm_task.py:142-143 | The head tilts about ±35° | **Partly.** A numeric `tilt_deg`, clamped in code, or a tilt derived from an ER point. |
| D3 | A look in a given direction checks only that view, then goes back to neutral; no search follows | live_conversation.py:430-437 | Honours what the user said | **Yes.** A `then_search` argument, or widen ±45° in an ER look loop. |
| D4 | After a find, the head visits every found object (up to 8; 0.8 s move and 0.5 s hold each, skipping targets past ±180°), then returns to neutral | scene_brain.py:369-396; live_conversation.py:481 | Shows what was found; timings tuned by hand | **Yes.** Live picks `gaze: none / first / all / label` and the hold time; ER ranks which object matters. |
| D5 | face_toward turns the body to the first person found and stays there, head level. On the current frame that is ER's first item; the largest Haar face is used only when ER finds nothing, in a sweep or in a directed look. | scene_brain.py:399-413; live_conversation.py:480 | Simple "eye contact" | **Partly.** Pitch toward the face with `elevation_deg`. Live picks which person, via ER and a description. |
| D6 | Back to neutral (body 0°, head level) after tours, misses, errors, cancelled calls and the end of an arm task | scene_brain.py:393-394; live_conversation.py:336-341, 377-381, 528-534, 617-627 | Known camera pose; recovery | **Partly.** Keep it on error, cancel, fault and task end. After a successful look, Live picks the end pose (stay, face the user, or neutral). |
| D7 | No expressive head motion. The prompt says the "head moves ONLY" through look_at_scene; there is no nod, shake, tilt or idle tool. | live_conversation.py:159, 172; tools at 955-1114 | Simplicity, honesty, no fight over the head | **Yes.** An `express` tool with clamped offsets, or named gestures with an intensity. It runs under `_MOTION_LOCK`, is refused during searches and tasks, and is rate-limited. |
| D8 | Antennas are always reset to the SDK rest position | scene_brain.py:331 | No antenna control was designed | **Yes.** Antenna patterns or angles in the same `express` tool. |
| D9 | No attention behaviour: no face tracking and no turning to a new speaker. The face box FaceCropper finds every frame is used only for the emotion crop. | scene_brain.py:399-411; faces.py:71-103 | It would compete with searches and tasks for the head | **Yes.** A local tracker at about 10 Hz whose mode (user / scene / off) Live switches. LLM latency is too high to drive the tracking itself. |
| D10 | Emotion never drives motion. `announce_emotion` ("I detect you seem …") is exported but never called. | live_conversation.py:1148-1191; tts_announcer.py:110-123; \_\_init\_\_.py:8, 12 | There is no motion tool to react with; announce_emotion is left over | **Yes.** Live picks a physical reaction through D7/D8. Delete announce_emotion. |
| D11 | Neutral pose: 0.8 s, min-jerk, antennas at rest | scene_brain.py:320-334 | SDK rest pose | **Partly.** Keep it as home. The motion style could come from the `express` tool. |
| D12 | Neutral at startup; the head isn't parked at shutdown | live_conversation.py:1412-1415, 1456-1467 | Lifecycle | **Keep.** Optionally add `goto_sleep` at exit. |
| D13 | Body yaw ±150° (the SDK allows 160° body and 65° head-to-body). Clamped only in `face_toward` and `work_view_deg`; the shared `_turn` clamps nothing and is safe only because `_FACING_DEG` tops out at 150. The gaze tour *skips* targets past ±180°. | scene_brain.py:56, 385-386, 406-409; arm_task.py:65, 122; live_conversation.py:300-309 | Kinematic limits | **Keep**, and move the clamp into `_turn` before any numeric `facing_deg` (D1). |
| D14 | Turns at most 90°/s and at least 0.6 s; turns across the back go via the front | live_conversation.py:300-309 | Motion blur; head and body spinning opposite ways | **Keep.** A style may only slow it down. |
| D15 | Wait for a frame at least 0.3 s after a move (give up after 2 s); video to Live at 1 fps; 0.1 s gap in the emotion loop | live_conversation.py:54-55, 248, 872-882 | Blur, pipeline latency, Live's input-rate limit | **Keep.** |
| D16 | Gaze targets computed from the pose the frame was captured at | scene_brain.py:337-366 | `look_at_image` uses the current pose | **Keep.** Route any model-chosen "look at a point" through it. |
| D17 | `_MOTION_LOCK`: head motions run one at a time, and an arm task holds the head for its whole run | live_conversation.py:244, 328, 357, 617, 839-852 | Each frame must match its head pose | **Keep.** Future expressive or tracking motions should be interruptible and dropped, not queued. |
| D18 | Directions reported in 8 fixed sectors (edges at 22.5/67.5/112.5/157.5°) plus "down low" / "up high" at ±20°; Live may not convert them to the user's left and right | scene_brain.py:276-298; live_conversation.py:150-155 | Live flips left and right | **Partly.** Also return the numeric bearing and elevation, and a user-relative direction computed in code. |
| D19 | The head always faces where the body faces (head yaw = body yaw), with roll and x/y/z at 0. Every look turns the whole body; the head's own ±65° yaw, lift, lean and roll are never used to see. | scene_brain.py:320-334 (`go_neutral`); live_conversation.py:300-309 | Simplicity | **Partly.** ER asks for a view (peek into a cup, over an obstacle, a quick glance without turning), code clamps it. The camera geometry already uses the full pose. |

### E. Search and perception

| # | Item | Where | Why it's fixed | Can an LLM decide it? How |
|---|---|---|---|---|
| E1 | Room-search headings: 0, +60, +120, +150, −60, −120, −150 | scene_brain.py:55; live_conversation.py:363 | Cover the full circle with the ~80° camera inside ±150°, with a known number of ER calls | **Partly.** An ER-driven loop that picks the next view: look and finish tools, a list of headings already seen, and memory as a hint. Keep the fixed sweep as the fallback and cap the number of views. |
| E2 | Fixed search order; headings within 20° of the start are skipped; the search stops at the first hit | live_conversation.py:344-382 | Latency | **Yes.** Order by a prior. A `find_all` option for "how many …". Re-check ambiguous hits. |
| E3 | The search always looks level | live_conversation.py:366 (`_turn` defaults to tilt 0, :300) | A second tilt doubles the ER calls | **Yes.** ER returns a tilt for each view. |
| E4 | Always glances 30° down before a room search for a non-person query | live_conversation.py:448-450 | Most things asked about are on the table | **Yes.** ER or Live suggests the first follow-up view; keep the glance as the default. |
| E5 | A query counts as a person search only if it is exactly one of 7 words (person, people, me, user, face, someone, human) | live_conversation.py:246-247, 265-270 | Routes to the fast face finder | **Yes.** A `target_kind` argument from Live, or an `is_person` flag from ER. Today "the guy" isn't treated as a person search. |
| E6 | Person search uses a frontal Haar face detector (640 px, scaleFactor 1.1, minNeighbors 5); profiles and faces beyond about 2 m are missed | live_conversation.py:278-287, 427, 440-447; faces.py:19, 41 | 10-20 ms against seconds per ER call; works offline | **Partly.** On the current frame ER already runs first. The gap is the room sweep and directed looks, which use Haar only: fall back to ER ("a person, any pose") or a better local detector there. |
| E7 | look_at_scene takes only query, facing and tilt, and its description promises a fixed sequence | live_conversation.py:1062-1111 | Live must know what will happen | **Yes.** Add arguments (max_items, find_all, gaze, end_pose, numeric facing/tilt, target_kind, hint), with today's behaviour as the defaults. |
| E8 | At most 8 objects per locate (prompt, schema and parser) | scene_brain.py:43, 62, 70, 163 | Latency, tokens, length of the gaze tour | **Yes.** `max_items` from Live, capped at about 20. |
| E9 | Fixed locate prompt; with no query it asks for "the distinct objects in view" | scene_brain.py:58-63, 483 | Google's documented ER pointing format | **Partly.** Keep the format; Live adds `context`. |
| E10 | Strict JSON schema for points; an unreadable reply is an error | scene_brain.py:64-88, 453-478; live_conversation.py:441-443 | A garbled answer must never mean "nothing there" (which would start a search or forget a place) | **Keep.** Extend it only with optional fields. |
| E11 | ER thinking: low for locate, high for plan | scene_brain.py:484, 498 | One call per heading while the user waits | **Partly.** Raise it automatically on empty or ambiguous results, or on a Live `careful` flag. |
| E12 | ER timeout 20 s, 1 retry | scene_brain.py:45-49, 420-431 | A stalled call must not hold the head lock | **Keep.** |
| E13 | ER always gets the full frame at JPEG quality 85; no crop or zoom | scene_brain.py:50, 459; arm_task.py:584 | Bandwidth against detail | **Partly.** Keep the quality. Add an ER `zoom` tool that crops around a point and maps its points back to the full frame. |
| E14 | Table geometry: head 17.7 cm above the table, 15 cm height ceiling, rays shallower than 10° below level refused | scene_brain.py:195-196, 219-233 | SDK head offset; shallow rays turn small pointing errors into large ones | **Keep.** It's a condition ER meets by looking down. |
| E15 | Places are remembered automatically: only for a single find whose label isn't a person or pet (word list); forgotten after a whole-room miss; stored as bearings relative to the robot | live_conversation.py:257-262, 273-275, 455-462, 477-478; memory.py:213-233 | Live forgot to save, and only code has the bearing | **Partly.** ER tags each object `movable`. Store positions relative to landmarks, because robot-relative bearings go stale whenever Reachy is moved. |
| E16 | Prompt rules for looking: always call look_at_scene; answer from memory unless it may have moved; always re-look for people and pets; use "the person" for the user | live_conversation.py:133-148 | Live otherwise eyeballs its 1 fps video and makes up locations | **Partly.** Keep the honesty rules. Make "memory or look again" Live's own call, using the memory's `seen` date (memory.py:216). |
| E17 | "ONE kind of thing per look_at_scene call": "find my keys and phone" becomes two calls, each of which can sweep the whole room | live_conversation.py:1084-1090 | Simple matching of a find to its query | **Yes.** A list of queries, or `find_all`, answered from the same frames. |

### F. Conversation and prompt-scripted behaviour

| # | Item | Where | Why it's fixed | Can an LLM decide it? How |
|---|---|---|---|---|
| F1 | Consent gate: arm_task only when the user asks, "never on your own" | live_conversation.py:174-175, 1009-1013 | No surprise manipulation. **A safety policy enforced only by the prompt**, like C16 and C17: no code checks it. | **Partly.** The user grants scoped standing permissions stored in memory ("keep the desk tidy"). Live announces each task first, and a code-side check confirms no person is in the workspace. |
| F2 | Honesty rules: never claim a look or task result before the tool answers | live_conversation.py:145-148, 163-166, 175-179 | Non-blocking tools mean Live keeps talking before results exist | **Keep.** |
| F3 | Which tools Live waits for: look, plan, task and move run in the background; remember and detect_emotion block | live_conversation.py:960, 1002, 1060 | Latency; fixed at connect by the Live API | **Keep.** |
| F4 | When results make Live speak: scene and task results once Reachy finishes speaking (WHEN_IDLE); successful `move_arm` results silently (SILENT) | live_conversation.py:536-541, 665-671, 723-731 | Avoids repetition and cut-off sentences | **Keep it in code.** Consider interrupting for faults and drops, which today wait up to 15 s. |
| F5 | System notices ending "Tell the user briefly", sent once Reachy stops speaking (at most 15 s) | live_conversation.py:759-782, 785-809 | Unprompted events must reach the user | **Partly.** Keep the delivery. Replace "tell the user" with "decide what to do", with a hard rule: no retry after a fault or an unreachable arm. |
| F6 | A dropped object is only reported | live_conversation.py:802-806 | Recovery is uncertain | **Yes.** Live chooses to look for it, then offers a new arm_task (the consent gate applies). |
| F7 | When the user talks over Reachy, a running search stops, its result is dropped and the head goes neutral | live_conversation.py:528-534, 1248-1251, 1267-1272 | A stale answer shouldn't be spoken | **Partly.** Pass the dropped result to Live silently, as context. |
| F8 | Memory layout: sections places/user/reachy/notes, 40 items, 60/300 characters, oldest place dropped when full | memory.py:34-42, 200-208 | Keeps the prompt small | **Keep.** |
| F9 | Prompt policy for what to save in memory | live_conversation.py:224-234 | Live already chooses the content | **Already decided by Live** (not counted); only the section list is fixed (F8). Add a "procedures" section for routines the user teaches, replayed through arm_task. |
| F10 | Persona and reply length (2-3 sentences, under 25 words, US English). The "TOOL & MOVEMENT RULES" heading contains no movement rules. | gemini_bridge.py:41-87; conversation_app.py:67-70 | Short, friendly demo voice | **Partly.** A persona file per deployment. Put gesture guidance under that heading once D7 exists. |
| F11 | Emotion only from detect_emotion, never judged from video | gemini_bridge.py:78-81 | The calibrated local model beats Live's guess | **Keep.** Treat "warming" or "no face visible" as unsure. |
| F12 | Emotion warm-up 3 s; readings older than 2 s are refreshed | local_inferencer.py:41, 44 | The model's 3 s window | **Keep.** |
| F13 | Fixed error strings ("camera frame not available yet", "I can't see well enough to do that.", busy refusals) | live_conversation.py:412, 620, 710, 713, 1288-1289, 1303 | They describe facts | **Keep.** Let Live word the spoken line at :620. |

### G. Text mode and other

| # | Item | Where | Why it's fixed | Can an LLM decide it? How |
|---|---|---|---|---|
| G1 | Text mode (`--text`) has no robot actions: its only tool is detect_emotion, with at most 5 tool calls per turn | gemini_bridge.py:39, 232; main.py:124 | An older debug path | **Yes.** One tool registry shared by Live and text mode. |
| G2 | gTTS with a blocking sleep; voice input with fixed energy thresholds on a code path main never takes (`voice_mode=False`) | tts_announcer.py:78, 98-102; voice_input.py:22-26; main.py:124 | Legacy | **Keep or remove.** voice_input could host the local "stop" spotter (C17). |
| G3 | Dev scripts with fixed motions (hello_move demo; camera_check's work view, 30° tilt, 1.5 s) | scripts/hello_move.py:24-45; scripts/camera_check.py:53, 62-63, 119 | Bench tools must be reproducible | **Keep.** |
| G4 | Arm CLI test defaults (`try --above` 4.6 cm, hold the close for 10 s, reach-map grid) | arm_server.py:1670, 1777, 1840-1848, 1880 | Bench only | **Keep.** |
| G5 | Connection settings (ARM_PORT, ARM_ID, LEADER_*, ARM_MOTIONS) | .env.example:26-31; arm_server.py:277-291 | Deployment config, not behaviour | Not applicable (not counted). |

---

## 3. What stays fixed and what becomes dynamic

**Must stay fixed in a production setting.** A model may *trigger* these or *add* constraints, never relax them:
- **Physical limits:** keep-out around Reachy, table floors, joint and wrist-roll limits, the 10° per-step clamp, speed caps, body yaw ±150°, head turn speed (C1-C4, B18 caps, D13-D14). A model error here means a collision or a damaged servo.
- **Anything in a loop faster than about 1 s:** guarded-descent contact detection, the compliant close, drop and fault polling (C5, C6, B16-B17 internals). An LLM takes seconds; these must react in about 60 ms or within the servo's ~2 s overload trip.
- **Emergency behaviour:** stop, freeze on a fault, the holding interlock, park/rest mechanics, shutdown put-back, watchdog timeouts (C6-C11, C15). They must be predictable and work when the app or the model is gone.
- **Operator gates and ownership:** `ARM_TASKS`, the head lock, one task at a time (C12, C18, D17).
- **Grounding:** points-only output from ER, one contact action per turn with a fresh frame, the strict JSON schema, the honesty rules (C13-C14, E10, F2). Without them, models invent coordinates and outcomes.
- **Calibration:** mount, joint offsets, jaw geometry, IK numerics, camera timing (C20-C23, D15-D16, E14). A model may *check* these, never override them.

**Fix these gaps before adding autonomy:**
- A stop that doesn't go through an LLM (C17), reaches the head and body too (C24), and stops every arm (C25).
- A code-side hand and person check before contact moves (C16), and a code check for "only when the user asks" (F1).
- A force-limited gripper close for gestures and rest (A4, A9).
- Geometric checks on gesture playback (A6).
- A hold timeout and a `put_down` (B33).

**Should become dynamic** (Live, ER 2 or both, through tool arguments the server clamps):
- **Gestures:** which gesture, its style and its target (A4-A5, A8-A10).
- **Grasp and place parameters per object:** depth, clearance, gripper opening, approach, grip profile, release style and height, place yaw, support surface, obstacles (B2-B16).
- **Task flow:** end state, chaining, retries, first view, budget, thinking level (B10, B20-B22, B25).
- **Head:** search, gaze, tilt and end pose (D1-D6, E1-E9).
- **Expressive head and antenna motion and attention mode:** D7-D10.
- **Memory and recovery decisions:** E15-E16, F5-F6.

---

## 4. Roadmap for dynamic production environments

**Dependency first: the camera-to-table calibration is unresolved.**
- Every ER-chosen metric parameter (heights from `height_above`, widths, obstacle positions, support surfaces) inherits this error.
- The only saved check, `captures/20261022-181500/capture.json`, has its single find (the blue tape) about 6.7 cm from the nearest measured mark.
- Switching to the square-pixel K that `scripts/camera_check.py` tests moves that point only about 0.3 cm, so the error source is still open.
- `_JAW_CENTRE_M` is marked unverified (arm_server.py:93), and Z is corrected by hand for the loaded arm sitting lower.
- Close these before giving ER more metric freedom.

1. **Safety floor for autonomy** (C16, C17, C24, C25, A4, A6, A9's idle rest, B33)
   - *Change:* a local "stop" keyword on the transcript or a local spotter, plus an e-stop, stopping every arm and the head directly; a local person and hand detector before every contact move; a force-limited gesture close; `Reach.check` on every gesture and pose move; an idle-rest timer and a hold timeout in code.
   - *Needs:* the transcript hook; a light local detector.
   - *Risk:* false stops, which are cheap.
2. **Calibration and limits reported by the server** (B23, C21-C22)
   - *Change:* fit the camera-to-table error with camera_check. The server reports its limits when it starts. A `can_reach` dry-run tool for ER, using `plan_grasp`/`plan_place` without moving. Optionally a periodic camera check of the mount.
   - *Risk:* a bad automatic fit shifts the whole workspace, so only apply bounded fits after confirmation.
3. **ER chooses grasp and place parameters per object** (A3, B2-B9, B11-B16, B32)
   - *Change:* ER points to base/top/edges/surface, code turns them into heights and widths, and the server clamps grasp depth, clearance, gripper opening, approach preference, grip profile, release style, place yaw and obstacles.
   - *Needs:* `height_above` reused for objects; 3-4 grip profiles validated on the hardware; the guard on for every set-down.
   - *Risk:* 1-2 cm monocular errors. The guarded descent stays as the safety net.
4. **Remove the ready/rest round trip** (A2, B10, B21-B22, B25)
   - *Change:* an end-state argument; via-points computed by IK; an empty grasp ends at the hover so ER re-aims; chaining across objects. Retry policy and thinking level come from the server's error reasons and the remaining budget in each caption.
   - *Risk:* the arm stays out and powered near people for longer, and the planner refuses more often.
5. **A primitive library for ER instead of taught gestures** (B1, B6, B19, E13, A5)
   - *Change:* `move_to`, `point_at`, `set_gripper`, `push/slide`, `handover`, `release_here`, `zoom`, plus gestures the server generates from parameters (a wave aimed at a person's bearing, a nod, a beckon). All are checked by the server, and recorded clips stay as a library.
   - *Needs:* contact and guard logic per primitive; new tests.
   - *Risk:* a larger vocabulary means more wrong tool choices and more ways to collide.
6. **ER-chosen search and gaze** (D1-D6, D19, E1-E8, E17, B20)
   - *Change:* an ER loop that picks the next view, with `look_at: [y,x]` and numeric facing and tilt (clamped), a coverage tracker, memory bearings as a hint, and the fixed sweep as the fallback. Live gets `gaze`, `end_pose`, `target_kind`, `max_items` and `find_all`.
   - *Risk:* slower searches and model loops, so a hard view cap is needed.
7. **Model judgement instead of prompt scripts, plus expressive tools** (A8-A9, D7-D10, F5-F6, B28, C18, F1)
   - *Change:* build `move_arm` from the server's actions and gesture descriptions; an `express` head and antenna tool; a switchable local attention tracker; notices that let Live decide what to do; drop recovery; `arm_task` hints, constraints and `confirm_before_contact`; `tell_task` corrections mid-task. Then scoped standing permissions, gated by item 1.
   - *Risk:* over-gesturing, Live's weakness with numbers, unrequested motion near people.

**Cheap fixes to do any time:**
- The left-arm `plan_task` prompt (B30).
- Arm tasks updating memory (B34).
- The body-yaw clamp in `_turn` (D13).
- Building the `move_arm` list from the server's actions (A8).
- The stale "2 s" hover docstring (arm_server.py:39).
- The dead `announce_emotion`.
- The unused `last_plan` (scene_brain.py:445, 507).

This audit only read the code and the saved captures. No code was edited, nothing ran on the hardware, and no APIs were called.