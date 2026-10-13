# LeRobot GitHub: SO-100/SO-101 follower wrist_roll overload, stalls, full-turn calibration and leader/follower mismatch

Research date: 2026-10-13. Method: `gh search issues/prs --repo huggingface/lerobot` (about 40 queries), full bodies, comments and reviews of about 30 issues and PRs, and the LeRobot source read at tags v0.2.0, v0.3.2, v0.4.0, v0.4.2, v0.4.3, v0.4.4, v0.5.0, v0.6.1 and `main` (as of 2026-10-13). No hardware was touched.

Release dates, for placing findings by version: v0.4.2 2025-11-27, v0.4.3 2026-01-22, v0.4.4 2026-02-27, v0.5.0 2026-03-09, v0.5.1 2026-04-07, v0.6.0 2026-07-06, v0.6.1 2026-08-03 — [Releases](https://github.com/huggingface/lerobot/releases)

GitHub Discussions are **disabled** on huggingface/lerobot (`has_discussions: false` from the GitHub API), so there are no Discussions to search. Some threads point to the LeRobot Discord instead, for example [#3196 comment](https://github.com/huggingface/lerobot/pull/3196#issuecomment-4164660444).

---

## Q1. Which GitHub issues report "Overload error" on so100/so101 followers (especially wrist_roll / ID 5)? Root causes and fixes?

### Takeaway
I found **no issue or PR reporting the exact symptom**: `Failed to write 'Torque_Enable' on id_=5 ... [RxPacketError] Overload error!` on wrist_roll. Overload reports in the repo are mostly about the **gripper (ID 6)**, or about joints driven into a **hard stop** after a calibration or direction mismatch. The fixes were a lower gripper torque limit (#1809), recalibration and drive-mode fixes, power cycling, or replacing the servo. Stall protection, a startup-mismatch guard and goal-bound checks are all still open PRs, none merged as of 2026-10-13.

### Cited Findings
- A search for `"overload error"` in issues and PRs returns only PR #4695 (disconnect cleanup). A broader `Overload` search returns #3585, #2819, #3942, #4447, #4783, #1809 and #2075. None of them is a wrist_roll/ID 5 overload report — [gh search: Overload](https://github.com/huggingface/lerobot/issues?q=Overload)
- **#2819** (2026-01-19, LeRobot 0.4.3, SO-101). The follower's gripper servo went stiff. Teleop then failed with `ConnectionError: Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6] after 1 tries. [TxRxResult] Incorrect status packet!`, and a power cycle did not help. Maintainer pkooij said the servo had most likely broken from being overloaded too long. His example cause: closing the leader gripper fully while the follower gripper is holding an object, so the follower motor keeps pushing. The user replaced the servo. Closed 2026-01-30 — [#2819](https://github.com/huggingface/lerobot/issues/2819), [maintainer comment](https://github.com/huggingface/lerobot/issues/2819#issuecomment-3768335194)
- **#1809** (merged 2025-08-29). Lowers the gripper's max current and torque by 50% on SO100/SO101 to avoid burnout. The PR says burnout happened more often because of miscalibration and because the leader gives no feedback when the operator closes it fully. Seeed Studio reported the problem — [PR #1809](https://github.com/huggingface/lerobot/pull/1809)
- In v0.6.1, `SOFollower.configure()` writes protection registers **only for the gripper**: `Max_Torque_Limit=500`, `Protection_Current=250`, `Overload_Torque=25`. wrist_roll and the other joints keep whatever is in servo EEPROM. That matches the user's servo 5 values: Max_Torque_Limit 1000, Overload_Torque 80, Protection_Time 200, Protection_Current 310 — [so_follower.py v0.6.1 L159-L171](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/so_follower.py#L159-L171)
- **#4783** (open, 2026-09-27) makes those gripper values configurable. It explains the STS3215 semantics: `Overload_Torque` is the load at which the overload countdown starts. After `Protection_Time` (2 s factory default) the servo drops its output to `Protective_Torque` (20% default). It also notes that the bus only checked byte width, so out-of-range values such as `Max_Torque_Limit=1500` would have reached the servo — [PR #4783](https://github.com/huggingface/lerobot/pull/4783)
- **#3942** (open, 2026-07-05, LeRobot 0.5.2, SO101). At teleop start the follower gripper (ID 6) opened to its maximum instead of matching the closed leader, then locked up. The reporter attributed this to Feetech overload/torque-off protection and lost control until a power cycle. The follower's gripper `drive_mode` stayed 0 and its `homing_offset` was near 0, against the leader's -436. Related PR #3948, "fix(calibration): detect inverted gripper drive_mode on SO follower/leader calibration", is closed — [#3942](https://github.com/huggingface/lerobot/issues/3942), [PR #3948](https://github.com/huggingface/lerobot/pull/3948)
- **#3585** (open, 2026-05-12) describes how the STS3215 handles Goal_Position writes outside `[Min_Position_Limit, Max_Position_Limit]`:
  - the firmware clamps the goal to the nearest limit, wrap-aware;
  - it sets `Torque_Enable=1` on the next PID tick, even if the host had disabled torque;
  - the Goal_Position register keeps the value the host wrote.
  - [#3585](https://github.com/huggingface/lerobot/issues/3585)
- In a #3585 comment (2026-09-24), a user with a new SO-101 follower reported that **shoulder_lift and wrist_flex moved the wrong way on every command and pushed into the hard stop until they overloaded**. Setting `Min_Position_Limit=0` / `Max_Position_Limit=4095` on those two motors fixed it, keeping the calibrated homing offset. Restoring the calibrated limits brought the fault back. Their calibrated ranges were shoulder_lift 856-3303 and wrist_flex 836-3221 — [#3585 comment](https://github.com/huggingface/lerobot/issues/3585#issuecomment-5816932404)
- **#3954** (open, 2026-07-06) adds a startup joint-mismatch guard to teleoperate and record. On the first frame, a large leader/follower mismatch is either blended over 1.5 s or aborted. Default threshold: 10 action units. The motivating SO-101 incident: a wrongly calibrated gripper drive direction caused a full-speed slam on the first teleop frame that stripped a follower motor gear — [PR #3954](https://github.com/huggingface/lerobot/pull/3954)
- **#2075** (open since 2025-09-28) adds an initial pose check. Its rationale is that large initial leader/follower differences make the follower move abruptly, causing overload or collisions — [PR #2075](https://github.com/huggingface/lerobot/pull/2075)
- **#4447** (open, 2026-08-13) adds opt-in SO follower stall protection:
  - new options `stall_current_thresholds`, `stall_position_tolerance` and `stall_timeout_s`;
  - it trips only when high current and an unresolved position error both persist;
  - on a trip it disables torque and raises `MotorStallError`.
  - Maintainer nicolas-rabault requested changes on 2026-09-30. He would rather freeze the arm at its present position than disable torque, because the arm then falls under gravity. He also wants the new fields forwarded for `bi_so_follower`, and flagged a `max_relative_target`/tolerance combination that turns the protection off.
  - A community tester triggered a real stall on an SO-101 gripper. Current ramped from 9 to 123 at a stuck position, and the error fired after 0.81 s.
  - [PR #4447](https://github.com/huggingface/lerobot/pull/4447), [review](https://github.com/huggingface/lerobot/pull/4447#issuecomment-5911331546)
- **#2693** (open, 2025-12-21, LeRobot 0.4.0): after recalibrating on another computer, the follower's wrist roll motor went partially stiff and stopped responding (checked in FT SCServo Debugger). No resolution has been posted — [#2693](https://github.com/huggingface/lerobot/issues/2693)
- **#1333** (2025-06-17, SO-100) reported severe wrist_roll instability and unwanted rotation during teleop, correlated with elbow_flex motion. The reporter suspected the default P_Coefficient of 16. It was closed as stale on 2026-04-14 with no code fix — [#1333](https://github.com/huggingface/lerobot/issues/1333)
- **#3131** (2026-03-11, LeRobot 0.4.4, Hiwonder servos) reported follower disconnects during teleop: `Failed to sync read 'Present_Position'`, then `RuntimeError: FeetechMotorsBus motor check failed`. The maintainer suspected wiring and suggested 30 fps instead of 60. The user worked around it with read retries. Closed 2026-07-29 — [#3131](https://github.com/huggingface/lerobot/issues/3131). Since then, v0.6.1 has `num_read_retries: int = 2` in `SOFollowerConfig` — [config_so_follower.py v0.6.1](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/config_so_follower.py#L49-L53)

### Inferences
- **Likely cause of the user's case** (an inference that combines the code facts in Q2/Q3 with the user's measurements):
  - Both arms' wrist_roll are calibrated as 0-4095, with the zero at the ENTER pose.
  - `use_degrees=True` maps the leader's angle 1:1 onto the follower.
  - So when the leader rolls more than about ±77° from its zero (the follower's 154° travel, if centred), the follower is commanded into its mechanical stop. If the follower's ENTER pose was not centred in its travel, this happens even sooner.
  - With `max_relative_target=None` and no position limits, the servo pushes at up to `Max_Torque_Limit`. Once load stays above `Overload_Torque` (80%) for `Protection_Time` (200 × 10 ms = 2 s per #4783's description), protection trips, the LED flashes and the overload status bit is set.
  - Teleop sends the leader's pose on the very first frame, which explains why the wrist moves at once and overloads soon after start (the same failure class #3954 and #2075 describe).
- The #3585 report (calibrated limits causing wrong-direction pushes into a hard stop) is a caution: writing tight position limits is not risk-free on every arm or firmware.

### Gaps
- No issue reports the exact `Failed to write 'Torque_Enable' on id_=5 ... [RxPacketError] Overload error!` string, or a wrist_roll overload at teleop start.
- No issue mentions Waveshare SO-ARM101 kits, Waveshare "ST3215 (SE)" leader servos, or STS3215 firmware 3.9.
- The Discord thread referenced in #3196 is not publicly indexable, so I could not read it.

---

## Q2. Why does LeRobot calibrate wrist_roll as full-turn (0-4095)? Are there proposals to record its range or add limits? Has this changed up to 0.6.x or main?

### Takeaway
SO-100 code always treated wrist_roll as full-turn. **SO-101 recorded wrist_roll's range like any other joint up to v0.4.2.** It became full-turn when the SO-100 and SO-101 classes were merged in **PR #2780** (merged 2026-01-12, shipped in v0.4.3). That behaviour is unchanged in v0.5.x, v0.6.0, v0.6.1 and `main` as of 2026-10-13.

Maintainers have said wrist_roll is *expected* to be full-turn (Maximellerbach, 2026-04-14). Several printed SO-101 arms, however, have **printed-part hard stops at about 338-345°**. Two open PRs would sweep or measure its range: **#4583** (manual sweep) and **#4826** (`--auto`). Neither is merged. In the user's version, **no real position limits are written** for wrist_roll.

### Cited Findings
- **SO-100 history.** `so100_follower.py` at v0.3.2 and v0.4.0 already had `full_turn_motor = "wrist_roll"` and `range_maxes[full_turn_motor] = 4095` (lines 129/137) — [so100_follower.py v0.4.0](https://github.com/huggingface/lerobot/blob/v0.4.0/src/lerobot/robots/so100_follower/so100_follower.py#L129-L137)
- **SO-101 before the merge.** At v0.4.2, `so101_follower.py` calls `self.bus.record_ranges_of_motion()` over **all** motors, with the prompt to move all joints through their entire ranges — [so101_follower.py v0.4.2](https://github.com/huggingface/lerobot/blob/v0.4.2/src/lerobot/robots/so101_follower/so101_follower.py#L125-L135)
- **The merge.** PR #2780 ("feat(robots): consolidates bi SO setups", merged 2026-01-12) deleted the SO10X classes in favour of one SO base class for SO100 and SO101, with aliases kept. It states that re-calibration is required. From v0.4.3, `so_follower.py` has `full_turn_motor = "wrist_roll"` (line 132) — [PR #2780](https://github.com/huggingface/lerobot/pull/2780), [so_follower.py v0.4.3](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/robots/so_follower/so_follower.py#L130-L140)
- **Current code, v0.6.1.** `SOFollower.calibrate()`:
  - disables torque;
  - prompts "Move ... to the middle of its range of motion and press ENTER....";
  - calls `set_half_turn_homings()` on every motor, including wrist_roll, so the ENTER pose reads about 2047;
  - records ranges for every joint except `wrist_roll`;
  - sets `range_mins["wrist_roll"]=0` and `range_maxes["wrist_roll"]=4095`.
  - `SOLeader.calibrate()` does the same (v0.6.1 so_leader.py L103-L111).
  - [so_follower.py v0.6.1 L115-L157](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/so_follower.py#L115-L157), [so_leader.py v0.6.1](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/teleoperators/so_leader/so_leader.py#L84-L111)
- `FeetechMotorsBus.write_calibration` writes `Homing_Offset`, `Min_Position_Limit=range_min` and `Max_Position_Limit=range_max`. For wrist_roll that is **0 and 4095, which in effect means no limit** — [feetech.py v0.6.1 L268-L276](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/motors/feetech/feetech.py#L268-L276)
- **`main` as of 2026-10-13** keeps the same logic (so_follower.py L160-L168). The differences from v0.6.1 are small:
  - an ASCII homing-pose diagram;
  - a link to the calibration video;
  - the prompt now reads "Move ... to the middle of its range of motion (shown above)";
  - `int()` casts on the calibration values;
  - `_get_half_turn_homings` clamps the offset to ±2047 instead of failing when a joint sits at 4095.
  - [so_follower.py main](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py), [feetech.py main](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/feetech.py)
- **#3193** (2026-03-21, LeRobot 0.5.1, SO-101, STS3215) reported two problems after wrist_roll became full-turn:
  - (1) calibration crashed with `ValueError: Magnitude 2073 exceeds 2047 (max for sign_bit_index=11)` inside `set_half_turn_homings()`;
  - (2) leader and follower wrist_roll zero points did not correspond, because each zero is just wherever the wrist was at ENTER.
  - The reporter noted that earlier versions calibrated wrist_roll like the other joints.
  - It was closed 2026-04-14 as fixed by #3373, which fixed the crash only.
  - [#3193](https://github.com/huggingface/lerobot/issues/3193), [closing comment](https://github.com/huggingface/lerobot/issues/3193#issuecomment-4242525173)
- **#3373** (merged 2026-04-13, so in v0.6.0+) clears bit 4 (0x10) of the STS3215 `Phase` register (0x12) in `configure_motors()`. This forces single-turn angle feedback so positions stay within 0-4095. Without it, the multi-turn firmware mode produced out-of-range readings and homing overflow. It lists #3196, #1296, #3193, #3348 and #3370 as fixed — [PR #3373](https://github.com/huggingface/lerobot/pull/3373), [feetech.py v0.6.1 L209-L225](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/motors/feetech/feetech.py#L209-L225)
- **#3196** (closed 2026-04-14) proposed putting wrist_roll back into range recording. Reviewer liyiteng said wrist_roll is designed as full-turn and that SO101 added mechanical wrist limits, making this a hardware/CAD issue. Joshuah143 said a Discord discussion had concluded the same. Maintainer Maximellerbach closed it, saying wrist roll is expected to be full-turn and the crash is fixed by #3373 — [PR #3196](https://github.com/huggingface/lerobot/pull/3196)
- **#3320** (closed 2026-04-10) proposed a symmetric wrist_roll range around the ENTER position. Maximellerbach measured only about ±90° of wrist travel in teleop afterwards, a loss of about 180°. The author withdrew it and floated two alternatives: (A) keep 0-4095 with an automatic offset, or (B) warn when the wrist is far from centre at calibration — [PR #3320](https://github.com/huggingface/lerobot/pull/3320), [comment](https://github.com/huggingface/lerobot/pull/3320#issuecomment-4224422669)
- **#2476** (closed 2026-04-14) proposed detecting angle wrap during calibration. It showed a wrist_roll range table of 0 / 129 / 4095 flagged as wrapped. Maximellerbach closed it because wrist_roll is now full-turn and no longer needs min/max calibration — [PR #2476](https://github.com/huggingface/lerobot/pull/2476), [closing comment](https://github.com/huggingface/lerobot/pull/2476#issuecomment-4244215403)
- **#3348** (docs PR, closed 2026-04-10) documented the overflow root cause. The workaround it found was unplugging the robot's power and USB and plugging them back in — [PR #3348](https://github.com/huggingface/lerobot/pull/3348)
- **#4583** (OPEN, 2026-09-08, s-tory), "fix(robots): sweep so_follower/so_leader wrist_roll instead of assuming it spins":
  - **Measurement.** On their SO-101 follower, wrist_roll stops in both directions at raw 113-3981 (3868 counts, about 340°). The stop is a corner of the printed gripper housing meeting the wrist bracket, not a gear stop. A second calibration recorded 70-3946 (340.7°).
  - **Change.** Stop excluding wrist_roll from `record_ranges_of_motion()` on both follower and leader. The prompt now asks for "a full revolution if it does [spin], stop to stop if it does not".
  - **Second commit.** It tracks wrist_roll across the encoder seam (`unwrap=[...]`) and re-homes it so the middle of the swept travel reads half a turn. If the sweep covers a full revolution, it falls back to 0-4095 with a warning.
  - **Why it matters.** The author found that a joint with about 340° of travel stays off the seam only if the ENTER pose is within about ±10° of mid-travel. Otherwise the leader and follower wrist_roll end up visibly misaligned.
  - **Status.** Not merged; last updated 2026-10-04.
  - [PR #4583](https://github.com/huggingface/lerobot/pull/4583), [author follow-up](https://github.com/huggingface/lerobot/pull/4583#issuecomment-5825878184)
- **#4583 reviews:**
  - stepankandrushin (2026-10-01) measured two more SO-101 arms with the auto-calibrator: follower 3902 counts (342.9°), leader 3846 counts (338.0°). Counting the PR author's arms, that makes four arms with a stop between 338° and 345°, and none that spins freely.
  - He supports placing the zero midway between the stops.
  - He flagged three issues: a `Homing_Offset` of 2048 edge case that the 11-bit sign-magnitude register cannot hold; LeKiwi still using an ENTER-pose zero; and migration — **both arms of a pair must be recalibrated together**.
  - GwenCCC (2026-09-24) pointed out that a partial sweep on the leader would record a narrower envelope and so narrow wrist_roll teleop travel.
  - [reviews on #4583](https://github.com/huggingface/lerobot/pull/4583)
- **#4826** (OPEN, 2026-10-01), `lerobot-calibrate --auto=true` for SO-101 leader and follower:
  - It drives every joint into both mechanical stops at limited torque, puts mid-travel at 2047 and takes the position limits from the measured span. A joint that turns a full revolution keeps 0-4095.
  - Measured travel (follower / leader): shoulder_pan 194.2° / 239.0°, shoulder_lift 213.1° / 211.5°, elbow_flex 197.4° / 194.9°, wrist_flex 207.0° / 202.8°, **wrist_roll 342.9° / 338.0°**, gripper 141.3° / 113.1°.
  - Firmware 3.10 observations: velocity mode reports Present_Position without Homing_Offset; after power-on, Goal_Position reads 0; a blocked joint driven to a far goal kept pushing at the torque limit for seconds, so moves are now stepped.
  - [PR #4826](https://github.com/huggingface/lerobot/pull/4826), [travel table in #4583 thread](https://github.com/huggingface/lerobot/pull/4583#issuecomment-5972135293)
- **#4867** (OPEN, 2026-10-07) refactors hand calibration into `SerialMotorsBus.record_calibration(mode, full_turn_motors, ...)`. Full-turn motors would span `resolution(model) - 1` instead of a hard-coded 4095. wrist_roll stays full-turn — [PR #4867](https://github.com/huggingface/lerobot/pull/4867)
- **#3499** (OPEN, 2026-05-03) adds `--motors=` to calibrate or set up a subset of motors. Full-turn motors such as wrist_roll still get a homing offset but skip range recording — [PR #3499](https://github.com/huggingface/lerobot/pull/3499)
- **#2924** (2026-02-08): a user saw only five motors (not wrist_roll) during calibration. A contributor explained that wrist_roll's values are hard-coded because the joint can rotate almost 360°. Maintainer imstevenpmwork advised following the positions in the official calibration video. Closed 2026-04-13 — [#2924](https://github.com/huggingface/lerobot/issues/2924), [comment](https://github.com/huggingface/lerobot/issues/2924#issuecomment-3973240595)
- **#1255** (2025-06-10, SO-101): turning wrist_roll toward its limits made the reading pass 0-4095 and wrap, which broke initialization and calibration. Closed 2025-06-22 with no comments — [#1255](https://github.com/huggingface/lerobot/issues/1255)

### Inferences
- **The user's arm is unusual.** The user's follower wrist_roll travels only about 154° and the leader about 255°. That is far below the 338-345° measured on four SO-101 arms in #4583/#4826. This hints that the Waveshare CAM-kit wrist has an extra obstruction, such as the camera mount or cable, a different housing, or a horn or bracket assembled at a different angle. The 0-4095 assumption is badly wrong for this arm.
- **What #4583 would do.** It would record about 154° of follower travel, write it as `Min/Max_Position_Limit`, and centre the zero on mid-travel. The firmware would then clamp goals before the stop instead of stalling against it (firmware clamping per #3585). It is unmerged, so the user would have to apply it locally or do the equivalent by hand.

### Gaps
- No merged change in any version up to 0.6.1 or `main` records a wrist_roll range or writes real wrist_roll position limits.
- I found no maintainer statement on whether #4583 or #4826 will be accepted.

---

## Q3. Known problems with leader/follower wrist_roll offset or sign/direction mismatch in teleop, and recommended fixes?

### Takeaway
With full-turn calibration, **each arm's wrist_roll zero is simply its pose when ENTER was pressed**, and `use_degrees=True` maps the leader's degrees 1:1 onto the follower with **no clamping**. The documented mismatches are:
- zero points that don't correspond (#3193);
- about 180° zero offsets on arms with hard stops (#3193 comments);
- different `use_degrees` settings on leader and follower (#3176);
- mirrored setups (#4832).

The maintainer's advice: put **both wrists in the same physical orientation at ENTER**, follow the calibration video, and recalibrate both arms.

### Cited Findings
- **Degrees mapping in v0.6.1.**
  - `_normalize` uses `mid = (min_ + max_) / 2` and returns `(val - mid) * 360 / max_res`. `_unnormalize` returns `int((val * max_res / 360) + mid)`.
  - Unlike the RANGE_M100_100 and RANGE_0_100 modes, the DEGREES path does **not** clamp to `[range_min, range_max]`.
  - With 0/4095 on both arms, `mid` is 2047.5, so the leader's degrees go straight through to the follower.
  - [motors_bus.py v0.6.1 L854-L911](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/motors/motors_bus.py#L854-L911)
- Maintainer nicolas-rabault confirmed in a #3615 review (2026-09-30) that DEGREES is the SO-101 default and that `_unnormalize` does not clamp in that mode. He also said a goal past the calibrated range is clamped by the firmware, and suggested clamping DEGREES in `_unnormalize` as a separate follow-up — [PR #3615 review](https://github.com/huggingface/lerobot/pull/3615)
- **#3193 problem 2.** Leader and follower wrist_roll zero points don't correspond, which causes incorrect teleop — [#3193](https://github.com/huggingface/lerobot/issues/3193)
  - Maintainer Maximellerbach (2026-08-23): this is not an issue if both grippers are positioned correctly before calibration, because the wrist pose at the start of calibration becomes the "0" — [comment](https://github.com/huggingface/lerobot/issues/3193#issuecomment-5388080872)
  - ayerone (2026-06-05): the wrist_roll pose at the start of calibration fully determines that joint's calibration. Their Seeed-printed SO-101 leader and follower **both have wrist_roll hard stops** (a little under 360°). Their apparent centre has the two pincers vertical and the wrist camera on the robot's left — [comment](https://github.com/huggingface/lerobot/issues/3193#issuecomment-4627743784)
  - twdent (2026-08-23 and 08-25), Seeed SO-101 Pro kit with wrist hard stops:
    - the leader and follower zeros stayed about **180° out of sync**, so the follower jumped when rolled past halfway and ended at a hard stop;
    - centring the joints and adding or removing wrist_roll from `full_turn_motor` did not help;
    - what fixed it was the **servo "middle" calibration in Seeed's RoboController tool** (the servo's midpoint was not at 2048), then recalibrating in LeRobot.
    - [comment 1](https://github.com/huggingface/lerobot/issues/3193#issuecomment-5386125382), [comment 2](https://github.com/huggingface/lerobot/issues/3193#issuecomment-5408003117), [Seeed RoboController](https://github.com/Seeed-Projects/Seeed_RoboController)
- **#3176** (2026-03-17, LeRobot 0.4.4, LeKiwi): the follower's wrist_roll rotated **much more** than the leader's. The cause was a `use_degrees` mismatch: LeKiwi's config defaulted to `False`, while the SO leader defaulted to `True`. Setting both the same fixed it. Fix PR #3185 was closed without merging; the issue closed 2026-05-07 — [#3176](https://github.com/huggingface/lerobot/issues/3176), [PR #3185](https://github.com/huggingface/lerobot/pull/3185)
- In v0.6.1, `use_degrees` defaults to `True` for both `SOFollowerConfig` and `SOLeaderConfig` — [config_so_follower.py v0.6.1 L41-L42](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/config_so_follower.py#L41-L42), [config_so_leader.py](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/teleoperators/so_leader/config_so_leader.py)
- **#4832** (OPEN, 2026-10-01) adds `--teleop.mirror=true` on SO leaders. When the follower is the leader's mirror image (for example, arms facing each other), copying joint for joint makes the follower roll its wrist the other way. Mirror mode negates `shoulder_pan` and `wrist_roll` about the middle of their calibrated range. It is off by default — [PR #4832](https://github.com/huggingface/lerobot/pull/4832)
- **#4583's second commit** re-homes wrist_roll to mid-travel. Its author found that with ENTER-pose zeros their leader and follower wrist_roll were visibly out of line while every other joint lined up. Reviewer stepankandrushin noted that today hand calibration (ENTER-pose zero) and auto calibration (mid-stop zero) disagree on wrist_roll, and that recalibrating only one arm of a pair would leave them misaligned — [PR #4583](https://github.com/huggingface/lerobot/pull/4583)
- Official SO-101 docs tell users to move the robot to where all joints are in the middle of their ranges, then sweep each joint, and link a calibration video. They give no wrist_roll-specific guidance about matching leader and follower orientation — [SO-101 docs](https://huggingface.co/docs/lerobot/so101), [so101.mdx source](https://github.com/huggingface/lerobot/blob/main/docs/source/so101.mdx)

### Inferences
- **The user's earlier "25% of the leader's rotation".** This does not match a pure `use_degrees` mismatch: the expected ratio for a ±100-vs-degrees mismatch on a 0-4095 joint is about 55%, not 25%. A unit mismatch like #3176 is still the first thing to rule out, for example a stale calibration or config from before v0.4.4 on one arm. Calibration files from before PR #2780 (v0.4.3) recorded a wrist_roll range on SO-101, which changes `mid` and the scaling. Mixing an old file on one arm with a new file on the other could produce asymmetric scaling.
- **Practical alignment for the user.** Put both arms' wrists in the same physical orientation, centred within the **follower's** roughly 154° travel, and press ENTER on both. Otherwise the leader's zero sits off the follower's mid-travel, and the follower reaches its stop with even less leader rotation.

### Gaps
- No issue reports a wrist_roll **sign/direction** inversion (as opposed to an offset) on SO-101 in non-mirrored setups.
- The leader's own `drive_mode` for wrist_roll is always 0 in SO calibration. I found no discussion of leader encoder direction differences specific to Waveshare SE servos.

---

## Q4. Known issue with Feetech "Overload error" status making disconnect / Torque_Enable writes fail, and recommended recovery?

### Takeaway
Yes. In v0.6.1, **any status packet that carries the overload bit makes `write()` raise a `RuntimeError`, even when the packet itself arrived fine**, and retries don't help. `disconnect()` disables torque motor by motor and **stops at the first failure**, so later motors (for example the gripper, ID 6) are left powered and the serial port stays open. **PR #4695** (open) fixes this. The repo's recovery advice is informal: **power cycle (unplug power and USB)**, recalibrate, and replace the servo if it was overloaded too long.

### Cited Findings
- **Where the message comes from (v0.6.1 `MotorsBus.write` / `_write`).**
  - `err_msg = f"Failed to write '{data_name}' on {id_=} with '{int_value}' after {num_retry + 1} tries."`
  - The retry loop exits as soon as the packet transfer succeeds.
  - If the status packet's error byte is nonzero, it raises `RuntimeError(f"{err_msg} {self.packet_handler.getRxPacketError(error)}")`.
  - [motors_bus.py v0.6.1 L1085-L1125](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/motors/motors_bus.py#L1085-L1125)
- In the Feetech SDK (`scservo_sdk/protocol_packet_handler.py`, feetech-servo-sdk 1.0.0), `ERRBIT_OVERLOAD = 32` (0x20) maps to the string `"[RxPacketError] Overload error!"`. The other bits are VOLTAGE=1, ANGLE=2, OVERHEAT=4 and OVERELE=8 — [feetech-servo-sdk on PyPI](https://pypi.org/project/feetech-servo-sdk/) (read from the locally installed copy in the project's `.venv-lerobot`)
- **v0.6.1 `SerialMotorsBus.disconnect()`** does three things in order: `clearPort()`, then `self.disable_torque(num_retry=5)`, then `closePort()`. Feetech's `disable_torque` loops over the motors, writing `Torque_Enable=0` then `Lock=0` for each. The first exception aborts the loop and skips `closePort()` — [motors_bus.py v0.6.1 L547-L562](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/motors/motors_bus.py#L547-L562), [feetech.py v0.6.1 L291-L294](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/motors/feetech/feetech.py#L291-L294)
- **PR #4695** (OPEN, 2026-09-21), "fix(motors): disable torque for every serial motor on disconnect", references #2819:
  - It tries to disable torque on every motor and logs failures as it goes. It always tries to close the port, then re-raises the first motor error.
  - Reviewer mmporong (2026-09-27) reproduced the exact failure with a fake port returning status `0x20` (overload) for `Torque_Enable` on motor id 2. On the current base, writes stopped at id 2 and the port stayed open. With the PR, id 3 was still disabled and `closePort()` was called.
  - Reviewer varun-projects (2026-10-10) confirmed the defect still exists on `main` (4 of the PR's cleanup tests fail against unmodified `motors_bus.py`).
  - Not merged as of 2026-10-13.
  - [PR #4695](https://github.com/huggingface/lerobot/pull/4695)
- **Recovery guidance found in the repo:**
  - **#3348** (2026-04-10): unplugging the robot's power and USB and plugging them back in cleared the bad position/overflow state — [PR #3348](https://github.com/huggingface/lerobot/pull/3348)
  - **#3942**: after a gripper overload or torque-off, control returned only after a power cycle — [#3942](https://github.com/huggingface/lerobot/issues/3942)
  - **#2819**: the maintainer said a servo can be permanently broken if overloaded for too long; the fix was replacing it — [#2819 comment](https://github.com/huggingface/lerobot/issues/2819#issuecomment-3768335194)
  - **#3193**: a user cleared a persistent offset problem by running the servo "middle" calibration in Seeed's RoboController — [#3193 comment](https://github.com/huggingface/lerobot/issues/3193#issuecomment-5408003117)
- The official SO-101 docs have no overload troubleshooting section. Their only troubleshooting block (motor setup) covers the power supply, USB cable and 3-pin cables, and says that on a Waveshare controller board both jumpers should be on the `B` channel (USB) — [SO-101 docs](https://huggingface.co/docs/lerobot/so101), [so101.mdx source L164-L178](https://github.com/huggingface/lerobot/blob/main/docs/source/so101.mdx)
- `disable_torque_on_disconnect` (default `True`) can be set to `False` (`--robot.disable_torque_on_disconnect=False`). #3131's reporter ran with it set that way. This skips the torque-off writes at disconnect entirely — [config_so_follower.py v0.6.1 L31](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/config_so_follower.py#L31), [#3131](https://github.com/huggingface/lerobot/issues/3131)

### Inferences
- **The user's disconnect error.** It is consistent with the code path above: servo 5's overload flag makes the `Torque_Enable=0` write raise. In 0.6.1, servo 6 (the gripper) is then never sent torque-off and the port is never closed. Practical recovery: cut the 12 V follower power, unplug and replug USB, and fix the cause (calibration or range) before running teleop again.
- **The retry count is misleading.** The message will say "after 6 tries" (`num_retry=5`), but the loop exits on the first successful packet transfer, so status-bit errors are never retried.

### Gaps
- No LeRobot source or issue documents whether the STS3215 actually carries out a `Torque_Enable=0` write whose reply carries the overload bit (that is, whether torque is in fact off despite the exception), or exactly what clears the overload latch (power cycle vs. torque toggle vs. the load dropping). The protection-register semantics in #4783 describe the torque reduction only, not how to clear it.

---

## Q5. Are position limits, `max_relative_target`, or stall protection recommended in newer versions?

### Takeaway
`max_relative_target` exists, defaults to `None`, and accepts a float or a per-motor dict in v0.6.1. **No official doc or maintainer comment recommends a value**, and the SO-101 page does not mention it. No merged feature limits wrist_roll's position or detects stalls. The relevant work (#4583, #4826, #4447, #3954, #3615, #2075) is all open.

### Cited Findings
- **v0.6.1 `SOFollowerConfig`:** `max_relative_target: float | dict[str, float] | None = None`. The comment says it limits the relative positional target for safety, as one scalar for all motors or a dict mapping motor names to values. `disable_torque_on_disconnect: bool = True`. Position PID defaults: P=16, I=0, D=32 — [config_so_follower.py v0.6.1 L31-L47](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/config_so_follower.py#L31-L47)
- **v0.6.1 `send_action`:** when `max_relative_target` is set, it reads `Present_Position` and calls `ensure_safe_goal_position`. That function clips each joint's goal to `present ± max_diff` (in the normalized units, so degrees with `use_degrees=True`) and logs `"Relative goal position magnitude had to be clamped to be safe."` The code notes that slower fps should be expected — [so_follower.py v0.6.1 L204-L230](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/so_follower.py#L204-L230), [robots/utils.py v0.6.1 L91-L123](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/utils.py#L91-L123)
- **#1483** (2025-07-11) asked how to choose `max_relative_target`. It got no maintainer answer and was auto-closed as stale on 2026-02-24. Related type bugs: #1527 ("Teleop 'max_relative_target' Type Error") and #1716 ("`max_relative_target` should be a float") — [#1483](https://github.com/huggingface/lerobot/issues/1483), [#1527](https://github.com/huggingface/lerobot/issues/1527), [#1716](https://github.com/huggingface/lerobot/issues/1716)
- **Docs coverage.** A GitHub code search for `max_relative_target` in `.mdx` docs finds it only in writing_docstrings, rebot_b601, openarm, reachy2 and integrate_hardware. It does not appear in so101.mdx — [GitHub code search](https://github.com/search?q=repo%3Ahuggingface%2Flerobot+max_relative_target+path%3A*.mdx&type=code)
- **Interaction with stall protection.** Maintainer review of #4447 flagged a `max_relative_target` / tolerance combination that turns the proposed stall protection off — [PR #4447](https://github.com/huggingface/lerobot/pull/4447)
- **Firmware limits.** Per #3585, when `Min/Max_Position_Limit` are set the STS3215 firmware clamps out-of-range goals to the nearest limit. So real wrist_roll limits, which today only #4583/#4826 would write, would let the firmware stop the wrist before its hard stop — [#3585](https://github.com/huggingface/lerobot/issues/3585)
- **#3615** (OPEN, 2026-05-16) would raise a `ValueError` on out-of-range `Goal_Position` writes. The maintainer requested changes, because in DEGREES mode this would stop teleop for goals a few ticks past the range — [PR #3615](https://github.com/huggingface/lerobot/pull/3615)

### Inferences
- **Why a small `max_relative_target` may help.** Setting it on the follower, for example as a per-joint dict with a small wrist_roll value such as 5-10°, caps how far the goal can lead the measured position. With position-mode P control, that also bounds stall torque against a stop, which may keep load under the 80% `Overload_Torque` threshold. It does not stop the wrist reaching the stop, and it costs an extra `Present_Position` read each loop. No source tested or recommended this, so treat it as a mitigation, not a fix.
- **What actually fixes it** (inference): make the commanded wrist_roll range fit the follower's real travel. Options:
  - real `Min/Max_Position_Limit` on servo 5, as with #4583/#4826;
  - a local clamp in DEGREES mode;
  - aligned ENTER poses on both arms;
  - removing the mechanical cause of the 154° restriction.
  - Combine this with an overload-safe disconnect (#4695) or `disable_torque_on_disconnect=False` plus a manual power cycle.

### Gaps
- I found no maintainer-endorsed `max_relative_target` value for SO-101, and no guidance tying it to overload prevention.
- No merged feature, in any release up to 0.6.1 or `main` as of 2026-10-13, clamps wrist_roll goals in DEGREES mode or detects stalls.
