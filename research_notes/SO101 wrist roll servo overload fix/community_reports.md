# Community reports: SO-100/SO-101 follower wrist_roll overload and stalls during LeRobot teleop (incl. Waveshare SE leader)

Research date: 2026-10-13. Scope: web only (GitHub issues/PRs, vendor wikis, HF forums, build-log repos, a TikTok listing). Nothing here touched hardware.

Coverage caveats, read first:
- Reddit was not reachable: the search tool returned "domains are not accessible to our user agent: ['reddit.com']". No Reddit threads are included.
- Discord is not indexed. The only Discord-adjacent material is maintainer comments on GitHub.
- A HF forums search found no thread on SO-101 wrist overload. The only relevant forum post (2026-10-13) has no replies.
- No Waveshare or Seeed forum thread, YouTube comment, Hackster or Medium build log described a wrist_roll overload. Nearly all first-hand material is in the huggingface/lerobot GitHub tracker plus a few community repos.

User's situation, for reference: Waveshare SO-ARM101 CAM Kit (SE). The leader has six encoder-only "ST3215 Servo (SE)" units, the follower six ST3215 12V servos, on LeRobot 0.6.1. The follower wrist_roll (ID 5) overloads as soon as teleop starts. Measured wrist roll travel is ~154° on the follower and ~255° on the leader.

---

## Q1. How common is follower wrist_roll overload in SO-101 teleop, and what are the most frequently confirmed root causes?

### Takeaway
No public thread matched the exact symptom: follower wrist_roll (ID 5) overloads with "Overload error!" on disconnect as soon as teleop starts. The closest and most common reports are wrist_roll calibration and range problems in LeRobot's SO-101 code. LeRobot treats wrist_roll as a free-spinning full turn (0–4095) with no range sweep, but every SO-101 wrist measured by the community has hard stops, and leader/follower zero points can disagree. Those reports end with the follower being driven into a hard stop. The only confirmed overload diagnosis is a broken servo after being "overloaded for too long" (on the gripper, not the wrist).

### Cited Findings

**Root cause A: wrist_roll is assumed to spin freely, but real arms have hard stops (most relevant)**
- LeRobot's SO follower and leader calibration "excluded `wrist_roll` from `record_ranges_of_motion` and set its range to 0-4095", on the assumption that the joint turns freely. PR #4583 by s-tory, opened 2026-09-08, is **open and not merged** as of the page fetch. — [lerobot PR #4583](https://github.com/huggingface/lerobot/pull/4583)
- Measured stop-to-stop wrist_roll travel from PR #4583:
  - s-tory's follower: 113–3981 counts, ~340°, measured 2026-09-03. The cause was "a corner of the printed gripper housing meets the wrist bracket … It is not a gear stop; it is two printed parts touching."
  - s-tory's leader: 3923 counts (345°).
  - stepankandrushin (2026-10-01): a follower at 342.9° and a leader at 338.0°. "That makes four arms with a stop, between 338° and 345°, and none that spins."
  — [lerobot PR #4583](https://github.com/huggingface/lerobot/pull/4583)
- In issue #3193, ayerone (2026-06-05) also reported physical stops on their wrist_roll ("My wrist_roll motor can rotate a little less than 360"). They said the start pose "determines 100% of calibration for that joint." — [lerobot issue #3193](https://github.com/huggingface/lerobot/issues/3193)
- The MuJoCo Menagerie SO-101 model was corrected (merged 2026-09-28) to a wrist_roll range of -2.744 to +2.841 rad, about -157° to +163° or ~320° total. — [mujoco_menagerie PR #324](https://github.com/google-deepmind/mujoco_menagerie/pull/324)

**Root cause B: leader/follower wrist_roll zero points don't correspond (calibration pose and offset)**
- ZhongChongWang opened issue #3193 on 2026-03-21 against v0.5.1 with STS3215 motors.
  - Calibration crashed with `Magnitude 2073 exceeds 2047` if wrist_roll was not near raw 2048.
  - Even when calibration succeeded, "leader and follower zero points don't correspond", so wrist_roll teleop was wrong.
  - The maintainer closed it on 2026-04-14 as "fixed in #3373".
  — [lerobot issue #3193](https://github.com/huggingface/lerobot/issues/3193)
- **Closest report to the user's symptom.** twdent (2026-08-25) reported:
  - a `Magnitude 30841 exceeds 2047` crash and a ~180° desync,
  - the follower "jumped when rolling past halfway and hit a hard stop",
  - the diagnosis that the servo's middle position was apparently not at 2048.
  — [lerobot issue #3193](https://github.com/huggingface/lerobot/issues/3193)
- PR #3373 (merged 2026-04-13) forces STS3215 single-turn mode by clearing bit 4 of the Phase register (0x12), because multi-turn firmware let readings leave [0, 4095]. It does not change the homing offset or middle-pose logic. — [lerobot PR #3373](https://github.com/huggingface/lerobot/pull/3373)
- Issue #1255 (2025-06-10): wrist_roll values wrapped past 4095 back to 0 near its limits, which "causes problems during initialization, leading to inaccurate calibration and control." It was closed 2025-06-22 with no visible fix comment. — [lerobot issue #1255](https://github.com/huggingface/lerobot/issues/1255)

**Root cause C: the servo is damaged from sustained overload (confirmed by a maintainer, gripper joint)**
- Issue #2819 (2026-01-19, lerobot 0.4.3): a follower servo "suddenly became extremely stiff", then teleop failed on sync read and on `Torque_Enable` writes during disconnect.
  - Maintainer pkooij: "I am afraid your servo has broken, this can happen if it is overloaded for to long." The likely cause was closing the leader gripper while an object blocked the follower gripper.
  - The reporter got a replacement ("Now I'm happy to make progress more with a new one", 2026-01-25).
  — [lerobot issue #2819](https://github.com/huggingface/lerobot/issues/2819)
- WowRobo (vendor) says servos are wear parts, and that "in most cases, the issues occur with the follower arm servos". They recommend buying spares. (Search-result summary; not fetched directly.) — [WowRobo SO-ARM101 kit page](https://shop.wowrobo.com/products/so-arm101-diy-kit-assembled-version-1)

**Root cause D: no software stop on stall; LeRobot keeps pushing**
- PR #4447 (darthjaja6, 2026-08-13, draft with changes requested, **not merged**) says SO-100/101 followers "do not stop when a motor keeps drawing high current while failing to reach its commanded position."
  - It adds opt-in `stall_current_thresholds`, `stall_position_tolerance` and `stall_timeout_s`.
  - A reviewer's gripper test (2026-09-30) shows current ramping to 123 while position stuck at 57–58 vs a target of 80.
  — [lerobot PR #4447](https://github.com/huggingface/lerobot/pull/4447)

**How the servo's overload protection behaves (explains the LED and "Overload error")**
- Feetech's STS3215 overload protection is position-based. When the servo is blocked before reaching its target, "the unloading force lasts for 2S". (Search-result quote of the Feetech page.) — [Feetech STS3215 evaluation](https://www.feetechrc.com/2020-05-13_56655.html)
- A retailer listing says overload "Triggers when exceeding 80% of stall torque for 25ms. Clear by sending new position command." (Search-result quote.) — [WowRobo STS3215 C018 listing](https://shop.wowrobo.com/products/feetech-sts3215-servo-12v-30kg-high-torque-servo-for-so-arm100)
- A bench test saw overload protection under prolonged heavy load cut torque to ~20%. The protection timeout and torque limit are configurable via registers. (Search-result quote.) — [Robonine STS3215 test](https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/)
- The rigbetellabs SO-101 build guide says:
  - "If the motor is mechanically blocked and cannot reach `2047`, the firmware will trigger an Overload protection flag".
  - Its troubleshooting entry for "Overload error" reads "The motor drove into a mechanical hard stop or was blocked and overloaded its own current limit".
  - Recovery is to clear the obstruction and power-cycle. The guide is undated.
  — [rigbetellabs/S0101](https://github.com/rigbetellabs/S0101)

**Other wrist_roll teleop problems (not overload)**
- Issue #1333 (SO-100, 2025-06-17): wrist_roll during teleop was "completely out of control in the default configuration".
  - sumanth-nirmal (2025-10-06): "I also saw this on SO-101 arm".
  - It was closed as stale on 2026-04-14 with no confirmed fix.
  — [lerobot issue #1333](https://github.com/huggingface/lerobot/issues/1333)
- Hiwonder NexArm (different servos, 2026-10-02): wrist_roll (servo 5) "does not reach targets past about ±90°. It lands short, in steps." The issue is open with no reply. — [Hiwonder nexarm-follower issue #2](https://github.com/Hiwonder-official/lerobot-robot-nexarm-follower/issues/2)
- Hiwonder SO-ARM101 community repo (2026-10-09): the follower shoulder stops ~9° short of the leader because the recorded range is wider than the follower can physically reach. The proposed fix is to recalibrate both arms; it is not yet confirmed. — [yolo-jared/hiwonder-SoArm-101 issue #5](https://github.com/yolo-jared/hiwonder-SoArm-101/issues/5)
- After a power cycle, gripper (ID 6) and wrist_roll (ID 5) could both come back as ID 5, giving `Missing motor IDs: 5, 6`. The fix is to re-lock the STS3215 EEPROM after ID writes. — [aakashvardhan/so101-lerobot](https://github.com/aakashvardhan/so101-lerobot)

**Power and voltage as a cause**
- Seeed wiki: "5V for the STS3215 7.4V motors and 12V for the STS3215 12V motors". On the Pro edition, "the Follower Arm should be calibrated and operated at every step using a 12V power supply." — [Seeed SO-Arm LeRobot tutorial](https://wiki.seeedstudio.com/lerobot_so100m_new/)
- No report was found that blamed a wrist_roll overload specifically on a weak or wrong-voltage supply.

### Inferences
- **Mechanism matching the user's symptom.** In LeRobot 0.6.1, PR #4583 is unmerged, so wrist_roll on both arms is normalized over a full 0–4095 turn with no range limit. A leader that rolls ~255° can then command follower positions the follower cannot physically reach. If the follower's real travel is ~154°, it drives into a mechanical stop and trips Feetech overload protection. That gives the flashing LED and the "Overload error!" when LeRobot writes `Torque_Enable=0` on disconnect. This is the pattern twdent described in #3193 ("follower jumped … and hit a hard stop") and the rigbetellabs troubleshooting entry.
- **The follower's ~154° is far below the community norm.** Community-measured wrist_roll travel is ~338–345° per arm, and the model range is ~320°. A follower stopping at ~154° suggests something specific to this follower:
  - cable binding at the wrist (gripper cable wrapping),
  - the gripper housing or wrist bracket colliding early,
  - the wrist horn mounted at an offset spline position so the stop falls near center.
  This is an inference; no source reports a ~154° follower.
- **Earlier "~25% of leader" symptom.** This is consistent with a mismatch in wrist_roll normalization: one arm's calibration file has a recorded (narrow or wide) range while the other has 0–4095. It is also consistent with the follower stalling early against an obstruction. Not verified by any source.
- The ~6–14° wrist_roll disagreement after recalibration is in line with PR #4583's note that the two arms agree only "as well as a hand posed them". Unequal travel is the bigger issue.

### Gaps
- No public report pairs an SE (encoder-only) leader with follower wrist_roll overload. No frequency data exists (no survey or issue count).
- Reddit (blocked), Discord (not indexed), YouTube comments and the Waveshare/Seeed forums yielded nothing usable.
- Issues #3585, #3586 and #3587 (goal positions outside min/max, calibration overflow protection, homing-offset wrap-around), referenced around 2026-05-12, were seen only as titles on PR #3373. Their contents and status are unknown.

---

## Q2. What practical fixes did people apply, and which were confirmed?

### Takeaway
The only explicitly confirmed fixes are:
- re-centering the servo's middle position (Seeed RoboController "write current position as 2048") and then re-running `lerobot-calibrate`,
- updating LeRobot past the April 2026 fix and recalibrating everything,
- replacing a servo that had been overloaded too long.

The rest are maintainer or community suggestions, or open PRs: pose both arms identically before calibrating, sweep or limit wrist_roll's range, check for mechanical binding, opt-in stall protection, lower P gain or speed/acceleration, and power-cycle to clear the overload flag.

### Cited Findings

**Confirmed (the person said it worked)**
- **Seeed middle-position recalibration, then LeRobot recalibration.** twdent (2026-08-25, #3193) had the ~180° wrist_roll desync and the hard-stop jump. They fixed it by recalibrating the servo's middle position with Seeed's tool, then re-running calibration with wrist_roll still in the full-turn list; it worked. — [lerobot issue #3193](https://github.com/huggingface/lerobot/issues/3193)
  - Seeed's FAQ for `Magnitude … exceeds 2047`: power off, restart and recalibrate. If it persists, "use the Seeed Studio SoARM quick calibration tool to perform middle-position calibration (write current position as **2048**)", then re-run `lerobot-calibrate`. — [Seeed SO-Arm LeRobot tutorial](https://wiki.seeedstudio.com/lerobot_so100m_new/)
- **Update LeRobot and recalibrate everything (self-report, video).** TikTok creator @mxjc.labs: "The wrist_roll calibration on my LeRobot SO-101 was broken — 'itching movement' … Hugging Face pushed a fix. After updating and recalibrating everything, the robot finally works again." The post ID decodes to about 2026-04-17, right after PR #3373 merged on 2026-04-13. Taken from the search-result listing; the video was not viewed. — [TikTok @mxjc.labs](https://www.tiktok.com/@mxjc.labs/video/7629741869035015456)
- **Replace the overloaded servo.** In #2819 (gripper), the maintainer diagnosed a broken servo from sustained overload. The reporter got a new one and proceeded (2026-01-25). The thread does not explicitly confirm the replacement fixed everything. — [lerobot issue #2819](https://github.com/huggingface/lerobot/issues/2819)

**Suggested by maintainers, vendors or contributors (not confirmed to fix overload)**
- **Pose both arms identically before calibrating.**
  - Maintainer Maximellerbach (2026-08-23, #3193): the zero-point mismatch is "not an issue if you position both grippers correctly before running the calibration script". The ENTER pose "will be the '0'".
  - twdent replied that aligning didn't fix their case when joints were "centred as best I could".
  — [lerobot issue #3193](https://github.com/huggingface/lerobot/issues/3193)
  - The official docs say to "move the robot to the position where all joints are in the middle of their ranges", then move each joint through its full range. — [LeRobot SO-101 docs](https://huggingface.co/docs/lerobot/so101)
- **Rotate wrist_roll to raw ~2048 before pressing ENTER.** This was the reporter's workaround in #3193. It avoids the crash but "does not fix Problem 2" (zero mismatch). — [lerobot issue #3193](https://github.com/huggingface/lerobot/issues/3193)
- **Sweep wrist_roll's real range instead of assuming 0–4095 (PR #4583, unmerged).**
  - It removes the hard-coded full turn, sweeps stop to stop, and (commit of 2026-09-25) zeroes wrist_roll at the middle of its swept travel instead of at the ENTER pose.
  - stepankandrushin's review note: "Recalibrating only one arm of a pair would leave the two wrist zeros misaligned", so recalibrate both.
  — [lerobot PR #4583](https://github.com/huggingface/lerobot/pull/4583)
- **Check for physical obstruction and cable tension at the wrist.**
  - The PR #4583 author found the gripper housing hitting the wrist bracket. — [lerobot PR #4583](https://github.com/huggingface/lerobot/pull/4583)
  - A community assembly guide says to leave cable slack and, after attaching the wrist, "move the wrist by hand to check for binding or cable tension" (search-result quote). — [StevenZhoooooooooooooou/so101-assembly-guide](https://github.com/StevenZhoooooooooooooou/so101-assembly-guide)
- **Clear the overload flag.** Remove the obstruction and power-cycle the arm. — [rigbetellabs/S0101](https://github.com/rigbetellabs/S0101)
  - The retailer listing says sending a new position command also clears it. — [WowRobo STS3215 listing](https://shop.wowrobo.com/products/feetech-sts3215-servo-12v-30kg-high-torque-servo-for-so-arm100)
  - Classmethod's write-up for wrapped readings: write 0 to Homing_Offset (register 33), then unplug **both USB and DC power** for ~5 s, because a small current can keep flowing over USB alone (search-result summary). — [Classmethod: SO-ARM101 assembly & teleop](https://dev.classmethod.jp/en/articles/lerobot-so-arm101-assembly-teleop/)
- **Delete old calibration files before recalibrating.**
  - Waveshare: delete files in `~/.cache/huggingface/lerobot/calibration/robots` or `.../teleoperators`, "otherwise error messages will appear". — [Waveshare calibration wiki](https://www.waveshare.com/wiki/SO-ARM100/101_Robotic_Arm_Calibration_and_Remote_Control)
  - Seeed: "If the hardware changes but the old calibration files remain, LeRobot may reuse outdated offsets." — [Seeed SO-Arm LeRobot tutorial](https://wiki.seeedstudio.com/lerobot_so100m_new/)
- **Opt-in stall protection (PR #4447, unmerged).** It trips on sustained high current with unresolved position error.
  - Maintainer nicolas-rabault (2026-09-30) prefers writing `Goal_Position = Present_Position` on trip rather than disabling torque.
  - He warned the check "can never trip" if `max_relative_target` is below `stall_position_tolerance`.
  — [lerobot PR #4447](https://github.com/huggingface/lerobot/pull/4447)
- **`max_relative_target` is a velocity cap, not a range limit.**
  - Issue #1527 found that it caps the per-loop step: values above 20 had no visible effect and ~3 gave a slow pace (search-result summary). — [lerobot issue #1527](https://github.com/huggingface/lerobot/issues/1527)
  - A Hiwonder SO-ARM101 PR dropped `--robot.max_relative_target=2.0` from its teleop command because the cap caused lag ("shoulder peak lag 8.4 deg (63 deg with the cap)", search-result quote). — [yolo-jared/hiwonder-SoArm-101 PR #3](https://github.com/yolo-jared/hiwonder-SoArm-101/pull/3)
- **Lower the P coefficient or speed/acceleration for wrist_roll.**
  - #1333 reporter: P_Coefficient 16→4 cut wrist_roll jitter ~37% (partial).
  - LaurentBerger (2025-12-05) suggested writing the speed and acceleration registers.
  - Neither is confirmed as a fix.
  — [lerobot issue #1333](https://github.com/huggingface/lerobot/issues/1333)
- **Use the right supply voltage.** The follower uses 12V servos on 12V; the leader uses 5V. — [Seeed SO-Arm LeRobot tutorial](https://wiki.seeedstudio.com/lerobot_so100m_new/)

### Inferences
- For the user's hardware, two community remedies address the actual mechanism, which is a full-turn command into a short physical range:
  - making the follower's mechanical wrist travel normal (~330–345°) by finding the binding, cable or horn-offset problem,
  - making software respect the real range (PR #4583-style sweep, or follower position limits / a clamp in the leader-to-follower mapping).
  No community report shows anyone setting STS3215 Min/Max_Position_Limit registers on wrist_roll as the fix. That remains a reasonable but unverified option.
- `max_relative_target` would only slow the follower's approach to the stop. It would not prevent the overload, per the velocity-cap behavior described in #1527.

### Gaps
- No source confirms that setting follower position limits, limiting the leader's roll, or re-routing cables fixed a wrist_roll overload. These remain suggestions.
- It is unclear whether PR #4583 or PR #4447 will ship in a release after 0.6.1.

---

## Q3. Any SE-leader-specific reports (e.g., leader wrist roll rotates further than the follower's) and recommended mitigations?

### Takeaway
No first-hand community report about the Waveshare SE (encoder-only) leader causing follower wrist overload was found. Waveshare documents the SE servo as a "virtual servo" with only an encoder. That explains why the leader rolls freely and further, but neither Waveshare nor LeRobot gives an SE-specific mitigation.

### Cited Findings
- Waveshare: the ST3215 Servo (SE) "is a virtual servo, which only has an encoder and no power section, making it very smooth to use." — [Waveshare SO-ARM100/101 Kit Assembly wiki](https://www.waveshare.com/wiki/SO-ARM100/101_Kit_Aassembly)
- The SE kits (No-3DP SE, 3DP SE, CAM Kit SE, Jetson Orin Nano SE) list "ST3215 Servo (SE)*6" for the leader and "ST3215 Servo *6" for the follower. The table is garbled on the page, so this is a best reading. The non-SE 3DP kit leader uses ST3215 for IDs 1–3 and ST3215-HS for IDs 4–6. — [Waveshare SO-ARM100/101 Kit Assembly wiki](https://www.waveshare.com/wiki/SO-ARM100/101_Kit_Aassembly)
- Waveshare's calibration page gives the same procedure for leader and follower: "Move the robotic arm to a position where all joints are at the midpoint of their range of motion", then sweep. It has nothing about SE servos, wrist roll, overload, LED or `max_relative_target`. — [Waveshare calibration wiki](https://www.waveshare.com/wiki/SO-ARM100/101_Robotic_Arm_Calibration_and_Remote_Control)
- The standard SO-101 leader uses geared STS3215 servos (wrist roll at 1/147); the follower uses 1/345 on all joints. — [LeRobot SO-101 docs](https://huggingface.co/docs/lerobot/so101)
- A HF forums post (2026-10-13) complains that STS3215 variants "look the same but aren't, and sellers don't always say which one you're getting". It has no replies. — [HF forum: Picking STS3215 servos](https://discuss.huggingface.co/t/picking-sts3215-servos-for-so-101-am-i-the-only-one-confused/190529)
- Every arm measured in PR #4583 (two leaders, two followers, all standard builds) had similar wrist_roll travel of 338–345°. In standard kits, leader and follower wrist ranges roughly match. — [lerobot PR #4583](https://github.com/huggingface/lerobot/pull/4583)
- PR #4583 discussion (GwenCCC, 2026-09-24) asked how a narrower recorded range on the leader would change its envelope, which is the leader-vs-follower range question in general form. The author's answer: a free-spinning wrist swept less than a full turn "would read as having stops". — [lerobot PR #4583](https://github.com/huggingface/lerobot/pull/4583)

### Inferences
- An encoder-only SE servo has no gear train or motor to resist the hand, so nothing on the leader limits wrist roll beyond the printed parts and cables. With LeRobot 0.6.1's full-turn wrist_roll mapping, nothing stops an SE leader from commanding follower roll beyond the follower's travel.
- The user's leader travel (~255°) and follower travel (~154°) are both well below the ~338–345° measured on standard arms. Both arms may have assembly-specific constraints, the follower more severely.
- Likely mitigations follow from Q2 but are unverified for SE kits:
  - restrict follower wrist_roll in software (range sweep à la PR #4583, or position limits),
  - free the follower's mechanical travel,
  - calibrate both arms in an identical pose.

### Gaps
- There is no Waveshare FAQ, forum post, Amazon review or YouTube comment on SE-leader wrist-range or overload issues.
- Waveshare doesn't say whether the SE "virtual servo" reports position at the same 4096 counts per turn and direction as a geared ST3215 output shaft. It is likely, since LeRobot works with it, but not documented in the sources.

---

## Q4. Any Discord announcements or FAQ entries from Hugging Face / LeRobot or vendors?

### Takeaway
No Discord announcement was found; Discord is not indexed. Official guidance is limited to:
- the LeRobot docs' "middle of their ranges" calibration instruction,
- maintainer comments in GitHub issues,
- Seeed's FAQ (Magnitude errors, middle-position tool, delete old calibration files, voltage),
- Waveshare's calibration wiki (delete old calibration files, midpoint pose).

None of these addresses wrist_roll overload directly.

### Cited Findings
- LeRobot docs: calibration ensures "the leader and follower arms have the same position values when they are in the same physical position". First "move the robot to the position where all joints are in the middle of their ranges". The docs have no troubleshooting for overload or wrist and point users to Discord for help. — [LeRobot SO-101 docs](https://huggingface.co/docs/lerobot/so101)
- Maintainer statements on GitHub, the closest thing to official guidance:
  - Maximellerbach (2026-04-14): #3193 "fixed in #3373".
  - Maximellerbach (2026-08-23): pose both arms correctly before calibrating.
  - Maximellerbach (2026-08-25): "that's weird"; asked for a new issue with Feetech debug-app readings.
  — [lerobot issue #3193](https://github.com/huggingface/lerobot/issues/3193)
  - pkooij (2026-01-19): a servo "has broken, this can happen if it is overloaded for to long". — [lerobot issue #2819](https://github.com/huggingface/lerobot/issues/2819)
  - nicolas-rabault (2026-09-30), reviewing stall protection: prefers holding pose over cutting torque. — [lerobot PR #4447](https://github.com/huggingface/lerobot/pull/4447)
- Seeed FAQ:
  - `Magnitude 30841 exceeds 2047`: restart, then use the SoARM quick calibration tool for middle-position calibration.
  - "If a servo LED is off, the cable before that servo may be loose".
  - "Delete the old calibration files and recalibrate."
  - Voltage rules (12V follower on Pro).
  - The fetched portion has no overload or overheating entry; the last ~5% of the page was not read.
  — [Seeed SO-Arm LeRobot tutorial](https://wiki.seeedstudio.com/lerobot_so100m_new/)
- Waveshare's calibration wiki: delete old calibration files before recalibrating; pose joints at "the midpoint of their range of motion". It has no FAQ on overload or wrist. — [Waveshare calibration wiki](https://www.waveshare.com/wiki/SO-ARM100/101_Robotic_Arm_Calibration_and_Remote_Control)
- A related maintainer-merged fix for bus errors: issue #3131 (Hiwonder servos, "Failed to sync read 'Present_Position'") closed 2026-07-29 via commit 7d615ac, "fix(robots): retry SO follower/leader bus reads on transient Feetech…". It is a communication retry, not an overload fix. — [lerobot issue #3131](https://github.com/huggingface/lerobot/issues/3131)

### Inferences
- LeRobot 0.6.1 ships no FAQ or announcement covering wrist_roll overload. The relevant upstream work (PR #4583 range sweep, PR #4447 stall protection) is still open, so users currently fix this by hand: mechanical inspection, pose discipline, or local code and register changes.

### Gaps
- The LeRobot Discord could not be searched; any announcements there are unknown.
- The final ~5% of the Seeed FAQ was not read, and could contain an overload entry.
- No Waveshare SE-specific FAQ was found.
