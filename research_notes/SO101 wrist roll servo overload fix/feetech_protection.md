# Feetech STS3215 / Waveshare ST3215 overload protection, register semantics, and SO-ARM100/101 vendor guidance

Scope: STS-series (STS3215 / ST3215, model number 777) only. Context: the user's follower servo ID 5 (wrist_roll, Waveshare ST3215 12V, firmware 3.9) reads Max_Torque_Limit 1000, Torque_Limit 1000, Protection_Current 310, Overload_Torque 80, Protective_Torque 20, Protection_Time 200, P/D 16/32, Acceleration 254, Min/Max_Position_Limit 0/4095, Homing_Offset 882.

Source-quality legend:
- **[OFFICIAL]**: Feetech datasheet or protocol manual.
- **[CODE]**: LeRobot source on GitHub.
- **[3RD-PARTY TABLE]**: AIFitLab wiki memory-table analysis. It does not name its upstream source, but its addresses match LeRobot's control table exactly.
- **[COMMUNITY]**: a measurement or opinion reported by a user.
- **[UNVERIFIED]**: anything else that could not be confirmed.

Register addresses and names come from LeRobot's `STS_SMS_SERIES_CONTROL_TABLE` ([lerobot tables.py](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/motors/feetech/tables.py)). I could not retrieve an official English Feetech/Waveshare memory table. The Waveshare ST3215 wiki only links an external "ST3215 memory register map-EN.xls" that I could not read ([Waveshare ST3215 wiki](https://www.waveshare.com/wiki/ST3215_Servo)).

## Q1. What triggers STS3215 overload protection, what happens once it trips, how is it cleared, and what does the LED mean?

### Takeaway
Feetech's official 12V ST3215 datasheet states the rule. If the servo is stalled above 80% of stall torque for 2 s, it enters overload protection. Re-sending a position command clears the overload flag, and both the percentage and the duration are configurable.

These map to Overload_Torque (36, 1% units) = 80 and Protection_Time (35, 10 ms units) = 200 → 2.0 s. Once protection is active, output is limited to Protective_Torque (34) = 20% rather than going fully limp. A separate over-current protection uses Protection_Current (28, 6.5 mA units): 310 ≈ 2.0 A for 2 s, after which output is switched off.

While a fault is latched, every status reply carries the fault bit in its ERROR byte. That is why LeRobot raises "[RxPacketError] Overload error!" on an ordinary Torque_Enable write. A flashing LED means a condition enabled in LED_Alarm_Condition (20) is active, and bit 5 of that register is the overload alarm.

### Cited Findings

**Official Feetech datasheet: trigger, clearing and other protections**
- **[OFFICIAL]** ST-3215-C047 datasheet (12V, 30 kg·cm, gear ratio 1/345, edition A/0, 2023-07-20), item 7-11 "Electronic Protection", Over Load. Paraphrased from the Chinese: protection is entered when the servo is stalled above 80% of stall torque for 2 s; re-sending a position command clears the overload protection flag; the stall percentage and duration are user-configurable. — [Feetech ST-3215-C047 datasheet (Seeed mirror)](https://files.seeedstudio.com/products/Feetech/108090003_FEETECH_ST-3215-C047-Datasheet.pdf). I extracted the text layer myself; the PDF is not machine-readable via WebFetch.
- **[OFFICIAL]** Same datasheet, Over Current. Paraphrased: when running current exceeds 2 A for 2 s, the servo switches its output off and enters over-current protection; re-sending a position command clears the flag; the current value and duration are configurable. — [same datasheet](https://files.seeedstudio.com/products/Feetech/108090003_FEETECH_ST-3215-C047-Datasheet.pdf)
- **[OFFICIAL]** Same datasheet, Over Voltage. Above 14 V or below 4 V the servo enters protection and releases it automatically once voltage returns to range (configurable). Over Hot: above 70 °C the servo turns torque output off. — [same datasheet](https://files.seeedstudio.com/products/Feetech/108090003_FEETECH_ST-3215-C047-Datasheet.pdf)
- **[OFFICIAL]** Same datasheet, electrical specification at 12 V:

  | Parameter | Value |
  |---|---|
  | Stall torque | 30 kg·cm |
  | Stall current | 2.7 A |
  | Rated load | 10 kg·cm |
  | Rated current | 900 mA |
  | No-load current | 180 mA |
  | Kt | 11 kg·cm/A |
  | Input range | 4–14 V |

  Mechanical limit angle is "No limit", with 360° magnetic 12-bit sensing. — [same datasheet](https://files.seeedstudio.com/products/Feetech/108090003_FEETECH_ST-3215-C047-Datasheet.pdf)
- **[OFFICIAL]** Same datasheet, item 7-14: writing 128 to address 40 calibrates the current position as the middle position (one-key centering). — [same datasheet](https://files.seeedstudio.com/products/Feetech/108090003_FEETECH_ST-3215-C047-Datasheet.pdf)

**Feetech marketing article: what the servo does once protection trips**
- **[OFFICIAL-ish]** Feetech staff article on STS3215 (Abby, Feetech Servo, LinkedIn, 2021-10-19). Overload protection works by position detection: if the servo is blocked by an obstacle before reaching its target, "the force will be unloaded after 2S (default 20% of the blocking force)". It adds: "Until a new command is triggered, the protection is released." Over-current protection unloads after 2 s with "the default torque is 0". — [LinkedIn / Feetech](https://www.linkedin.com/pulse/servo-program-any-angle-feetech-releases-ultra-high-abby-smith)

**Reseller specs**
- **[RESELLER]** DFRobot's "ST3215 Serial Bus Servo (12V, 30kg)" (SKU SER0070) gives the same overload rule (80% of stall for 2 s, cleared by a new position command) and the same over-current rule (above 2 A for 2 s, cleared by a new position command). It lists over-voltage as above 7.4 V or below 4 V, which conflicts with the official 12V datasheet's 14 V and is probably copied from the 7.4V model. — [DFRobot](https://www.dfrobot.com/product-2962.html)

**Register semantics** (addresses are from LeRobot; descriptions are from the AIFitLab STS table)

| Register (address, size) | Unit / default | Meaning per the AIFitLab STS table |
|---|---|---|
| Protective_Torque (34, 1) | 1% units, default 20 | "Holding Torque": "Output torque after overload protection (e.g., 20 = 20%)" |
| Protection_Time (35, 1) | 10 ms units, default 200 | "Duration exceeding overload torque (e.g., 200 = 2 s, max 2.5 s)" |
| Overload_Torque (36, 1) | 1% units, default 80 | "Overload Torque Threshold": "Threshold to trigger overload protection timing" |
| Protection_Current (28, 2) | 6.5 mA units, default 511, range 0–2047 | Protection current |
| Over_Current_Protection_Time (38, 1) | 10 ms units, default 200 | Over-current protection time (2 s at default) |
| Max_Torque_Limit (16, 2) | 0.1% units, default 1000 | "Assigned to address 48 (Torque Limit) at power-on" |
| Torque_Limit (48, 2) | 0.1% units | Torque limit |
| Present_Load (60, 2) | 0.1% units | "PWM duty cycle driving the motor; Bit10 is direction" |

  Sources: [AIFitLab STS memory table](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis); addresses from [lerobot tables.py](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/motors/feetech/tables.py).
- **[3RD-PARTY TABLE]** Bit definitions in the same table:
  - **Unloading_Condition (19):** "Bit=1 enables corresponding protection". BIT0 (1) voltage, BIT2 (4) overheat, BIT3 (8) overcurrent, BIT5 (32) load overload.
  - **LED_Alarm_Condition (20):** "Bit=1 enables alarm flashing". BIT0 (1) voltage alarm, BIT2 (4) overheat alarm, BIT5 (32) load overload alarm.
  - **Status (65):** "Bit=1 indicates corresponding error". BIT0 (1) voltage, BIT1 (2) magnetic encoder, BIT3 (8) current, BIT5 (32) load.

  — [AIFitLab STS memory table](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis)
- **[3RD-PARTY TABLE]** The AIFitLab SCSCL table uses the same semantics. Holding torque is "Output torque after entering overload protection (e.g., 20 = 20%)", protection time is "200 = 2 s, max 2.5 s", and overload torque is the "Threshold that triggers overload protection timing (e.g., 80 = 80% max torque)". — [AIFitLab SCSCL table](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-scscl-servo-memory-table-analysis)

**Why LeRobot reports "Overload error!" on a Torque_Enable write**
- **[OFFICIAL]** Feetech Communication Protocol Manual (v1.01, 2019, "Universal SCS and SMS Series"). Every reply packet is `0xFF 0xFF ID Length ERROR Params… Checksum`. Paraphrased: the reply carries the servo's current status in the ERROR byte; if the status is abnormal it shows up in this byte (meanings are in the memory control table); ERROR = 0 means no error. — [Feetech protocol manual (Seeed mirror)](https://files.seeedstudio.com/wiki/robotics/Actuator/feetech/Communication_Protocol_Manual.pdf). I extracted the text myself.
- **[CODE]** In the Feetech SDK ports, `getRxPacketError` maps ERRBIT_OVERLOAD (bit value 32) to the string "[RxPacketError] Overload error!". — [feetech.js lowLevelSDK.mjs](https://huggingface.co/spaces/blanchon/LeRobot-Arena/blob/main/packages/feetech.js/lowLevelSDK.mjs); [feetech.js constants](https://huggingface.co/spaces/blanchon/LeRobot-Arena/blob/main/packages/feetech.js/scsservo_constants.mjs). I did not open the Python `scservo_sdk` itself. **[UNVERIFIED]** that the Python SDK is identical, though LeRobot's error string matches.
- **[CODE]** LeRobot `SerialMotorsBus._write()` calls `writeTxRx`. If communication succeeded but the status packet carries an error bit, it raises `RuntimeError(f"{err_msg} {self.packet_handler.getRxPacketError(error)}")`. `write()` always passes `raise_on_error=True`. — [lerobot motors_bus.py](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/motors/motors_bus.py)
- **[CODE]** LeRobot Feetech `_disable_torque` writes Torque_Enable = 0 and then Lock = 0. `enable_torque` writes Torque_Enable = 1 and then Lock = 1. — [lerobot feetech.py](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/motors/feetech/feetech.py)

**Community reports**
- **[COMMUNITY]** The Seeed SO-ARM debugging tool shows per-servo "Status" as "Normal, Overcurrent, or Overload". — [Seeed lerobot steering gear debugging tool](https://wiki.seeedstudio.com/lerobot_steering_gear_debugging_tool/)
- **[COMMUNITY]** A user on LeRobot issue #2387 reports that during calibration one motor's red LED blinked fast while the others did not, and that motor errored. No resolution was recorded. — [lerobot #2387](https://github.com/huggingface/lerobot/issues/2387) (seen only via a search snippet)
- **[COMMUNITY]** The Beam Bots Elixir driver README says the "servo's own overload protection can wind the ceiling down under sustained load". It also lists status-byte bit 4 as "Torque enabled (not an error)" in the Feetech column. — [bb_servo_feetech README](https://readme.hex.pm/bb_servo_feetech/0.7.0)

### Inferences
- **What tripped servo 5.** With these register values, overload protection starts when the servo's output (Present_Load, a PWM duty in 0.1%) stays at or above 80% of maximum for 200 × 10 ms = 2.0 s. After that it drops to a 20% torque ceiling (Protective_Torque) and sets Status bit 5. Wrist roll needs little torque in free motion, so the likely cause is a persistent position error the joint cannot close. Candidates are a hard stop, a cable or wire wrap, a gripper collision, or a commanded target on the far side of a wrap. Any of these drives the P-controller output to saturation for more than 2 s.
- **The over-current path is separate.** Protection_Current 310 × 6.5 mA ≈ 2.0 A, which matches the datasheet's "> 2 A for 2 s". If Unloading_Condition bit 3 is enabled, this path cuts output to 0 and sets Status bit 3 (an "over-current" error, not "Overload"). The user's error string names overload, so bit 5 is the latched condition.
- **Why the Torque_Enable write "fails".** It is very likely that the Torque_Enable write is executed but the reply's ERROR byte still has bit 5 set. LeRobot turns any non-zero ERROR into a `RuntimeError`. So the "failure" is the latched overload status being reported, not necessarily a rejected write. Reading Status (65) with error-raising disabled would confirm it. That read is a hardware action, not done here.
- **How to clear it.** Officially: send a new position command, meaning write Goal_Position. A practical, low-motion way is to write Goal_Position = Present_Position, which clears the flag without commanding a jump. Power-cycling also clears it, because the flag lives in RAM and protection state is not stored in EEPROM. The power-cycle point is **[UNVERIFIED]** from official text; it is consistent with community advice that power-cycling clears it.
- **Torque_Enable is not documented as a way to clear the flag.** None of the sources say that writing Torque_Enable clears it.
- **The LED flash** is the LED_Alarm_Condition mechanism. If bit 5 is set in register 20, the LED flashes while the overload condition or flag is active. Bit 0 (voltage) and bit 2 (temperature) can also make it flash, so read Status (65) to see which.

### Gaps
- I could not open an official English Feetech/Waveshare STS memory table. Waveshare's xls and Feetech's own pages were unreachable; feetechrc.com did not resolve. The exact default values of Unloading_Condition and LED_Alarm_Condition on STS3215 firmware 3.9 are therefore unconfirmed.
- Whether protection compares Present_Load (PWM) or current against Overload_Torque is not stated explicitly by an official source. The "% of stall torque" wording suggests output duty.
- Whether the overload flag also clears on its own once load drops (without a new command) is not documented. Both Feetech sources say "until a new command".
- I found no official description of the LED blink pattern (rate or colour) per fault.

## Q2. Does writing Goal_Position (42) with torque off auto-enable torque? What does Torque_Enable=1 do, and what is Goal_Position at power-up?

### Takeaway
Yes, in practice. Two independent community measurements on STS3215 show that writing Goal_Position while Torque_Enable = 0 sets Torque_Enable back to 1 and moves the horn. Writes to other registers do not do this. That matches the user's observation. Feetech does not document it explicitly, but its datasheet treats a "position command" as the thing that re-arms the servo after protection.

Writing Torque_Enable = 1 alone produced no motion in a controlled test, so it appears to hold the present position. Goal_Position (SRAM) defaults to 0 at power-up, consistent with the user's reading. This is why LeRobot should never re-enable a servo by writing a stale goal.

### Cited Findings
- **[COMMUNITY]** feetech-cli 0.1.0 (Iori Yanokura, PyPI, released 2026-09-24): "A position command energises the servo by itself." It continues: "Measured on an STS3215: with `torque_enable` explicitly set to 0, writing `goal_position` took it back to 1 and the horn moved." Also: "Writes to `lock`, `homing_offset` and `acceleration` did not do this, so it is specific to the position command." The tool re-points the goal at the present reading so that re-energising does not jump. — [feetech-cli on PyPI](https://pypi.org/project/feetech-cli/0.1.0/)
- **[COMMUNITY]** LeRobot issue #3585 (opened 2026-05-12 by 0o8o0-blip; SO-101 follower shoulder_lift; limits 2045–3919):
  - The firmware sets "`Torque_Enable = 1` on the next PID tick regardless of the host's previous `disable_torque` call" after a Goal write.
  - Test A: writing Goal = Present (228, outside the limits) moved the joint to about 3472 and Torque_Enable read 1, with no `enable_torque` call from the host.
  - Test B: `disable_torque`, then writing Torque_Enable = 1 with **no** Goal write, gave "Motor sits still" (Present 227).
  - The author concludes the Goal write is the trigger.

  — [lerobot #3585](https://github.com/huggingface/lerobot/issues/3585)
- **[COMMUNITY]** Beam Bots driver README: "A joint left passive by a `Stop` comes back under power on the next motion command, at the position it came to rest — rather than the goal it was chasing when torque was cut." This is driver-level behaviour and may reflect driver logic rather than firmware. — [bb_servo_feetech README](https://readme.hex.pm/bb_servo_feetech/0.7.0)
- **[OFFICIAL]** The ST-3215-C047 datasheet says re-sending a position command clears the overload and over-current protection flags. The 2021 Feetech article says protection is released when "a new command is triggered". — [datasheet](https://files.seeedstudio.com/products/Feetech/108090003_FEETECH_ST-3215-C047-Datasheet.pdf); [Feetech LinkedIn](https://www.linkedin.com/pulse/servo-program-any-angle-feetech-releases-ultra-high-abby-smith)
- **[3RD-PARTY TABLE]** Address 42 "Target Position" (SRAM, 2 bytes) has default **0**, range −32767~32767, 0.087°/unit: "Absolute position control … Bit15 is direction bit". Address 40 "Torque Switch" has default 0: 0 = disable torque, 1 = enable torque, 2 = damping mode (strength set by Torque_Limit at 48), 128 = "Calibrate current position to 2048". — [AIFitLab STS memory table](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis)
- **[CODE]** LeRobot SO follower `send_action()` writes Goal_Position, optionally clipped by `max_relative_target` ("Cap goal position when too far away from present position"). `configure()` runs inside `with self.bus.torque_disabled():`, which re-enables torque on exit. — [lerobot so_follower.py](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/robots/so_follower/so_follower.py)

### Inferences
- **Sequence to avoid a jump.** The user's "torque turned itself on" is consistent with the documented community behaviour: any Goal_Position write re-energises the servo. To re-enable without a jump:
  1. Read Present_Position (56).
  2. Write Goal_Position = Present_Position. This also clears a latched overload flag, per the datasheet.
  3. Only then rely on Torque_Enable = 1.
- **Why the order matters.** After a power cycle, Goal_Position = 0 (SRAM default). If a tool ever "re-arms" by writing back the register contents, it would command raw position 0. For wrist_roll with Homing_Offset 882, that is a large rotation.
- **Torque_Enable = 1 on its own.** Test B in #3585 indicates it holds the present position rather than chasing the stored goal. This is a single community test on one unit; firmware 3.9 may differ.
- **LeRobot teleop re-arms the follower with the first leader frame.** The first `send_action()` after connect writes the leader's pose as Goal_Position. That instantly energises the follower and drives it there. With wrist_roll as a full-turn joint (0..4095), a leader/follower offset or a wrap at the 0/4095 seam can command a large rotation into a cable or gripper limit, giving a stall, then overload after 2 s.

### Gaps
- No official Feetech statement documents "Goal_Position write sets Torque_Enable = 1". Evidence is two community measurements: firmware unknown for feetech-cli; for #3585 the comment by hkder is on firmware 3.10.
- No source explicitly documents what Torque_Enable = 1 does with a stale Goal_Position on firmware 3.9. Test B in #3585 did not report the stored goal value.
- An official power-up value for Goal_Position was not found. The default of 0 comes from the third-party table and the user's own reading.

## Q3. What do Min/Max_Position_Limit (9/11) do in position mode? Are they a recommended way to prevent stalling, and do they persist in EEPROM?

### Takeaway
Min_Position_Limit (9) and Max_Position_Limit (11) are EEPROM angle limits in 0.087° steps. Setting both to 0 selects multi-turn mode. Community testing shows the firmware clamps each goal into [min, max] (wrap-aware) before using it as the PID target.

They persist only if written while Lock (55) = 0. They can keep commands away from a mechanical stop, but no vendor recommends them as anti-stall protection. Limits that are inconsistent with Homing_Offset have caused STS3215 joints to drive the wrong way into a hard stop until they overloaded. For wrist_roll at 0/4095 they provide no protection at all.

### Cited Findings
- **[3RD-PARTY TABLE]** Address 9 "Minimum Angle Limit": 2 bytes, default 0, range 0~4094, 0.087°. Address 11 "Maximum Angle Limit": default 4095, range 1~4095. Both are "0 when multi-turn absolute position control is used". — [AIFitLab STS memory table](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis). The SCSCL table adds "must be less/greater than" the other limit. — [AIFitLab SCSCL](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-scscl-servo-memory-table-analysis)
- **[3RD-PARTY TABLE]** Address 55 "Lock Flag" (default 1): "0: Disable write lock (EPROM values saved after power-off); 1: Enable write lock (EPROM values not saved after power-off)". — [AIFitLab STS table](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis)
- **[OFFICIAL]** Waveshare ST3215 FAQ: "Remember to unlock the EPROM area before modifying parameters in the memory table". Otherwise changes are lost at power-off. — [Waveshare ST3215 wiki](https://www.waveshare.com/wiki/ST3215_Servo)
- **[CODE]** LeRobot `write_calibration` writes Homing_Offset, Min_Position_Limit and Max_Position_Limit. `disable_torque` sets Lock = 0 (so the writes persist) and `enable_torque` sets Lock = 1. — [lerobot feetech.py](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/motors/feetech/feetech.py). In `calibrate()` the follower's `full_turn_motor = "wrist_roll"` gets `range_mins = 0` and `range_maxes = 4095`. — [lerobot so_follower.py](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/robots/so_follower/so_follower.py)
- **[COMMUNITY]** In #3585 the firmware is reported as "internally clamping the goal to the nearest edge of [Min_Position_Limit, Max_Position_Limit] (wrap-aware) and using that as the PID target". The Goal register keeps the host's value, so the substitution is invisible on the bus. — [lerobot #3585](https://github.com/huggingface/lerobot/issues/3585)
- **[COMMUNITY]** Comment on #3585 by hkder (2026-09-24; new SO-101 follower, STS3215 C018 12V, firmware 3.10):
  - shoulder_lift and wrist_flex "moved the wrong way on every position command" and pushed into the hard stop until they overloaded.
  - Setting the limits to 0/4095 fixed it; restoring the calibrated limits brought the fault back.
  - Local patch: always write 0/4095 and clamp on the host instead.

  — [lerobot #3585](https://github.com/huggingface/lerobot/issues/3585)
- **[CODE]** LeRobot's own SO follower limits gripper effort using protection registers, not position limits:
  - `Max_Torque_Limit = 500`: "50% of max torque to avoid burnout"
  - `Protection_Current = 250`: "50% of max current to avoid burnout"
  - `Overload_Torque = 25`: "25% torque when overloaded"

  — [lerobot so_follower.py](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/robots/so_follower/so_follower.py)
- **[CONFLICT]** LeRobot's comment reads Overload_Torque (36) as "25% torque when overloaded". The memory-table descriptions say 36 is the *trigger threshold* and 34 (Protective_Torque) is the output torque after protection. — [lerobot so_follower.py](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/robots/so_follower/so_follower.py) vs. [AIFitLab STS table](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis). Either way, a lower value of 36 makes protection trip earlier.

### Inferences
- **Position limits only help when they sit inside the mechanical range.** If they are inside the stop, the clamped goal never reaches it, so the servo cannot stall there. For a full-turn wrist_roll (0..4095) they constrain nothing, so they cannot prevent a cable-wrap or gripper-collision stall.
- **Narrowing wrist_roll limits has two risks:**
  1. The wrap-aware clamping is undocumented, and limits that interact with Homing_Offset have produced wrong-direction drives (#3585 comment).
  2. LeRobot's calibration code would overwrite them on the next calibration write.
- **Host-side alternatives are safer for anti-stall:**
  - clamp wrist_roll goals on the host;
  - use `max_relative_target`;
  - lower Torque_Limit or Max_Torque_Limit;
  - lower Overload_Torque or Protection_Time, so protection trips earlier and the servo spends less time at full stall current. This trades nuisance trips against heat.
- **Persistence.** Values written with Lock = 0 persist across power cycles. Torque_Limit (48) is SRAM and is reloaded from Max_Torque_Limit (16) at power-on, so persistent torque caps must go in 16.

### Gaps
- No vendor document (Feetech, Waveshare or Seeed) recommends using position limits as stall protection, or describes the firmware's clamping or wrap behaviour.
- I found no official statement on whether goals outside the limits are clamped or rejected.

## Q4. What do Waveshare, Seeed and Feetech say about SO-ARM100/101 overload, flashing LEDs, overheating, the wrist servo, and the SE leader servos?

### Takeaway
None of the vendor wikis has a dedicated overload or flashing-LED FAQ for SO-ARM100/101. I checked the Waveshare ST3215 and SO-ARM100/101 pages and the Seeed lerobot_so100m_new page, including its FAQ.

The relevant vendor facts are:
- Feetech's protection thresholds (80% for 2 s; 2 A for 2 s; 70 °C).
- Waveshare's EEPROM-unlock note.
- Waveshare's warning that the No. 5 (wrist) servo cable tears easily during assembly.
- The Seeed debug tool's Normal/Overcurrent/Overload status.

Waveshare's SE leader servos are "virtual" encoder-only units with no motor. The leader therefore never resists, so it can be driven into poses the follower physically cannot reach. For a full-turn wrist_roll, the risk is a follower stall into cable or gripper limits, or a wrap-around jump.

### Cited Findings
- **[OFFICIAL]** Waveshare SO-ARM100/101 Kit Assembly wiki:
  - "The ST3215 Servo (SE) is a virtual servo, which only has an encoder and no power section, making it very smooth to use."
  - SE kits list "ST3215 Servo (SE)*6".
  - Leader (3DP kit) uses IDs 1–3 ST3215 and IDs 4–6 ST3215-HS; follower uses six ST3215.
  - Wrist warning: "Be careful when inserting the No. 5 servo, otherwise it is easy to tear the wire."
  - No FAQ is linked.

  — [Waveshare SO-ARM100/101 Kit Assembly](https://www.waveshare.com/wiki/SO-ARM100/101_Kit_Aassembly)
- **[OFFICIAL]** Waveshare ST3215 wiki: the position range is 0–4095 with middle 2047. The FAQ says to unlock EPROM before modifying parameters. There is no overload or LED FAQ. — [Waveshare ST3215 wiki](https://www.waveshare.com/wiki/ST3215_Servo)
- **[OFFICIAL / UNVERIFIED]** The Waveshare "Set Servo ID" page reportedly notes that USB does not power the servos and gives SO-ARM101 power as 12V 5A. I saw this only in a search summary and did not open the page. — [Waveshare SO-ARM100/101 Set Servo ID](https://www.waveshare.com/wiki/SO-ARM100/101_Set_Servo_ID)
- **[OFFICIAL]** Seeed SO-Arm LeRobot tutorial: the gear ratios of the first three SO-ARM101 leader joints differ from SO-ARM100, and "the leader arm always uses the 7.4V motors". Its FAQ covers connection, calibration ("Magnitude … exceeds 2047"), sync-read and dataset errors, but **no overload, LED or overheating entry**. — [Seeed lerobot_so100m_new](https://wiki.seeedstudio.com/lerobot_so100m_new/)
- **[OFFICIAL]** Seeed's Feetech hub lists datasheets for ST3215-C047 (12V 1:345), C046 (7.4V 1:147), C044 (7.4V 1:191) and C001 (7.4V 1:345), plus the FT SDKs and the FD debug software. — [Seeed Feetech hub](https://wiki.seeedstudio.com/feetech_servo/)
- **[OFFICIAL]** Seeed debugging tool: status is Normal / Overcurrent / Overload. "Write neutral" stores the current pose as the neutral value. Disabling torque makes the arm go limp. — [Seeed debugging tool](https://wiki.seeedstudio.com/lerobot_steering_gear_debugging_tool/)
- **[MAINTAINER]** LeRobot issue #2819 (SO-101 follower gripper went stiff, then errors): maintainer @pkooij said the servo was likely broken from overload. Fully closing the leader gripper while the follower gripper holds an object can overload and damage the motor. The user replaced the servo. — [lerobot #2819](https://github.com/huggingface/lerobot/issues/2819)
- **[COMMUNITY]** andymai/so-arm101 README:
  - The Seeed-spec 12 V **2 A** supply browned out (the whole arm went limp); "a 12 V 5 A adapter" is "the de-facto community recommendation".
  - `soarm set-protection` caps acceleration to limit current spikes and applies uniform voltage/overload limits.
  - One follower servo shipped with Max_Voltage_Limit = 8.0 V on a 12 V rail and "refused all bus traffic".
  - "A continuous joint (wrist_roll) calibrated to an arbitrary zero was 111° out of sync", fixed with `calibrate-leader --align-wrist-roll`.

  — [andymai/so-arm101](https://github.com/andymai/so-arm101)
- **[CODE]** LeRobot `configure_motors` sets `Maximum_Acceleration = 254` and `Acceleration = 254` ("to speedup acceleration and deceleration"). This is the source of the user's Acceleration 254. — [lerobot feetech.py](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/motors/feetech/feetech.py)
- **[COMMUNITY]** A Waveshare-reseller assembly write-up warns that servo models differ by joint and are easy to swap by mistake. — seen only via search summary ([classmethod article](https://dev.classmethod.jp/en/articles/lerobot-so-arm101-assembly-teleop/)) **[UNVERIFIED detail]**

### Inferences
- **The SE leader adds a leader/follower mismatch risk.** SE leader servos have no motor or gear resistance, and Waveshare's 3DP leader mixes ST3215 and ST3215-HS units. The leader's wrist roll can therefore be turned freely and quickly to any angle.
- **Follower wrist_roll range mismatch.** The follower is full-turn in LeRobot but physically limited by the gripper cable and the servo 5 wire that Waveshare warns about. A leader rotation beyond that range, or across the 0/4095 seam if the two arms' zeros differ (andymai's 111° case), commands the follower into its own cable or stop.
- **What happens next.** At 80% load for 2 s it trips overload. The LED flashes, the servo drops to 20%, and every reply returns "Overload error!".
- **Maximum acceleration makes it worse.** Acceleration 254 (maximum) makes these moves violent and current-spiky, which adds to brownout risk on an undersized supply.
- **Mitigations consistent with the sources** (none is a vendor-published SO-101 procedure):
  - align the wrist_roll zero between leader and follower;
  - clamp or limit wrist_roll travel on the host, for example with `max_relative_target`;
  - check the servo 5 cable routing and slack;
  - use a 12 V supply of at least 5 A;
  - optionally lower Acceleration and Max_Torque_Limit on ID 5;
  - clear a trip by writing Goal_Position = Present_Position, or by power-cycling.

### Gaps
- I found no Waveshare or Seeed FAQ entry on overload errors, flashing LEDs, or servo overheating specific to SO-ARM100/101.
- I found no vendor statement on leader/follower range mismatch specific to the SE (encoder-only) leader servos, or on wrist_roll specifically, beyond Waveshare's cable-tearing warning.
- Gear ratios and voltage of the Waveshare SE servo (encoder-only) are not given on the wiki page I read.
- The exact Waveshare ST3215 12V variant (C047 vs C018) for the user's kit is unconfirmed. The official datasheet used above is for ST-3215-C047 (12V, 1/345). Protection thresholds are expected to match within the family, but this is **[UNVERIFIED]**.
