# Adapting Gemini Robotics VLAs to a custom embodiment (SO-101 follower arm + Reachy Mini head), as of 15 Oct 2026

Scope note: these notes come from web research on 15 Oct 2026. Primary sources are DeepMind blogs, arXiv papers, and the public `google-deepmind/gemini-robotics-sdk` repo. I read the repo's source directly through the GitHub API at version 2.156.0 (latest commit 12 Oct 2026). The trusted-tester docs page on ai.google.dev returns 404 to the public, so anything documented only there is out of reach. Secondary press coverage is labelled as such.

Model timeline, for orientation:
- Gemini Robotics 1.0 (VLA) and ER, Mar 2025.
- Gemini Robotics On-Device ("GROD" v1), 24 Jun 2025.
- Gemini Robotics 1.5 and ER 1.5, 25 Sep 2025.
- Gemini Robotics 2, ER 2 and On-Device 2 (GROD v2), 30 Jul 2026.
- Gemini 4 Argon for Robotics, a general physical-reasoning model rather than a VLA, 13 Oct 2026.

## 1. Does Google allow fine-tuning Gemini Robotics models on user data? Through what SDK or program, how many demos, and in what data format?

### Takeaway
Yes, but only for approved trusted testers or early-access partners, and only as a closed-weight managed service. You record or convert episodes to MCAP with the Safari SDK (`safari-sdk` on PyPI). You upload them with `flywheel-cli upload_data` and launch training on Google's side with `flywheel-cli train`. You then get a checkpoint and Docker inference image to serve locally. A LeRobot-to-MCAP converter ships in the SDK. GROD v2's `--embodiment` flag lists `so101` as one of only four allowed embodiments. Google cites 50–100 demos for task adaptation (On-Device v1) and "less than 200 examples" plus "a few hours" for a new bi-arm embodiment (On-Device 2).

### Cited Findings
**Access and program**
- On-Device v1 (24 Jun 2025) was "the first VLA model we're making available for fine-tuning". It was released to "a select group of trusted testers". Developers access the SDK by signing up to the trusted tester program. — [DeepMind blog, On-Device](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/); [Trusted tester form](https://docs.google.com/forms/d/1sM5GqcVMWv-KmKY3TOMpVtQ-lDFeAftQ-d9xQn92jCE/edit?ts=67cef986)
- The On-Device model page lists availability as "Trusted tester". — [DeepMind On-Device model page](https://deepmind.google/models/gemini-robotics/on-device/)
- Gemini Robotics 2 (30 Jul 2026): the VLA and On-Device models are available to "early-access partners" through a form, alongside a Trusted Tester Program sign-up. ER 2 is in Google AI Studio and in private preview on the Gemini Enterprise Agent Platform. — [DeepMind blog, Gemini Robotics 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)
- Secondary coverage says the two hardware-facing models (VLA and On-Device 2) are "waitlist-only". Only ER 2 is usable now through the Gemini API. — [eesel.ai summary](https://www.eesel.ai/blog/gemini-robotics-2)
- Commercial licensing terms for robot makers have not been published. — [TechWire Asia](https://techwireasia.com/2026/07/google-gemini-robotics-2-whole-body-control/)
- Gemini Robotics 1.5 (Sep 2025) was "currently available to select partners", and ER 1.5 was available via the Gemini API. — [DeepMind blog, GR 1.5](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/)

**SDK and pipeline**
- The SDK repo (Apache-2.0, "not an officially supported Google product") covers the full lifecycle: "access checkpoint, serving a model, evaluate the model on robot and in sim, upload data, finetuning the model, download the finetuned checkpoint". "Most of the functionality requires you to join Gemini Robotics Trusted Tester Program". "Trusted Testers can access the Gemini Robotics On Device model from SDK v2.4.1." — [gemini-robotics-sdk README](https://github.com/google-deepmind/gemini-robotics-sdk)
- The `flywheel-cli` commands are `train`, `serve`, `list`, `list_serve`, `data_stats`, `download`, `upload_data` and `version`. `train` needs a task ID plus start and end dates, meaning it trains on data previously uploaded for that task. — [README](https://github.com/google-deepmind/gemini-robotics-sdk); [flywheel_cli.py](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/safari_sdk/flywheel/flywheel_cli.py)
- Training recipes in the CLI are `narrow`, `gemini_robotics_v1`, `gemini_robotics_on_device_v1` and `gemini_robotics_on_device_v2`. — [flywheel_cli.py](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/safari_sdk/flywheel/flywheel_cli.py)
- **The GROD v2 recipe requires `--embodiment`.** Its allowed values are `["trossen", "so101", "dexmate", "fr3_duo"]`, and the help text reads "Required for gemini_robotics_on_device_v2 to specify the embodiment." — [flywheel_cli.py, SDK 2.156.0](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/safari_sdk/flywheel/flywheel_cli.py)
- GROD v1 instead uses `--checkpoint_type` with values `aloha` or `franka_duo`. — [flywheel_cli.py](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/safari_sdk/flywheel/flywheel_cli.py)
- Other GROD v1/v2 training flags:
  - `--image_keys` and `--proprioception_keys`: these must be a subset of the keys logged by `EpisodicLogger`. Proprioception keys are required. For v2, the keys "cannot contain dots".
  - `--max_episodes`, `--seed`, `--only_successful_episodes`.
  - `--max_training_steps`, capped at 50,000 by backend validation.
  - `--checkpoint_every_n_steps`, minimum 100.
  
  — [flywheel_cli.py](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/safari_sdk/flywheel/flywheel_cli.py)

**Data format**
- The SDK's logging stack writes MCAP files, through C++ `mcap_file_handle` and an `EpisodicLogger`. The SDK also ships an MCAP loader for offline evaluation. — [SDK repo tree, safari_sdk/logging](https://github.com/google-deepmind/gemini-robotics-sdk/tree/main/safari_sdk/logging); [examples/offline_eval](https://github.com/google-deepmind/gemini-robotics-sdk/tree/main/examples/offline_eval)
- `examples/logging/lerobot_data_conversion_script.py` is documented as "A script to convert LeRobot datasets to MCAP format".
  - It requires `lerobot >= 0.4.0` and loads data with `LeRobotDataset(...)`.
  - It writes MCAP tagged with a `task_id` "used to identify data for later finetuning".
  - Its example dataset name is `lerobot/aloha_static_cups_open`.
  - The underlying `mcap_lerobot_logger` maps LeRobot's `action` key and `observation.*` keys into image and proprioception specs.
  
  — [lerobot_data_conversion_script.py](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/examples/logging/lerobot_data_conversion_script.py); [mcap_lerobot_logger.py](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/safari_sdk/logging/python/mcap_lerobot_logger.py)
- A third-party description of the CLIFT work says the API maps "observation–instruction–action-chunk tuples" to a fine-tuned policy. GROD is "a closed-weight model exposed through a managed SFT API" with no access to "model weights, gradients, or training internals". — [CLIFT, arXiv 2607.29172](https://arxiv.org/html/2607.29172)
- I found no evidence that RLDS is the user-facing upload format for GROD. The public SDK exposes MCAP plus a LeRobot converter. — [SDK repo](https://github.com/google-deepmind/gemini-robotics-sdk)

**Number of demonstrations**
- On-Device v1 can be adapted "with as few as 50 to 100 demonstrations". It was tested on seven dexterous tasks; average fast-adaptation success was about 0.68 for On-Device versus about 0.8 for the flagship model. — [DeepMind blog, On-Device](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/)
- Gemini Robotics 1.0 tech report: with at most 100 demos, success was above 70% on 7 of 8 sub-tasks. That amount of data is roughly 15 min to 1 h of demonstrations. — [arXiv 2503.20020](https://arxiv.org/html/2503.20020)
- On-Device 2 can adapt to "new bi-arm robot embodiments with just a few hours of adaptation time, typically with less than 200 examples", even ones with "drastically different shapes, sensors and degrees of freedom". — [DeepMind blog, GR 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)
- CLIFT (Unitree G1 humanoid) collected "2 hours of human teleoperated data" per task. Each flywheel cycle trained 10 epochs through the managed API. — [CLIFT, arXiv 2607.29172](https://arxiv.org/html/2607.29172)

### Inferences
- The developer cannot fine-tune Gemini Robotics weights themselves. The weights are closed, and training runs on Google's backend via `flywheel-cli train`. The realistic route is to apply as a trusted tester or early-access partner and, if accepted, upload SO-101 data.
- The developer's LeRobot 0.6.1 recordings are a plausible input, since the converter only checks `>= 0.4.0` and uses the v3-style `meta.episodes[...]['dataset_to_index']` API. I have not verified this against 0.6.1.
- LeRobot keys such as `observation.images.front` contain dots, which GROD v2 rejects as `--image_keys`. The converter appears to strip the `observation.` prefix, but key naming should be checked.
- Because `so101` is a fixed enum value, Google has presumably built a base "SO-101 embodiment" head or normalization into GROD v2. Matching Google's expected SO-101 camera and joint layout may matter more than the raw demo count. That expected layout isn't public; see Gaps.

### Gaps
- The trusted-tester docs on ai.google.dev (`/gemini-api/docs/eap/robotics/robotics-ondevice`) return 404 publicly. These gated docs are probably where the expected SO-101 observation and action spec, camera names, control rate, recommended demo counts and turnaround time live.
- Not found: whether `so101` means a single arm or a bimanual SO-101 pair. The GR 2 blog frames the adaptation as "bi-arm", and LeRobot has a `bi_so_follower` robot class. `replay_eval.py` references `safari_sdk.examples.so101.env`, but that module isn't in the public tree.
- Not found: pricing, quotas, and acceptance criteria for individual hobbyist applicants versus companies.
- Not found: whether a fully custom embodiment, outside the four enum values, can be requested.

## 2. Which new embodiments have been demonstrated, and how was cross-embodiment transfer done?

### Takeaway
DeepMind's documented path runs as follows:
- Gemini Robotics 1.0 and On-Device v1 were trained on ALOHA 2, then adapted by fine-tuning on small amounts of target-robot data to a bi-arm Franka FR3 and to the Apptronik Apollo humanoid.
- GR 1.5 added "Motion Transfer", where one model controls ALOHA 2, Franka and Apollo without per-robot specialization.
- On-Device 2 (Jul 2026) inherits Motion Transfer and is reported adapted to SO101, Dexmate, Trossen and Franka Duo.

DeepMind reports that SO101 adaptation success rose from 6.7% to 53.3%; this figure comes via secondary coverage.

### Cited Findings
**Gemini Robotics 1.0 and On-Device v1**
- On-Device was trained only on ALOHA, then adapted to the bi-arm Franka FR3 and to Apptronik's Apollo humanoid. On the FR3 it handled general instruction following, unseen objects, dress folding and belt assembly; on Apollo it followed instructions with unseen objects. — [DeepMind blog, On-Device](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/)
- Gemini Robotics 1.0 was trained on ALOHA 2 teleop data, "thousands of hours" collected over 12 months. It was adapted with "a small amount of data on the target platforms". The adapted bi-arm Franka model averaged 63% success on the tasks considered; no Apollo success rate is given. — [arXiv 2503.20020](https://arxiv.org/html/2503.20020)

**Gemini Robotics 1.5 and Motion Transfer**
- GR 1.5 "can transfer motions learned from one robot to another, without needing to specialize the model to each new embodiment". Tasks "only presented to the ALOHA 2 robot during training, also just work on the Apptronik's humanoid robot, Apollo, and the bi-arm Franka robot, and vice versa." — [DeepMind blog, GR 1.5](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/); [GR 1.5 tech report PDF](https://storage.googleapis.com/deepmind-media/gemini-robotics/Gemini-Robotics-1-5-Tech-Report.pdf)

**Gemini Robotics 2 and On-Device 2**
- On-Device 2 "is natively multi-embodiment and inherits our advanced 'motion transfer' techniques from Gemini Robotics 1.5". Embodiments shown include Apollo 2 (with SharpaWave 22-DoF hands and Inspire hands) and Franka Duo with Robotiq grippers. Dexmate, SO101 and Trossen platforms are shown for On-Device 2. — [DeepMind blog, GR 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)
- Reported adaptation numbers:
  - SO101: On-Device 2 went from 6.7% to 53.3%, versus 0.0% to 6.7% for the previous on-device model.
  - Dexmate: 24.4% to 75.6%, versus 13.3% to 33.3% for the previous model.
  
  These are DeepMind's own figures relayed by press; I could not fetch the primary chart. — [MarkTechPost](https://www.marktechpost.com/2026/07/30/google-deepmind-gemini-robotics-2-whole-body-control-dexterity-multi-robot-collaboration/); [Robozaps](https://blog.robozaps.com/b/gemini-robotics-2-humanoid-robot-ai) (Robozaps gives a different baseline for the prior model)
- Support for a demonstration platform "does not mean every configuration is commercially supported". — [eesel.ai](https://www.eesel.ai/blog/gemini-robotics-2)
- The model card is reported to list limits on out-of-distribution tasks and on high-DoF robots. — [MarkTechPost via search summary](https://www.marktechpost.com/2026/07/30/google-deepmind-gemini-robotics-2-whole-body-control-dexterity-multi-robot-collaboration/)
- GR 2's VLA controls the full Apollo 2 body, including walking and crouching. Earlier versions handled only the upper body. — [humanoid.guide](https://humanoid.guide/google-deepmind-demonstrates-gemini-robotics-2-humanoid-control/)
- Date conflict: one outlet's headline labels the GR 2 news "(October 2026)", but DeepMind's post and most coverage date the release to 30 Jul 2026. — [SmashingRobotics](https://www.smashingrobotics.com/google-deepmind-says-gemini-robotics-2-enables-full-body-control/) vs [DeepMind blog](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)

**SDK evidence for these embodiments**
- The SDK's policy example defines `robot_type` choices `aloha`, `trossen` and `franka_duo`, with specs:
  - ALOHA/Trossen: four 480x848 cameras and `joints_pos` of dim 14.
  - Franka Duo: two head RGB cameras, two 7-DoF arms and two 1-D grasp primitives.
  
  — [gemini_robotics_policy_example.py](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/examples/model/gemini_robotics_policy_example.py)
- The Docker inference server release notes list "2.0.0 – Release of GRODv2 (Gemini Robotics On-Device v2)". v1.1.0 added "customization of image names and proprioception key names". — [RELEASE_NOTES_DOCKER.md](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/RELEASE_NOTES_DOCKER.md)

**Third-party new-embodiment case**
- CLIFT (UC Berkeley, Google DeepMind and NVIDIA Research; 31 Jul 2026) fine-tuned GROD through the managed SFT API onto a Unitree G1 humanoid.
  - Box packing rose from 93% to 100% and cup insertion from 70% to 98% after two CLIFT cycles.
  - Plain GROD SFT "substantially outperforms" π0.5 trained on the same demos.
  
  — [CLIFT, arXiv 2607.29172](https://arxiv.org/abs/2607.29172)

### Inferences
- For new embodiments, the publicly visible mechanism is a pretrained multi-embodiment model with Motion Transfer, fine-tuned on a small target-robot dataset that Google ingests. The base model already includes SO-101 data, since `so101` is a GROD v2 embodiment. A single SO-101 is therefore likely the easiest of the developer's two robots to get supported.
- The 53.3% SO101 figure suggests that even after adaptation, success on SO-101 tasks is moderate. Expect tens of percent, not near-perfect, unless iterative data collection (CLIFT-style) is used.

### Gaps
- Not found: DeepMind's primary chart or task definitions for the SO101 and Dexmate adaptation numbers, including how many demos and which tasks. MarkTechPost returned 403 to the fetch.
- Not found: a public Motion Transfer algorithm description at the action-space level. The GR 1.5 tech report PDF was linked but not read in full here.

## 3. Is there any public example of Gemini Robotics on SO-100/SO-101, Koch or other LeRobot arms, or on Reachy / Reachy Mini? Any HF or Pollen collaboration with DeepMind?

### Takeaway
SO-101 is DeepMind-supported:
- It is a named On-Device 2 embodiment and a `--embodiment so101` value in the SDK.
- It appears in a Gemini 4 Argon demo (13 Oct 2026), where Argon wrote code; no VLA was involved there.
- It has an official MuJoCo Menagerie model.

There is no public evidence of Gemini Robotics VLAs on SO-100, Koch, Reachy 2 or Reachy Mini, and no announced Hugging Face/Pollen–DeepMind partnership on Gemini Robotics. The closest Reachy Mini link is a Google DevRel demo of Gemma, not Gemini Robotics.

### Cited Findings
**SO-101**
- SO101 is named among the On-Device 2 platforms. — [DeepMind blog, GR 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)
- The SDK's GROD v2 `--embodiment` enum includes `so101`. `replay_eval.py` defaults to `--env_module=safari_sdk.examples.so101.env` / `create_so101_environment`, a module absent from the public repo. — [flywheel_cli.py](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/safari_sdk/flywheel/flywheel_cli.py); [replay_eval.py](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/examples/model/replay_eval.py)
- Gemini 4 Argon for Robotics (13 Oct 2026) shows "SO-101 Vision-Guided Pick-and-Place — Iterative Hardware Self-Improvement". Argon writes and debugs a planner from logs and wrist-camera footage, cutting task time from 58.9 s to 17.4 s over three iterations. The post doesn't name a VLA controlling the arm. — [Google Developers Blog, Gemini 4 Argon for Robotics](https://developers.googleblog.com/gemini-4-argon-for-robotics/)
- The same Argon post says Argon "can generate new data in simulation to improve performance of generalist vision-language-action (VLA) models", including "4,500+ domain-randomized episodes". — [Google Developers Blog](https://developers.googleblog.com/gemini-4-argon-for-robotics/)
- MuJoCo Menagerie contains `robotstudio_so101`, last changed 28 Sep 2026 ("Fix SO101 wrist_roll joint range upper limit"). It also contains `aloha`, `apptronik_apollo`, `franka_fr3`/`franka_fr3_v2` and `trossen_*` models. — [mujoco_menagerie repo](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)
- A community repo "GeminiRobotics-SO101" appeared in search results described as combining Gemini Robotics models with the SO-101. It returned HTTP 404 (deleted or private) on 15 Oct 2026, so its contents are unverifiable. — [github.com/johannstark/GeminiRobotics-SO101 (404)](https://github.com/johannstark/GeminiRobotics-SO101)

**Reachy Mini and Hugging Face**
- MuJoCo Menagerie has no Reachy Mini model. Pollen ships its own MJCF at `src/reachy_mini/descriptions/reachy_mini/mjcf/reachy_mini.xml`, used by `reachy-mini-daemon --sim`. — [reachy_mini repo](https://github.com/pollen-robotics/reachy_mini/blob/main/src/reachy_mini/descriptions/reachy_mini/mjcf/reachy_mini.xml); [Seeed Reachy Mini sim guide](https://wiki.seeedstudio.com/es/reachymini_platforms_simulation_get_started/)
- LeRobot's `src/lerobot/robots` (main branch, Oct 2026) includes `reachy2`, `so_follower`, `bi_so_follower`, `koch_follower`, `lekiwi`, `unitree_g1` and others, but no `reachy_mini` class. — [LeRobot robots directory](https://github.com/huggingface/lerobot/tree/main/src/lerobot/robots)
- A Google DeepMind developer-relations demo ran Gemma (not Gemini Robotics) on Reachy Mini, taking pictures, moving the head and showing emotion. This was found only via a low-quality aggregator page. — [taxheal.com mirror](https://www.taxheal.com/run-gemma-on-reachy-mini-an-open-source-robot.html)
- Hugging Face acquired Pollen Robotics in 2025, and Reachy Mini was co-developed by the two. — [HF blog, Reachy Mini](https://huggingface.co/blog/reachy-mini)
- Searches for a Gemini Robotics × LeRobot/HF partnership returned nothing. — (no source found; see Gaps)

### Inferences
- In this public material, SO-101 is the only LeRobot-ecosystem arm with first-party Gemini Robotics VLA support. Koch, SO-100 and Reachy Mini would need Google to add an embodiment.
- The SO-100 is kinematically very close to the SO-101, so it might work under the `so101` embodiment. This is untested speculation.
- The Reachy Mini is not a manipulator. A Gemini Robotics VLA has no documented reason or path to drive it. Gemini ER, Live or Argon via the API (perception, pointing, planning) is the more natural fit for the head.

### Gaps
- No HF forum or Discord evidence was found of hobbyists running GROD v2 on SO-101. Trusted-tester NDAs may suppress public reports.
- Not found: the original Google source for the Gemma-on-Reachy-Mini demo.

## 4. How would a head (camera pose, gaze, Stewart platform) be included in a VLA's action space alongside an arm? Do Gemini Robotics models support controlling camera or head motion?

### Takeaway
No Gemini Robotics source documents head, neck, gaze or camera-pose actions. Published Gemini embodiments use fixed or head-mounted cameras purely as observations. Examples are ALOHA's four cameras, Franka Duo's two head cameras, and the G1 head cameras in CLIFT, where the action space covers arms, hands, base and torso but not the head. The fixed GROD v2 embodiment list (`trossen`, `so101`, `dexmate`, `fr3_duo`) gives no public way to append Reachy Mini head DoFs. Head-as-action is established in academic VLA work: Vision in Action, ActiveScale on π0.5, EyeVLA and ActiveVLA. Today that would mean an open model rather than Gemini.

### Cited Findings
**Gemini Robotics**
- In CLIFT's GROD setup on Unitree G1, the cameras are two head-mounted RGB cameras. Actions are upper-body joint targets for arms and hands plus base velocity and yaw, executed through an RL whole-body controller. No head or neck action is listed, and the policy outputs 1.6 s action chunks. — [CLIFT, arXiv 2607.29172](https://arxiv.org/html/2607.29172)
- SDK example specs: Franka Duo uses `head_cam_left_rgb` and `head_cam_right_rgb` as observations, with actions limited to arm joints and grasp primitives. ALOHA uses four cameras plus 14 joint positions. — [gemini_robotics_policy_example.py](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/examples/model/gemini_robotics_policy_example.py)
- The GR 2 blog doesn't mention head, camera or gaze control. Its "feet to fingertips" whole-body claim concerns locomotion and manipulation. — [DeepMind blog, GR 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/); [humanoid.guide](https://humanoid.guide/google-deepmind-demonstrates-gemini-robotics-2-humanoid-control/)
- The Gemini Robotics 1.0 tech report describes action chunks for robot joints, with no head or camera control mentioned. — [arXiv 2503.20020](https://arxiv.org/html/2503.20020)

**Academic head-as-action work**
- Vision in Action (CoRL 2025) uses a 6-DoF ARX5 arm as a "neck" carrying an iPhone camera. Head pose is part of the learned action, taught from VR teleop where the operator's head pose drives the neck. It learns search, track and focus behaviours on bimanual occluded tasks. — [arXiv 2506.15666](https://arxiv.org/abs/2506.15666); [PMLR](https://proceedings.mlr.press/v305/xiong25a.html)
- ActiveScale (arXiv 2609.18514, Sep 2026) builds on π0.5 and treats camera motion as part of the action space. An auxiliary head predicts a 9-D camera pose in training and is dropped at inference. This is from a search-result summary; the full paper was not read. — [arXiv 2609.18514](https://arxiv.org/pdf/2609.18514)
- EyeVLA outputs pan, tilt and zoom for a PTZ camera as discrete action tokens. — [arXiv 2511.15279](https://arxiv.org/html/2511.15279v2)
- ActiveVLA (CVPR 2026) adds active viewpoint selection to VLAs. — [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/36656)

### Inferences
- For Gemini Robotics as publicly offered, the Reachy Mini head can only be an observation source (a "head cam") for the SO-101 policy, not an action. Holding the head at a fixed "look at workspace" pose during both data collection and deployment would keep the camera distribution stable.
- Head motion would then be driven separately. Options include:
  - scripted gaze;
  - Gemini ER or Live pointing outputs mapped to the reachy_mini `look_at` or head-pose API;
  - the developer's existing emotion and gesture logic, gated to pause manipulation while the head moves.
- If a joint arm+head policy is wanted, an open VLA in LeRobot could take a concatenated action vector. Examples are SmolVLA, π0/π0.5 or GR00T. The vector would be 6 SO-101 joints plus head pose (6-D, or a reduced yaw/pitch gaze) plus body yaw, with head teleop from a VR or phone tracker as in ViA. This requires writing a LeRobot `Robot` wrapper for Reachy Mini, since none exists upstream.
- With a moving head camera, actions in camera or world frame need the head pose logged as proprioception. Otherwise the arm policy sees an unexplained shifting viewpoint.

### Gaps
- Not found: any statement from Google on roadmap support for user-defined DoFs such as neck or gaze, or for multi-robot composite embodiments (arm plus separate head).
- The GR 1.5 and GR 2 tech reports' detailed action-space tables were not read. Apollo has a head, but there is no evidence its neck is actuated by the VLA.

## 5. What compute does on-device inference and fine-tuning need? Can it run or be fine-tuned on Apple Silicon? What are the latency figures?

### Takeaway
Fine-tuning runs on Google's backend, so no local GPU is needed for training. Local inference uses a Google-supplied Docker image (`google-deepmind/gemini_robotics_on_device`, JAX/XLA) launched with `--gpus device=0`. That implies an NVIDIA GPU on Linux, with a slower `--use_cpu` fallback. The SDK's PyPI wheels are Linux x86_64 only, and the SDK also offers CLOUD and CLOUD_GENAI connection modes. An M3 Pro MacBook is not a documented inference host for GROD. It could act as a client to a Linux/NVIDIA box or to cloud serving. Public latency figures exist only for Gemini Robotics 1.0 (cloud backbone): under 160 ms query, about 250 ms end-to-end, 50 Hz effective control. No numbers have been published for On-Device v1 or v2.

### Cited Findings
**Serving and hardware**
- `flywheel-cli serve` for GROD v1/v2 "requires Docker". It "will download the model and start a serving docker container". The container runs `docker run ... --gpus device=0 -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e XLA_PYTHON_CLIENT_MEM_FRACTION=... google-deepmind/gemini_robotics_on_device:<tag>` and exposes port 60061. Flags include `--use_cpu` ("Whether to use CPU for serving. If False, GPU will be used."), `--gpu_mem_fraction`, `--shm_size` (default 8g) and `--license_path` (a "license file corresponding to model checkpoint"). — [flywheel_cli.py](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/safari_sdk/flywheel/flywheel_cli.py)
- `RoboticsApiConnectionType` has `CLOUD` ("Connects to a Google Cloud-based server"), `LOCAL` ("Connects to a local server") and `CLOUD_GENAI` ("Connects via GenAI API"). The ALOHA example covers "LOCAL, CLOUD, and CLOUD_GENAI connection types". — [constants.py](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/safari_sdk/model/constants.py); [examples/model/README.md](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/examples/model/README.md)
- `safari-sdk` 2.156.0 (12 Oct 2026) on PyPI ships only `manylinux_2_27/2_28 x86_64` wheels for CPython 3.11–3.13, plus an sdist, and requires Python ">=3.10,<3.14". The package includes "compiled C++ logging extensions". — [PyPI safari-sdk](https://pypi.org/project/safari-sdk/); [README](https://github.com/google-deepmind/gemini-robotics-sdk)
- The CI Dockerfile builds from `quay.io/pypa/manylinux_2_28_x86_64`. — [Dockerfile](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/Dockerfile)
- DeepMind describes On-Device as "engineered to require minimal computational resources" with "low-latency inference", but gives no hardware spec or latency numbers. — [DeepMind blog, On-Device](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/)
- On-Device 2 is "optimized to run locally", but no numbers are given. — [DeepMind blog, GR 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)
- Third-party claims about suitable hardware are unsourced and unreliable, so treat them as noise. A Medium post says the action decoder can run on ARM CPUs, and it also includes an apparently fabricated case study. A sider.ai guide suggests Jetson AGX Orin or Orin NX. InfoQ gives no hardware figures. — [Medium](https://medium.com/@babarranjha/gemini-robotics-on-device-bringing-real-time-robot-smarts-to-the-edge-c67aa115506b); [sider.ai](https://sider.ai/blog/ai-tools/how-to-use-gemini-robotics-on-device-for-offline-robot-control-tasks); [InfoQ](https://www.infoq.com/news/2025/07/google-gemini-robotics/)

**Latency**
- Gemini Robotics 1.0 pairs a cloud-hosted VLA backbone with a local action decoder. Backbone query-to-response latency is "under 160ms", end-to-end latency is about 250 ms, and "the effective control frequency is 50Hz". — [arXiv 2503.20020](https://arxiv.org/html/2503.20020)
- GROD (CLIFT setup) emits 1.6 s action chunks and replans after execution. — [CLIFT, arXiv 2607.29172](https://arxiv.org/html/2607.29172)
- The SDK has `action_chunk_resampling.py` and an `ASYNCHRONOUS` inference mode. Its `genai_robotics_example.py` benchmarks raw `generate_content` latency. — [SDK model dir](https://github.com/google-deepmind/gemini-robotics-sdk/tree/main/safari_sdk/model); [genai_robotics_example.py](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/examples/model/genai_robotics_example.py)

**Fine-tuning compute**
- Training is requested with `flywheel-cli train` against uploaded data. Artifacts, including Docker images, are then fetched with `download`. — [README](https://github.com/google-deepmind/gemini-robotics-sdk); [flywheel_cli.py](https://github.com/google-deepmind/gemini-robotics-sdk/blob/main/safari_sdk/flywheel/flywheel_cli.py)
- CLIFT confirms it is a managed SFT API with no access to weights or gradients. — [CLIFT, arXiv 2607.29172](https://arxiv.org/html/2607.29172)

### Inferences
- **Fine-tuning on the Mac:** not applicable. It runs on Google's infrastructure, and the Mac only records, converts and uploads data.
- **Local inference on the Mac:** effectively unsupported.
  - Docker Desktop on macOS cannot pass through the Apple GPU (`--gpus` is NVIDIA-container-toolkit-specific).
  - The image is almost certainly linux/amd64 (inferred from the x86_64-only wheels and build image), so it would run under emulation.
  - `--use_cpu` under emulation on an M3 Pro would probably be far too slow for closed-loop control, though this is untested.
  - The 18 GB of unified RAM is another constraint, since the model size is undisclosed.
- **Practical options:**
  - (a) A Linux x86_64 box with an NVIDIA GPU running the GROD container, with the Mac or robot host as a LOCAL-mode client over LAN.
  - (b) CLOUD or CLOUD_GENAI serving if Google grants it, accepting network latency on the order of the GR 1.0 figures (about 160–250 ms).
  - (c) For local Apple Silicon inference, an open policy instead: SmolVLA or ACT via LeRobot, which run on MPS.
- **Installing the SDK on macOS:** the `safari-sdk` client may need a source build (sdist with C++ MCAP logging), or a Linux VM or container. Untested.

### Gaps
- No public GPU model or VRAM minimum, and no measured inference latency or control rate, for On-Device v1 or v2. These are likely in the gated EAP docs.
- Not found: whether the Docker image has a linux/arm64 variant, for Jetson or Apple Silicon Linux VMs.
- Not found: fine-tuning turnaround time or cost per job. The CLI default timeout is 15 min per request, which says nothing about training duration.
