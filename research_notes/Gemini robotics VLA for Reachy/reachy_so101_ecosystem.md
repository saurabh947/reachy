# Reachy Mini and SO-101 ecosystem for a VLA and an LLM "brain" (state as of 15 October 2026)

Scope: what Pollen Robotics, Hugging Face (HF), Google, and the community offer for (a) conversational "brains" on Reachy Mini, (b) LeRobot support for Reachy Mini and for pairing it with SO-100/101 arms, (c) VLA results on SO-101 and Apple Silicon inference, (d) handling a moving head camera in VLA training, and (e) emotion and expressive behavior. Web research only, no hardware. Dates are publication or last-activity dates seen on the cited page. Items marked **[UNVERIFIED]** could not be traced to a primary source.

Context flag for the developer: the latest `reachy_mini` SDK release is **v1.12.0 (2 Oct 2026)**, and the official emotions dataset needs **v1.8.4 or later**. The developer's setup notes say v1.6, so several features below assume an SDK upgrade (see Q5).

---

## Q1. Do Pollen Robotics or Hugging Face provide Reachy Mini apps or examples using Gemini (Live API, Robotics-ER, Gemini Robotics VLA), OpenAI realtime, or other conversational brains? What do they look like?

### Takeaway
Pollen's official app, `reachy_mini_conversation_app`, follows one pattern: a realtime speech-to-speech backend plus a tool layer (`move_head`, `play_emotion`, `dance`, `camera`, `head_tracking`, and more) driven by a 60 Hz movement manager. Its current README describes only the Hugging Face realtime backend. Google's own walkthrough of the same repo (April 2026) runs it on `gemini-3.1-flash-live-preview` with a `gpt-realtime` switch. Gemini Robotics-ER 2 appears only in a community app (`max-lm/reachy_mini_er2`, Sep 2026), which uses ER 2 for voice and for pointing the head. I found no Pollen or HF example that uses the Gemini Robotics VLA (On-Device) on Reachy Mini.

### Cited Findings
**Official Pollen conversation app**
- `pollen-robotics/reachy_mini_conversation_app` (Apache 2.0, 688 commits on main, no tagged releases) combines "realtime voice, vision, personality-aware tools, and choreographed motion." Its README (as fetched Oct 2026) documents only the Hugging Face realtime backend, in two modes: "deployed" (the default, an app-managed Space proxy that needs no API key) or "local" via `HF_REALTIME_WS_URL`, for example HF's `speech-to-speech` server. The README does not mention OpenAI or Gemini. — [GitHub: pollen-robotics/reachy_mini_conversation_app](https://github.com/pollen-robotics/reachy_mini_conversation_app)
- Built-in tools: `move_head` (left/right/up/down/front), `head_tracking` (follow a face), `sweep_look`, `play_emotion`/`stop_emotion`, `dance`/`stop_dance`, `camera` (sends the latest frame to the backend), `idle_do_nothing`, `go_to_sleep`, `remember`/`forget`, `volume_control`, and `robot_status`. MCP Spaces for search, weather, time, and knowledge come preinstalled. — [GitHub README](https://github.com/pollen-robotics/reachy_mini_conversation_app)
- Extension points:
  - **Custom tools:** a Python `Tool` subclass with `name`, `description`, a JSON Schema `parameters_schema`, and an async `__call__`. It goes in `tools/` or in `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY`.
  - **Remote tools:** any Gradio Space that exposes MCP at `/gradio_api/mcp/`.
  - **Profiles:** each is a `profile.md` with TOML metadata (`default_tools`) followed by a Markdown prompt.
  - Source: [GitHub README](https://github.com/pollen-robotics/reachy_mini_conversation_app)
- Google's walkthrough, "Build a Talking Robot with Gemini Live and Reachy Mini" (Thor Schaeff, Google AI on DEV, 13 April, year not shown and inferred as 2026), uses this same Pollen repo:
  - The default `MODEL_NAME` is `gemini-3.1-flash-live-preview`. Setting it to `gpt-realtime` switches to OpenAI Realtime.
  - Audio runs over `fastrtc` WebRTC (16 kHz in, 24 kHz out), and the camera sends JPEGs at 1 FPS.
  - A `BackgroundToolManager` runs tool calls so they don't block audio.
  - A 60 Hz `MovementManager` sequences primary moves (emotions, dances) and layers additive offsets on top (speech wobble, face tracking, breathing).
  - No latency numbers are given, and Robotics-ER, Gemma, and arms are not mentioned.
  - Source: [DEV / Google AI](https://dev.to/googleai/build-a-talking-robot-with-gemini-live-and-reachy-mini-20e2)
  - **Conflict:** the upstream README now describes only the HF backend, while the April article shows Gemini and OpenAI in the same repo. Either the multi-provider code moved, or the README is out of date. **[UNVERIFIED which is current]**
- A community fork (`eduardoyi/reachy_mini_conversation_app`, active through 13 Oct 2026) adds a `REALTIME_PROVIDER` switch:
  - `huggingface` (the default).
  - `gemini`: defaults to `gemini-3.1-flash-live-preview` with voice `Puck`, and lists `gemini-3.8-live-extended-thinking` as a migration target "still pending."
  - `openai_live`: `gpt-live-1`, backend `gpt-5.6-luna`, voice `marin`.
  - Opt-in flags: `GEMINI_LIVE_VIDEO_ENABLED`, `HOUSEHOLD_MEMORY_ENABLED`, and a hybrid local wake-word/STT runtime.
  - Source: [GitHub: eduardoyi fork](https://github.com/eduardoyi/reachy_mini_conversation_app)

**Gemini Robotics-ER 2 on Reachy Mini (community)**
- `max-lm/reachy_mini_er2` (HF Space, created 15 Sep 2026, 0 likes) is described as "Gemini Robotics ER 2 as the brain of your Reachy Mini."
  - **Voice mode:** a Live session on `gemini-robotics-er-2-streaming-preview` takes mic audio and about 1 fps video, and replies through Gemini TTS.
  - **Text mode:** uses `gemini-robotics-er-2-preview` through the Interactions API, with a fresh frame on each turn.
  - **Tools:** `look_at(target)` (ER 2 points to the object and the head aims at that pixel), `find_objects`, `track_object` (re-locates the target about once per second), `follow_face`, `move_head`, `play_emotion`, `watch_for(description)`, `describe_view`, `stop_motion`, and `stop_watching`.
  - It works on Reachy Mini Lite over USB with the daemon running. The author patched the SDK to use a laptop mic.
  - Sources: [HF Space README](https://huggingface.co/spaces/max-lm/reachy_mini_er2/raw/main/README.md); [HF Spaces API listing](https://huggingface.co/api/spaces?filter=reachy_mini&search=er&limit=100)
- Gemini Robotics ER 2 public preview launched 30 Jul 2026, with model code `gemini-robotics-er-2-preview`, a 131,072-token input limit and a 65,536-token output limit. A third-party guide notes that its Live API output is text only, so it cannot speak directly through that path. — [Google AI for Developers model page](https://ai.google.dev/gemini-api/docs/models/gemini-robotics-er-2-preview?hl=fr); [agentpedia guide (third-party)](https://agentpedia.codes/blog/gemini-robotics-er-2-developer-guide.md)
  - **Conflict:** the `max-lm` app uses an ER 2 "streaming" variant and adds Gemini TTS for speech, which fits text-only Live output. **[UNVERIFIED: `gemini-robotics-er-2-streaming-preview` is not confirmed on an official Google page I fetched]**

**Other Gemini, Gemma, and OpenAI apps (community)**
- HF Spaces tagged `reachy_mini` that match "gemini" (7 as of Oct 2026):
  - gamellama/reachy-mini-gemini (8 likes, Jan 2026)
  - n0012/reachy_mini_conversation_app_gemini (6 likes, May 2026)
  - ThomasEvotion/reachy_mini_qr_code_gemini
  - p-2HighOnRunningSlow/reachy-mini-powered-by-gemini (Jul 2026)
  - LightVII/reachy-mini-gemini-eigen (Jul 2026)
  - Nevescr/reachy-mini-gemini-2026 (Sep 2026)
  - karstenskyt/gemini-config-helper
  - Source: [HF Spaces API](https://huggingface.co/api/spaces?filter=reachy_mini&search=gemini&limit=50)
- `gamepop/reachy-mini-gemini` streams voice and camera to the Gemini Live API, with head and antenna motion and function calling. It needs the Reachy Mini daemon running. — [GitHub: gamepop/reachy-mini-gemini](https://github.com/gamepop/reachy-mini-gemini)
- Pollen's community page ("Reachy Mini in the Wild") lists:
  - A Gemini 3.8 Live low-latency multilingual app (Tim Messerschmidt, X, 16 Sep 2026).
  - A Gemini Live conversation app vibe-coded in Antigravity, "full emotion controls not yet implemented" (Glen Yu, LinkedIn, 17 Jul 2026).
  - Reachy Mini on GPT Realtime 2 (Reddit, 31 May 2026).
  - Reachy Mini plus GPT-live-1 through OpenClaw as a Microsoft Foundry hosted agent (Jesús Rubio Sainz, LinkedIn, 21 Sep 2026).
  - Gemma 4 E2B fully offline on a Raspberry Pi 5 via LiteRT, with voice, YOLO vision, and motion (Google Gemma on X, 12 Aug 2026).
  - Gemma 4 in the browser on WebGPU over WebSerial (Reddit u/xenovatech, 11 May 2026).
  - NVIDIA PersonaPlex running on Reachy Mini (Andi Marafioti, X, 28 Jan 2026).
  - An HF pipeline (Silero VAD, then Parakeet STT, then any LLM, then Qwen3-TTS) "reportedly running on thousands of Reachy Mini robots" (Bluesky, 30 Jul 2026).
  - Source: [Pollen: Reachy Mini in the Wild](https://pollen-robotics.com/reachy-mini/community/)
- Seeed Studio has two related wiki pages:
  - A fully local voice LLM for Reachy Mini on a reComputer Jetson, using Ollama with qwen2.5:7b. — [Seeed wiki](https://wiki.seeedstudio.com/local_voice_llm_on_recomputer_jetson_for_reachy_mini/)
  - A case that voice-controls an SO-ARM gripper from the conversation app (covered in Q2). — [Seeed wiki](https://wiki.seeedstudio.com/reachymini_development_cases_gripper_voice_control/)

**SDK plumbing relevant to a cloud brain**
- Recent `reachy_mini` releases:
  - v1.9.0 (9 Jul 2026): `get_frame_jpeg()`, on-board face tracking, and auto-start of a default app.
  - v1.10.0 (13 Aug 2026): a JSON-RPC 2.0 relay over WebRTC, `play_recorded_move` over the data channel (the sound plays on the robot speaker), and an aiortc consumer that "gives cloud backends the latest camera frame."
  - v1.11.0 (11 Sep 2026): on-demand hardware H.264.
  - v1.12.0 (2 Oct 2026): daemon hardening against DNS rebinding and cross-site writes; HF credentials moved to a 0600 store, so robots must be re-linked.
  - None of these release notes mention LeRobot, MCP, or Gemini.
  - Source: [GitHub releases: pollen-robotics/reachy_mini](https://github.com/pollen-robotics/reachy_mini/releases)

### Inferences
- The house pattern (Pollen, Google's walkthrough, and community apps) is "realtime voice model + function-calling tools that wrap SDK moves + a separate motion loop." The developer's Gemini 3.8 Live app with ER 2 spatial tools already matches it. The Pollen tool interface (a `Tool` subclass or a Gradio MCP Space) is the natural place to expose the emotion SDK and an "arm skill" tool, if they want to stay compatible with the official app.
- No official source puts the Gemini Robotics *VLA* (cloud or On-Device) on Reachy Mini. Reachy Mini has no manipulator, so a VLA would only matter once an arm is attached.
- The `max-lm` ER 2 app shows ER 2 can serve as both the voice brain and the pointing and look-at layer. The developer runs Gemini 3.8 Live for voice and ER 2 as tools. That split looks better for natural speech, since ER 2 Live output is text only per the third-party guide.

### Gaps
- I could not read the official Pollen app store page (`pollen-robotics-reachy-mini.hf.space/apps`); it rendered empty to the fetcher. Whether Pollen curates any Gemini app as "official" is unknown.
- I could not confirm whether the upstream Pollen conversation app still ships Gemini and OpenAI backends; see the conflict above.
- I found no first-party latency benchmarks for Gemini Live on Reachy Mini.

---

## Q2. Is Reachy Mini a supported robot or teleoperator type in LeRobot (record, train, evaluate)? Can one LeRobot policy drive the Reachy Mini head and an SO-101 arm together? Any community projects pairing them?

### Takeaway
As of mid-October 2026, Reachy Mini is **not** a merged LeRobot robot type. The only integration is community PR #2726 (opened Dec 2025, rebased and tested 25 Sep 2026, review still pending), and LeRobot main's `robots/` folder has no `reachy_mini`. LeRobot has no generic "compose two different robots" feature. The supported patterns are dedicated combined robot classes (`bi_so_follower`) or per-part flags on one robot (`reachy2 --robot.with_neck`). A head plus SO-101 policy therefore means writing a custom combined `Robot` class. Community pairings so far use Reachy Mini as a static front camera or as a voice front-end that triggers an SO-101 policy or tool, not as a jointly controlled head.

### Cited Findings
**Reachy Mini in LeRobot**
- LeRobot main `src/lerobot/robots/` contains: bi_openarm_follower, bi_rebot_b601_follower, bi_so_follower, earthrover_mini_plus, hope_jr, koch_follower, lekiwi, omx_follower, openarm_follower, reachy2, rebot_b601_follower, so_follower, unitree_g1. There is **no `reachy_mini`**. — [GitHub: lerobot/src/lerobot/robots](https://github.com/huggingface/lerobot/tree/main/src/lerobot/robots) (fetched 15 Oct 2026)
- PR #2726, "feat(robots): Integrate Reachy Mini" (by ravediamond), is **open and unmerged**.
  - Opened 27 Dec 2025; review requested from CarolinePascal 4 Jan 2026 and still pending.
  - The 25 Sep 2026 rebase adds a `ReachyMini` robot driver and config, an IK processor step wrapping the SDK's `AnalyticalKinematics`, `KeyboardReachyMiniTeleop` for head and antenna keyboard control, an optional `reachy_mini` extra, docs, `examples/reachy_mini/teleoperate.py`, and tests.
  - It targets Reachy Mini **Lite over USB**. Wi-Fi "may require further adjustments."
  - Teleop uses joint-space coordinates.
  - A maintainer wrote on 24 Feb 2026: "nothing wrong with the PR, I just haven't had the time to dig into it yet". The author noted Reachy Mini "is not a classic robot arm and more about HRI."
  - Multi-robot use and SO-101 pairing are not discussed.
  - Source: [GitHub PR #2726](https://github.com/huggingface/lerobot/pull/2726)
- The LeRobot v0.6.0 release post (7 Jul 2026) mentions no Reachy Mini, Reachy 2, multi-robot, or robot-composition feature. Highlights:
  - World-model policies: VLA-JEPA, FastWAM, LingBot-VA.
  - New VLAs: EO-1, MolmoAct2, Multitask DiT, EVO1, and GR00T N1.7 (replacing N1.5).
  - The new `lerobot-rollout` CLI, with base, sentry, highlight, episodic, and dagger strategies plus Real-Time Chunking.
  - Depth support, VLM dataset annotation, and HF Jobs training.
  - Source: [HF blog: LeRobot v0.6.0](https://huggingface.co/blog/lerobot-release-v060)
- v0.6.1 was tagged 3 Aug 2026 with "one more rename," per a third-party migration guide. **[UNVERIFIED against the official changelog]** — [iotdigitaltwinplm guide](https://iotdigitaltwinplm.com/lerobot-v0-6-vs-v0-5-migration-guide-2026/)

**Patterns for "head + arm(s)" in LeRobot**
- **Reachy 2 (Pollen's humanoid) as the model for head-in-the-action-space:** `--robot.type=reachy2` exposes per-part flags `with_neck`, `with_antennas`, `with_l_arm`, `with_r_arm`, and `with_mobile_base`, plus `with_left/right_teleop_camera` (head cameras) and `with_torso_camera`.
  - The teleoperator must enable the same parts.
  - The docs warn that "recording all joints when only some are moving can reduce model quality with certain policies."
  - The docs' training example uses ACT with `--policy.device=mps`.
  - Source: [LeRobot docs: Reachy 2](https://huggingface.co/docs/lerobot/reachy2)
- **Bimanual SO-101:** a dedicated `bi_so_follower` robot type with separate left and right ports. One policy (for example ACT) is trained on the combined dataset with one task string, and each arm is calibrated separately with its own ID. — [GitHub: lerobot robots folder](https://github.com/huggingface/lerobot/tree/main/src/lerobot/robots); [Seeed double-arm guide](https://wiki.seeedstudio.com/lerobot_double_arm_so_arm_training/); [AIDASLab lerobot-so101-bimanual](https://github.com/AIDASLab/lerobot-so101-bimanual)
- **XLeRobot (community, not in LeRobot main):** community datasets with `robot_type: xlerobot` record `head_pan.pos` and `head_tilt.pos` in the same state and action vectors as two arms and a base. Examples: `Odog16/test_transfer_block` (154 episodes, 30 fps) and `block_sorting_single` (256 episodes, 4 tasks). — [HF dataset meta: test_transfer_block](https://huggingface.co/datasets/Odog16/test_transfer_block/blob/main/meta/info.json); [HF dataset meta: block_sorting_single](https://huggingface.co/datasets/Odog16/block_sorting_single/blob/main/meta/info.json)
- **No generic multi-robot composition:** the LeRobot paper says the dataset format supports different robot configurations, but neither it nor the docs describe merging two heterogeneous robots into one robot at record time. — [LeRobot paper (arXiv 2602.22818)](https://arxiv.org/html/2602.22818v1)

**Community projects pairing Reachy Mini with SO-100/101**
- Yannis Tevissen (LinkedIn, 6 Mar 2026) fine-tuned **SmolVLA** to pick up a pen.
  - Setup: an SO-101 with a wrist camera, plus Reachy Mini as the **front camera**.
  - Data: about 50 teleop episodes recorded with his own phone app.
  - Training: about 9 h on a Mac M3.
  - Main difficulties: changing light and a reflective table.
  - No success rate given.
  - Sources: [Pollen community page](https://pollen-robotics.com/reachy-mini/community/); [LinkedIn post](https://www.linkedin.com/posts/yannis-tevissen_over-the-past-few-months-my-holidays-have-activity-7435685741151739904-EqLa)
- Innobox, "Reachy mini control So101 lerobot arm" (YouTube, 15 Mar 2026, 2.3k views). Only the title was visible, with no policy or result details. — [YouTube](https://www.youtube.com/watch?v=jdETTLSyA4c) via [Pollen community page](https://pollen-robotics.com/reachy-mini/community/)
- Seeed Studio, "Voice-Controlled Robot Arm: Reachy Mini Commands SO-ARM101 | Imitation Learning Demo" (YouTube, 7 Aug 2026). — [YouTube](https://www.youtube.com/watch?v=HgReXV3z20A) via [Pollen community page](https://pollen-robotics.com/reachy-mini/community/)
- Seeed wiki case, "gripper voice control":
  - Chain: voice command → `gripper_control` **external tool** in the official conversation app (no source changes) → subprocess `soarm_gripper.py` → LeRobot `SOFollower`.
  - Ports: Reachy Mini on `/dev/ttyACM0`, the follower on `/dev/ttyACM1`.
  - Only gripper actions are sent so far. Other joints (`shoulder_pan.pos` … `gripper.pos`, normalized 0–100) can be added the same way.
  - This is LLM function calling, not a learned policy.
  - Source: [Seeed wiki](https://wiki.seeedstudio.com/reachymini_development_cases_gripper_voice_control/)
- **[UNVERIFIED]** A demo where Reachy Mini, asked by voice to pick up a LEGO brick, "scans the table with its head camera," checks the brick's position, and triggers an SO-101 running a home-trained ACT policy, with a success rate "still well below 10%." This came from a search-engine summary only; I could not find the original post. Possibly linked to the Seeed video above.

### Inferences
- To have one LeRobot policy drive the Reachy head and an SO-101 together today, the developer would write a custom `Robot` (and `Teleoperator`) that wraps both, along the lines of `bi_so_follower`. Its `observation_features` and `action_features` would merge the SO-101 joints with Reachy head pose (or joint) keys and cameras. PR #2726's driver and IK step are the best starting code for the Reachy half. The Reachy 2 `with_neck` flag and the XLeRobot `head_pan/head_tilt` datasets show the expected feature layout.
- The simpler architecture, used by every community pairing found, keeps the arm policy and the head separate: Reachy handles voice, gaze, and LLM tools, and the SO-101 runs its own LeRobot policy, invoked as a tool or a subprocess.
- PR #2726 is Lite and USB only and works in joint space, which fits the developer's USB-tethered Lite. Note that the developer's app talks to the daemon; whether the PR goes through the daemon or the SDK directly is not stated.

### Gaps
- I could not access the PR #2726 diff to see the exact feature names (head pose vs. Stewart joints, body yaw, camera key) or whether it requires a running daemon.
- I found no report of any policy trained with **Reachy Mini head motion as an action** alongside an SO-101.
- I could not open the Seeed blog post "Cool SO-ARM101 projects" (HTTP 403). — [Seeed blog URL](https://www.seeedstudio.com/blog/2026/08/31/cool-so-arm101-projects-to-try-now-from-grasping-to-imitation-learning/)

---

## Q3. What has the community done with VLAs on SO-101 (SmolVLA, pi0/pi0.5, GR00T N1.x), with what dataset sizes and success rates? Can they run inference on Apple Silicon (MPS)?

### Takeaway
The cleanest controlled comparison is a June 2026 arXiv benchmark. On a real SO-101 with 100 demos per task, π0.5 averaged 56% success, Wall-X 51%, ACT 34%, and SmolVLA 33%. All models were strongly task-dependent, ranging from 95% on pen transfer to 0–10% on color sorting. SmolVLA is the practical Mac option: it is 450M parameters, HF recommends about 50 episodes, community users have trained it on M-series Macs, and PyTorch MPS inference has been measured at about 0.2 s per 50-action chunk on an M5 Pro. π0.5 and GR00T are GPU/CUDA workflows (GR00T fine-tuning needs more than 48 GB VRAM). Gemini Robotics On-Device 2 has been evaluated on SO-101 but is available to trusted testers only.

### Cited Findings
**Controlled SO-101 benchmark**
- "Benchmarking VLA Models on SO-101: Failure and Recovery Analysis" (Yi Yu and Xinchuan Qiu, arXiv 2606.08881, v1 7 Jun 2026, v2 11 Jun 2026).
  - Four tasks, **100 teleoperated demos per task** (400 total), 20 trials per model-task pair.
  - Success rates (%), in task order Pen Transfer / Color Sorting / Object Packing / Pen Placement:
    - π0.5: 95 / 10 / 55 / 65 (avg **56.25**)
    - Wall-X: 95 / 0 / 30 / 80 (avg 51.25)
    - ACT: 75 / 0 / 10 / 50 (avg 33.75)
    - SmolVLA: 70 / 5 / 10 / 45 (avg **32.5**)
  - Recovery rate after an error: π0.5 30.8%, Wall-X 20.5%, ACT 6.5%, SmolVLA 3.2%.
  - "Execution instability" is the dominant failure source.
  - The paper does not report the camera setup or inference hardware.
  - Source: [arXiv HTML](https://arxiv.org/html/2606.08881); [abstract](https://arxiv.org/abs/2606.08881)

**SmolVLA (HF, 450M)**
- The official docs recommend about **50 episodes** as a starting point, with repeated variations. The reference dataset `svla_so100_pickplace` has 50 episodes across 5 cube positions, 10 each. "25 episodes … was not enough leading to a bad performance."
  - Training: 20k steps takes about 4 h on one A100.
  - Deployment now goes through `lerobot-rollout --robot.type=so101_follower`, with optional RTC "when running on low power hardware."
  - Source: [LeRobot docs: SmolVLA](https://huggingface.co/docs/lerobot/smolvla)
- Community real-arm result: the arm touched the block about 95% of the time but grasped it about 40% of the time, over about 30 trials (author says not rigorous). — [Medium: Fine-tuning SmolVLA to pick and place](https://medium.com/@henryhu1607/genai-for-robotics-fine-tuning-smolvla-to-pick-and-place-940b485e6c9b)
- Sim fine-tune: 80% success on 50 sim episodes on a plain desk, but **0/20** on a visually different desk texture, showing a strong visual-domain sensitivity. — [HF model card: Ethgar/smolvla-so101-pen-sim-50](https://huggingface.co/Ethgar/smolvla-so101-pen-sim-50_2026-10-05_09-15-19)
- An "87.66%" figure on one SO-101 SmolVLA card is an **offline per-joint-within-5% metric**, not task success. — [HF model card: Sa74ll/smolvla_so101_pickandplace](https://huggingface.co/Sa74ll/smolvla_so101_pickandplace)

**π0 / π0.5**
- SO-101 Bench: a full π0.5 fine-tune on about 3,600 *simulated* SO-101 episodes (3,629 episodes, about 2M frames) reached 16/39 (about 41%) validation success at the official checkpoint and 18/39 at the best checkpoint. — [HF model card: 5hadytru/so101_bench_pi05](https://huggingface.co/5hadytru/so101_bench_pi05)
- Real-arm π0.5 LoRA fine-tunes (openpi JAX) used 25-episode and 49-episode datasets; success rates were not reported. — [GitHub: ljt228/pi05-so101-finetune](https://github.com/ljt228/pi05-so101-finetune)
- A LeRobot `pi05_base` fine-tune on 57 episodes (5k steps, batch 2) reported no metrics. — [HF model card: CQCGO/pi05_so101_bottle_finetuned](https://huggingface.co/CQCGO/pi05_so101_bottle_finetuned)
- A bimanual SO-101 π0.5 used 90 episodes and 10k steps on an H100. — [HF model card: izlley/h101-pi05-expert-v1](https://huggingface.co/izlley/h101-pi05-expert-v1)
- An RL-token pipeline for π0.5 on SO-101 (fine-tune, then online RL). — [GitHub: Zhengsw03/RL-Token-Pi05-open](https://github.com/Zhengsw03/RL-Token-Pi05-open)

**GR00T**
- LeRobot now supports **GR00T N1.7 only**. N1.5 was removed in v0.6.0; pin `lerobot==0.5.1` if you need N1.5. SO-101 uses the `new_embodiment` tag. — [LeRobot docs: GR00T](https://huggingface.co/docs/lerobot/en/groot); [HF blog v0.6.0](https://huggingface.co/blog/lerobot-release-v060)
- NVIDIA's "Train an SO-101 Robot From Sim-to-Real" course covers GR00T **N1.6** end to end, using NVIDIA's Isaac-GR00T repo rather than LeRobot. — [NVIDIA course: GR00T](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/10-groot.html); [Isaac Teleop docs](https://nvidia.github.io/IsaacTeleop/main/getting_started/lerobot/training_groot.html)
- Seeed's GR00T N1.6 SO-101 guide says fine-tuning needs a device with **more than 48 GB VRAM** (rent a server) and deploys on a Jetson AGX Orin. — [Seeed wiki](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.6_for_lerobot_so_arm_and_deploy_on_agx_orin/)

**Gemini Robotics VLA**
- Gemini Robotics On-Device 2 model card (30 Jul 2026):
  - "Currently available to a select group of trusted testers."
  - Focused on bi-arm robots, evaluated on **SO-101** and Dexmate.
  - SO-101 success rose with hours of data per task: GRODv2 went from 6.7% to 53.3%, GRODv1 from 0% to 6.7%.
  - The task and demo counts are not named, and hardware requirements are not given.
  - Source: [DeepMind model card](https://deepmind.google/models/model-cards/gemini-robotics-on-device-2/)
- The Gemini Robotics SDK (Safari SDK) is on GitHub. Most functionality requires the Trusted Tester program, and On-Device model access starts at SDK v2.4.1. — [GitHub: google-deepmind/gemini-robotics-sdk](https://github.com/google-deepmind/gemini-robotics-sdk)

**Apple Silicon / MPS**
- SmolVLA inference on an Apple M5 Pro (`mlx-smolvla`, a third-party port): a 50-action chunk took about **204.6 ms with PyTorch-MPS** and about 110.8 ms with MLX fp32. The author calls this a bounded integration test, not proof of task success. — [PyPI: mlx-smolvla](https://pypi.org/project/mlx-smolvla/0.1.1/); [HF forum thread](https://discuss.huggingface.co/t/mlx-smolvla-smolvla-checkpoint-execution-and-validation-on-apple-silicon/179927)
  - **Conflict:** another Mac-first benchmark harness says to expect "seconds per chunk" on MPS. — [GitHub: imtisaalm/vla-inference-bench](https://github.com/imtisaalm/vla-inference-bench)
- SmolVLA *training* on a Mac M3 took about 9 h for about 50 episodes (Tevissen, Mar 2026). — [Pollen community page](https://pollen-robotics.com/reachy-mini/community/)
- LeRobot's own Reachy 2 docs show `--policy.device=mps` for ACT training. — [LeRobot docs: Reachy 2](https://huggingface.co/docs/lerobot/reachy2)
- LeRobot async inference splits a policy server from the robot client and exposes `actions_per_chunk` and `chunk_size_threshold`. This lets a Mac run the robot while a GPU box runs the policy. — [LeRobot docs: Async inference](https://huggingface.co/docs/lerobot/async)
- An arXiv study (Sep 2026) found that faster deployment variants of SmolVLA (ONNX) can *change* closed-loop task success, not just latency. — [arXiv 2609.14146](https://arxiv.org/pdf/2609.14146)

### Inferences
- On the developer's Mac-tethered setup, SmolVLA (and ACT) is the realistic local VLA. Expect roughly 30–70% success on simple SO-101 pick-and-place with 50–100 well-varied episodes. π0.5 gives about 20 points higher average success but realistically needs a CUDA GPU (local or rented) behind LeRobot async inference. GR00T and Gemini Robotics On-Device are not practical on Mac today.
- Sensitivity to visual domain and lighting (the 0/20 desk-texture result; Tevissen's lighting and reflective-table issues) matters for a camera on a Reachy head. Moving that camera changes the view, so head motion makes this worse (see Q4).

### Gaps
- I found no source measuring π0/π0.5 or GR00T **inference on MPS**; whether LeRobot's π0.5 runs on MPS at usable rates is unknown.
- I found no community SO-101 GR00T N1.6/N1.7 success-rate numbers.
- The camera configuration in the arXiv benchmark is not stated.

---

## Q4. How do people handle the head (gaze or active perception) in VLA training, e.g. a head camera that moves vs a fixed or wrist camera?

### Takeaway
Practitioners mostly **freeze the head** during data collection and inference so the head camera behaves like a fixed camera, and pair it with a wrist camera. Platforms that do record head motion (Reachy 2 `with_neck`, XLeRobot `head_pan/head_tilt`) put head joints into the same state and action vector as the arms. Research in 2026 has moved toward learned active perception (camera pose as an action) and toward gaze as a training or prompting signal, mostly on π0/π0.5 backbones. None of this work is packaged in LeRobot for small robots yet.

### Cited Findings
- XLeRobot's VLA tutorial: "for both VLA dataset collection and inference you need to keep consistent head servos angles," otherwise the policy may fail. A utility returns the head to a fixed VLA pose "for perfect repeatability of the camera angle." The workspace must be fully visible from the head camera. XLeRobot has two wrist cameras plus one head camera. — [XLeRobot docs: VLA ACT](https://xlerobot.readthedocs.io/en/latest/software/getting_started/VLA_ACT.html)
- XLeRobot community datasets record `head_pan.pos` and `head_tilt.pos` as both state and action, so a policy *can* learn head motion. — [HF: Odog16/test_transfer_block meta](https://huggingface.co/datasets/Odog16/test_transfer_block/blob/main/meta/info.json)
- LeRobot Reachy 2: neck and antennas are optional action parts. Recording joints that don't move "can reduce model quality with certain policies," so turn them off unless the task uses them. The head teleop cameras are optional observations. — [LeRobot docs: Reachy 2](https://huggingface.co/docs/lerobot/reachy2)
- Community Reachy Mini + SO-101 practice (Tevissen, Mar 2026): Reachy Mini served as a static **front camera** with an SO-101 wrist camera. — [Pollen community page](https://pollen-robotics.com/reachy-mini/community/)
- Active-perception research, 2026:
  - **SaPaVe** (CVPR 2026) puts camera movement in the policy's action space, pre-learns camera motion, then jointly optimizes it with manipulation. — [CVF open access](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_SaPaVe_Towards_Active_Perception_and_Manipulation_in_Vision-Language_Action_Models_CVPR_2026_paper.pdf)
  - **ActiveScale** (arXiv 2609.18514) extends **π0.5** with camera-pose tokens and an auxiliary head that predicts camera translation, rotation, and FOV. It mid-trains on more than 1,000 h of egocentric and robot data, with human head-camera poses as supervision. — [arXiv 2609.18514](https://arxiv.org/html/2609.18514)
  - **The Moving Eye** (arXiv 2607.02322) instead trains on broadly sampled moving-view data so the policy becomes viewpoint-invariant, an approach it calls complementary to learned gaze. — [arXiv 2607.02322](https://arxiv.org/pdf/2607.02322)
  - **Gaze Prompts** (arXiv 2609.34550) records human gaze during teleop and renders it as a crosshair on the head-camera image. A predictor supplies gaze at inference to a fine-tuned π0. — [arXiv 2609.34550](https://arxiv.org/html/2609.34550)
  - **Gaze-Regularized VLA** (arXiv 2603.23202) uses human gaze only at training time to shape attention, with no eye tracker at deployment. — [arXiv 2603.23202](https://arxiv.org/html/2603.23202v1)
  - **ActGaze** (arXiv 2609.28955) derives gaze from the policy's own action sensitivity. — [arXiv 2609.28955](https://arxiv.org/html/2609.28955)

### Inferences
- For a first Reachy + SO-101 VLA, the low-risk recipe that matches community practice:
  1. Park the Reachy head at a fixed "work pose" whenever the arm policy runs.
  2. Use the Reachy camera as the front/scene camera plus the SO-101 wrist camera.
  3. Let the LLM brain handle gaze and expressiveness *between* manipulation episodes, through Gemini 3.8 Live and ER 2 `look_at`-style tools.
- Adding head pose to the action space is possible with a custom combined robot (Q2). It needs teleoperating the head during demos and much more data, because the view changes. The research above suggests this pays off mainly with large pretrained backbones (π0.5-class), not SmolVLA-scale models. This inference is not directly tested on Reachy Mini.
- A middle ground follows "The Moving Eye": record demos while deliberately varying the (fixed-per-episode) head pose, so the policy tolerates the head being anywhere in a range.

### Gaps
- I found no published experiment on any Stewart-platform head (Reachy Mini) used as an active camera for manipulation.
- I did not retrieve the Vision-in-Action and ActiveUMI papers; they are cited only secondhand in The Moving Eye.

---

## Q5. Are there Reachy Mini emotion or expressive-behavior projects (emotion detection driving moves, the HF "emotions" library) relevant to integrating an emotion-detection tool?

### Takeaway
Pollen's expressive layer is a **dataset of recorded moves**, not a model: `pollen-robotics/reachy-mini-emotions-library` holds 81 head, antenna, and body-yaw trajectories at 50 Hz with Opus audio sidecars, loaded through `RecordedMoves`. It needs SDK v1.8.4 or later. Pollen's "Emotions" Space is a pure playback UI ("No AI in the loop"). Official apps let the LLM choose an emotion through a `play_emotion` tool. Detection-driven projects are community-built and map a detected label to a library move, which is the integration pattern the developer's emotion SDK would follow.

### Cited Findings
- `pollen-robotics/reachy-mini-emotions-library`:
  - 81 moves, each a `<emotion>.json` trajectory (head pose, antennas, body yaw at 50 Hz) plus a `<emotion>.ogg` mono Opus file. Size 8.27 MB, Apache-2.0.
  - **Requires `reachy_mini` ≥ v1.8.4** (non-WAV audio sidecars).
  - Usage: `RecordedMoves("pollen-robotics/reachy-mini-emotions-library")` then `reachy.play_move(library.get("amazed1"))`.
  - Names include: amazed1, anxiety1, attentive1/2, boredom1/2, cheerful1, confused1, curious1, disgusted1, enthusiastic1/2, fear1, frustrated1, furious1, grateful1, laughing1/2, lonely1, loving1, no1, proud1–3, rage1, relief1/2, sad1/2, scared1, serenity1, shy1, surprised1/2, thoughtful1/2, tired1, uncertain1, welcoming1/2, yes1, and others.
  - Derivatives should keep the `reachy_mini_community_moves` tag.
  - Source: [HF dataset card](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library)
- SDK release history:
  - v1.8.4 (25 Jun 2026): moves can carry non-WAV sound.
  - v1.10.0 (13 Aug 2026): `play_recorded_move` over WebRTC, with sound on the robot speaker.
  - v1.12.0 (2 Oct 2026): public emotions and dances repos no longer need auth.
  - Source: [GitHub releases](https://github.com/pollen-robotics/reachy_mini/releases)
- Pollen's **"Emotions" Space** (`pollen-robotics/emotions`, created 15 May 2026, 32 likes) is a mobile-first JS app.
  - It shows a 79-emotion Plutchik-style wheel (8 sectors × 3 intensity rings).
  - It streams `set_full_target` commands per animation frame over the SDK WebRTC data channel and plays the `.ogg` in the browser.
  - "No AI in the loop - pure dataset playback."
  - Source: [HF Space README](https://huggingface.co/spaces/pollen-robotics/emotions/raw/main/README.md)
  - **Count mismatch:** 79 in the Space vs. 81 in the dataset; possibly dances or variants are excluded. **[UNVERIFIED]**
- The emotions were created by teleoperation: "recorded 80+ emotions (motion + sound) using teleoperation." An LLM (originally GPT-4o-realtime) picks the emotion from conversation, not from face analysis. — [Remi Fabre LinkedIn, 29 Jul 2025](https://www.linkedin.com/posts/remifabre_boom-emotions-on-reachy-mini-maybe-this-activity-7355975480530903042-zLO9); [Coralie Deplanne LinkedIn, 25 Jul 2025](https://www.linkedin.com/posts/coralie-deplanne_were-currently-developing-a-full-set-of-activity-7354423922521513985-nofC)
- In the official conversation app, `play_emotion`/`stop_emotion` play clips from this dataset, and dances come from `pollen-robotics/reachy-mini-dances-library`. — [GitHub README](https://github.com/pollen-robotics/reachy_mini_conversation_app)
- Pollen's expressiveness work, 2026 (Remi Fabre):
  - "Prosody Wobbler" ties head motion to the voice's sound, mixed with library emotions (TikTok, 23 Sep 2026).
  - "How far can we push Reachy Mini's expressiveness?" (YouTube, 24 Sep 2026).
  - "Marionette," a tool for authoring moves by hand (YouTube, 12 May 2026). Its moves are on the Hub as `apirrone/marionette-moves`.
  - Sources: [Pollen community page](https://pollen-robotics.com/reachy-mini/community/); [YouTube: expressiveness](https://www.youtube.com/watch?v=p-SECvclPHc); [YouTube: Marionette](https://www.youtube.com/watch?v=glMBdZlfjYI); [HF dataset: marionette-moves](https://huggingface.co/datasets/apirrone/marionette-moves)
- Community **detection-driven** apps:
  - **NathanSegers/reachy-mini-emotion-mirror** (May 2026):
    - Default detector: `trpakov/vit-face-expression` (HF); Ollama (e.g. `gemma4:31b`) and OpenAI are alternative providers.
    - Labels: joy, sadness, anger, surprise, fear, disgust, and neutral.
    - Debounce: confidence ≥ 0.45, 2 consecutive agreeing checks, checks every 2 s, 3 s cooldown.
    - Each label maps to a matching move in the Pollen emotions library, with a deterministic fallback pose.
    - It is built as a `ReachyMiniApp` with an optional Gradio debug UI.
    - Source: [HF Space README](https://huggingface.co/spaces/NathanSegers/reachy-mini-emotion-mirror/raw/main/README.md)
  - **pearlyjam21/reachy_mini_multimodal_emotion** (27 Sep 2026):
    - Fuses face, voice (speech emotion), and Traditional-Chinese text emotion with adjustable weights.
    - Antennas perk up when the user starts speaking, and an "empathetic move" plays after each sentence.
    - Replies come from an LLM via OpenRouter, with a 3 s deadline and a prepared emotion-matched fallback line.
    - It warns the models are "near chance on natural conversation" and that the output is "a soft social cue, not a measurement."
    - Source: [HF Space README](https://huggingface.co/spaces/pearlyjam21/reachy_mini_multimodal_emotion/raw/main/README.md)
  - Other emotion-tagged Reachy Mini Spaces: `RemiFabre/emotions` (Dec 2025), `RemiFabre/chess_emotions_app` (Jan 2026), and a fork, `NatashaN/chess_emotions_app`. — [HF Spaces API](https://huggingface.co/api/spaces?filter=reachy_mini&search=emotion&limit=50)
- Text-to-motion datasets derived from the library:
  - `binhpham/reachy-mini-motion-synth`: about 2,100 generated clips plus the real Pollen clips, paired with text prompts.
  - `JosephAusterweil/reachy-meme-expressions`.
  - Sources: [HF: reachy-mini-motion-synth](https://huggingface.co/datasets/binhpham/reachy-mini-motion-synth); [HF: reachy-meme-expressions](https://huggingface.co/datasets/JosephAusterweil/reachy-meme-expressions)

### Inferences
- The integration pattern that fits Pollen's design is to expose the developer's emotion SDK as a **tool** that returns a label and confidence. The LLM, or a deterministic mapper as in emotion-mirror, then calls `play_emotion(<library name>)`. A debounce with a confidence threshold, consecutive agreement, and a cooldown (as emotion-mirror does) prevents jittery reactions.
- Because the developer is on SDK v1.6 and the library needs v1.8.4 or later for its Opus sidecars, upgrading the SDK and daemon (latest v1.12.0) is likely required before using the official emotions library. Otherwise the audio sidecars may fail to load. Note the v1.12.0 re-linking requirement for HF credentials.
- Emotion moves and arm-policy rollouts both command the head. An arbiter is needed, such as "no expressive head moves while the arm policy runs with a fixed head camera," consistent with Q4.

### Gaps
- I found no official Pollen or HF emotion-*detection* model or app; all detection apps found are community-built.
- I could not confirm the exact Reachy SDK behavior when loading `.ogg` sidecars on versions below 1.8.4 (failure mode not documented).
