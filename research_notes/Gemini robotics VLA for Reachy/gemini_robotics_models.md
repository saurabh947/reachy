# Gemini Robotics model family: inventory, outputs, access and fit for a Reachy Mini + SO-101 robot (as of 2026-10-15)

Research date: 2026-10-15. Web research only. Every finding is dated. **SUPERSEDED** marks items replaced by newer versions. **UNVERIFIED** marks claims that come only from secondary or aggregator sources, or that conflict between sources.

---

## 1. Which Gemini Robotics models exist as of Oct 2026 (IDs, release dates, status, API surface)?

### Takeaway
Three models make up the current family, all announced 2026-07-30 as "Gemini Robotics 2": the **Gemini Robotics 2** VLA (early-access partners / private preview only), **Gemini Robotics ER 2** (public preview on the Gemini API and AI Studio as `gemini-robotics-er-2-preview`, plus `gemini-robotics-er-2-streaming-preview` for the Live API), and **Gemini Robotics On-Device 2** (trusted testers only). ER 2 is the only model anyone can call without approval. ER 1.0, 1.5 and 1.6, the Gemini Robotics 1.0 and 1.5 VLAs, and On-Device 1 are all superseded. ER 1.5 and ER 1.6 have passed their published shutdown dates.

### Cited Findings

**Inventory table (status as of 2026-10-15)**

| Model | Model ID (if any) | Announced / released | Status now | API surface | Notes |
|---|---|---|---|---|---|
| Gemini Robotics (1.0) VLA | none public | 2025-03-12 | **SUPERSEDED**; trusted testers / partners only | none public | Based on Gemini 2.0; cloud backbone plus a local action decoder |
| Gemini Robotics-ER (1.0) | none public | 2025-03-12 | **SUPERSEDED**; trusted testers only | none public | Embodied-reasoning VLM |
| Gemini Robotics On-Device (1) | none (SDK) | 2025-06-24 | **SUPERSEDED** by On-Device 2; was trusted tester only | Gemini Robotics SDK (`safari-sdk`), trusted testers | First fine-tunable Gemini VLA |
| Gemini Robotics 1.5 VLA | none public | 2025-09-25 | **SUPERSEDED**; select partners only | none public | "Thinking" VLA with Motion Transfer |
| Gemini Robotics-ER 1.5 | `gemini-robotics-er-1.5-preview` | 2025-09-25 | **SHUT DOWN** (shutdown date 2026-04-30) | Gemini API / AI Studio (preview) | First broadly available model in the family |
| Gemini Robotics-ER 1.6 | `gemini-robotics-er-1.6-preview` | 2026-04-14 | **DEPRECATED / SHUT DOWN** (shutdown date 2026-08-31) | Gemini API / AI Studio (preview) | Added instrument reading (with Boston Dynamics) |
| **Gemini Robotics 2** (VLA) | none public | 2026-07-30 | Early-access partners ("private preview") via the trusted tester waitlist | none public | Whole-body humanoid control, dexterous hands |
| **Gemini Robotics ER 2** | `gemini-robotics-er-2-preview` | 2026-07-30 | **Public preview** | Gemini API, Google AI Studio; private preview on Gemini Enterprise Agent Platform | Built on Gemini 3.5 Flash |
| **Gemini Robotics ER 2 Streaming** | `gemini-robotics-er-2-streaming-preview` | 2026-07 | **Public preview** | Gemini Live API | Low-latency bidirectional streaming variant |
| **Gemini Robotics On-Device 2** | none public | 2026-07-30 | Trusted testers / early-access partners | presumably the SDK (not confirmed for v2) | Runs locally; adapts with fewer than 200 demos |

**Current generation (Gemini Robotics 2, announced 2026-07-30)**
- On 2026-07-30 DeepMind announced Gemini Robotics 2 as a family of three models. **Gemini Robotics 2** is a VLA that "turns vision and language input into motor control" for full humanoids and bi-arm robots. **Gemini Robotics ER 2** is the embodied-reasoning "brain" that talks with humans, plans multi-step tasks lasting several minutes, and coordinates multiple robots. **Gemini Robotics On-Device 2** is a smaller, efficient VLA that runs locally on robot hardware. — [DeepMind blog, 2026-07-30](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)
- Availability wording in the launch post: ER 2 "is now available on Google AI Studio" and is in "private preview on Gemini Enterprise Agent Platform". The VLA and On-Device models are "available to early-access partners", with a Trusted Tester sign-up form linked. — [DeepMind blog, 2026-07-30](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)
- The DeepMind Gemini Robotics hub lists exactly three models (Gemini Robotics 2, ER 2, On-Device 2). It links "Try Gemini Robotics ER 2" to AI Studio, says DeepMind is "working with 100+ trusted testers", and invites visitors to "join the waitlist for early access". It does not list older versions. — [DeepMind Gemini Robotics models page, accessed 2026-10-15](https://deepmind.google/models/gemini-robotics/)
- The Gemini Robotics 2 VLA page gives its status as "Private preview", with early access through a waitlist. — [DeepMind VLA page, accessed 2026-10-15](https://deepmind.google/models/gemini-robotics/vla/)
- The On-Device 2 page gives its status as "Trusted testers" and links "Join waitlist for early access". — [DeepMind On-Device page, accessed 2026-10-15](https://deepmind.google/models/gemini-robotics/on-device/)
- The ER 2 Gemini API model page lists `gemini-robotics-er-2-preview` (Preview) and `gemini-robotics-er-2-streaming-preview` (Preview, streaming variant), both with "Latest update: July 2026". The same page still lists `gemini-robotics-er-1.6-preview` as "Preview" (latest update December 2025, knowledge cutoff January 2025). The page was last updated 2026-10-08. — [Gemini API model page: ER 2 preview](https://ai.google.dev/gemini-api/docs/models/gemini-robotics-er-2-preview)
- The Gemini API robotics overview (last updated 2026-10-06) says "The Gemini Robotics ER 2 model is the latest model in Gemini Robotics". It names ER 2 as built on Gemini 3.5 Flash and says ER 1.6 was "deprecated on August 31, 2026". — [Gemini API Robotics overview](https://ai.google.dev/gemini-api/docs/robotics-overview)
- The ER 2 model card gives the publication date as July 2026, the base model as Gemini 3.5 Flash, and the distribution channels as Google AI Studio, the Gemini API, and Gemini Enterprise Agent Platform (private preview). — [ER 2 model card](https://deepmind.google/models/model-cards/gemini-robotics-er-2/)
- The Google blog post "Introducing Gemini Robotics ER 2" is dated 2026-07-30. It says ER 2 is public on the Gemini API and AI Studio and in private preview on Gemini Enterprise Agent Platform. — [blog.google, 2026-07-30](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/)
- Independent coverage (dated 2026-08-02) confirms the late-July launch and the status of each model: ER 2 on AI Studio, the VLA and On-Device 2 for early-access partners. — [The Robot Report, 2026-08-02](https://www.therobotreport.com/google-deepmind-says-gemini-robotics-2-enables-full-body-control/)
- One DEV Community post claims Gemini Robotics 2 "has not been announced". DeepMind's own blog and the trade press contradict it, so it is treated as wrong. — [DEV Community](https://dev.to/alifar/gemini-robotics-2-has-not-been-announced-what-google-deepmind-actually-offers-l72); contradicted by [DeepMind blog](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)
- **UNVERIFIED:** Eden AI lists ER 2 as "released August 5, 2026". This conflicts with the official 2026-07-30 announcement and may be an API or Vertex listing date. — [Eden AI](https://www.edenai.co/models/gemini-robotics-er-2-preview)

**Deprecations (SUPERSEDED models)**
- The Gemini API deprecations page (last updated 2026-10-12) lists `gemini-robotics-er-1.5-preview` (released 2025-09-25, shutdown 2026-04-30, replacement `gemini-robotics-er-1.6-preview`) and `gemini-robotics-er-1.6-preview` (released 2026-04-14, shutdown 2026-08-31, replacement `gemini-robotics-er-2-preview`). The page calls these dates the "earliest possible" retirement dates. — [Gemini API deprecations](https://ai.google.dev/gemini-api/docs/deprecations)
- ER 1.6 (April 2026) improved on both ER 1.5 and Gemini 3.0 Flash in pointing, counting and success detection, and added "instrument reading", a use case found with partner Boston Dynamics. — [Google Korea blog (ER 1.6)](https://blog.google/intl/ko-kr/company-news/technology/gemini-robotics-er-16-kr/); [Interesting Engineering](https://interestingengineering.com/ai-robotics/google-gemini-robotics-er16-embodied-reasoning)
- **UNVERIFIED / CONFLICTING:** Secondary sources disagree on ER 1.6's instrument-reading accuracy (93% vs 98%) and on its release date (2026-04-14 vs 2026-04-15). Neither accuracy figure was checked against a Google primary source. — [DEV Community guide](https://dev.to/akaranjkar08/gemini-robotics-er-16-physical-ai-developer-guide-2026-o27); [Towards AI](https://pub.towardsai.net/gemini-robotics-er-1-6-boston-dynamics-secret-weapon-just-went-public-9e6cabd49fc4)

**1.5 generation (SUPERSEDED)**
- On 2025-09-25 DeepMind introduced Gemini Robotics 1.5 (VLA) and Gemini Robotics-ER 1.5. ER 1.5 was "available to developers now through the Gemini API in Google AI Studio". The 1.5 VLA was "currently available only to select partners". — [DeepMind blog, 2025-09-25](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/)
- ER 1.5 was the first Gemini Robotics model made broadly available, as a preview. — [Google Developers Blog (ER 1.5)](https://developers.googleblog.com/building-the-next-generation-of-physical-agents-with-gemini-robotics-er-15/)
- The tech report is arXiv:2510.03342, "Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer" (v1 2025-10-02, v3 2025-11-28). — [arXiv 2510.03342](https://arxiv.org/abs/2510.03342)

**1.0 generation and On-Device 1 (SUPERSEDED)**
- Gemini Robotics and Gemini Robotics-ER launched 2025-03-12, based on Gemini 2.0. Access to the early models was limited to trusted testers, including Agile Robots, Agility Robotics, Boston Dynamics and Enchanted Tools. — [Wikipedia: Gemini Robotics](https://en.wikipedia.org/wiki/Gemini_Robotics) (citing DeepMind 2025-03-12 and The Verge 2025-03-12)
- Gemini Robotics On-Device was announced 2025-06-24. It was initially released to "a select group of trusted testers", and developers could get the SDK by "signing up to our trusted tester program". — [DeepMind blog, 2025-06-24](https://deepmind.google/discover/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/)

### Inferences
- For a hobby or indie developer, the **only model usable today without approval is ER 2** (`gemini-robotics-er-2-preview`, or `-streaming-preview` for the Live API). Every Gemini Robotics VLA (the 1.0, 1.5 and 2 VLAs, On-Device 1 and On-Device 2) has stayed gated behind partner or trusted-tester access in every generation since March 2025.
- ER preview models have had short lives. ER 1.5 lasted about 7 months (2025-09-25 to 2026-04-30) and ER 1.6 about 4.5 months (2026-04-14 to 2026-08-31). The developer's `gemini-robotics-er-2-preview` will probably be replaced too, so they should keep the model ID configurable and watch the deprecations page.
- Google does not name the "Gemini Enterprise Agent Platform" channel as Vertex AI in any source read here. The ER 2 model card says Google Cloud Platform Terms govern it, which suggests it is Google Cloud's enterprise channel (possibly a rebrand or successor of Vertex AI). This was not verified.

### Gaps
- No public model ID, API endpoint or Vertex/Model Garden listing exists for the Gemini Robotics 2 VLA or On-Device 2. Google has not published a GA timeline for either.
- No Gemini Robotics 2 (VLA) technical report was found on arXiv. ER 2 references a separate "safety technical report", which was not retrieved.
- I could not confirm whether the original ER 1.0 ever had a public model ID.
- The ER 2 model page still shows ER 1.6 as "Preview" while the overview and deprecations pages say it was deprecated or shut down on 2026-08-31. Whether calls to ER 1.6 still work was not tested.

---

## 2. What does each model output, at what control rate and latency, where does it run, and on what hardware (including Apple Silicon)?

### Takeaway
The ER models output **text only**: JSON points (normalized `[y, x]` on a 0–1000 grid), bounding boxes, trajectories, plans, code and function calls. The VLAs output **actions**. Google has published concrete timing figures only for the original 2025 Gemini Robotics VLA: a cloud backbone with a local action decoder, about 250 ms end-to-end, and an effective 50 Hz control rate through action chunks. No control rate, latency, action format or compute requirement has been published for Gemini Robotics 2 or On-Device 2, and **no source mentions Apple Silicon support**.

### Cited Findings

**ER models (ER 2, ER 2 streaming, and earlier ER)**
- ER 2: inputs are text, image, video and audio (interleaved). Output is text, up to 64K tokens. The context window is up to 128k. — [ER 2 model card](https://deepmind.google/models/model-cards/gemini-robotics-er-2/)
- Exact limits for `gemini-robotics-er-2-preview` and `gemini-robotics-er-2-streaming-preview`: 131,072 input tokens and 65,536 output tokens. Output is text only, and "Audio generation" and "Image generation" are Not supported for both. — [Gemini API ER 2 model page](https://ai.google.dev/gemini-api/docs/models/gemini-robotics-er-2-preview)
- ER capabilities across the family: pointing (example output uses normalized `[y, x]` coordinates on a 0–1000 scale), bounding-box detection, trajectory planning, object tracking in video, and multi-step task planning. The model breaks tasks into sub-tasks and runs them "through function calls or generated code". Video progress estimation and moment finding are ER 2 only. The docs recommend "medium" thinking to balance latency and quality. — [Gemini API Robotics overview](https://ai.google.dev/gemini-api/docs/robotics-overview)
- ER 2 benchmarks reported by Google: progress classification into five bins (0–20% … 80–100%) at 57.4% accuracy; moment finding at 91.3% accuracy with a mean absolute distance of 0.96 s. ER 2 also extends instrument reading to digital displays, linear scales, rulers and liquid thermometers, tested across 10 instrument types. — [blog.google ER 2 post, 2026-07-30](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/)
- ER 2 "can also think about its next step while actions are running" when orchestrating tools. — [blog.google ER 2 post](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/)
- The ER 1.5 VLM produced natural-language step instructions that the 1.5 VLA executed (**SUPERSEDED** by ER 2). — [DeepMind blog, 2025-09-25](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/)

**Gemini Robotics 2 VLA (current)**
- Input is text and image. Output is "Actions", with no published action format. The VLA handles whole-body control (balancing center of gravity to step, squat and bend) and 22-DoF hands. — [DeepMind VLA page](https://deepmind.google/models/gemini-robotics/vla/)
- DeepMind's published success rates for the VLA: pick up from table 68.4%, floor 45.7%, shelf 76.3%; unscrew bulb 92%, screw bulb 36%, tie trash bag 44%; gripper pick-and-place 74.2%, precise insertion 89.6%. — [DeepMind VLA page](https://deepmind.google/models/gemini-robotics/vla/)
- Launch coverage says DeepMind admits robots "still have room to improve in movement speed". Coverage gives no control rate or latency. — [The Robot Report, 2026-08-02](https://www.therobotreport.com/google-deepmind-says-gemini-robotics-2-enables-full-body-control/)
- No source says whether the Gemini Robotics 2 VLA runs in the cloud or locally. Only On-Device 2 is described as cloud-free. — [TechTimes, 2026-07-30](https://www.techtimes.com/articles/322309/20260730/gemini-robotics-2-controls-full-humanoids-legs-torso-arms-fingers-under-one-policy.htm)

**On-Device 2 (current)**
- Inputs are image, text and **action** (proprioceptive state). Output is action. It is built to "run locally on robotic devices" with no network dependency. The page gives no hardware or latency figures. — [DeepMind On-Device page](https://deepmind.google/models/gemini-robotics/on-device/)
- TechTimes describes it as "designed to run entirely on local hardware without cloud connectivity", with no compute or latency specs. — [TechTimes, 2026-07-30](https://www.techtimes.com/articles/322309/20260730/gemini-robotics-2-controls-full-humanoids-legs-torso-arms-fingers-under-one-policy.htm)

**Original Gemini Robotics VLA (March 2025, SUPERSEDED; the only published timing figures)**
- Architecture: a "VLA backbone hosted in the cloud", which is a distilled Gemini Robotics-ER, plus a "local action decoder" on the robot's onboard computer. Backbone query-to-response latency was "optimized from seconds to under 160ms". End-to-end latency from observations to low-level action chunks is "approximately 250ms". "With multiple actions in the chunk … the effective control frequency is 50Hz." — [arXiv 2503.20020 (Gemini Robotics tech report)](https://arxiv.org/html/2503.20020)
- On-Device 1 (June 2025) was "optimized for low-latency inference". DeepMind gave no hardware specs. — [DeepMind blog, 2025-06-24](https://deepmind.google/discover/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/)

**SDK and hardware**
- The Gemini Robotics SDK repo (`google-deepmind/gemini-robotics-sdk`) installs as `pip install safari-sdk` and is labelled "not an officially supported Google product". Trusted testers can access On-Device from SDK v2.4.1. Models are served and fine-tuned through `flywheel-cli` (`serve`, `download` of fine-tuned checkpoints). There is no public weight download. The README states no GPU, CUDA, OS or Apple Silicon requirements, and the only robot example is ALOHA. — [GitHub: google-deepmind/gemini-robotics-sdk](https://github.com/google-deepmind/gemini-robotics-sdk)
- **UNVERIFIED (third-party tutorial):** one blog claims the On-Device action decoder can run on ARM CPUs or modest integrated GPUs. No DeepMind source supports this. — [BinaryVerse AI tutorial](https://binaryverseai.com/gemini-robotics-on-device/)

### Inferences
- ER 2 cannot drive SO-101 joints directly. It produces points, boxes, trajectories, plans and tool calls as text. Something else (inverse kinematics on ER 2 points or trajectories, a LeRobot-trained policy, or a Gemini VLA if access is granted) has to turn those into joint commands.
- If Gemini Robotics 2 follows the 1.0 design (cloud backbone plus local decoder), a cloud-hosted VLA would be roughly 250 ms behind the scene with 50 Hz chunked control. This is an extrapolation and has not been confirmed for v2.
- Apple Silicon: the SDK documents no macOS support, and model weights are served through DeepMind's platform rather than downloaded. Running On-Device 2 locally on a Mac should not be assumed.

### Gaps
- No action representation (joint positions vs end-effector deltas), chunk length, control rate or latency has been published for Gemini Robotics 2 or On-Device 2.
- No GPU class or compute requirement for On-Device 1 or 2 was found in any primary source. Apple Silicon support is undocumented anywhere.
- The ER 2 Live API streaming latency in milliseconds is not published. The 0.96 s figure is a moment-finding accuracy metric, not latency.

---

## 3. Which robot embodiments are officially supported, and has Google said anything about LeRobot, SO-100/SO-101, or Pollen Robotics / Reachy?

### Takeaway
DeepMind's officially demonstrated embodiments are ALOHA / ALOHA 2, bi-arm Franka (FR3, later Franka Duo / Mobile FR3 Duo with Robotiq grippers), and Apptronik Apollo / Apollo 2 (with SharpaWave 22-DoF and Inspire hands), plus Boston Dynamics Spot for ER. On 2026-07-30 DeepMind said **On-Device 2 was adapted to SO101**, along with Dexmate and Trossen platforms, none of which were in its training data. This is the only Google mention of the SO-10x arms found. Google has said nothing about LeRobot as software or about Pollen / Reachy in connection with Gemini Robotics. A DeepMind DevRel demo ran *Gemma* (not Gemini Robotics) on Reachy Mini.

### Cited Findings
- Gemini Robotics 2 launch embodiments: Apptronik Apollo 2 with SharpaWave (5-finger, 22-DoF) and Inspire hands; Franka Duo with Robotiq grippers; and "Dexmate, SO101, and Trossen platforms" for the on-device model. Acknowledged partners are Apptronik, Boston Dynamics and Agile Robots. — [DeepMind blog, 2026-07-30](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)
- On-Device 2 "adapts to new bi-arm embodiments in a few hours, typically with fewer than 200 examples". It is "natively multi-embodiment" and inherits Motion Transfer from Gemini Robotics 1.5. — [DeepMind blog, 2026-07-30](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/); [The Robot Report](https://www.therobotreport.com/google-deepmind-says-gemini-robotics-2-enables-full-body-control/)
- TechTimes reports that the Dexmate, SO101 and Trossen demos used platforms not in the training data, across "substantially different joint configurations, sensor arrays, and degrees of freedom". It gives no per-platform demo counts or success rates. — [TechTimes, 2026-07-30](https://www.techtimes.com/articles/322309/20260730/gemini-robotics-2-controls-full-humanoids-legs-torso-arms-fingers-under-one-policy.htm)
- **UNVERIFIED:** a MarkTechPost benchmark table, seen only as a search summary (the page returned 403), reportedly shows On-Device 2 raising SO101 task success from 6.7% to 53.3%, against 0.0% to 6.7% for On-Device 1. It was not possible to confirm whether these are zero-shot vs fine-tuned figures. — [MarkTechPost, 2026-07-30](https://www.marktechpost.com/2026/07/30/google-deepmind-gemini-robotics-2-whole-body-control-dexterity-multi-robot-collaboration/)
- Kingy AI cautions that a demonstration platform does not mean every configuration is commercially supported. — [Kingy AI](https://kingy.ai/news/gemini-robotics-2-whole-body-ai-robots/)
- DeepMind's hub, VLA page and On-Device page do not mention SO101, SO-100, LeRobot, Hugging Face, Reachy or Pollen. The SO101 mention appears only in the launch blog. — [DeepMind models page](https://deepmind.google/models/gemini-robotics/); [On-Device page](https://deepmind.google/models/gemini-robotics/on-device/)
- The Robot Report's coverage mentions neither LeRobot, SO-101, Pollen nor Reachy. Its only Hugging Face reference is the ASIMOV-Agentic dataset link. — [The Robot Report](https://www.therobotreport.com/google-deepmind-says-gemini-robotics-2-enables-full-body-control/)
- ER 2 multi-robot demo: Apptronik Apollo 2 and Franka Mobile FR3 Duo collaborating. Boston Dynamics Spot demo: ER 2 orchestrates the Spot navigation and manipulator APIs, with sample code under the Gemini live-api samples on GitHub. — [blog.google ER 2 post](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/)
- 1.5 generation (**SUPERSEDED**): Motion Transfer let skills learned on ALOHA 2 work on Apollo and bi-arm Franka without per-robot specialisation. — [DeepMind blog, 2025-09-25](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/)
- On-Device 1 (**SUPERSEDED**) was trained on ALOHA, adapted to bi-arm Franka FR3 and Apptronik Apollo, and fine-tuned with as few as 50–100 demonstrations. — [DeepMind blog, 2025-06-24](https://deepmind.google/discover/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/)
- Gemini Robotics 1.0 (**SUPERSEDED**) used ALOHA 2 for most data and experiments, and adapted to bi-arm Franka and the Apollo humanoid. — [arXiv 2503.20020](https://arxiv.org/html/2503.20020)
- The SDK's only robot example is ALOHA (`examples/aloha/`). There is no LeRobot or SO-10x integration. — [GitHub gemini-robotics-sdk](https://github.com/google-deepmind/gemini-robotics-sdk)
- The trusted-tester form asks for robot types (industrial 6-DoF arms, research arms, humanoids, quadrupeds, AMR/AGV, other) and a robot brand list with "Other". — [Trusted Tester waitlist form](https://docs.google.com/forms/d/1sM5GqcVMWv-KmKY3TOMpVtQ-lDFeAftQ-d9xQn92jCE/viewform)
- Reachy Mini: Google DeepMind developer-relations engineer Ian Ballantyne showed **Gemma** (not Gemini Robotics) running locally on Reachy Mini with camera, emotion/head motion, mic and speaker. **Weak source:** this is a repost on an aggregator, and the original Google video was not located. — [taxheal.com repost](https://www.taxheal.com/run-gemma-on-reachy-mini-an-open-source-robot.html)
- Context: SO-101 was announced 2025-04-28 as the successor to SO-100 and is integrated with LeRobot. Pollen Robotics is owned by Hugging Face. — [GIGAZINE](https://gigazine.net/gsc_news/en/20250501-hugging-face-robotic-arm); [Hugging Face Reachy Mini blog](https://huggingface.co/blog/reachy-mini)

### Inferences
- SO-101 is now a Google-demonstrated target for On-Device 2. This is the strongest signal so far that a Gemini VLA could drive the developer's arm, but only through trusted-tester access. Google's adaptation claim is about "bi-arm" embodiments, and sources do not say whether the SO101 demo was a single arm or a bimanual SO-101 pair.
- Reachy Mini's head (a Stewart-platform neck with antennas) is not a manipulator and resembles no demonstrated embodiment. No Gemini VLA is described as controlling head-only or expressive robots. Head motion would most likely stay under the developer's own code, called as tools by Gemini Live or ER 2.

### Gaps
- No official statement was found about LeRobot (the software stack), LeRobot dataset formats, Pollen Robotics or Reachy in connection with Gemini Robotics.
- The SO101 demo details (single vs bimanual, number of demos, success rates, which cameras) were not found in a primary source.
- I could not confirm whether On-Device 2 accepts arbitrary DoF counts (SO-101 has 5 DoF plus a gripper) or needs a bi-arm layout.

---

## 4. Do these models support function/tool calling and audio/voice I/O, and how does Google describe ER (orchestrator) + VLA (executor)?

### Takeaway
Yes on tool calling: ER 2 natively calls Google Search and **any user-defined function**, and Google's pattern is to declare the VLA, navigation APIs and safety stops as tools that ER 2 orchestrates. ER 2 **accepts audio input**, and the streaming variant works with the **Live API**, but **both ER 2 endpoints output text only** (audio generation is not supported). ER 2 therefore cannot replace Gemini Live for spoken replies. The VLAs take text and images (On-Device also takes action/state) and expose no tool calling.

### Cited Findings
- Capability matrix (ER 2 / ER 2 Streaming / ER 1.6): function calling Supported / Supported / Supported; Live API Not supported / Supported / Not supported; audio generation Not supported for all three; thinking Supported for all three; code execution Supported / Not supported / Supported; structured outputs Supported / Not supported / Supported; search grounding Supported for all three; computer use Supported / Not supported / Supported; Batch API Supported / Not supported / Supported. — [Gemini API ER 2 model page](https://ai.google.dev/gemini-api/docs/models/gemini-robotics-er-2-preview)
- "Developers declare low-level control interfaces, such as VLA models or navigation APIs, as tools", and ER 2 orchestrates them. ER 2 natively calls Google Search and any user-defined function. ER 2 integrates with the Gemini Live API "over a bidirectional streaming endpoint optimized for latency", and developers can stream video, audio or text into it. — [blog.google ER 2 post, 2026-07-30](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/)
- ER 2 "observes the room, reasons about the steps needed, directs the VLA to carry out actions, and tracks progress". It self-corrects on failure and can "trigger safety tool calls and bring the robot to a stop when a person gets too close". — [DeepMind blog, 2026-07-30](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)
- ER output can be passed to "a robotics API or a VLA model". Orchestration uses function calls or generated code. — [Gemini API Robotics overview](https://ai.google.dev/gemini-api/docs/robotics-overview)
- The ER 2 model card lists audio as an input modality and "tool orchestration" as an intended use. — [ER 2 model card](https://deepmind.google/models/model-cards/gemini-robotics-er-2/)
- ER 1.5 (**SUPERSEDED**) already acted as a high-level planner. It called Google Search or user-defined functions, estimated progress and success, and issued natural-language step instructions that the 1.5 VLA executed. — [DeepMind blog, 2025-09-25](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/); [Google Developers Blog](https://developers.googleblog.com/building-the-next-generation-of-physical-agents-with-gemini-robotics-er-15/)
- The 1.5 VLA "thinks before acting": it interleaves actions with multi-level natural-language reasoning. — [arXiv 2510.03342](https://arxiv.org/abs/2510.03342)
- The Gemini Robotics 2 VLA "can think and then interact with the physical world". Its inputs are text and image only. — [DeepMind VLA page](https://deepmind.google/models/gemini-robotics/vla/)
- On-Device 1 (**SUPERSEDED**): semantic and content safety was handled "through the Live API", with the model connected to low-level safety-critical controllers. — [DeepMind blog, 2025-06-24](https://deepmind.google/discover/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/)
- Developer resources: a getting-started notebook (`gemini_robotics_er.ipynb`), a robotics-samples repo, and live-api examples, including the Spot orchestration sample. — [blog.google ER 2 post](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/)

### Inferences
- The developer's current design (Gemini 3.8 Live for voice and function calling, ER 2 for pointing and planning) already matches Google's recommended "ER orchestrates tools" pattern. Their emotion detector, Reachy head moves and SO-101 skills can all be declared as user-defined functions for ER 2, or kept on Gemini Live.
- ER 2 Streaming takes audio in but produces no audio out. Replacing Gemini Live with ER 2 Streaming would therefore need a separate TTS step. Keeping Live for conversation and calling ER 2 as a tool (or a sub-agent) is the lower-friction option.
- A Gemini VLA would sit beneath ER 2 as one tool, such as `run_vla(instruction)`. Without VLA access the developer can put their own executor behind the same interface, for example a LeRobot-trained policy or scripted IK on ER 2 points, and swap in On-Device 2 later if granted.

### Gaps
- No documentation was found for the actual tool schema or interface used to call the Gemini Robotics 2 VLA from ER 2, such as the instruction format or a status/progress callback.
- No figures were published for Live API latency or for audio-in turn latency with ER 2 Streaming.
- I could not confirm whether the VLAs accept audio, or whether On-Device 2 accepts free-form language instructions at runtime (it lists text as an input).

---

## 5. What are the licensing, pricing, access requirements, safety restrictions and usage limits, and how do you get access?

### Takeaway
ER 2 is self-serve: a Gemini API key from AI Studio, a **free tier**, and paid pricing of **$1.00 / $5.00 per 1M input/output tokens until 2026-12-31, doubling to $2.00 / $10.00 from 2027-01-01**. The Gemini API Additional Terms of Service govern it. The VLAs (Gemini Robotics 2, On-Device 2) require joining the **Trusted Tester waitlist** through a Google Form, which is open to individuals ("personal interest") aged 18+. Acceptance is not guaranteed and no pricing has been published. The SDK is Apache-2.0, but the model weights are not public. Google excludes safety-critical uses and puts responsibility for physical safety and bystander privacy on the developer.

### Cited Findings

**Pricing (ER 2, official)**
- `gemini-robotics-er-2-preview`, Standard tier. Free tier: input and output "Free of charge" (context caching not available). Paid per 1M tokens: input (text/image/video/audio) $1.00 through 2026-12-31, then $2.00 from 2027-01-01; output $5.00, then $10.00. Caching $0.10, then $0.20, with storage $0.50/1M tokens/hour, then $1.00. Batch input $0.50, then $1.00; batch output $2.50, then $5.00. There is no separate audio rate. — [Gemini API pricing (last updated 2026-10-15)](https://ai.google.dev/gemini-api/docs/pricing)
- `gemini-robotics-er-2-streaming-preview` has a free tier and the same $1.00/$5.00, then $2.00/$10.00 schedule. Caching and batch are not listed. ER 1.5 and 1.6 no longer appear on the pricing page. — [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- Aggregator pages that show $1/$5 (Eden AI, getmaxim.ai) or $2/$10 (cloudprice.net, getmaxim.ai) are explained by the scheduled 2027-01-01 increase. — [Eden AI](https://www.edenai.co/models/gemini-robotics-er-2-preview); [cloudprice.net](https://cloudprice.net/models/gemini/gemini-robotics-er-2-preview)
- **UNVERIFIED:** tech-now.io claims ER 2 costs "double ER 1.6's rate". This could not be checked because ER 1.6 pricing has been removed. — [tech-now.io](https://tech-now.io/en/blogs/gemini-robotics-er-2/)
- No pricing has been announced for the Gemini Robotics 2 VLA or On-Device 2. — [TechTimes](https://www.techtimes.com/articles/322309/20260730/gemini-robotics-2-controls-full-humanoids-legs-torso-arms-fingers-under-one-policy.htm)

**Access**
- ER 2 is reached through the Gemini Developer API with an API key from Google AI Studio or the Google Cloud console. API keys must be restricted, or requests return `403 Forbidden`. `gs://` URIs are not directly accessible, so local files or the Files API must be used. — [Gemini API Robotics overview](https://ai.google.dev/gemini-api/docs/robotics-overview)
- The Trusted Tester waitlist form ("Gemini Robotics Trusted Tester Program") covers the Gemini Robotics VLA, On-Device and ER. It asks for contact details, an 18+ confirmation, organisation, robotics category (including "personal interest"), industry, role, fleet size, robot types and brands, and which models are relevant. Submitting it joins a waitlist and does not mean acceptance. Responses may be used under the Google Privacy Policy, and the opt-out contact is robotics-trusted-testers-admin@google.com. — [Trusted Tester waitlist form](https://docs.google.com/forms/d/1sM5GqcVMWv-KmKY3TOMpVtQ-lDFeAftQ-d9xQn92jCE/viewform)
- DeepMind says it is "working with 100+ trusted testers". — [DeepMind Gemini Robotics page](https://deepmind.google/models/gemini-robotics/)
- SDK: most functionality requires the Trusted Tester Program, and On-Device is available to trusted testers from v2.4.1. — [GitHub gemini-robotics-sdk](https://github.com/google-deepmind/gemini-robotics-sdk)

**Licensing / terms**
- ER 2 is governed by the Gemini API Additional Terms of Service (AI Studio / Gemini API) and the Google Cloud Platform Terms of Service (Enterprise Agent Platform). — [ER 2 model card](https://deepmind.google/models/model-cards/gemini-robotics-er-2/)
- The Gemini Robotics SDK (`safari-sdk`) is Apache-2.0 and "not an officially supported Google product". No public weight distribution is mentioned. — [GitHub gemini-robotics-sdk](https://github.com/google-deepmind/gemini-robotics-sdk)

**Usage limits**
- ER 2 and ER 2 Streaming allow 131,072 input tokens and 65,536 output tokens. Flex and Priority inference are not supported. Rate limits are on a separate page that was not retrieved. — [Gemini API ER 2 model page](https://ai.google.dev/gemini-api/docs/models/gemini-robotics-er-2-preview)

**Safety restrictions**
- Out-of-scope uses for ER 2: safety-critical applications "including healthcare, transportation, and other settings where a malfunction could foreseeably cause death, injury, or property damage". The card also advises discretion before production, commercial or public deployment. — [ER 2 model card](https://deepmind.google/models/model-cards/gemini-robotics-er-2/)
- The docs say "it is your responsibility to maintain a safe environment around the robot". On privacy, the robotics models collect video and audio that may contain personal data, so identifiable people must be notified and give consent. The docs suggest face blurring or operating away from identifiable people. Known limitations include hallucinations and prompt sensitivity. — [Gemini API Robotics overview](https://ai.google.dev/gemini-api/docs/robotics-overview)
- ASIMOV-Agentic is a new benchmark (dataset on Hugging Face). It tests whether ER refuses unsafe VLA tool calls, predicts whether a task is feasible, and asks humans for help when uncertain. DeepMind calls ER 2 its safest robotics model on safety-constraint and human-proximity benchmarks. — [DeepMind blog, 2026-07-30](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/); [The Robot Report](https://www.therobotreport.com/google-deepmind-says-gemini-robotics-2-enables-full-body-control/)
- On-Device 1 guidance (**SUPERSEDED**, likely still relevant): evaluate the full system on a semantic safety benchmark and red-team at every level. — [DeepMind blog, 2025-06-24](https://deepmind.google/discover/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/)

### Inferences
- Today's practical path is to keep using ER 2 under the free or paid tier, budget for the 2027-01-01 price doubling, and apply to the Trusted Tester waitlist by naming SO-101 / Hugging Face under robot brands and picking On-Device. Acceptance is uncertain for an individual with a hobby arm, so plan on owning the low-level control layer.
- Putting a home-use robot with a camera and microphone near family members brings in the docs' notice-and-consent expectations for identifiable people.

### Gaps
- The terms of the Trusted Tester Program (NDA, commercial use, data sharing, any fees) are not public.
- The Gemini API rate-limit numbers (RPM/TPM/RPD) for ER 2 on the free and paid tiers were not retrieved.
- No pricing or GA date has been published for the VLA or On-Device 2, and none has been published for ER 2 on Gemini Enterprise Agent Platform.
- Observed but not investigated: a banner on the Gemini API model page (2026-10-08) says "Gemini 4 Argon is now available". This may affect the developer's Gemini 3.8 Live choice but is outside this scope. — [Gemini API ER 2 model page](https://ai.google.dev/gemini-api/docs/models/gemini-robotics-er-2-preview)
