# Alternatives to Gemini Robotics as the VLA for an SO-101 arm + Reachy Mini head (as of 15 October 2026)

Scope: open or self-hostable VLA and imitation-learning policies that could drive a LeRobot SO-101 follower arm (and possibly a Reachy Mini Lite head). The developer has a MacBook M3 Pro with 18 GB and no NVIDIA GPU. Gemini 3.8 Live stays as the voice and conversation layer. Web research only; no hardware was tested. "Reported" means a number published by the source. "Claim" means a qualitative statement with no measurement behind it. Sources marked "(search summary only)" were seen only in a search-engine digest, not read in full, so they are weaker.

## Q1. Which VLAs are integrated into LeRobot 0.6.x, which have documented SO-100/SO-101 fine-tunes, and what dataset sizes and success rates were reported?

### Takeaway
LeRobot 0.6.1 (3 Aug 2026) ships 10 VLA policy types: Pi0, Pi0Fast, Pi0.5, GR00T N1.7, SmolVLA, XVLA, EO-1, MolmoAct2, WALL-OSS and EVO1. It also ships the non-VLA baselines ACT, Diffusion, VQ-BeT and Multitask DiT. SmolVLA has by far the best-documented SO-100/SO-101 results: about 50 episodes per task, and 78.3% average real-world success on SO-100 against 61.7% for pi0 and 48.3% for ACT in its paper. GR00T, pi0.5 and MolmoAct2 have SO-101 recipes or checkpoints but few published SO-101 success rates. OpenVLA/OpenVLA-OFT, Octo and RDT are not LeRobot policy types.

### Cited Findings

**LeRobot release state**
- PyPI shows lerobot 0.6.1, released Aug 3, 2026. Earlier releases are 0.6.0 (Jul 6, 2026), 0.5.1 (Apr 7, 2026) and 0.5.0 (Mar 9, 2026). It requires Python >=3.12. — [PyPI lerobot](https://pypi.org/project/lerobot/)
- The 0.6.1 policy list on PyPI:
  - Imitation learning: ACT, Diffusion, VQ-BeT, Multitask DiT Policy.
  - RL: HIL-SERL and TDMPC, with QC-FQL "coming soon".
  - VLAs: Pi0, Pi0Fast, Pi0.5, GR00T N1.7, SmolVLA, XVLA, EO-1, MolmoAct2, WALL-OSS, EVO1.
  - World models: VLA-JEPA, LingBot-VA, FastWAM.
  - Reward models: SARM, TOPReward, Robometer.
  - — [PyPI lerobot](https://pypi.org/project/lerobot/)
- The GitHub `main` README also lists FineART-VLA, DM05, LaWAM and "FLUX 3 Action". The DM05 docs page says it "doesn't exist in v0.6.1, but exists on the main version", so these extra entries are main-branch only, not in the 0.6.1 release. — [LeRobot GitHub README](https://github.com/huggingface/lerobot); [LeRobot DM05 docs redirect](https://huggingface.co/docs/lerobot/en/dm05)
- LeRobot's "Supported Hardware" list includes SO100, LeKiwi, Koch, HopeJR, OMX, Reachy2 and others. Reachy Mini is not listed. — [LeRobot GitHub README](https://github.com/huggingface/lerobot)

**SmolVLA (Hugging Face, `policy.type=smolvla`)**
- 450M parameters, of which about 100M are the action expert. Paper dated June 2025. — [SmolVLA blog, 3 Jun 2025](https://huggingface.co/blog/smolvla); [SmolVLA paper arXiv 2506.01844](https://arxiv.org/html/2506.01844)
- Pretraining data: the paper says 481 community datasets, 22.9K episodes and 10.6M frames. The blog says "487 high-quality datasets" and "around 10 million frames", a small mismatch between the two. — [SmolVLA paper](https://arxiv.org/html/2506.01844); [SmolVLA blog](https://huggingface.co/blog/smolvla)
- Reported real-world results on SO-100 (Table 3):

  | Policy | Pick-Place | Stacking | Sorting | Average |
  |---|---|---|---|---|
  | ACT (single-task) | 70 | 50 | 25 | 48.3 |
  | π0 3.5B (multi-task) | 100 | 40 | 45 | 61.7 |
  | SmolVLA 0.45B (multi-task) | 75 | 90 | 70 | 78.3 |

  — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- Reported SO-101 Pick-Place-Lego (Table 4): SmolVLA 90% in-distribution and 50% out-of-distribution, against ACT at 70% and 40%. No π0 result is given for SO-101. — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- Reported effect of community pretraining: 51.7% without it, 78.3% with it (SO-100 multi-task). — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- Data used: 50 demonstrations per dataset, as 10 trajectories at each of 5 starting positions. — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- LeRobot recommends "~50 episodes" as a starting point. The docs add that a similar 25-episode dataset "was not enough leading to a bad performance". — [LeRobot SmolVLA docs](https://huggingface.co/docs/lerobot/en/smolvla)
- Community SO-101 fine-tune by ggando (1 Mar 2026). Reported results:
  - Final data: 75 clean episodes in a 10 cm workspace. An earlier 50 episodes spread over 30 cm "grasped at air".
  - Real-robot success over 5 trials each: SmolVLA dual-camera 5/5, SmolVLA wrist-only 4/5, ACT dual-camera 4/5.
  - Typical rate across 5-episode eval runs: "60-80%".
  - Colour generalisation: 1/3.
  - — [ggando.com blog](https://ggando.com/blog/smolvla-so101/)
- Community bimanual SO-101 fine-tune (izlley, 23 Sep 2026): 34 episodes, 20k steps. Run on a real robot with Mac MPS at 0.45 s per chunk. The card says "Task T1 grasps succeed" but gives no rate. — [HF model card izlley/h101-smolvla-t1-v6](https://huggingface.co/izlley/h101-smolvla-t1-v6)
- Other community numbers, not real-robot success rates:
  - An "87.66%" from 40 episodes is an offline metric that counts a joint as correct within 5% of its range. — [Sa74ll/smolvla_so101_pickandplace](https://huggingface.co/Sa74ll/smolvla_so101_pickandplace) (search summary only)
  - A Medium write-up reports 40% grasp success. — [Correll Lab, Medium](https://medium.com/correll-lab/fine-tuning-smolvla-for-new-environments-code-included-af266c56d632) (search summary only)

**ACT / Diffusion Policy (non-VLA baselines in LeRobot)**
- ACT has about 80M parameters. Its SO-100 and SO-101 numbers are in the SmolVLA tables above. — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- On an SO-ARM101 with an M-series Mac (MPS), ACT trained on 30 demonstrations for 30,000 steps in about 6h40m. The author saw overfitting, and the 25k-step checkpoint was better. — [DevelopersIO / classmethod (Japanese)](https://dev.classmethod.jp/articles/so-arm101-duck-pick-imitation-learning-mac-training/) (search summary only)
- In SmolVLA's LIBERO table, Diffusion Policy averages 72.4%. — [SmolVLA paper](https://arxiv.org/html/2506.01844)

**GR00T (NVIDIA, `policy.type=groot`)**
- LeRobot removed GR00T N1.5 support and "current releases support GR00T N1.7 only". To keep N1.5, pin `lerobot==0.5.1`. — [LeRobot GR00T docs](https://huggingface.co/docs/lerobot/en/groot)
- LeRobot's SO-101 recipe:
  - Uses `--policy.embodiment_tag=new_embodiment`, `chunk_size=16`, relative actions, bf16, batch size 64 and 20,000 steps.
  - The rollout example targets `so101_follower` with wrist and front cameras, RTC enabled and `n_action_steps=8`.
  - — [LeRobot GR00T docs](https://huggingface.co/docs/lerobot/en/groot)
- LeRobot gives LIBERO results for N1.7 only: 96.5% average. It gives no SO-101 success rate. — [LeRobot GR00T docs](https://huggingface.co/docs/lerobot/en/groot)
- The NVIDIA GR00T N1.5 SO-101 tutorial (11 Jun 2025):
  - Data: the `youliangtan/so101-table-cleanup` dataset (80 episodes per the dataset card), 2 cameras, `new_embodiment`, about 25 GB VRAM with default settings, `--max-steps 10000`.
  - Results: no success rate reported.
  - — [HF blog, Post-Training GR00T N1.5 for SO-101](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- The Isaac-GR00T repo includes an SO100 demo dataset of 5 episodes (`cube_to_bowl_5`) and an `examples/SO100` walkthrough. It reports no SO-100/SO-101 success rates. — [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- A Seeed Studio wiki covers fine-tuning GR00T N1.6 for SO-101 and deploying on an AGX Orin 64G. It says fine-tuning needs ">48GB of VRAM". — [Seeed wiki N1.6 SO-101](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.6_for_lerobot_so_arm_and_deploy_on_agx_orin/) (search summary only)

**π0 / π0-FAST / π0.5 (Physical Intelligence, `pi0`, `pi0fast`, `pi05`)**
- LeRobot ports these from openpi. On LIBERO, the LeRobot π0.5 port averages 97.5% against 96.85% for the OpenPI reference. — [LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/en/pi05)
- A LeRobot issue reports that fine-tuned π0 and π0.5 come out about 10–20 points below reported LIBERO success rates. — [lerobot issue #2114](https://github.com/huggingface/lerobot/issues/2114) (search summary only)
- In the SmolVLA paper, π0 (3.5B) averages 61.7% on SO-100 multi-task (table above). — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- Community π0.5 on SO-101:
  - A full fine-tune on 50 episodes claims no task success rate. A bottle-pickup card lists 57 episodes and no rate.
  - A simulated SO-101 benchmark reports 16/39 (41%) after training on 3,629 simulated episodes.
  - — [sun3689/pi05-so101-cube-fullft-30k](https://huggingface.co/sun3689/pi05-so101-cube-fullft-30k-7799251), [CQCGO/pi05_so101_bottle_finetuned](https://huggingface.co/CQCGO/pi05_so101_bottle_finetuned), [5hadytru/so101_bench_pi05](https://huggingface.co/5hadytru/so101_bench_pi05) (search summary only)
- A third-party training-platform guide (9 Aug 2026) recommends at least 50 kept episodes and "a few hundred" for robustness across the workspace. It reports no success rates. — [ay-robots: Train Pi0.5 on an SO-101](https://www.ay-robots.com/train/pi0-5-on-so-101)

**X-VLA (`policy.type=xvla`)**
- 0.9B parameters, pretrained on 290K episodes across 7 platforms. New robots are adapted with learned "soft prompts". — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/en/xvla)
- Has a built-in `so101_bimanual` action mode and an example bimanual SO-101 run of 3,000 steps. — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/en/xvla)
- Released checkpoints: base, LIBERO (93%), WidowX, folding, AgiBot and Google Robot. There is no single-arm SO-101 checkpoint and no reported SO-101 success rate. — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/en/xvla)

**MolmoAct2 (Ai2, `policy.type=molmoact2`), new in 2026**
- Released May 5, 2026; full code and LeRobot integration followed on May 28. — [Ai2 MolmoAct2 blog](https://allenai.org/blog/molmoact2)
- Trained partly on "large-scale SO-100/SO-101 datasets from low-cost open-source robot arms". The amount of SO-100/SO-101 data is not given. — [Ai2 MolmoAct2 blog](https://allenai.org/blog/molmoact2)
- LeRobot hosts `lerobot/MolmoAct2-SO100_101-LeRobot` for zero-shot SO-100/SO-101 rollout:
  - It is the only checkpoint found that is pretrained specifically for SO-100/101.
  - It needs a joint-frame correction (`joint_signs`, `joint_offsets`) because its training data used the pre-0.5.0 calibration convention.
  - — [LeRobot MolmoAct2 docs](https://huggingface.co/docs/lerobot/en/molmoact2)
- For real-world datasets under 200 demos, the docs suggest a global batch size of 16–32 with LoRA-VLM or action-expert-only fine-tuning. — [LeRobot MolmoAct2 docs](https://huggingface.co/docs/lerobot/en/molmoact2)
- LIBERO average: 98.25% for the LeRobot port, 97.20% for the original. — [LeRobot MolmoAct2 docs](https://huggingface.co/docs/lerobot/en/molmoact2)
- No SO-100/SO-101 real-world success rates are reported. — [Ai2 MolmoAct2 blog](https://allenai.org/blog/molmoact2)

**EVO1 (`policy.type=evo1`), new in 2026**
- InternVL3-1B backbone with a flow-matching head. A single run scored 96.65% on LIBERO. No SO-101 results are given. — [LeRobot EVO1 docs](https://huggingface.co/docs/lerobot/en/evo1)
- The CVPR 2026 paper describes Evo-1 as 0.77B parameters. — [CVPR 2026 Evo-1](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_Evo-1_Lightweight_Vision-Language-Action_Model_with_Preserved_Semantic_Alignment_CVPR_2026_paper.html) (search summary only)

**Not LeRobot policy types**
- **OpenVLA / OpenVLA-OFT**:
  - The OFT recipe (Feb 2025) reports LIBERO going from 76.5% to 97.1%, with 26× faster action generation. — [OpenVLA-OFT paper](https://arxiv.org/html/2502.19645v1)
  - The repo documents only LIBERO and ALOHA. — [openvla-oft GitHub](https://github.com/moojink/openvla-oft)
- **Octo**: Octo-Small has 27M parameters and Octo-Base 93M. It is JAX-based and has no LeRobot integration. — [Octo GitHub](https://github.com/octo-models/octo)
- **RDT**:
  - RDT-1B is from October 2024. — [RDT-1B paper mirror](https://fugumt.com/fugumt/paper_check/2410.07864v2_enmode)
  - RDT2 (Feb 2026) uses a 7B VLM and was trained on 10,000+ hours of data collected with the UMI device. — [RDT2 arXiv 2602.03310](https://arxiv.org/abs/2602.03310v1)
  - Neither is in LeRobot.

### Inferences
- For an SO-101 fine-tuned on about 50–100 demos per task, SmolVLA is the only VLA with peer-reviewed and community-replicated SO-101 success evidence. ACT is the reliable non-language baseline.
- MolmoAct2's SO-100/101 checkpoint is the most interesting new option for a zero-shot first trial. It is unproven on SO-101 in public and needs an NVIDIA GPU (see Q2).
- GR00T N1.7 and π0.5 have SO-101 recipes, but their community SO-101 evidence is mostly "model trained, no measured success rate". Expect to measure success yourself.
- Plan on 50+ episodes per skill, with about 10 repeats per object position. Keep the workspace tight. The ggando result suggests density per position matters more than the raw episode count.
- At-a-glance summary. Every cell is drawn from the cited findings in Q1–Q5; "Mac?" means runs on an M3 Pro (18 GB) via MPS.

  | Model | In LeRobot 0.6.1 | SO-101 evidence | Fine-tune compute | Mac? | Language | Licence |
  |---|---|---|---|---|---|---|
  | ACT (~80M) | yes (`act`) | SO-101 lego 70/40%; community 4/5 | trains on Mac MPS (hours) | yes | no | Apache-2.0 (LeRobot) |
  | SmolVLA (450M) | yes (`smolvla`) | SO-101 lego 90/50%; community 5/5 | ~4 h A100 for 20k steps; Mac MPS possible | yes, ~0.2–0.45 s per 50-step chunk | yes | Apache-2.0 |
  | π0 / π0-FAST / π0.5 (~3.3B) | yes (`pi0`, `pi0fast`, `pi05`) | SO-100 61.7% (π0); π0.5 SO-101 rates unpublished | LoRA >22.5 GB; full >70 GB (80 GB GPU) | no (~14 GB inference; openpi is Ubuntu-only) | yes | Apache-2.0 + Gemma terms |
  | GR00T N1.7 (3B) | yes (`groot`) | recipe only, no rate | 40 GB+ GPU recommended | no (CUDA) | yes | NVIDIA Open Model License |
  | X-VLA (0.9B) | yes (`xvla`) | bimanual SO-101 action mode, no rate | GPU, bf16 | unknown | yes | not stated |
  | MolmoAct2 | yes (`molmoact2`) | SO100_101 zero-shot checkpoint, no rate | 16.5–48 GiB GPU | no (NVIDIA GPU required) | yes | Apache-2.0, research/educational |
  | EVO1 (~0.77B) | yes (`evo1`) | none | reference run on 2×H100 | unknown (flash-attn optional) | yes | upstream checkpoints: check |
  | OpenVLA-OFT (7B) | no | none (LIBERO/ALOHA only) | 27–80 GB GPUs | no | yes | code MIT; weights unstated |
  | Octo (27M / 93M) | no | none | small (JAX) | unknown | yes, or goal image | MIT |
  | RDT-1B / RDT2 | no | none | not researched | no | yes | RDT-1B MIT (third-party listing) |

### Gaps
- I found no public SO-101 real-robot success rates for GR00T N1.7, X-VLA (single-arm), EVO1, EO-1, WALL-OSS or MolmoAct2.
- I did not investigate EO-1, WALL-OSS, FineART-VLA or DM05: parameters, licenses and hardware are unknown here.
- I found no head-to-head comparison of π0 against π0.5 on SO-101 using one dataset and one protocol.
- The AGENT_GUIDE.md in the LeRobot repo reportedly gives SmolVLA a 3.93 GB peak against about 15.5 GB for π0 (batch 1), and recommends proving an ACT baseline first. I saw this only in a search summary and did not read it. — [lerobot AGENT_GUIDE.md](https://github.com/huggingface/lerobot/blob/main/AGENT_GUIDE.md)

## Q2. Training and inference hardware: which can be fine-tuned cheaply on a rented GPU, which run on an M3 Pro (MPS) at a usable rate, and what are the latency and chunk sizes?

### Takeaway
On an 18 GB M3 Pro, only ACT, Diffusion and SmolVLA (about 2 GB at inference) are realistic local options:
- SmolVLA's PyTorch-MPS chunk latency is roughly 0.2–0.45 s for a 50-action chunk (about 1.7 s of motion at 30 fps). With LeRobot async inference or RTC, that is usable.
- π0/π0.5 need about 14 GB at inference and the official openpi supports only Ubuntu. GR00T N1.7, MolmoAct2 and OpenVLA-OFT need CUDA GPUs with 12–18+ GB.
- For all of these, the practical path is to fine-tune on a rented A100/H100 (roughly US$5–12 per run) and run a remote LeRobot policy server, with the Mac acting as robot client.

### Cited Findings

**LeRobot device support and async inference**
- LeRobot supports CPU, CUDA, MPS and XPU backends, selected with `--policy.device`. — [LeRobot torch accelerators docs](https://huggingface.co/docs/lerobot/torch_accelerators) (search summary only)
- Async inference splits work between a `PolicyServer` (local or on another machine) and a `RobotClient`. Settings shown in the docs:
  - `--policy_device` can be `cuda`, `mps`, `xpu` or `cpu`.
  - `actions_per_chunk` defaults to 50; typical values are 10–50.
  - A `chunk_size_threshold` of about 0.5–0.6 is recommended.
  - The docs say "PI0 occupies 14GB of memory at inference time, while SmolVLA requires only ~2GB."
  - "inference can run on powerful remote hardware while maintaining real-time robot control."
  - — [LeRobot async inference docs](https://huggingface.co/docs/lerobot/en/async)

**SmolVLA**
- Defaults: `chunk_size=50`, `n_action_steps=50`, `num_steps=10` (flow-matching denoising), `max_action_dim=32`, and `train_expert_only=True` with `freeze_vision_encoder=True`. — [configuration_smolvla.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/configuration_smolvla.py)
- Training cost:
  - "20k steps will roughly take ~4 hrs on a single A100" (LeRobot docs). — [LeRobot SmolVLA docs](https://huggingface.co/docs/lerobot/en/smolvla)
  - About 10 h for 20k steps at batch size 64 on an RTX 3090 (community). — [ggando.com](https://ggando.com/blog/smolvla-so101/)
  - About 40% faster to train than π0 with 6× less memory (paper). — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- On-Mac training: SmolVLA 10k steps on an M4 Max (64 GB) via MPS took 2 h 23 min (27 Aug 2026). It was evaluated only on synthetic fixture episodes. — [langgraph-vla-agent](https://github.com/ajmcastro/langgraph-vla-agent)
- Inference speed on Apple Silicon and other hardware:

  | Hardware | Setup | Latency per 50-action chunk | Source |
  |---|---|---|---|
  | Apple M5 Pro | PyTorch-MPS | 204.58 ms (median) | [mlx-smolvla PyPI](https://pypi.org/project/mlx-smolvla/) |
  | Apple M5 Pro | MLX fp32 | 110.75 ms (median), 1.85× faster | [mlx-smolvla PyPI](https://pypi.org/project/mlx-smolvla/) |
  | Mac (chip not stated) | MPS, real bimanual SO-101 | 0.45 s, with RTC execution_horizon 20 | [izlley/h101-smolvla-t1-v6](https://huggingface.co/izlley/h101-smolvla-t1-v6) |
  | 6-core Intel i5 CPU | CPU only | 7.1 s median; 2.84× too slow for a 20 Hz controller | [krm.fyi, 2 Aug 2026](https://krm.fyi/blog/robotics-inference-latency.html) |
  | Jetson AGX Orin, SO-ARM101 | AdaVLA paper | about 807 ms, cut to about 488 ms by the paper's method | [AdaVLA arXiv 2608.29208](https://arxiv.org/pdf/2608.29208) (search summary only) |

  The mlx-smolvla package (v0.1.2, 5 Sep 2026) was tested only in a limited way on a Hiwonder SO-101: "reliable task completion and sustained operation remain unvalidated". — [mlx-smolvla PyPI](https://pypi.org/project/mlx-smolvla/)
- Async inference cut average task time from 13.75 s to 9.70 s and placed 19 cubes against 9 in a fixed window, at about the same success (sync 78.3%, async 73.3%). — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- The Hugging Face blog claims SmolVLA is "Small enough to run on CPU, train on a single consumer GPU, or even a MacBook!" — [SmolVLA blog](https://huggingface.co/blog/smolvla)

**ACT**
- About 80M parameters. — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- Trains on Mac MPS (about 6h40m for 30k steps on 30 episodes). — [classmethod](https://dev.classmethod.jp/articles/so-arm101-duck-pick-imitation-learning-mac-training/) (search summary only)
- Per XLeRobot, ACT "can run on a Raspberry Pi". Other policies need a different computer. — [XLeRobot LLM agent docs](https://xlerobot.readthedocs.io/en/latest/software/getting_started/LLM_agent.html)

**π0 / π0.5 (openpi and LeRobot)**
- openpi hardware table:
  - Inference: >8 GB (RTX 4090).
  - LoRA fine-tuning: >22.5 GB.
  - Full fine-tuning: >70 GB (A100/H100 80 GB).
  - "tested with Ubuntu 22.04, we do not currently support other operating systems."
  - Remote inference over a websocket (port 8000).
  - — [openpi GitHub](https://github.com/Physical-Intelligence/openpi)
- The openpi PyTorch port (Sept 2025) does not support π0-FAST, mixed precision, FSDP or LoRA. — [openpi GitHub](https://github.com/Physical-Intelligence/openpi)
- LeRobot π0.5:
  - The example recipe is "Sized for a single 80 GB GPU".
  - The default chunk is 50 and the LIBERO example uses `n_action_steps=10`.
  - Datasets need q01/q99 quantile stats, or you switch to MEAN_STD normalisation.
  - `train_expert_only=true` freezes the VLM to save memory.
  - — [LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/en/pi05)
- On a 14 GiB-RAM CPU machine, restoring the 12 GB π0 checkpoint was killed by signal 9 at about 8.75 GiB peak. — [krm.fyi](https://krm.fyi/blog/robotics-inference-latency.html)
- A third-party platform quotes 3–6 h for 30k steps on an A100/H100 80 GB, costing "about 4 to 12 USD per run" at US$1.20–2.00/h. It quotes about 485 ms latency "per action step" on its hosted platform. These are vendor claims. — [ay-robots](https://www.ay-robots.com/train/pi0-5-on-so-101)

**GR00T N1.7**
- 3B parameters, about a 6 GB download, Cosmos-Reason2-2B (Qwen3-VL) backbone. — [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- Action horizon expanded from 16 to 40, and state/action dims from 29 to 132. — [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- Hardware:
  - Inference: 1 GPU with 16 GB+ (RTX 4090, L40, H100, Jetson AGX Thor/Orin, DGX Spark).
  - Fine-tuning: 40 GB+ recommended (H100/L40).
  - Supported platforms are CUDA Linux, Jetson and Spark only. macOS is not mentioned.
  - — [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- LeRobot: "GR00T is intended for NVIDIA GPU-accelerated systems". — [LeRobot GR00T docs](https://huggingface.co/docs/lerobot/en/groot)

**MolmoAct2**
- "you need an NVIDIA GPU"; tested on Ubuntu 22.04. — [LeRobot MolmoAct2 docs](https://huggingface.co/docs/lerobot/en/molmoact2)
- Peak memory measured on an H100:
  - Inference (bs 1, CUDA graph): 12.1 GiB.
  - Action-expert-only fine-tune: 16.5 GiB at bs 8.
  - LoRA-VLM fine-tune: 20.2 GiB at bs 8.
  - Full fine-tune: 48.3 GiB at bs 8.
  - Chunk size 10 for LIBERO.
  - — [LeRobot MolmoAct2 docs](https://huggingface.co/docs/lerobot/en/molmoact2)
- About 180 ms per action call (base model) on 1 H100, or 790 ms with adaptive depth reasoning. — [Ai2 MolmoAct2 blog](https://allenai.org/blog/molmoact2)

**X-VLA**
- Defaults: `chunk_size=32` and 10 denoising steps. — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/en/xvla)
- Recommends bf16 to avoid running out of memory. — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/en/xvla)
- Tuning only 1% of parameters (9M) reaches near-π0 on LIBERO and Simpler-WidowX. — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/en/xvla)

**EVO1**
- FlashAttention is optional: "EVO1 falls back to standard attention when flash_attn is not available." Chunk size 50. — [LeRobot EVO1 docs](https://huggingface.co/docs/lerobot/en/evo1)
- The reference run used 2 H100s. — [LeRobot EVO1 docs](https://huggingface.co/docs/lerobot/en/evo1)

**OpenVLA-OFT**
- Inference needs about 16 GB VRAM (LIBERO) or about 18 GB (ALOHA). Training needs 1–8 GPUs with 27–80 GB. — [openvla-oft GitHub](https://github.com/moojink/openvla-oft)
- The authors report 25 Hz control for OFT+ on ALOHA. — [OpenVLA-OFT paper](https://arxiv.org/html/2502.19645v1) (search summary only)

**Octo**
- Runs at 13 it/s (Base) and 17 it/s (Small) on an RTX 4090. JAX-based, pretraining chunk size 4. — [Octo GitHub](https://github.com/octo-models/octo)

### Inferences
- **What fits on the M3 Pro (18 GB unified memory):**
  - ACT, Diffusion and SmolVLA fit comfortably.
  - π0/π0.5 at about 14 GB inference would leave almost nothing for macOS, the Gemini Live client, the emotion model and the Reachy daemon, and openpi does not support macOS at all.
  - GR00T, MolmoAct2 and OpenVLA-OFT all assume CUDA.
- **SmolVLA latency on the M3 Pro:** expect worse than the M5 Pro's 205 ms (MPS) and 111 ms (MLX). A rough guess is 0.3–0.5 s per 50-action chunk, consistent with the 0.45 s Mac report. A 50-action chunk at 30 fps covers about 1.67 s, so async inference or RTC with a 0.5 threshold should keep the queue full. This has not been measured on an M3 Pro.
- **Cheapest training path:** fine-tune SmolVLA on a rented A100 for about 4 h (20k steps). At the US$1.20–2.00/h A100 rate quoted by ay-robots, that is roughly US$5–8. Then run it locally on MPS.
- **Big models (π0.5, GR00T N1.7, MolmoAct2):** fine-tune on an 80 GB A100/H100 (a few hours) and run them on a rented or home CUDA box as a LeRobot `PolicyServer`. The Mac runs the `RobotClient` over the network. Network round-trip adds latency, but chunked and async execution hides much of it.
- **Training on the Mac itself:** feasible for ACT and SmolVLA (hours to an overnight run), not for the 3B-class models.

### Gaps
- I found no M3 Pro or M3-generation measurement for SmolVLA, ACT or π0 inference. The only Apple chips with numbers are the M5 Pro, the M4 Max (training only) and an unspecified Mac.
- I found no source confirming π0/π0.5 inference on MPS through LeRobot's PyTorch port, either way.
- I found no measured latency for GR00T N1.7 on any hardware. Neither the README nor the LeRobot docs give Hz or ms figures.
- Cloud GPU pricing comes from one third-party vendor page. Rental prices vary by provider and month.

## Q3. Which models take language instructions, so that Gemini Live could call "pick up the cup" as a tool that runs the policy, and how have people wired an LLM or voice agent to VLA skills?

### Takeaway
Every VLA here takes a text instruction: SmolVLA, π0/π0.5, GR00T, X-VLA, EVO1, MolmoAct2, Octo and OpenVLA-OFT. ACT and Diffusion do not, so with them each skill is a separate checkpoint. The common pattern is a hierarchy: an LLM or voice agent plans and calls skill tools, and a tool starts a LeRobot policy (often via the async policy server) with a task string and a time budget. Examples include XLeRobot's RoboCrew with Gemini, LangGraph over SmolVLA, Gemini Robotics-ER 2's Live API with blocking function calls, and Physical Intelligence's Hi Robot.

### Cited Findings

**Which models take language**
- SmolVLA's inputs are multi-view images, robot state and "a natural language instruction". At rollout you pass `--task=` and should "Use the same task description you used in your dataset recording". — [LeRobot SmolVLA docs](https://huggingface.co/docs/lerobot/en/smolvla)
- In LeRobot async inference, `--task` is "Not necessarily defined for all policies, such as `act`", meaning ACT ignores language. — [LeRobot async docs](https://huggingface.co/docs/lerobot/en/async)
- SmolVLA was evaluated multi-task (pick-place, stacking and sorting in one model). — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- π0.5 co-trains on "Verbal Instructions" and "Subtask Commands" for open-world generalisation. — [LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/en/pi05)
- openpi notes π0-DROID "may have weaker language following" than π0-FAST-DROID, and that π0.5-DROID has "good language-following". — [openpi GitHub](https://github.com/Physical-Intelligence/openpi)
- A third-party π0.5 SO-101 guide advises one task string, identical in every episode, then supplying that string at inference. — [ay-robots](https://www.ay-robots.com/train/pi0-5-on-so-101)
- GR00T N1.5 on SO-101 can be steered by task prompts from the dataset's `tasks.jsonl`, e.g. "Grab pens and place into pen holder". The GR00T N1.7 rollout example uses `--task="place the vial in the rack"`. — [HF GR00T N1.5 SO-101 blog](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning); [LeRobot GR00T docs](https://huggingface.co/docs/lerobot/en/groot)
- MolmoAct2 builds its prompt from the task plus `setup_type` (e.g. "single franka robotic arm in libero") and `control_mode` (e.g. "absolute joint pose"). — [LeRobot MolmoAct2 docs](https://huggingface.co/docs/lerobot/en/molmoact2)
- EVO1 reads the dataset `task` field as its language prompt. — [LeRobot EVO1 docs](https://huggingface.co/docs/lerobot/en/evo1)
- Octo "can be instructed via language commands or goal images", e.g. `create_tasks(texts=["pick up the spoon"])`. — [Octo GitHub](https://github.com/octo-models/octo)
- OpenVLA-OFT takes a `task_description` and has a `use_film` option. — [openvla-oft GitHub](https://github.com/moojink/openvla-oft)

**Orchestrator + VLA executor examples**
- **XLeRobot / RoboCrew** (LLM Agent Control docs):
  - Agent: `LLMAgent` with `model="google_genai:gemini-3-flash-preview"` and a `tools=[...]` list.
  - Arm skills: each is built with `create_vla_single_arm_manipulation(tool_name, tool_description, task_prompt, server_address, policy_name, policy_type, ...)`, with `execution_time` (45 s in the example).
  - "Our tool is a policy client, but all VLA computations are run on the server side". The server is LeRobot's `lerobot.async_inference.policy_server`.
  - Voice: wake word "hey robot", and `tts=True` for speech output.
  - Head: moved by a separate `create_look_around` tool, not by the VLA.
  - The example uses `policy_type="act"`.
  - — [XLeRobot LLM agent docs](https://xlerobot.readthedocs.io/en/latest/software/getting_started/LLM_agent.html)
- **langgraph-vla-agent** (2026): a LangGraph planner (understand_goal → create_plan → select_subtask → safety_check) feeds subtasks to a SmolVLA executor loop. "The LLM layer never generates joint torques, servo positions, or high-frequency trajectories." Developed on an M4 Max via MPS; a physical SO-101 is "Future work only". — [langgraph-vla-agent GitHub](https://github.com/ajmcastro/langgraph-vla-agent)
- **Google Gemini Robotics-ER 2 streaming Live API** (docs updated 11 Sep 2026):
  - Model: `gemini-robotics-er-2-streaming-preview`, with function calling.
  - "Only blocking function calls are supported when using the Live API for robotics". Physical actions use `"behavior": "BLOCKING"`.
  - The sample stub is `execute_tool(name, args)`, to "replace with real robot SDK calls". Results return via `session.send_tool_response`.
  - Output is text only, with TTS handled externally.
  - — [Gemini API: Robotics with streaming](https://ai.google.dev/gemini-api/docs/robotics-streaming)
  - A third-party summary notes the ER 2 model page lists "Live API: Not supported", which contradicts the streaming docs. — [Gemini Robotics ER 2 streaming model page](https://ai.google.dev/gemini-api/docs/models/gemini-robotics-er-2-streaming-preview) (search summary only)
- **Hi Robot** (Physical Intelligence, 26 Feb 2025):
  - A high-level VLM ("System 2") turns open-ended prompts and user interjections into low-level language commands for π0 ("System 1").
  - Reported average instruction accuracy: 76 for Hi Robot, 30 for a GPT-4o high-level planner, 36 for a flat VLA.
  - Reported task progress: 81, 64 and 44 respectively.
  - — [PI Hi Robot](https://www.pi.website/research/hirobot)
- **Other reported patterns** (search summary only):
  - An MCP server exposing SO-ARM100/101 control to Claude, Gemini and GPT agents. — [glama.ai MCP server](https://glama.ai/mcp/servers/smsj0erz66)
  - A MiniMax tutorial pairing LLM planning plus MCP visual checks with a π0.5 executor. — [MiniMax robot-agent docs](https://platform.minimax.io/docs/solutions/robot-agent)

### Inferences
- **A workable wiring for this project:** Gemini 3.8 Live exposes a small set of function declarations, e.g. `run_arm_skill(skill: enum, object?: str)`, `look_at(target)` and `stop()`. The tool handler:
  1. Maps the call to a task string identical to the dataset's (VLAs are brittle to paraphrase when fine-tuned on one string per task).
  2. Starts a LeRobot `RobotClient`, or keeps one running and swaps the `task`.
  3. Runs under a timeout or success check.
  4. Returns a result to Gemini.

  This is the RoboCrew / Hi Robot pattern, with Gemini Live in the planner role.
- **Tool-call style:** blocking or long-running calls suit physical actions. The Gemini Robotics streaming docs require BLOCKING, which suggests Gemini Live should await the skill's completion rather than fire-and-forget.
- **SmolVLA versus ACT for this pattern:** a multi-task SmolVLA can serve several skills from one checkpoint (one language string per skill). With ACT the tool would have to load or switch checkpoints per skill.
- **Success detection:** the policies themselves do not report done or failed. A time budget, a gripper-state heuristic, or a VLM check (Gemini looking at the camera) is needed. This is consistent with langgraph-vla-agent's verify step.

### Gaps
- I found no public project that wires Gemini Live (consumer or Live API, not Robotics-ER) directly to a LeRobot policy on an SO-101. The XLeRobot example uses Gemini 3 Flash via LangChain, not the Live audio API.
- I found no measurements of how well fine-tuned SmolVLA follows paraphrased instructions it was not trained on.

## Q4. Can any of these models control extra degrees of freedom such as a head or neck, through multi-embodiment action spaces?

### Takeaway
In principle, yes. All the flow-matching VLAs pad state and action to a fixed maximum: SmolVLA 32, π0.5 32, MolmoAct2 32, EVO1 24, X-VLA 20, GR00T N1.7 up to 132. So an SO-101 (6 joints) plus Reachy Mini head, body and antennas (about 9 values) would fit. GR00T (embodiment tags), X-VLA (soft prompts and domain IDs) and π0.5 (cross-embodiment co-training) are designed for new bodies. But no public source shows any VLA controlling a Reachy Mini, and LeRobot has no Reachy Mini robot class. Joint arm-and-head control would need a custom combined LeRobot robot, head teleoperation during recording, and your own data.

### Cited Findings
- **SmolVLA:** `max_state_dim=32` and `max_action_dim=32`. — [configuration_smolvla.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/configuration_smolvla.py)
- **π0.5:**
  - LIBERO's 7-D action is "padded to 32 internally; the loss uses the first 7 dims". — [LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/en/pi05)
  - Pretraining includes "Cross-Embodiment Robot Data" and "Mobile Manipulation Data". — [LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/en/pi05)
- **GR00T N1.7:**
  - State/action dimensions expanded "from 29 to 132".
  - Custom robots use the `NEW_EMBODIMENT` tag; named tags include `UNITREE_G1` and `LIBERO_PANDA`.
  - — [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
  - LeRobot uses `new_embodiment` for SO-101. — [LeRobot GR00T docs](https://huggingface.co/docs/lerobot/en/groot)
  - GR00T N1.5 SO-101 was trained "as a new embodiment" because SO-100/101 data was not in pretraining. — [HF GR00T N1.5 SO-101 blog](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- **X-VLA:**
  - "treats each hardware setup as a 'task'" and encodes it with soft prompts.
  - `action_mode=auto` reads the dataset action dimension and pads it to `max_action_dim=20`.
  - Custom action spaces can be registered with `@register_action`.
  - `so101_bimanual` maps the 20-D model output to a 12-D robot.
  - — [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/en/xvla)
- **EVO1:** "pads robot state/action vectors to fixed maximum dimensions", default 24. — [LeRobot EVO1 docs](https://huggingface.co/docs/lerobot/en/evo1)
- **MolmoAct2:** released checkpoints use `expected_max_action_dim=32`. — [LeRobot MolmoAct2 docs](https://huggingface.co/docs/lerobot/en/molmoact2)
- **Reachy Mini hardware:**
  - A 6-DOF head, two antennas, a body that rotates about the vertical axis, a wide-angle camera, 4 microphones, and no arms. — [Robots Guide: Reachy Mini](https://robotsguide.com/robots/reachy-mini) (search summary only)
  - The SDK sends head poses with `goto_target(...)`, built with `create_head_pose` and a duration. The Lite version "Runs on your PC" over USB. Software license is Apache 2.0. — [pollen-robotics/reachy_mini GitHub](https://github.com/pollen-robotics/reachy_mini)
  - A Hugging Face FAQ space says the SDK's `start_recording()` / `stop_recording()` capture all `set_target` calls, which unpack into time, head, antenna and body-yaw values. Tools to record and upload datasets are in `reachy_mini_toolbox/tools/moves`. — [Reachy Mini FAQ space](https://anne-charlotte-reachy-mini-faq.static.hf.space/index.html) (search summary only; page returned 401 when fetched)
- **LeRobot hardware list:** includes "Reachy2" but not Reachy Mini. — [LeRobot GitHub README](https://github.com/huggingface/lerobot); [PyPI lerobot](https://pypi.org/project/lerobot/)
- **XLeRobot's split:** head and camera motion is a separate scripted LLM tool (`create_look_around`), while VLA policies drive only the arm. — [XLeRobot LLM agent docs](https://xlerobot.readthedocs.io/en/latest/software/getting_started/LLM_agent.html)

### Inferences
- **Technically feasible path:**
  1. Write a LeRobot `Robot` subclass that combines the SO-101 follower (6 joint targets) with Reachy Mini head pose, body yaw and antennas via the daemon/SDK.
  2. Record demos where the head moves purposefully, for example by teleoperating the head from a phone or with scripted gaze while the leader arm drives the follower.
  3. Fine-tune SmolVLA, π0.5 or GR00T on the combined ~15-D action. The SO-101 wrist camera and/or the Reachy head camera would be the image inputs.
- **Caveat:** the Reachy head camera moves with the head. A policy that controls the head also controls its own viewpoint, which makes learning harder.
- **Lower-risk design:** keep the head out of the VLA and drive it with scripted or LLM tools (look-at-object, emote), as XLeRobot does. Use the head camera only as a fixed-pose observation while an arm skill runs.
- **Ranking for new bodies:** GR00T N1.7 has the largest action space (132) and an explicit embodiment mechanism, but needs CUDA. SmolVLA's 32-D padding is enough and runs on the Mac.

### Gaps
- I found no report of any VLA, or of ACT, being trained on Reachy Mini head data, or on a combined arm-plus-head embodiment, with a desktop arm.
- I found no official LeRobot `reachy_mini` robot type or teleoperator, nor documentation of the head-pose representation (Euler, matrix or Stewart actuator space) suitable for a LeRobot dataset.
- How well these models generalise to added non-manipulation DoF such as gaze is not covered by any source found.

## Q5. Licensing terms

### Takeaway
- **Permissive (Apache-2.0 or MIT):** SmolVLA, LeRobot code, openpi π0/π0.5 code and weights (with a Gemma licence file and the gated PaliGemma tokenizer), Octo (MIT), OpenVLA-OFT code (MIT) and the Reachy Mini SDK.
- **MolmoAct2:** Apache 2.0, plus Ai2's research and educational responsible-use guidelines.
- **GR00T N1.7 weights:** NVIDIA Open Model License. Commercial use and derivatives are allowed, but there are guardrail and litigation termination clauses and attribution rules.
- **Newer Physical Intelligence models** (π\*0.6, π0.7, MEM): not open-weight. The October 2026 Pi API beta is hosted.

### Cited Findings
- **SmolVLA:** `lerobot/smolvla_base` is Apache-2.0. — [HF model card smolvla_base](https://huggingface.co/lerobot/smolvla_base)
- **π0 / π0.5:**
  - The openpi repo is Apache-2.0 and its file listing includes `LICENSE_GEMMA.txt`. — [openpi GitHub](https://github.com/Physical-Intelligence/openpi)
  - LeRobot's π0.5 page says "This model follows the Apache 2.0 License". It uses the gated `google/paligemma-3b-pt-224` tokenizer: "accept its license on the Hub". — [LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/en/pi05)
- **Newer Physical Intelligence models:**
  - The PI site lists π\*0.6 (17 Nov 2025), MEM (3 Mar 2026), π0.7 (16 Apr 2026), and "Frontier Models for Every Robot" (13 Oct 2026, Pi API beta: data upload, fine-tuning and cloud-hosted inference). — [Physical Intelligence site](https://www.pi.website/)
  - The only open-source post on the site is "Open Sourcing π0" (4 Feb 2025); π0.5 weights came via openpi in Sept 2025. — [Physical Intelligence site](https://www.pi.website/); [openpi GitHub](https://github.com/Physical-Intelligence/openpi)
  - A third-party "OpenPIE-0.6" reimplementation exists but is unverified. — [exla-ai/openpie-0.6](https://huggingface.co/exla-ai/openpie-0.6) (search summary only)
- **GR00T N1.7:**
  - Code is Apache 2.0 and weights use the NVIDIA Open Model License. The same README also says N1.7 is "fully commercially licensable under Apache 2.0", an internal inconsistency. — [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
  - LeRobot: "GR00T N1.7 is released under the NVIDIA Open Model License Agreement". — [LeRobot GR00T docs](https://huggingface.co/docs/lerobot/en/groot)
- **NVIDIA Open Model License** (version dated 24 Oct 2025):
  - "Models are commercially usable". You may create and distribute derivative models, and NVIDIA "claims no ownership rights in outputs".
  - Distribution must include the licence text and a "Licensed by NVIDIA Corporation under the NVIDIA Open Model License" notice.
  - Rights "will automatically terminate" if you bypass safety guardrails without a substantially similar replacement, or if you file copyright or patent litigation over the model.
  - The licence is revocable and NVIDIA may update it.
  - — [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
- **GR00T N1.7 status and timing:**
  - The README excerpt in one search result calls the release "Early Access (EA)" with limited support guarantees until GA. A later fetch of the GitHub page described it as "General Availability". This is a conflict, or a status change over time. — [Isaac-GR00T README raw](https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/main/README.md) (search summary); [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
  - A third-party blog dates the N1.7 unveiling to GTC, March 2026. — [agentmarketcap.ai](https://agentmarketcap.ai/blog/2026/04/10/nvidia-groot-n1-7-cosmos-reason-2-gtc-2026-physical-ai) (search summary only)
- **MolmoAct2:** "licensed under Apache 2.0. It is intended for research and educational use in accordance with Ai2's Responsible Use Guidelines". — [LeRobot MolmoAct2 docs](https://huggingface.co/docs/lerobot/en/molmoact2)
- **EVO1:** the LeRobot integration "follows the Apache 2.0 License"; check the upstream EVO1 and InternVL3 pages for checkpoint licences. — [LeRobot EVO1 docs](https://huggingface.co/docs/lerobot/en/evo1)
- **OpenVLA-OFT:** code is MIT. The README states no licence for the model weights. — [openvla-oft GitHub](https://github.com/moojink/openvla-oft)
- **Octo:** MIT. — [Octo GitHub](https://github.com/octo-models/octo)
- **RDT-1B:** MIT according to a third-party listing. — [PromptLayer RDT-1B](https://www.promptlayer.com/models/rdt-1b) (search summary only)
- **Reachy Mini:** software is Apache 2.0; hardware design files are CC BY-SA-NC. — [pollen-robotics/reachy_mini GitHub](https://github.com/pollen-robotics/reachy_mini)
- **Third-party Apple Silicon port:** mlx-smolvla is Apache-2.0. — [mlx-smolvla PyPI](https://pypi.org/project/mlx-smolvla/)

### Inferences
- For a hobbyist, non-commercial project, none of these licences is a blocker.
- SmolVLA plus LeRobot is the cleanest stack (Apache-2.0 end to end).
- GR00T's NVIDIA licence adds attribution and a guardrail clause, and is revocable, which matters only if the project is distributed or commercialised.
- π0/π0.5 weights carry Gemma/PaliGemma-derived terms alongside Apache-2.0 code. Read `LICENSE_GEMMA.txt` before redistributing a fine-tune.

### Gaps
- I found no licence statement for X-VLA weights, EO-1, WALL-OSS or the OpenVLA-OFT checkpoints. The OpenVLA base model's Llama-2 heritage, and the terms that implies, was not verified here.
- I did not read the contents of `LICENSE_GEMMA.txt` in openpi, so the exact downstream obligations for π0/π0.5 fine-tunes are unconfirmed.
- I did not resolve whether the GR00T N1.7 "Early Access" label changed to "General Availability", or when.
