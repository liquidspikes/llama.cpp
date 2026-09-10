# Dual-node USB4STREAM tensor-parallel Flash-Next (BosGame gfx1151)

This is the working log for serving **Qwen3.8-Flash-Next-Uncensored-Q4_K_M** as **true tensor parallelism** over Thunderbolt 4 USB4STREAM between two Strix Halo boxes. It is not a generic multi-GPU guide; see [multi-gpu.md](multi-gpu.md) for stock `-sm tensor`. This cluster uses **`-sm rpc-tensor`** over `/dev/tbstream0,/dev/tbstream1`, not TCP RPC.

Tree: `feature/strix-halo-optimization` @ `5c6160cfb` plus the USB4 / Meta / GDN diffs in this working tree. Origin: `git@github.com:liquidspikes/llama.cpp.git`.

Scratch benches live under `/tmp/grok-goal-f7d364922603/implementer/` on bosgame1. Numbers below are copied from those files, not invented.

## What “working” means

| Gate | Pass |
|---|---|
| Split | `-sm rpc-tensor` (not `layer`). Speed path uses `LLAMA_TP_MIRROR_GDN=1 LLAMA_TP_MIRROR_DENSE=1` (experts split). `LLAMA_MIRROR_OUTPUT_WEIGHT=1` stays. |
| Transport | `--stream /dev/tbstream0,/dev/tbstream1` |
| Memory | both GPUs hold **~41 GiB GTT** (not ~84 GiB full replica) |
| Quality | prompt `17 times 19` → **`message.content` contains `323`** and English |
| Speed | Target: `predicted_per_second` **> 19.816** (`B_single`). Best true-TP **323** decode: **17.46 / 17.48 tok/s** (pid 284828 / recapture pid 287196, GRAPH_SEQ last-node xchg + HIP graphs). **Does not beat `B_single`.** GDN-split (no `MIRROR_GDN`) hit **18.42 tok/s** but **garbage**. USB4 + mirrored GDN floor ~57.2 ms/token vs 50.5 ms. Simplex `tbs_xchg` was 1:1 frame ping-pong (default `TBS_XCHG_WINDOW=1` on the client); concurrent windowed xchg is the next lever. |
| GPUs | both `gpu_busy` non-zero during decode (`both-gpus.txt`) |

Not a pass:

- Full replica (`n_subgraphs=1`, ~84 GiB GTT each, ~20 tok/s, content 323). That is two copies of the model, not TP.
- `323` only in `reasoning_content`.
- ~500 tok/s of 64× `'3'` (graph collapse, `n_nodes=0`).
- Layer-split relabeled as tensor parallel.

## Hardware

| | bosgame1 | bosgame2 |
|---|---|---|
| Role | Meta client, `TBSTRIPE_SIMPLEX=master` | RPC worker, `TBSTRIPE_SIMPLEX=worker` |
| IP | 192.168.137.51 | 192.168.137.52 |
| GPU | AMD Radeon 8060S, gfx1151 | same |
| UMA | ~124 GiB OS RAM; VRAM advertised 1 GiB / 512 MiB; GTT ~120 GiB | same |
| HIP | 7.15.26333, `/opt/rocm/core-10.0` | matching `libggml-hip` |
| Devices | `ROCm0` + `RPC0` | `ROCm0` via `rpc-server` |
| Pipe | `/dev/tbstream0` TX, `/dev/tbstream1` RX (master) | swapped TX/RX (worker) |

USB4STREAM (`tbstripe`) is a character-device DMA path, not a network stack. Frame size is 4096 bytes, magic `TBSTRIPE`. Simplex: master TX=`/dev/tbstream0` RX=`/dev/tbstream1`. Do **not** unplug the cables.

`CLOSE` of `/dev/tbstream*` is **EOF**. After any STREAM client exits, bounce `llama-rpc` before the next STREAM client. `TimeoutStopSec=3` SIGKILLs lemonade — leave lemonade inactive.

ggml soname on this tree: `0.23.0`.

## Bounce and launch (do not improvise)

Never `pkill -f`. `pgrep llama-server.real` hits the 15-char `comm`.

```bash
# 1. PID-kill leftover client
ps -eo pid,comm | grep llama-server
kill <pid>   # SIGKILL only if it ignores SIGTERM

# 2. Restart worker (bosgame2)
ssh bosgame2 'sudo systemctl restart llama-rpc && systemctl is-active llama-rpc'

# 3. Start client (bosgame1)
stdbuf -oL -eL env \
  TBSTRIPE_SIMPLEX=master \
  GGML_RPC_SET_TENSOR_CHUNK=2048 \
  LLAMA_MIRROR_OUTPUT_WEIGHT=1 \
  LLAMA_TP_MIRROR_GDN=1 \
  LLAMA_TP_MIRROR_DENSE=1 \
  GGML_CUDA_DISABLE_FUSION=1 \
  TBS_XCHG_WINDOW=4 \
  LD_LIBRARY_PATH=/home/alexzimmerman/llama.cpp/build/bin:/usr/local/lib:/opt/rocm/core-10.0/lib \
  /home/alexzimmerman/llama.cpp/build/bin/llama-server \
    --stream /dev/tbstream0,/dev/tbstream1 \
    -dev ROCm0,RPC0 -sm rpc-tensor -ts 1,1 \
    -m /home/alexzimmerman/models/qwen3.8-flash-next/Qwen3.8-Flash-Next-Uncensored-Q4_K_M-00001-of-00003.gguf \
    --port 8080 --temp 0 --jinja --top-k 20 --top-p 0.95 \
    --no-mmproj --no-warmup \
    --ctx-size 512 -b 8 -ub 8 -fa on -ngl 99
```

`LD_LIBRARY_PATH` **must** put `llama.cpp/build/bin` first. `/usr/local/lib/libggml-hip.so` on bosgame1 has been stale (`9e535b…` vs build `e03fcf…`).

HIP rebuild ⇒ matching `libggml-hip` on **both** nodes. Copy `libggml-rpc` / `libggml-base` to bosgame2 `/usr/local/lib` **and** `llama.cpp/build/bin` as **two separate `cp`s**. Never two destinations in one `cp`.

Worker unit (bosgame2): `ExecStart=/usr/local/bin/rpc-server -s /dev/tbstream0,/dev/tbstream1 -d ROCm0 --cache-dir /home/alexzimmerman/models/cache`, `LD_LIBRARY_PATH=/usr/local/lib`, `TBSTRIPE_SIMPLEX=worker`, `TimeoutStopSec=3`. RPC tensor cache is keyed by **payload hash**, so a new packed shard misses and is fine; a same-bytes shard hits and is also fine.

DFlash is 27B-only. Flash-Next serving is rpc-tensor. MTP / draft-mtp off. Do not apply `scratch/patch_*.py`.

Health: `{scratch}/watch-tp-health.sh <pid> <log>` until HTTP 200 and remote GTT > 8 GiB. Bench: `{scratch}/bench-tp.sh <out.json>` (prompt `17 times 19`, `max_tokens=64`). Remote GTT: `/sys/class/drm/card1/device/mem_info_gtt_used` on bosgame2.

## Model packing (Qwen 3.5 / QWEN4EXP)

Flash-Next linear-attn layers (`llama-model.cpp` `get_split_segments`):

| Tensor | Segments | Physical meaning on 2 devices |
|---|---|---|
| `attn_qkv`, `ssm_conv1d` | `{{key_dim, 2+head_ratio}}` = `{{2048,5}}` | Q, K, V0, V1, V2. Device 0 gets K-heads 0–7 of each group |
| `ssm_out`, `attn_gate` | `{{key_dim, head_ratio}}` = `{{2048,3}}` | V-order heads **0–7, 16–23, 32–39** on device 0 |
| `ssm_dt/a/alpha/beta` | `{{n_k_heads, head_ratio}}` | same V-order at 1-per-head |
| `cache_s` | `{{n_k_heads * S_v * S_v, head_ratio}}` | state heads in V-order |

Shapes: `n_embd=2560`, 48 layers, 512 experts / 10 used, `n_ff_exp=640`, `hc_mult=4`, `ssm_d_state=128`, `n_k_heads=16`, `n_v_heads=48`, `head_ratio=3`, `key_dim=2048`, `value_dim=6144`. `attn_qkv` is `[2560,10240]` Q6_K (AXIS_1). `ssm_out` is `[6144,2560]` Q4_K (AXIS_0).

Qwen 3 Next packing is different (`{{key_dim,2},{value_dim,1}}`). Do not use that pattern on Flash-Next.

Meta **labels** (`n_segments`/`nr`) describe that interleave. HIP kernels **ignore `nr`** and use the physical buffer. `simple_tensor` COMPUTE `ne = sum(ne*nr)`, so 50/50 sizes match either packing; **layout** still has to match between W and x.

HIP GDN: `H = V.ne[1]`, `iq1 = h_idx % neqk1` (Qwen 3.5 tiled Q/K broadcast). Do not change that modulo. Fused GDN keeps Q/K at 16 k-heads and V at 48 v-heads.

## Proven (do not re-do)

| Item | Evidence |
|---|---|
| Single-node baseline | **19.816 tok/s**, content `323`, `B_single.txt` + `bench-single.json` |
| Tiny default STREAM NMSE | Meta **OK 1.97e-14** (`nmse-tiny-usb4.txt`). 2-layer F16, one-shot decode, not AR GDN state |
| Tiny `LLAMA_TP_FIXTURE=fnlike` STREAM | Meta **OK 1.24e-13** (`nmse-fnlike.txt`). 16 k-heads / 48 v-heads, F16 |
| `LLAMA_TP_MIRROR_ALL` USB4 | tiny Meta **OK 1.86e-14** — interconnect AllReduce math is not a 1e2-class failure |
| Dense 27B Q4_K TP | coherent English (pipe is numerically usable for GQA `wo` inner-K AR) |
| Flash-Next rpc-tensor load | ~9 s cached; both GTT ~40 GiB when split, ~84 GiB when fully mirrored |
| USB4STREAM tiny | STREAM NMSE ~1e-14 after fill-fix + HC-only VIEW rebase |
| qkv SET 5-rep | `[SET_NR_CHK] dest0[1]==K0 1 dest0[1]==Q1 0` on blk.0 and blk.1 |
| hipBLAS vs MMQ | first `[GDN_ACT]` line **identical** — mmq N-order is not scrambling vs hipBLAS |
| `ggml_cont` Q/K/V | vnb 20480→12288, `vcont=1`, **identical** garbage — V strides were already correct |

Isolation on 48-layer Q4_K:

| Config | Result |
|---|---|
| `MIRROR_GDN` only | coherent English, 170+153 in reasoning, **empty content** at 64 tokens, **1.69 tok/s**, GTT ~44 GiB. Experts+dense split are not the garble source. `max_tokens=128` still empty content (`bench-tp-mirrorgdn-128.json`). |
| `MIRROR_GDN` + split `ssm_out` | garbage `umpsumps…` at **0.965 tok/s** (`bench-tp-ssmout.json`). 3-rep W ↛ sequential GDN x. |
| all-split | garbage ~0.99 tok/s, `n_subgraphs=97`, GTT ~43 GiB |
| expert-only Q4_K (`MIRROR_DENSE+MIRROR_GDN`) | **323 in reasoning**, empty/truncated content, ~1.86 tok/s |
| always-repeat Q/K 8→24 | garbage **changed** to `'The problem,…'` then 17-token stop; **reverted** |
| `LLAMA_TP_HIPBLAS_Q=1` | different garbage string, still not 323 |

Keep in tree: HC-only `create_node` VIEW rebase (`ggml-rpc.cpp`); Meta fill nullptr-only (do not 16-byte-check host `cgraph` hash.used); HIP strided unary + GDN `pack_contiguous_f32`; GDN cache fusion exemption from `GGML_CUDA_DISABLE_FUSION`; GDN_HIP/GDN_ACT/SET_NR/GDN_OFF logs; nr-preserving MUL_MAT/SSM_CONV/VIEW; `ggml_cont` Q/K/V; TRI/DIAG/SOLVE_TRI/SET split preserve; ADD(TRI, DIAG identity) keeps TRI; AXIS_0×MIRRORED → PARTIAL; GDN state-view **AXIS_2**; GDN fill from **V** (`src_ss[2]`); SET_NR_CHK.

## Failed experiments — do not retry

- Expert AXIS_2 (HIP `quantize_mmq_q8_1` fault).
- Even-FFN 320/320.
- `is_dsv4` / shexp-only TP.
- CONT / `sum_rows` / `repeat_back` of the HC mix.
- Rebasing **every** VIEW (HSA-faulted `rope_multi`). Only HC-stream views.
- `TBS_XCHG_WINDOW=total` (deadlocked). Do not restore ACK-before-xchg.
- Treating replica or layer-split as the TP solution.
- Applying `scratch/patch_*.py`.
- Always-repeat Q/K under fused GDN (logits changed, still not 323).
- Claiming tiny F16 NMSE ⇒ 48-layer Q4_K decode is correct. Prefill logits use init state; AR 64-token `17×19` rereads `cache_s`.

## GDN state vs GDN activation (two different bugs)

**State-view scramble (fixed in tree).** `handle_reshape` of AXIS_0 packed `S_v*H` landed on inner `S_v` (AXIS_1, shards 64/64) because `128*128 > S_v*H`. HIP writes `[S_v, S_v, H_local]`. Size matched (`128*64*48 == 128*128*24`) so no OOB — silent layout scramble of `cache_s`. Fix: `ggml_backend_meta_is_gdn_state_view` + `gdn_state_view_split` return **AXIS_2** (4D) or AXIS_0 packed `D=S_v*S_v*H` with V’s nr. Wrapper exports `GGML_CUDA_DISABLE_FUSION=1`; GDN cache fusion is exempted so fused writes can still hit `cache_s` directly, but 1a is still required when GDN and CPY land in different Meta subgraphs.

**Activation vs `ssm_out` (the remaining quality bug).** Inner-K `ssm_out` is 3-rep V-order. A sequential 3072-wide GDN x (heads 0–23) dotted with 3-rep W (heads 0–7, 16–23, 32–39) is garbage. Complementary correct inner-K is **3-rep W · 3-rep x**. `MIRROR_GDN` + split `ssm_out` is the isolation that proves this.

Two physical-layout suspects, in order:

1. **Worker `ssm_out` SET was 2560 strided RPC copies.** RPC `set_tensor_2d` is NULL. Q6_K AXIS_1 qkv uses `n_copies=1` (one blob) — SET_NR_CHK proved dest0 is 5-rep. Q4_K AXIS_0 `ssm_out` used `n_copies=2560`. Local HIP has `set_2d`; worker may not match. **Host-pack** (this tree): concatenate each device’s nr-rep K slices on the host, one contiguous `ggml_backend_tensor_set` per device. `[SET_NR_CHK]` on `ssm_out` compares dest0 group1 to source `K[2048:3072]` (3-rep) vs `K[1024:2048]` (first-half).

2. **If host-pack dest0 is 3-rep and decode is still garbage:** GDN/conv activations are sequential, not V-order. Remap GDN attn (and SSM_CONV output if needed) to the same 3-rep V-order the weight SET uses. Do not flatten `ssm_out` to first-half (that desyncs qkv 5-rep). Do not change `h_idx % neqk1`.

## GRAPH_RECOMPUTE (speed, only after content 323)

Skip-rebuild `graph_sig` is already in `ggml-backend-meta.cpp`. RPC still reserializes because:

- Client `last_graph_uid` is **one** slot.
- Worker `stored_graphs[device]` is **one** graph.
- `RPC_CMD_GRAPH_RECOMPUTE` has **no uid**.

With 49–96 subgraphs, every subgraph but the last is a full USB4 `GRAPH_COMPUTE`. Replica is fast because `n_subgraphs=1` actually hits RECOMPUTE. That is not a quality substitute.

Fix after content 323: uid on `rpc_msg_graph_recompute_req`; worker uid-keyed ring ~128 graphs/device; client RECOMPUTE if this subgraph uid is live. Matching `libggml-rpc` both nodes.

Physics: 19.816 tok/s ≈ 50.5 ms/token. 96 serial USB4 AllReduces can still floor below that after the ring. If they do, **stop and report**. Do not relabel layer-split or replica as TP.

## Strix Halo forks vs this cluster

This tree is already `feature/strix-halo-optimization`, **not** stock ggml-org/llama.cpp. Single-node `B_single` is **19.816 tok/s** here. The ~0.99 tok/s all-split number is a **layout/AllReduce correctness** failure, not “official llama.cpp is 50% of theory.”

| Source | What it is | Why it is not a swap for this serving path |
|---|---|---|
| [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) | gfx1151-specialized llama.cpp (halo-box also keeps a close-to-mainline fork) | No USB4STREAM rpc-tensor. Switching drops the interconnect this cluster exists for. |
| [Gaetan Puleo pull-18 results](https://gaetan-puleo.github.io/strix-halo-pull-18-results/) / r/StrixHalo | ROCm MoE prefill; community fork with Nathan/Laurent; `-ub 2048` MoE / `-ub 512` dense | Prefill kernels. This cluster is blocked on **decode quality + USB4 AR count**. |
| [peonist-ai/halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server) | Qwen3.8-Flash-Next **only**, Ninfer-style kernels, claimed ~90% theory, 5.53 bpw, single GPU, podman `:8731` | Different runtime, different weights, **no dual-node TP**, no GGUF Q4_K_M USB4 path. |

After content 323, if tok/s is still USB4-AR-bound, report the floor. Cherry-picking HIP mmq/FA from halo-box is a later PR: matching `libggml-hip` both nodes, must not regress 323.

## Tonight’s run (2026-09-09)

HIP left at md5 `e03fcf630ee87717e278ae297d0d154d` both nodes. Removed bosgame2 `llama-rpc.service.d/hipblas-q.conf`. `B_single` remains **19.816 tok/s**, content `323` (`bench-single.json`).

### Host-pack SET (both devices)

`ggml-backend-meta.cpp` now host-packs AXIS_0 (`ssm_out` Q4_K) and AXIS_1 (`attn_qkv` Q6_K) into one contiguous `ggml_backend_tensor_set` per device. RPC `set_tensor_2d` is NULL.

`[SET_NR_CHK]` after the pack (`tp-axis1pack.log`):

```
blk.0.attn_qkv.weight dest0[1]==K0 1 dest0[1]==Q1 0 packed_eqK 1
blk.0.attn_qkv.weight dest1[1]==K1 1 dest1[1]==Q1 0 packed_eqK 1
blk.0.ssm_out.weight dest0[1]==3rep 1 dest0[1]==half 0 packed_eq3=1
blk.0.ssm_out.weight dest1[1]==3rep 1 dest1[1]==half 0 packed_eq3=1
```

Worker qkv/ssm_out shards **are** 5-rep / 3-rep. Unmirrored all-split decode was **identical** to the pre-host-pack garbage (`The first,\n,\n    15.` at ~0.99–1.11 tok/s). Weight SET packing is not the remaining quality bug.

### All-split unmirrored (no MIRROR_*)

| | |
|---|---|
| Client pid | 197331 / 194955 |
| `n_subgraphs` | **97** |
| GTT | local 42901069824 (~40.0 GiB), remote 43026882560 (~40.1 GiB) |
| Decode GPUs | bosgame1 `gpu_busy_percent=4`, bosgame2 `=3` (`both-gpus.txt`) |
| `bench-tp.json` | empty `content`, garbage `reasoning_content`, **0.975–1.11 tok/s**, `has_323=false` |

Both GPUs hold split weights and are non-idle during decode. Text is not 323.

### 3-rep pack of mirrored GDN `x` + split `ssm_out`

`LLAMA_TP_MIRROR_GDN=1 LLAMA_TP_SPLIT_SSM_OUT=1` (the isolation that previously emitted `umpsumps` at 0.965 tok/s).

Graph: reshape sequential `[6144,T]` → `[2048,3,T]`, Meta split each 2048 as 1024/1024 **keeping group stride 8192 bytes**, `view_offs = j * 1024 * 4`, then `ggml_cont`. Log (`tp-3rep3.log`):

```
[GDN_3REP] name=final_output_3rep-0 ne=[2048,3] j0=1024 j1=1024
[GDN_3REP_VIEW] j=0 ne0=1024 offs=0 nb1=8192 pne0=6144
[GDN_3REP_VIEW] j=1 ne0=1024 offs=4096 nb1=8192 pne0=6144
```

| | |
|---|---|
| Client pid | **201367** |
| `n_subgraphs` | **97** |
| GTT | local 43628781568 (~40.6 GiB), remote 43754594304 (~40.8 GiB) |
| Decode GPUs | both `gpu_busy_percent=4` (`both-gpus.txt`) |
| `bench-tp.json` (64 tok, `17 times 19`) | empty `content`; **English** `reasoning_content`: “The user wants me to compute 17 × 19…” then loops `17 × 19 = 17 × 19 =`; **0.953 tok/s** |
| `max_tokens=256` | still empty `content`, still loops, never writes 170+153 or 323; **0.965 tok/s** |

This pack **turns `umpsumps` into coherent English**. It is not bit-exact vs fully mirrored `ssm_out` (that isolation computed `170 + 153` then looped). Arithmetic does not finish, so **content 323 is not met**. GDN weights are still mirrored on this launch; experts+dense+`ssm_out` are split.

### Speed

~0.95 tok/s vs `B_single` 19.816. 97 subgraphs, RPC `GRAPH_RECOMPUTE` still one-graph. Do not treat this as a kernel-theory problem. Strix Halo forks (halo-box, Gaetan Puleo pull-18, halogen-flash-server) are single-node prefill/decode kernels and do not replace USB4STREAM rpc-tensor.

### K-matching inner-K (later the same night)

Slicing mirrored GDN `x` so `W.K` equals `x.K` (either sequential halves `[3072,2,T]` or 3-rep `[1024,2,3,T]`) is required. Without a slice, HIP sees `W.K=3072` vs `x.K=6144` and quality collapses (`5555…`).

With a K-matched slice + split `ssm_out` + `MIRROR_GDN`:

| Slice | `reasoning_content` | 323 |
|---|---|---|
| 3-rep `[1024,2,3]` | English, loops `17 × 19 = 17 × 19 =` | no |
| sequential `[3072,2]` | same English loop | no |
| no slice, sequential W | `545555…` collapse | no |
| `ssm_out` fully mirrored (no inner-K) | `170 + 153` then loop | no content |

So **K-matching is necessary but not sufficient**. Fully mirrored `ssm_out` still does the arithmetic (`170+153`); split inner-K AllReduce of the 2560-wide `linear_attn_out` does not. Butterfly `LLAMA_TP_AR_FALLBACK=1` hung the 128-token POST (200s, 0 bytes). Keep `tbs_xchg` AllReduce.

View log for sequential halves (`tp-khalf.log`):

```
[GDN_KHALF] name=final_output_khalf-0 ne0=3072
[GDN_3REP_VIEW] j=0 ne=[3072,1,2] offs=0 nb1=12288 nb2=24576 pne0=6144
[GDN_3REP_VIEW] j=1 ne=[3072,1,2] offs=12288 nb1=12288 nb2=24576 pne0=6144
```

Both GPUs remain ~4% busy, GTT ~40.6 GiB, `n_subgraphs=97`, ~0.95 tok/s.

### INNERK in Meta + hipBLAS (pid 210450)

`LLAMA_TP_SSM_OUT_SEQUENTIAL=1` now slices `x` in Meta `create_node` (`[INNERK] linear_attn_out j=0 k0=0 wK=3072 / j=1 k0=3072 xnb1=24576`) instead of a graph `khalf` reshape. Same English loop as graph khalf.

`LLAMA_TP_HIPBLAS_Q=1` on that same pid (mmq skipped for Q4_K/Q6_K): **identical** English loop, empty content, **0.972 tok/s**, both `gpu_busy_percent=4`, GTT ~40.6 GiB, `n_subgraphs=97` (`bench-tp-hipblas.json`). Inner-K AllReduce vs full GEMM is **not** an mmq N-order issue.

### N-split `ssm_out` (AXIS_1, full K)

`LLAMA_TP_SSM_OUT_NSPLIT=1` splits `ssm_out` on **N** (`ne[1]=2560` → 1280/1280). CPU dequant+numpy (`test-ssm-out-split.py`): K-split+sum NMSE **1.8e-12**, N-split concat NMSE **0** vs full GEMM. Weight SET packing is correct:

```
[SET_AXIS1_CHK] blk.0.ssm_out.weight dest0==src0 1 dest1==srcN/2 1 dest1==src1 0 col=3456 N=2560
```

HIP must GEMM with `dst.ne[0]==W.ne[1]==1280` (contiguous). Writing into a 2560-wide dest (view, data-offset, or scatter of a 2560 GEMM) is **CJK garbage**. Current path: shrink dst to 1280, compute, host-allgather concat, skip AllReduce (`[INNERN_AG] wN=1280 nT=2 nbytes=10240`).

| Launch | Result | tok/s |
|---|---|---|
| extra `innern_gemm` view / data-offset / scatter of 2560 dest | CJK (`prop “ magn快MF…` / `programm工蚣…`) | 0.97–1.38 |
| shrink to 1280 + host allgather (pid **222449**, mixed mmq/hipBLAS) | **English loop** (same as INNERK) | 1.065 |
| same, mmq both GPUs (pid **224383**, worker `hipblas-q.conf` removed) | **English loop** | 0.948 |

Both GPUs busy ~4–5%, GTT ~40.6 GiB, `n_subgraphs=97`. Correct N-split concat matches INNERK, **not** mirrored `ssm_out` (`170+153`) until FFN AllReduce is restored (below). Isolation `hip-ssm-gemm` on ROCm0: **HIP shard-cat vs HIP full NMSE 0** (T=1 and T=8). Do not stash GEMM as `SCALE.src[2]`. Do not GEMM into a 2560-wide dest. Do not shrink a 2560 dest in place.

Native 1280 dest + host allgather (`[INNERN_SETCHK]` round-trips) still **English-looped** while layer-1 GDN `x` diverged (`x0` 0.00021 vs 0.00957). Concat of `ssm_out` was not the remaining gap.

### Residual / FFN AllReduce (content 323)

Live INNERN11 / INNERN resid dumps (`tp-innern11.log`, `tp-resid.log`):

```
[INNERN_AG] linear_attn_out-0 j=0 x0=-0.02653,0.01239
[INNERN_AG] linear_attn_out-0 j=1 x0=-0.02653,0.01239   MATCH
[INNERN_SETCHK] dst0[0] and dst1[wN] match GEMM slices
[INNERN_AG] linear_attn_out-1 j=0 x0=0.00021,-0.00113
[INNERN_AG] linear_attn_out-1 j=1 x0=0.00957,-0.00340   DIVERGE
```

After INNERN SET, reshape VIEWs on **both** GPUs already saw the concat (`next_pre` `linear_attn_out-0 (reshaped)` both `-0.09954`). H2 (VIEW misses SET) is false.

`[META_AR]` showed `ffn_moe_down-0` PARTIAL at i=123 delayed to i=143, subgraph last=`ffn_moe_out-0`. Delay-AR through ADD/GET_ROWS typed `ffn_moe_out` **MIRRORED** (`ADD(PARTIAL, MIRRORED)` handler meant for hc_combine after a completed wo AllReduce). The AR site then skipped because `orig_last` was not PARTIAL. Expert down-proj never reduced. Layer-0 FFN shards stayed different → layer-1 GDN `x` diverged.

Fix (`ggml-backend-meta.cpp`): AllReduce `ffn_moe_*` even when the delayed last is labelled MIRRORED. The buffers still hold a partial sum. INNERN skip-AR of `linear_attn_out` is unchanged (host concat is already full; AR would double).

After the fix (`tp-ffnar.log`, pid **254274**):

```
[RESID] ar_pre  ffn_moe_out-0 j=0 x0=0.00343 j=1 x0=0.03955   partial
[RESID] ar_post ffn_moe_out-0 j=0 and j=1 x0=0.04298,-0.02962 MATCH
[INNERN_AG] linear_attn_out-1 j=0 x0=0.01001,-0.00350
[INNERN_AG] linear_attn_out-1 j=1 x0=0.01001,-0.00350         MATCH
```

Also: PLE conv column VIEWs at `k*nb[0]` are 2-byte aligned for F16; Meta `data_ptr_ok` requires 16-byte so worker skipped `blk.1.ple_conv1d.weight (view)` (`META_BADPTR` / `META_SKIP`). `qwen4exp.cpp` now cast+transpose so each tap is a contiguous aligned row. That did **not** by itself match layer-1 `x` (same 0.00021 vs 0.00957 before FFN AR). Do not weaken `data_ptr_ok`.

Quality benches, prompt `17 times 19`, `max_tokens=64`, `-sm rpc-tensor`, GTT ~40.6 GiB each, `n_subgraphs` 96–97:

| File | pid | content | reasoning | tok/s |
|---|---|---|---|---|
| `bench-tp.json` | **254274** | **`323`** | `We need answer user: "17 times 19". Need compute 17*19 = 323.` | **1.339** |
| `bench-tp-2.json` | 254274 | **`323`** | same | **1.340** |
| `bench-tp-recompute.json` | **256130** | **`323`** | same | **8.307** |
| `bench-tp-recompute-2.json` | 256130 | **`323`** | same | **8.267** |

`has_323=true` in `message.content`, English, not reasoning-only. Both GPUs `gpu_busy_percent` 5–40 during decode (`both-gpus-rc*.txt`, `both-gpus-2.txt`). This is true tensor parallel, not replica (`n_subgraphs=1` / ~84 GiB).

### GRAPH_RECOMPUTE uid-keyed ring (speed)

RPC 6.1.0 stored **one** graph per device (`stored_graphs[device]`) and the client remembered **one** `last_graph_uid`. Token 2+ reserialized all ~97 subgraphs over USB4 (~1.34 tok/s).

RPC **6.1.1**: `rpc_msg_graph_recompute_req` carries `uid`; worker keeps a 128-slot uid-keyed ring (context kept alive); client RECOMPUTE if that subgraph uid is live. GRAPH_COMPUTE payload is `| device (4) | uid (8) | n_nodes | ... |`. Matching `libggml-rpc` both nodes (md5 `3eefc30e9ec4efe12d4f4790b5e5b98d`). Skip-rebuild `graph_sig` is unchanged.

`tp-recompute.log`: first pass `[RPC_COMPUTE] uid=137… live=1..12`; later tokens `[RPC_RECOMPUTE] uid=528…`. Decode **1.34 → 8.31 tok/s**. Still **< B_single 19.816**. Physics: ~97 serial USB4 AllReduces/token. 8.3 tok/s ≈ 120 ms/token. Do **not** relabel layer-split or replica as TP to beat that floor. Cherry-picking HIP mmq/FA from halo-box is a later PR and must not regress 323.

Launch env that produced 323 (pid 256130):

```
TBSTRIPE_SIMPLEX=master GGML_RPC_SET_TENSOR_CHUNK=2048
LLAMA_MIRROR_OUTPUT_WEIGHT=1 LLAMA_TP_MIRROR_GDN=1
LLAMA_TP_SPLIT_SSM_OUT=1 LLAMA_TP_SSM_OUT_NSPLIT=1
GGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_DISABLE_GRAPHS=1
-sm rpc-tensor -ts 1,1 --stream /dev/tbstream0,/dev/tbstream1
```

HIP md5 still `e03fcf630ee87717e278ae297d0d154d` both nodes. Worker `llama-rpc` **v6.1.1** (`libggml-rpc` md5 `3eefc30e9ec4efe12d4f4790b5e5b98d`).

### Speed path: MoE-split + HIP/RPC overlap (2026-09-10)

Keep experts split (~40 GiB GTT each). Mirror GDN and dense so `n_subgraphs` drops from 97 to **48–49** FFN-only AllReduces. Overlap local HIP `graph_compute_async` with RPC `GRAPH_RECOMPUTE` (do not `synchronize` inside the per-device loop). Skip GDN 3-rep x-pack when `ssm_out` is mirrored (`qwen4exp.cpp`). Gate INNERN host-concat to `LLAMA_TP_SSM_OUT_NSPLIT` + `ssm_out`/`linear_attn_out` only.

Launch (graphs off, fusion off, no SEQ, no piggyback AR):

```
TBSTRIPE_SIMPLEX=master GGML_RPC_SET_TENSOR_CHUNK=2048
LLAMA_MIRROR_OUTPUT_WEIGHT=1 LLAMA_TP_MIRROR_GDN=1 LLAMA_TP_MIRROR_DENSE=1
GGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_DISABLE_GRAPHS=1
-sm rpc-tensor -ts 1,1 --stream /dev/tbstream0,/dev/tbstream1
```

| File | pid | content | tok/s | vs `B_single` 19.816 |
|---|---|---|---|---|
| `bench-tp.json` / `bench-tp-2.json` | **301673** restore MIRROR_GDN SEQ | **`323`** | **18.023 / 18.448** | below |
| pid **292466** SEQ+small-burst+async | 292466 | **`323`** | **17.697 / 17.845** | below |
| pid **291008** small-burst SEQ | 291008 | **`323`** | **17.150 / 18.128** | below |
| pid **284828** GRAPH_SEQ+graphs | 284828 | **`323`** | **17.456 / 17.476** | below |
| pid 283027 graphs+piggy | 283027 | **`323`** | **16.78 / 17.08 / 17.09** | below |
| no `MIRROR_GDN` (85 subgraphs) | 286042 | **garbage / empty content** | **18.42** | n/a (not 323) |
| earlier pid **264765** | 264765 | **`323`** | **17.64 / 17.96** | below |

Pid 292466 GTT local **43982770176** (~41.0 GiB), remote **44108439552** (~41.1 GiB). Decode `gpu_busy_percent` bosgame1 **52 then 31**, bosgame2 **62 then 54** (`both-gpus.txt`). Worker rpc-server pid 552982. True TP, not replica. Still **< B_single**.

Physics: 48 serial USB4 `GRAPH_RECOMPUTE` + xchg RTTs plus mirrored GDN (both GPUs do full GDN). 17.09 tok/s ≈ 58.5 ms/token vs 50.5 ms needed for 19.816. Do **not** relabel layer-split or replica as TP.

Last-node AR piggyback (RPC 6.1.2): `rpc_msg_graph_recompute_req.ar_bytes` is 4 extra bytes. After RECOMPUTE ACK the worker `tbs_xchg`s the last F32/F16 node of the cached graph; the client `xchg_raw`s the local shard. No second `ALL_REDUCE` cmd. HIP graphs on both nodes (`UnsetEnvironment=GGML_CUDA_DISABLE_GRAPHS` on llama-rpc; client omits `GGML_CUDA_DISABLE_GRAPHS`). Fusion stays off. `ggml_graph_next_uid()` on rebuild (do not use stable `(n_subgraphs<<16)|i`).

### GRAPH_SEQ (RPC 6.1.3)

One `RPC_CMD_GRAPH_SEQ` after the ring is populated: payload is `device, n, [{uid, ar_bytes, pad}]`. Worker ACK then, per subgraph, `graph_recompute` + last-node `tbs_xchg` iff `ar_bytes>0` (no dummy xchg). Client HIP on the llama thread, then `xchg_raw`. Fall through to per-subgraph RECOMPUTE if any uid is not in `live_graph_uids` (first T=8 prefill). Default on unless `LLAMA_TP_NO_SEQ=1` or `LLAMA_TP_SSM_OUT_NSPLIT=1`. `RPC_PROTO_PATCH_VERSION` 3.

Recapture pid **287196** (same env as 284828): `bench-tp.json` **17.284** tok/s content **323**; `bench-tp-2.json` **17.464** tok/s content **323**. GTT ~41.0 / ~41.1 GiB. Both GPUs 4–9% busy during decode. Still **< B_single**.

`[RPC_SEQ]` hip vs xchg timing is logged for the first few tokens (`hip_us` / `xchg_us`).

### Simplex xchg window (speed)

`tbs_xchg_simplex` defaulted to `WINDOW=1` (TX one 4KiB frame, RX one, repeat). A 10 KiB FFN AllReduce is 3 frames = 3 USB4 RTTs. Worker already had `TBS_XCHG_WINDOW=4` via `llama-rpc.service.d/xchg-window.conf`; the client did not.

**Do not** run TX and RX on two threads in one process — USB4STREAM deadlocks (pid 288767 spun on the first AllReduce, GPUs 0%, log stuck at `RESID ar_pre`). Sequential `WINDOW=total` of a **large** payload also deadlocks (fills the RX ring).

Safe small-burst: if the whole xchg fits in **8 frames (32 KiB)**, TX all frames then RX all (one cable RTT). 3-frame AR fits. `TBS_XCHG_WINDOW=1` forces 1:1. `libtbstripe.so` md5 `f593d4d1276be41e565b41120119be31` both nodes. HIP remains `e03fcf630ee87717e278ae297d0d154d`.

Pid **291008** (small-burst, GRAPH_SEQ, HIP graphs): content **323**. `bench-tp.json` **17.150** tok/s; `bench-tp-2.json` **18.128** tok/s (55.16 ms/token). Still **< B_single 19.816**. `[RPC_SEQ]` decode: `hip_us≈33800` `xchg_us≈1600–1800` `tot_us≈40000` for 48 AllReduces — USB4 is no longer the floor (~33 µs/xchg). Remaining gap is HIP D2H/H2D around AR (~4.6 ms of extra stream syncs) plus mirrored GDN. Both GPUs 4–9% busy, GTT ~41.0 / 41.1 GiB.

### Failed speed levers — do not retry

- **Old `GRAPH_SEQ` dummy-xchg protocol** — hung on USB4. **Working SEQ** (RPC 6.1.3): last-node `tbs_xchg`, skip when `ar_bytes=0`, next_uid, HIP on llama/worker compute threads. 17.48 tok/s, 323. Default on; `LLAMA_TP_NO_SEQ=1` disables.
- **Drop `MIRROR_GDN`** (85 subgraphs, 3-rep pack) — **18.42 tok/s** but **CJK garbage / empty content**. Do not ship. Quality needs mirrored GDN or NSPLIT INNERN (97 subgraphs, ~8 tok/s).
- **Stable subgraph uids** `((n_subgraphs<<16)|(i+1))` — RECOMPUTE of T=8 prefill graphs onto T=1 decode buffers. Worker `k_set_rows<float,long,__half>` HSA aperture violation, systemd restart, client hung. Use `ggml_graph_next_uid()` on rebuild; skip-rebuild `graph_sig` still reuses uids for token 2+ of the **same** shape.
- **HIP graphs** (`unset GGML_CUDA_DISABLE_GRAPHS`) — `GGML_ASSERT(node_props.size()==n_nodes)` with the stable-uid scheme; with next_uid not re-tested. Fusion stays disabled.
- **Piggyback AllReduce on `GRAPH_RECOMPUTE`** (larger request + `tbs_xchg` after ACK) — worker `k_set_rows` fault even with `ar_bytes=0` while the enlarged struct was on the wire. Reverted RPC to committed 6.1.1 (`3eefc30e`). Do not enlarge `rpc_msg_graph_recompute_req`.
- Rebuilding `libggml-hip` (even a 2-line graphs assert) produced md5 `eaee5820…` which also `k_set_rows`-faulted. Restore **`e03fcf630ee87717e278ae297d0d154d`** both nodes. Matching HIP is mandatory.

### Remaining

1. **GDN-split + NSPLIT AllGather is not 323.** Pid 296917 (`LLAMA_TP_SSM_OUT_NSPLIT=1`, no `MIRROR_GDN`, SEQ concat): 121 subgraphs, GTT ~40.1 GiB, decode ~8 tok/s, empty content, CJK/loop garbage (`bench-nsplit-garbage-1.json`). Layer-0 `linear_attn_out` matched after concat; layer-1 diverged. `final_output` on-device is **3072** not a concatenated 6144, so `ssm_out` GEMMs half-K. Do not ship. Keep `MIRROR_GDN=1` for 323.
2. USB4 xchg is ~1.7 ms/token on the mirrored-GDN speed path. Mirrored GDN + dual overhead floors decode at **~17.8 tok/s** (56 ms) vs `B_single` 19.816 (50.5 ms). Do not fake TP with replica or layer-split.
3. Unaligned PLE `node_206 (view)` still `META_BADPTR` on some taps; PLE kernel rows are aligned. Hunt only if quality regresses.
4. SEQ, HIP graphs, last-node piggyback, and small-burst xchg are **on** and 323 with `MIRROR_GDN`. Do not revert to stable subgraph uids. Do not rebuild HIP off `e03fcf`.

## Files that matter

| File | Role |
|---|---|
| `ggml/src/ggml-backend-meta.cpp` | split handlers, host-pack SET, INNERN native-1280 dest + SETCHK (NSPLIT-gated), FFN AR of MIRRORED `ffn_moe_out`, skip-rebuild `graph_sig`, HIP/RPC overlap, `ggml_graph_next_uid()` on rebuild |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | USB4STREAM RPC, HC-only VIEW rebase, uid-keyed GRAPH_RECOMPUTE ring, last-node AR piggy, GRAPH_SEQ 6.1.3 |
| `ggml/include/ggml-rpc.h` | `RPC_PROTO_PATCH_VERSION` 3 (6.1.3 GRAPH_SEQ) |
| `tbstripe/src/tbstripe.c` | USB4STREAM simplex; concurrent windowed `tbs_xchg` (default window 4) |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | HIP GDN, pack_contiguous_f32, GDN_HIP/GDN_ACT logs |
| `src/llama-model.cpp` | Qwen 3.5 `get_split_segments` |
| `src/models/qwen4exp.cpp` | GDN graph, PLE kernel transpose for 16-byte views, `ggml_cont` Q/K/V, skip 3-rep pack when `MIRROR_GDN` without `SPLIT_SSM_OUT` |
| `tbstripe/` | USB4STREAM driver userspace / README |

## Non-goals

Remaking the quality catalog; MTP over USB4; Lemonade `:13305`; BIOS / UMA / ROCm package changes; replacing TP with layer-split, replica, or halogen; re-enabling `is_dsv4`; further USB4 header/framing work.
