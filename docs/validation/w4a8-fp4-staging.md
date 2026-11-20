# W4A8 FP4 decode page staging

The native FP4 decode scorer consumes64-record canonical pages. For non64-record physical pages,route the existing gather/repack operation through Adapter `fp4_staging_triton.stage_decode_pages` on the admitted gfx93864CU path. Native provider identity,device,dtype and layout checks retain the original path for custom collaborators and unsupported inputs. Physical64-record pages remain zero-copy. Prefill and native scoring/selection are unchanged.

This integrates Adapter MR1 and requires its matching helper commit. All previous accepted FULL,DeepEP/finalPost,activation,mHC Post launch,MLA and WO_A changes remain. Delayed mHC stage2 stays reverted.

Operator validation:72 eager byte-exact/input-preservation cases,72 changing graph replays,4 actual native LightOp decode consumers;86 existing CPU regressions passed. The boundary fixture originally supplied an undersized source table at context65;the unchanged baseline rejected it before any fused execution. The corrected table follows its existing rounded64page contract;all original failure logs are retained.

Context4096 full-wrapper operator medians(R32) were99.36→10.04us(rows8),317.39→50.89us(rows48),3shuffled repeats and8transactions per captured graph. These measure the staging wrapper. No capacity,parallelism,synthetic rates or testclient parameters change;no new flag.

Full service validation completed on nmz18 on 2026-11-20. The pinned runtime/test sources were Plugin `1c935df` and Adapter `1e11880`; the published sources match byte-for-byte outside documentation/CI. Real-prefill HumanEval passed159/164(96.95%),first32=32,zero errors/missing/truncations.

Unprofiled median TPOT was19.065356/18.688054ms,p9922.367298/23.229673ms,64completed/0failed per repeat. Relative to the frozen accepted WO_A composition(19.932703/19.343608ms),each repeat improved4.35%/3.39%;the mean of repeat medians improved3.88%. The unchanged median2%,p995%,mean2% regression limits and>=2%-per-repeat gain threshold passed. Performance uses fake zero KV and five literal unconditional0.6 acceptance rates;accuracy uses real prefill/rejection.

All8 traces confirm3 fused staging calls and0 native gather calls per target span,with8 native FP4 scorer calls. FULL target/draft execution,40/3 DeepEP layers and40/3 activation calls,83/6-or-more mHC Post calls,and40 WO_A BMMs without following layout copies remain. Delayed mHC stays absent. DP0 target-through-draft step measured77.608710ms,versus79.812687ms for the prior accepted trace. The supplied SGLang reference is65.098274ms;hardware,batch,context and providers differ,so whole-step parity remains unfinished.

Replay the probe with `--original` pointing at a saved baseline `fp4_indexer_lightop.py` from `fbde474` (SHA25615af0c4c56f25b2c14606da6375b7577e0f00599ca81ce93f6dcc87473fdce76). The workspace immutable equivalent is `vllm-hcu033-wt-w4a8-wo-a-token-major-20261119/adapter_extension/src/hcu_vllm_das/fp4_indexer_lightop.py`. Do not use an advancing publication worktree as a later baseline.

Evidence in `run/dsv4.1/033/w4a8_reverify_20261119/`: `FP4_STAGING_UNIT_RESULTS.json`, `FP4_STAGING_V2_EXPERIMENT.json`, `FP4_STAGING_SERVICE_EXPERIMENT.json`, `nmz18/fp4_staging_service_20261120_v2/acceptance.json` and `nmz18/fp4_staging_step_reference_20261120.json`.
