# W4A8 FP4 decode page staging

The native FP4 decode scorer consumes64-record canonical pages. For non64-record physical pages,route the existing gather/repack operation through Adapter `fp4_staging_triton.stage_decode_pages` on the admitted gfx93864CU path. Native provider identity,device,dtype and layout checks retain the original path for custom collaborators and unsupported inputs. Physical64-record pages remain zero-copy. Prefill and native scoring/selection are unchanged.

This integrates Adapter MR1 and requires its matching helper commit. All previous accepted FULL,DeepEP/finalPost,activation,mHC Post launch,MLA and WO_A changes remain. Delayed mHC stage2 stays reverted.

Operator validation:72 eager byte-exact/input-preservation cases,72 changing graph replays,4 actual native LightOp decode consumers;86 existing CPU regressions passed. The boundary fixture originally supplied an undersized source table at context65;the unchanged baseline rejected it before any fused execution. The corrected table follows its existing rounded64page contract;all original failure logs are retained.

Context4096 full-wrapper operator medians(R32) were99.36→10.04us(rows8),317.39→50.89us(rows48),3shuffled repeats and8transactions per captured graph. No model speedup claim follows from this alone. Full164 accuracy and fixed fakezeroKV/5literal0.6 performance gates are pending at publication,as authorized by user. No capacity,parallelism,synthetic rates or testclient parameters change;no new flag.

Replay the probe with `--original` pointing at a saved baseline `fp4_indexer_lightop.py` from `fbde474` (SHA25615af0c4c56f25b2c14606da6375b7577e0f00599ca81ce93f6dcc87473fdce76). The workspace immutable equivalent is `vllm-hcu033-wt-w4a8-wo-a-token-major-20261119/adapter_extension/src/hcu_vllm_das/fp4_indexer_lightop.py`. Do not use an advancing publication worktree as a later baseline.

Evidence in the experiment workspace: `FP4_STAGING_UNIT_RESULTS.json`, `FP4_STAGING_V2_EXPERIMENT.json`, `FP4_STAGING_SERVICE_EXPERIMENT.json`.
