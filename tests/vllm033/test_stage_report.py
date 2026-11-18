"""Full kernel symbols must not turn a singleton family into characters."""
from tools.moe_reverify.stage_report import stage


def test_full_qkv_symbols_remain_distinct_from_post_and_unknown():
    prefix = 'void fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel<c10::BFloat16, '
    assert stage(prefix + '0, false, false>') == 'native_kv_insert'
    assert stage(prefix + '64, true, false>') == 'native_qkv_with_norm'
    assert stage(prefix + '64, false, false>') == 'native_qkv_fused'
    assert stage('_mhc_post_5120') == 'mhc_post_5120'
    assert stage('_compact_selectors') == 'mla_selectors'
    assert stage('_mask_copy_native_mla') == 'mla_output_mask_copy'
    assert stage('_mask_copy_inverse_rope') == 'mla_mask_inverse_rope'
    assert stage('_fresh_native_plan') == 'mla_fresh_plan'
    assert stage('gfx9::decode::get_mla_metadata_kernel(GetDecodeSchedMetaParams)') == 'mla_native_plan'
    assert stage('unrelated_vectorized_elementwise_kernel') is None
