"""Reuse the verified one-time heartbeat binding for the numerical storage fix."""
from pathlib import Path

base = Path('E:/GAN/research/N01/receipts/wp1_013_20261015')
source = (base / 'native_import_fix01/record_followup.py').read_text('utf-8')
source = source.replace('native_import_fix01', 'native_repeat_fix01')
source = source.replace('wp1_013_runtime_fix01_20261015', 'wp1_013_repeat_fix01_20261015')
source = source.replace('_02', '_03').replace('attempt=2', 'attempt=3')
start = source.index("    prompt += ('\\n本次是")
end = source.index("    with prompt_path.open", start)
source = source[:start] + '''    prompt += ('\\n本次是经过保存数组定量审查后登记的实际重复信号保存修复attempt3。'
        '原attempt1因遗漏原011 CDF依赖退出1、attempt2因A逐字节门强于原数值回放标准退出1；'
        '25/38文件现场和已耗成本都保存，不记为科学或工程完成。'
        'saved variance只有最后6592父、4row尾批非零，pairwise严格上界7.657714e-9V，'
        '不是已观测exactmax，也不能单凭moments唯一确定CUDA kernel。'
        '采用原既有baseline回放1e-7V限值并保存全部实际A逐draw float64及精确数值残差；'
        '不把舍入残差称物理随机，不假广播、不平均硬波形、不改变实际ADC/裁窗。'
        '新REFERENCE_0/1均含actual A_signal_pre384，按完整实际两bank阵列重建preADC covariance；'
        'REFERENCE_ENCODING里A broadcast必须false。其他4个科学数值源码、全部人口/方向/尺度/观察/T公式不变。'
        '同一snapshot如果FAILED/phasefailed/native非0，historical FIRST128不足以宣布健康启动。'
        '本attempt的原科学计算cap不变；整个013累计必须包含前两次各52744 A/head/005行、'
        '各316464 ADC/gather、各CP3。累计MAX A/head403992、linear40960、head_total444952、'
        'noise158232、ADC2202768、gather3677328、CP9，其余零预算不变。'
        '先读native_repeat_fix01/NUMERICAL_STORAGE_REPAIR_REGISTRATION.json，分别报告三次attempt与总实际成本；'
        '严格区分actual raw行可重建性和已保存summary可复用范围。'
        '若此attempt再次失败，回收现场及原计数并报告缺项，禁止自动开启第四次同范围重跑或抹去成本。\\n')
''' + source[end:]
with (base / 'native_repeat_fix01/record_followup.py').open('x', encoding='utf-8', newline='\n') as handle:
    handle.write(source)
print('Reused followup helper saved; no app update or scientific execution.')
