"""Reuse verified metadata/backup helpers for the explicitly registered attempt3."""
from pathlib import Path

root = Path('E:/GAN/research/N01/receipts/wp1_013_20261015')
old = root / 'native_import_fix01'
new = root / 'native_repeat_fix01'
source = (old / 'record_attempt.py').read_text('utf-8')
for before, after in (
    ('native_import_fix01', 'native_repeat_fix01'),
    ('wp1_013_runtime_fix01_20261015', 'wp1_013_repeat_fix01_20261015'),
    ('RUNTIME_REPAIR_REGISTRATION', 'NUMERICAL_STORAGE_REPAIR_REGISTRATION'),
    ("cfg['engineering_repair']['attempt'] == 2", "cfg['engineering_repair']['attempt'] == 3"),
    ('attempt=2', 'attempt=3'),
    ('WP1_013_runtime_fix01_', 'WP1_013_repeat_fix01_'),
    ("['runtime_repair']", "['numerical_storage_repair']"),
    ("original_source_commit='ffe88e43518cd79f01c116df421052550b1c874d'", "original_source_commit='190553e063d5d4d2c11bdd9f68124aaef423fb59'"),
):
    assert before in source, before
    source = source.replace(before, after)
start = source.index("old_text = '013")
end = source.index('assert expected_text in readme', start)
source = source[:start] + '''old_text = ('013 的包装修复通过了导入，但在重复纯信号逐字节检查处原生退出1；正在量化保存的方差并核对身份。'
    ' 原尝试退出1、25文件失败现场和52744行参考计算成本保留；补入原011评价依赖，没有改变科学设置。')
ready_text = '013 的实际重复纯信号保存修复已冻结，等待新服务器目录启动。'
live_text = '013 的实际重复纯信号保存修复已在新服务器目录启动，首批与当时无失败均已核验。'
history_text = ' 前两次原生失败、25/38文件现场和各52744行参考计算成本保留；此次保存全部实际A输出并使用原数值回放标准。'
new_text = (live_text if live else ready_text) + history_text
expected_text = old_text if not live else ready_text + history_text
''' + source[end:]
with (new / 'record_attempt.py').open('x', encoding='utf-8', newline='\n') as handle:
    handle.write(source)
backup = (old / 'backup_attempt.py').read_text('utf-8')
for before, after in (
    ('native_import_fix01', 'native_repeat_fix01'),
    ("args.stage + '_02'", "args.stage + '_03'"),
    ('RUNTIME_REPAIR_REGISTRATION', 'NUMERICAL_STORAGE_REPAIR_REGISTRATION'),
    ('PRELAUNCH_RUNTIME_FIX_REVIEW', 'PRELAUNCH_REPEAT_FIX_REVIEW'), ('attempt=2', 'attempt=3'),
):
    assert before in backup, before
    backup = backup.replace(before, after)
with (new / 'backup_attempt.py').open('x', encoding='utf-8', newline='\n') as handle:
    handle.write(backup)
print('Two reused lifecycle helpers saved; no scientific execution or remote calls.')
