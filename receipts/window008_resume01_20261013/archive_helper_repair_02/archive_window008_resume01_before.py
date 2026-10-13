"""Private byte-preserving N01 snapshot, or a small final-ledger supplement.

IMPLEMENTED ONLY: do not execute before report QA and the final commit.
Only E/D are writable. No numpy/torch, waveform decode, GPU, model call, job
launch, Git commit, deletion, overwrite, compression, or archive extraction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = Path('E:/GAN/private_archives/N01')
D_ROOT = Path('D:/GAN_backups/research/N01')
POINTER = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
RUN_ID = 'window008_resume01_20261013'
BLOCK = 1024 * 1024
GIB = 1024**3
D_MARGIN = 256 * 1024**2
SUPPLEMENT_LIMIT = 64 * 1024**2
TEXT_SUFFIXES = {'.py', '.sh', '.ps1', '.bat', '.cmd', '.md', '.txt', '.json', '.toml',
                 '.yaml', '.yml', '.html', '.htm', '.css', '.js', '.ts', '.tsx', '.svg',
                 '.tex', '.bib', '.rst', '.ini'}
DATA_SUFFIXES = {'.pt', '.pth', '.ckpt', '.npz', '.npy', '.h5', '.hdf5', '.mat',
                 '.parquet', '.arrow', '.csv', '.tsv', '.bin', '.safetensors'}


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf8')


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique)


def write_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(json_bytes(value)); stream.flush(); os.fsync(stream.fileno())


def digest_stream(stream):
    digest = hashlib.sha256(); size = 0
    for block in iter(lambda: stream.read(BLOCK), b''):
        size += len(block); digest.update(block)
    return size, digest.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return digest_stream(stream)[1]


def reparse(path):
    s = Path(path).lstat()
    return stat.S_ISLNK(s.st_mode) or bool(getattr(s, 'st_file_attributes', 0) & 0x400)


def require_unlinked(path):
    path = Path(path).absolute()
    for part in [path] + list(path.parents):
        try: s=part.lstat()
        except FileNotFoundError: continue
        require(not stat.S_ISLNK(s.st_mode) and not (getattr(s,'st_file_attributes',0)&0x400),
                'Do not traverse linked/reparse input or output: ' + str(part))
    return path


def e_path(path, *, inside=PRIVATE_ROOT):
    path = Path(path).absolute()
    require(os.name == 'nt' and path.drive.upper() == 'E:', 'Private snapshot outputs require Windows E:')
    require_unlinked(path)
    require(path != inside.absolute() and path.is_relative_to(inside.absolute()), 'Output must be a fresh child of ' + str(inside))
    require(not path.exists(), 'Preserve existing archive/receipt: ' + str(path))
    return path


def fingerprint(s):
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, mode=s.st_mode,
                device=s.st_dev, inode=s.st_ino)


def excluded(relative):
    parts = Path(relative).parts
    if '__pycache__' in parts or Path(relative).suffix.lower() in {'.pyc', '.pyo'}:
        return 'transient Python bytecode cache'
    if parts and parts[0] == '.git' and Path(relative).name.endswith('.lock'):
        return 'transient Git lock; historical run.lock and .start.lock remain included'
    return None


def scan(root):
    """lstat walk: every directory/file/link is explicit, no link traversal."""
    root = require_unlinked(root); require(root.is_dir(), 'Missing verified N01 root')
    entries, exclusions = {}, []
    def walk(folder):
        with os.scandir(folder) as scan_entries:
            children = sorted(scan_entries, key=lambda x: x.name)
        for child in children:
            path = Path(child.path); relative = path.relative_to(root).as_posix()
            s = path.lstat(); reason = excluded(relative)
            if reason:
                exclusions.append(dict(path=relative, reason=reason, lstat=fingerprint(s)))
                continue
            row = dict(path=relative, source=str(path), fingerprint=fingerprint(s))
            if stat.S_ISLNK(s.st_mode) or bool(getattr(s, 'st_file_attributes', 0) & 0x400):
                target = os.readlink(path)
                row.update(kind='symlink_or_junction', link_target=target,
                           reparse_tag=getattr(s, 'st_reparse_tag', None), bytes=0,
                           sha256=hashlib.sha256(target.encode('utf8')).hexdigest())
            elif stat.S_ISDIR(s.st_mode):
                row.update(kind='directory', bytes=0, sha256=hashlib.sha256(b'').hexdigest())
            elif stat.S_ISREG(s.st_mode):
                row.update(kind='file', bytes=s.st_size)
            else:
                raise ValueError('Unsupported nonregular workspace entry: ' + relative)
            entries[relative] = row
            if row['kind'] == 'directory': walk(path)
    walk(root)
    require(len({name.casefold() for name in entries}) == len(entries), 'Case-aliased workspace paths')
    return entries, exclusions


def hash_row(row):
    require(row['kind'] == 'file', 'Expected regular inventory file')
    path = Path(row['source']); require(not reparse(path), 'Regular file became linked: ' + str(path))
    with path.open('rb') as stream:
        require(fingerprint(os.fstat(stream.fileno())) == row['fingerprint'], 'File changed before hash: ' + str(path))
        size, digest = digest_stream(stream)
        require(fingerprint(os.fstat(stream.fileno())) == row['fingerprint'], 'File changed while hashing: ' + str(path))
    require(size == row['bytes'] and fingerprint(path.lstat()) == row['fingerprint'], 'File changed after hash: ' + str(path))
    row['sha256'] = digest


def external_row(path, archive_name):
    path = require_unlinked(path)
    require(path.drive.upper() in ('E:', 'D:') and path.is_file(), 'External private input must be an E/D regular file')
    row = dict(path=archive_name, source=str(path), kind='file', bytes=path.stat().st_size,
               fingerprint=fingerprint(path.stat()))
    hash_row(row)
    return row


def git(arguments, root=ROOT, *, timeout=600):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', TMP='E:/GAN/runtime_tmp', TEMP='E:/GAN/runtime_tmp')
    result = subprocess.run(['git', '-C', str(root)] + list(arguments), shell=False, capture_output=True,
                            text=True, encoding='utf8', errors='strict', timeout=timeout, env=env)
    require(result.returncode == 0, 'Git read/bundle command failed: ' + result.stderr.strip())
    return result.stdout.strip()


def commit_gate(expected, root=ROOT):
    require(len(expected) in (40, 64) and all(c in '0123456789abcdef' for c in expected), 'Full expected commit hash required')
    require(Path(git(['rev-parse', '--show-toplevel'], root)).resolve() == root.resolve(), 'N01 must be its own repository')
    require(git(['rev-parse', 'HEAD'], root) == expected, 'HEAD differs from the requested final commit')
    require(not git(['status', '--porcelain=v1', '--untracked-files=no'], root), 'Tracked or staged files changed after final commit')
    return dict(commit=expected, tracked_worktree_clean=True, untracked_and_ignored_workspace_files_included=True,
                refs=git(['show-ref'], root).splitlines())


def report_gate(pdf, pdf_hash, qa, qa_hash):
    pdf, qa = require_unlinked(pdf), require_unlinked(qa)
    require(pdf.is_relative_to(ROOT) and qa.is_relative_to(ROOT) and pdf.suffix.lower() == '.pdf', 'Report and QA must belong to N01')
    require(sha(pdf) == pdf_hash and sha(qa) == qa_hash, 'Report/QA bytes changed')
    value = read(qa)
    count = value.get('pages_expected')
    require(value.get('status') == 'passed' and Path(value.get('pdf_path', '')).absolute() == pdf
            and value.get('pdf_sha256') == pdf_hash and type(count) is int and count > 0
            and type(value.get('pages_rendered')) is int and type(value.get('pages_visually_reviewed')) is int
            and value.get('pages_rendered') == value.get('pages_visually_reviewed') == count
            and value.get('text_extraction_passed') is True, 'Complete hash-bound report QA is required')
    return dict(pdf_path=str(pdf), pdf_sha256=pdf_hash, qa_path=str(qa), qa_sha256=qa_hash,
                pages_from_hash_bound_QA=count, QA_status='passed', PDF_not_rerendered_by_archive_helper=True)


def tar_info(name, row):
    info = tarfile.TarInfo(name); info.mode = stat.S_IMODE(row.get('fingerprint', {}).get('mode', 0o644))
    info.mtime = row.get('fingerprint', {}).get('mtime_ns', 0) / 1e9
    if row['kind'] == 'directory': info.type = tarfile.DIRTYPE
    elif row['kind'] == 'symlink_or_junction':
        info.type = tarfile.SYMTYPE; info.linkname = row['link_target']
    else: info.type = tarfile.REGTYPE; info.size = row['bytes']
    return info


def tar_upper_size(entries):
    size = sum(len(tar_info(name, row).tobuf(format=tarfile.PAX_FORMAT, encoding='utf8'))
               + ((row['bytes'] + 511)//512)*512 for name, row in entries.items()) + 1024
    return ((size + tarfile.RECORDSIZE - 1)//tarfile.RECORDSIZE)*tarfile.RECORDSIZE


class HashingReader:
    def __init__(self, stream): self.stream=stream; self.digest=hashlib.sha256(); self.size=0
    def read(self, count=-1):
        value=self.stream.read(count); self.digest.update(value); self.size += len(value); return value


def write_tar(path, entries, metadata):
    require(not Path(path).exists(), 'Archive exists')
    expected = dict(entries)
    with tarfile.open(path, 'x', format=tarfile.PAX_FORMAT, dereference=False) as archive:
        for name, row in sorted(entries.items()):
            info = tar_info(name, row)
            if row['kind'] == 'file':
                source = Path(row['source']); require(not reparse(source), 'Input became linked')
                with source.open('rb') as stream:
                    require(fingerprint(os.fstat(stream.fileno())) == row['fingerprint'], 'Input changed before archive: ' + str(source))
                    reader=HashingReader(stream); archive.addfile(info, reader)
                    require(reader.size == row['bytes'] and reader.digest.hexdigest() == row['sha256'], 'Archive source bytes changed: ' + name)
                    require(fingerprint(os.fstat(stream.fileno())) == row['fingerprint'], 'Input changed during archive: ' + name)
            else:
                source = Path(row['source'])
                require(fingerprint(source.lstat()) == row['fingerprint'], 'Directory/link changed: ' + name)
                if row['kind'] == 'symlink_or_junction': require(os.readlink(source) == row['link_target'], 'Link target changed')
                archive.addfile(info)
        for name, payload in sorted(metadata.items()):
            require(name not in expected, 'Metadata name collision')
            row=dict(kind='file', bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            archive.addfile(tar_info(name, row), io.BytesIO(payload)); expected[name]=row
    return expected


def verify_tar(path, expected):
    seen=set(); details=[]
    with tarfile.open(path, 'r|') as archive:
        for member in archive:
            require(member.name in expected and member.name not in seen, 'Unexpected/duplicate tar member: ' + member.name)
            seen.add(member.name); row=expected[member.name]
            if row['kind'] == 'file':
                require(member.isreg() and member.size == row['bytes'], 'Tar file type/size mismatch')
                stream=archive.extractfile(member); require(stream is not None, 'Missing member stream')
                size,digest=digest_stream(stream)
            elif row['kind'] == 'directory':
                require(member.isdir(), 'Tar directory type mismatch'); size=0; digest=hashlib.sha256(b'').hexdigest()
            else:
                require(member.issym() and member.linkname == row['link_target'], 'Tar link target mismatch')
                size=0; digest=hashlib.sha256(member.linkname.encode('utf8')).hexdigest()
            require(size == row['bytes'] and digest == row['sha256'], 'Reopened member differs: ' + member.name)
            details.append(dict(path=member.name, kind=row['kind'], bytes=size, sha256=digest))
    require(seen == set(expected), 'Tar omits inventoried members')
    return dict(members_verified=len(seen), all_member_sizes_sha256_match=True, streaming_reopen=True,
                member_sha256=details, archive_bytes=Path(path).stat().st_size, archive_sha256=sha(path))


def stable_tree(before, root=ROOT):
    after, exclusions = scan(root)
    require(set(before) == set(after), 'Workspace membership changed during snapshot')
    for name,row in before.items():
        require(row['kind'] == after[name]['kind'] and row['fingerprint'] == after[name]['fingerprint']
                and row.get('link_target') == after[name].get('link_target'), 'Workspace metadata changed: ' + name)
    return exclusions


def d_backup(output, workspace, bundle, report):
    before=shutil.disk_usage('D:/').free
    eligible=[]
    for name,row in workspace.items():
        if row['kind'] != 'file': continue
        parts=Path(name).parts; suffix=Path(name).suffix.lower()
        if ((parts[0] in ('src','scripts','docs') and suffix in TEXT_SUFFIXES)
                or (len(parts)==1 and (suffix in TEXT_SUFFIXES or name=='.gitignore'))):
            require(suffix not in DATA_SUFFIXES, 'Data/weights may not enter source-only D backup')
            eligible.append((name,row))
    history_data=[line.split(' ',1)[1] for line in git(['rev-list','--objects','--all','HEAD']).splitlines()
                  if ' ' in line and Path(line.split(' ',1)[1]).suffix.lower() in DATA_SUFFIXES]
    omitted_bundle=None
    if history_data:
        omitted_bundle=dict(path=str(bundle),sha256=sha(bundle),status='omitted_from_D_source_copy',
                            reason='Historical Git includes data/weight-extension paths; full Git and bundle remain in the E private snapshot.',
                            historical_data_paths=history_data)
    else:
        eligible += [('N01_LOCAL_HISTORY.bundle', external_row(bundle,'N01_LOCAL_HISTORY.bundle'))]
    pdf=Path(report['pdf_path']); name=pdf.relative_to(ROOT).as_posix()
    if name not in {x[0] for x in eligible}: eligible.append((name,external_row(pdf,name)))
    listing=[dict(path=name,bytes=row['bytes'],sha256=row['sha256']) for name,row in eligible]
    manifest=json_bytes(dict(schema='n01/source-only-backup-inventory/v1', files=listing,
                             no_direct_weight_or_data_array_files=True, git_bundle_included=not history_data,
                             omitted_git_bundle=omitted_bundle))
    sums=(''.join(x['sha256']+'  '+x['path']+'\n' for x in listing)).encode('utf8')
    exact_bytes=sum(row['bytes'] for _,row in eligible)+len(manifest)+len(sums)
    allocation_allowance=(len(eligible)+32)*4096
    if before < D_MARGIN+exact_bytes+allocation_allowance:
        return dict(status='skipped_insufficient_D_space', exact_payload_bytes_before_copy=exact_bytes,
                    allocation_allowance_bytes=allocation_allowance, D_free_before=before,
                    D_free_after=shutil.disk_usage('D:/').free, required_free_margin_bytes=D_MARGIN,copied_bytes=0,
                    omitted_git_bundle=omitted_bundle)
    destination=D_ROOT/output.name/'source_only'
    require_unlinked(destination); require(not destination.exists(), 'Preserve prior D source backup')
    destination.mkdir(parents=True,exist_ok=False)
    for name,row in eligible:
        target=destination/name; target.parent.mkdir(parents=True,exist_ok=True)
        with Path(row['source']).open('rb') as source, target.open('xb') as copied:
            shutil.copyfileobj(source,copied,BLOCK); copied.flush(); os.fsync(copied.fileno())
        require(target.stat().st_size==row['bytes'] and sha(target)==row['sha256'], 'D source copy differs: '+name)
    with (destination/'SOURCE_INVENTORY.json').open('xb') as stream: stream.write(manifest)
    with (destination/'SHA256SUMS.txt').open('xb') as stream: stream.write(sums)
    actual=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
    after=shutil.disk_usage('D:/').free
    require(actual==exact_bytes and after>=D_MARGIN, 'D payload size/margin changed during copy; preserve files')
    return dict(status='source_only_copy_reopened_and_sha256_verified',path=str(destination),files=len(eligible),
                exact_payload_bytes_before_copy=exact_bytes,actual_payload_bytes=actual,D_free_before=before,
                D_free_after=after,required_free_margin_bytes=D_MARGIN,no_direct_weight_or_data_arrays=True,
                git_bundle_included=not history_data,omitted_git_bundle=omitted_bundle)


def full(args):
    snapshot_started=utc()
    require_unlinked(ROOT)
    commit=commit_gate(args.expected_commit)
    report=report_gate(Path(args.report_pdf).absolute(),args.report_pdf_sha256,
                       Path(args.report_qa).absolute(),args.report_qa_sha256)
    pointer=read(POINTER)
    require(pointer.get('program_id')=='N01' and pointer.get('active_scientific_jobs')==0,
            'Current program pointer must identify N01 with zero active scientific jobs')
    output=e_path(args.out or PRIVATE_ROOT/(RUN_ID+'_full_'+stamp()))
    workspace,exclusions=scan(ROOT)
    source_bytes=sum(row['bytes'] for row in workspace.values())
    # Full .git is included except transient locks. The bundle is additional,
    # portable refs+HEAD history; unreachable objects/reflogs remain in .git.
    git_bytes=sum(row['bytes'] for name,row in workspace.items() if name.startswith('.git/'))
    required_early=source_bytes+2*git_bytes+4*GIB+len(workspace)*4096
    free_before=shutil.disk_usage('E:/').free
    require(free_before>=required_early, 'Insufficient E space for full snapshot, history bundle and4GiB margin')
    output.mkdir(parents=True,exist_ok=False)
    write_new(output/'ATTEMPT.json',dict(UTC=utc(),kind='private_full',expected_commit=args.expected_commit,
        E_free_before=free_before,workspace_payload_bytes=source_bytes,required_early_bytes=required_early,
        overwrite_allowed=False,automatic_retry_allowed=False))
    try:
        for row in workspace.values():
            if row['kind']=='file': hash_row(row)
        cfg=read(ROOT/'window008_config.json'); refs=read(ROOT/'DATA_REFERENCES.json')
        extras={}
        raw=external_row(cfg['local_training_container'],'external/TRAIN_PRIVATE.npz')
        require(raw['sha256']==cfg['train_container_sha256']==refs['training_container']['container_sha256'],
                'Frozen external TRAIN_PRIVATE SHA256 differs')
        extras[raw['path']]=raw
        statuses=[]
        for index,reference in enumerate(refs.get('references',[])):
            path=Path(reference['path']); expected=reference.get('sha256')
            row=dict(path=str(path),expected_sha256=expected,role=reference.get('role'),current_sha256=None,archived=False)
            try:
                payload=external_row(path,'external/references/'+str(index).zfill(2)+'/'+path.name)
            except (OSError,ValueError) as error:
                row.update(status='unavailable_or_not_unlinked_E_D_regular_file',error=str(error))
            else:
                extras[payload['path']]=payload
                row.update(current_sha256=payload['sha256'],bytes=payload['bytes'],archived=True,
                           archive_path=payload['path'],status='matches_expected' if payload['sha256']==expected else 'current_bytes_differ_from_recorded_expected')
            statuses.append(row)
        current=external_row(POINTER,'GAN/CURRENT_SIM2REAL_PROGRAM.json'); extras[current['path']]=current
        bundle=output/'N01_LOCAL_HISTORY.bundle'
        git(['bundle','create',str(bundle),'--all','HEAD'])
        verify_text=git(['bundle','verify',str(bundle)])
        write_new(output/'GIT_BUNDLE_VERIFICATION.json',dict(UTC=utc(),native_exit_code=0,stdout=verify_text,
            refs_and_HEAD_included=True,raw_git_metadata_also_in_workspace_snapshot=True))
        bundled=external_row(bundle,'metadata/N01_LOCAL_HISTORY.bundle'); extras[bundled['path']]=bundled
        stable_tree(workspace); commit_gate(args.expected_commit)
        inventory=dict(schema='n01/private-full-byte-inventory/v1',root=str(ROOT),snapshot_started_UTC=snapshot_started,
            commit=commit,report_QA=report,workspace=list(workspace.values()),external=list(extras.values()),
            exclusions=exclusions,symlinks_traversed=False,scientific_arrays_decoded=0,scientific_model_calls=0,
            content_scope='Main regular-file byte streams, explicit link text and lstat metadata; not an NTFS ACL/alternate-stream image.',
            snapshot_boundary='Inventoried source bytes remain unchanged through write and streaming reopen. Later final-ledger/receipt changes belong to a separate supplement.')
        external=dict(schema='n01/external-reference-archive-status/v1',references=statuses,
                      training_container_sha256=raw['sha256'],CURRENT_SIM2REAL_PROGRAM_sha256=current['sha256'])
        write_new(output/'INVENTORY.json',inventory);write_new(output/'EXTERNAL_REFERENCE_STATUS.json',external)
        entries={'N01/'+name:row for name,row in workspace.items()}; entries.update(extras)
        metadata={'_ARCHIVE_METADATA/INVENTORY.json':json_bytes(inventory),
                  '_ARCHIVE_METADATA/EXTERNAL_REFERENCE_STATUS.json':json_bytes(external)}
        size_bound=tar_upper_size(entries)+sum(len(v)+4096 for v in metadata.values())+tarfile.RECORDSIZE
        require(shutil.disk_usage('E:/').free>=size_bound+4*GIB,'Insufficient E space after exact inventory/bundle sizing')
        pending=output/'N01_PRIVATE_FULL.partial.tar';expected=write_tar(pending,entries,metadata)
        reopened=verify_tar(pending,expected)
        stable_tree(workspace);commit_gate(args.expected_commit)
        for row in extras.values():
            require(fingerprint(Path(row['source']).stat())==row['fingerprint'],'External input changed during archive')
        write_new(output/'ARCHIVE_MEMBER_VERIFICATION.json',reopened)
        archive=output/'N01_PRIVATE_FULL.tar';pending.rename(archive)
        try:
            backup=d_backup(output,workspace,bundle,report)
        except (OSError,ValueError) as error:
            backup=dict(status='optional_D_source_backup_failed_preserved',error=repr(error),
                        partial_copy_path=str(D_ROOT/output.name/'source_only'),copied_files_not_deleted=True)
        stable_tree(workspace);commit_gate(args.expected_commit)
        receipt=dict(schema='n01/window008-resume01-private-archive/v1',UTC=utc(),status='private_full_snapshot_reopened_and_verified',
            root=str(ROOT),archive=str(archive),archive_bytes=reopened['archive_bytes'],archive_sha256=reopened['archive_sha256'],
            inventory_sha256=sha(output/'INVENTORY.json'),members_verified=reopened['members_verified'],
            workspace_entries=len(workspace),workspace_files=sum(x['kind']=='file' for x in workspace.values()),
            workspace_bytes=source_bytes,external_reference_status=str(output/'EXTERNAL_REFERENCE_STATUS.json'),
            external_reference_all_available=all(x['archived'] for x in statuses),
            external_reference_all_expected_hashes_match=all(x['status']=='matches_expected' for x in statuses),
            git_bundle=str(bundle),git_bundle_sha256=sha(bundle),commit=commit,report_QA=report,
            E_free_before=free_before,E_free_after=shutil.disk_usage('E:/').free,D_backup=backup,
            source_receipts_written_inside_N01=False,final_ledger_receipt_supplement_allowed=True,
            no_scientific_array_decode=True,no_model_GPU_or_optimizer_calls=True,no_deletion_or_overwrite=True)
        write_new(output/'ARCHIVE_RECEIPT.json',receipt)
        print(json.dumps(dict(status=receipt['status'],receipt=str(output/'ARCHIVE_RECEIPT.json'),archive=str(archive),sha256=receipt['archive_sha256'])))
    except BaseException as error:
        write_new(output/'FAILED.json',dict(UTC=utc(),status='archive_attempt_failed_preserved',error=repr(error),
            partial_files_preserved=True,automatic_retry_allowed=False))
        raise


def supplement_allowed(name):
    if name in ('PROGRAM.json','jobs.json'): return True
    return name.startswith('receipts/'+RUN_ID+'/') and Path(name).suffix.lower() in {'.json','.md','.txt','.log'}


def supplement(args):
    archive_root=require_unlinked(Path(args.supplement_archive).absolute())
    require(archive_root.is_relative_to(PRIVATE_ROOT.absolute()),'Supplement must bind an E private N01 archive')
    receipt=read(archive_root/'ARCHIVE_RECEIPT.json'); inventory=read(archive_root/'INVENTORY.json')
    require(receipt['status']=='private_full_snapshot_reopened_and_verified'
            and sha(archive_root/'INVENTORY.json')==receipt['inventory_sha256'], 'Original snapshot receipt/inventory differs')
    base=receipt['commit']['commit'];commit=commit_gate(args.expected_commit)
    require(sha(receipt['archive'])==receipt['archive_sha256'],'Original full archive bytes changed')
    current,exclusions=scan(ROOT);baseline={row['path']:row for row in inventory['workspace']}
    entries={};changes=[]
    for name,old in baseline.items():
        if name.startswith('.git/'): continue
        require(name in current,'Supplement cannot conceal a deleted snapshot file: '+name)
        row=current[name]
        if row['kind']!=old['kind'] or row.get('link_target')!=old.get('link_target'):
            raise ValueError('Supplement cannot change file/link types: '+name)
        if row['kind']=='file' and row['fingerprint']!=old['fingerprint']:
            hash_row(row)
            if row['sha256']!=old['sha256']:
                require(supplement_allowed(name),'Non-ledger/receipt snapshot bytes changed; use a fresh full snapshot: '+name)
                entries['N01/'+name]=row;changes.append(name)
    for name,row in current.items():
        if name in baseline or name.startswith('.git/') or row['kind']=='directory': continue
        require(row['kind']=='file' and supplement_allowed(name),'New non-receipt payload requires a fresh full snapshot: '+name)
        hash_row(row);entries['N01/'+name]=row;changes.append(name)
    pointer=external_row(POINTER,'GAN/CURRENT_SIM2REAL_PROGRAM.json');entries[pointer['path']]=pointer
    require(sum(x['bytes'] for x in entries.values())<=SUPPLEMENT_LIMIT,'Final receipt/ledger supplement exceeds64MiB')
    output=e_path(args.out or archive_root/'supplements'/('final_'+stamp()),inside=archive_root)
    require(shutil.disk_usage('E:/').free>=SUPPLEMENT_LIMIT+GIB,'Supplement needs64MiB maximum payload plus1GiB E margin')
    output.mkdir(parents=True,exist_ok=False)
    write_new(output/'ATTEMPT.json',dict(UTC=utc(),kind='final_ledger_receipt_supplement',base_commit=base,current_commit=args.expected_commit))
    try:
        if base!=args.expected_commit:
            git(['merge-base','--is-ancestor',base,args.expected_commit])
            tracked_changes=git(['diff','--name-only',base,args.expected_commit]).splitlines()
            require(all(supplement_allowed(name) for name in tracked_changes),'Post-snapshot commit changes more than final ledger/receipts')
            delta=output/'FINAL_LEDGER_HISTORY.bundle'
            git(['bundle','create',str(delta),base+'..HEAD']);git(['bundle','verify',str(delta)])
            row=external_row(delta,'metadata/FINAL_LEDGER_HISTORY.bundle');entries[row['path']]=row
        else: tracked_changes=[]
        require(sum(x['bytes'] for x in entries.values())<=SUPPLEMENT_LIMIT,'Supplement including Git delta exceeds64MiB')
        require(shutil.disk_usage('E:/').free>=tar_upper_size(entries)+GIB,'Supplement requires1GiB E margin')
        value=dict(schema='n01/final-ledger-receipt-supplement-inventory/v1',UTC=utc(),base_archive=receipt['archive'],
            base_archive_sha256=receipt['archive_sha256'],base_inventory_sha256=receipt['inventory_sha256'],
            base_commit=base,commit=commit,changed_workspace_paths=changes,tracked_commit_paths=tracked_changes,
            files=list(entries.values()),no_duplicate_full_workspace=True,exclusions=exclusions)
        write_new(output/'SUPPLEMENT_INVENTORY.json',value)
        pending=output/'FINAL_LEDGER_RECEIPTS.partial.tar'
        expected=write_tar(pending,entries,{'_ARCHIVE_METADATA/SUPPLEMENT_INVENTORY.json':json_bytes(value)})
        verified=verify_tar(pending,expected);stable_tree(current);commit_gate(args.expected_commit)
        require(fingerprint(POINTER.stat())==pointer['fingerprint'],'Final external program pointer changed during supplement')
        final=output/'FINAL_LEDGER_RECEIPTS.tar';pending.rename(final)
        write_new(output/'MEMBER_VERIFICATION.json',verified)
        result=dict(schema='n01/window008-resume01-private-supplement/v1',UTC=utc(),status='final_metadata_supplement_reopened_and_verified',
                    archive=str(final),archive_sha256=verified['archive_sha256'],archive_bytes=verified['archive_bytes'],
                    base_archive_sha256=receipt['archive_sha256'],members_verified=verified['members_verified'],
                    commit=commit,changed_workspace_paths=changes,source_receipts_written_inside_N01=False,
                    no_deletion_or_overwrite=True,no_scientific_array_decode=True)
        write_new(output/'SUPPLEMENT_RECEIPT.json',result)
        print(json.dumps(dict(status=result['status'],receipt=str(output/'SUPPLEMENT_RECEIPT.json'),sha256=result['archive_sha256'])))
    except BaseException as error:
        write_new(output/'FAILED.json',dict(UTC=utc(),error=repr(error),partial_files_preserved=True,automatic_retry_allowed=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit',required=True,help='Exact committed HEAD after reportQA or final ledger reconciliation')
    parser.add_argument('--out',help='Fresh E private archive/supplement child; default timestamped directory')
    parser.add_argument('--supplement-archive',help='Existing verified full archive directory; capture only final ledger/receipts and Git delta')
    parser.add_argument('--report-pdf');parser.add_argument('--report-pdf-sha256')
    parser.add_argument('--report-qa');parser.add_argument('--report-qa-sha256')
    args=parser.parse_args()
    if args.supplement_archive: supplement(args)
    else:
        require(all((args.report_pdf,args.report_pdf_sha256,args.report_qa,args.report_qa_sha256)),
                'Full archive requires final reportPDF and reportQA paths plus exact SHA256 values')
        full(args)


if __name__=='__main__':main()
