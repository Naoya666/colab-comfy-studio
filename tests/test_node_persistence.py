"""Offline lifecycle acceptance tests. Real archives/Git, simulated durable Drive and Manager CLI."""
import hashlib
import json
import shutil
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from app.gpu import node_persistence as p


class Drive:
    def __init__(self, path):
        self.path = path
        self.uploads = []
        self.fail = None

    def get(self, key, target):
        source = self.path / key
        if not source.exists():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return True

    def put(self, source, key):
        if self.fail == key:
            raise RuntimeError('interrupted upload')
        self.uploads.append(key)
        target = self.path / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    remote = Drive(tmp_path / 'drive')
    def new(name):
        root = tmp_path / name
        comfy = root / 'ComfyUI'
        (comfy / 'custom_nodes').mkdir(parents=True)
        obj = p.Persistence(root, comfy, root / 'user', remote)
        return obj
    monkeypatch.setattr(p.Persistence, 'snapshot', lambda self: {'git_custom_nodes': {}})
    return new, remote


def add(obj, name='demo', version=1):
    folder = obj.nodes / name
    folder.mkdir(exist_ok=True)
    (folder / '__init__.py').write_text(f'NODE_CLASS_MAPPINGS = {{}}\nVERSION = {version}\n')
    (folder / 'pyproject.toml').write_text(f'[project]\nversion = "{version}.0"\n')
    return folder


def sync(obj, verified=True):
    if verified:
        obj.boot = p.digest(p.inventory(obj.nodes))
    obj.tick(now=0)
    assert obj.tick(now=31)
    return obj.status()


def test_first_install_and_new_vm_restores_without_clone(setup):
    new, drive = setup
    first = new('first')
    add(first)
    status = sync(first)
    assert status['manifest']['plugins'][0]['version'] == '1.0'
    assert not status['restart_required']
    second = new('second')
    second.restore()
    assert (second.nodes / 'demo/__init__.py').read_bytes() == (first.nodes / 'demo/__init__.py').read_bytes()
    assert not second.nodes.is_symlink()
    assert sync(second)['manifest']['generation'] == status['manifest']['generation']


def test_manager_new_install_survives_death_before_restart_but_is_not_verified(setup):
    new, drive = setup
    first = new('first')
    add(first)
    head = sync(first)['manifest']
    add(first, 'new')
    status = sync(first, verified=False)
    assert status['restart_required'] and status['durable']
    assert json.loads((drive.path / 'head.json').read_text()) == head
    second = new('second')
    second.restore()
    assert (second.nodes / 'new').is_dir()
    assert len(sync(second)['manifest']['plugins']) == 2


def test_update_rollback_disable_delete_are_complete_sets(setup):
    new, drive = setup
    first = new('first')
    add(first)
    v1 = sync(first)['manifest']
    add(first, version=2)
    v2 = sync(first)['manifest']
    assert v2['generation'] != v1['generation']
    rollback = new('rollback')
    rollback.restore(v1['generation'])
    assert 'VERSION = 1' in (rollback.nodes / 'demo/__init__.py').read_text()
    (first.nodes / 'demo').rename(first.nodes / 'demo.disabled')
    assert not sync(first)['manifest']['plugins'][0]['enabled']
    disabled = new('disabled')
    disabled.restore()
    assert (disabled.nodes / 'demo.disabled').is_dir()
    shutil.rmtree(first.nodes / 'demo.disabled')
    assert sync(first)['manifest']['plugins'] == []
    empty = new('empty')
    add(empty, 'stale')
    empty.restore()
    assert list(empty.nodes.iterdir()) == []


def test_corrupt_remote_archive_fails_before_replacing_local_nodes(setup):
    new, drive = setup
    first = new('first')
    add(first)
    manifest = sync(first)['manifest']
    archive = manifest['plugins'][0]['archive']
    (drive.path / 'objects' / (archive + '.tar.gz')).write_bytes(b'broken')
    second = new('second')
    add(second, 'existing')
    with pytest.raises(RuntimeError, match='Corrupt plugin cache'):
        second.restore()
    assert (second.nodes / 'existing').is_dir()
    assert not (second.nodes / 'demo').exists()


def test_corrupt_local_cache_redownloads_verified_remote(setup):
    new, drive = setup
    first = new('first')
    add(first)
    manifest = sync(first)['manifest']
    archive = first.cache / (manifest['plugins'][0]['archive'] + '.tar.gz')
    archive.write_bytes(b'broken')
    first.restore()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == archive.stem.split('.')[0]


def test_interrupted_commit_keeps_previous_head_and_retries(setup):
    new, drive = setup
    first = new('first')
    add(first)
    v1 = sync(first)['manifest']
    add(first, version=2)
    drive.fail = 'pending.json'
    with pytest.raises(RuntimeError, match='interrupted'):
        sync(first)
    assert json.loads((drive.path / 'head.json').read_text()) == v1
    drive.fail = None
    v2 = sync(first)['manifest']
    assert v2['generation'] != v1['generation']


def test_unstable_or_busy_install_does_not_publish(setup):
    new, drive = setup
    first = new('first')
    add(first)
    first.tick(now=0)
    first.tick(now=29)
    add(first, version=2)
    first.tick(now=32)
    first.tick(busy=True, now=60)
    assert drive.uploads == []
    first.tick(now=91)
    assert 'pending.json' in drive.uploads
    assert 'head.json' not in drive.uploads


def test_unchanged_restart_reuses_archives(setup):
    new, drive = setup
    first = new('first')
    add(first)
    sync(first)
    count = len([x for x in drive.uploads if x.startswith('objects/')])
    for i in range(3):
        other = new('restart' + str(i))
        other.restore()
        sync(other)
    assert len([x for x in drive.uploads if x.startswith('objects/')]) == count


def test_git_origin_commit_and_manager_update_support_survive_archive(setup):
    new, drive = setup
    first = new('first')
    node = add(first)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(node), *args], stderr=subprocess.DEVNULL, text=True).strip()
    git('init')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.test', 'add', '.')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.test', 'commit', '-m', 'first')
    git('remote', 'add', 'origin', 'https://github.com/example/demo.git')
    commit = git('rev-parse', 'HEAD')
    manifest = sync(first)['manifest']
    assert manifest['plugins'][0]['commit'] == commit
    second = new('second')
    second.restore()
    restored = p.inventory(second.nodes)[0]
    assert restored['commit'] == commit
    assert restored['repo'] == 'https://github.com/example/demo.git'
    assert restored['checksum'] == manifest['plugins'][0]['checksum']


def test_credentials_symlinks_and_traversal_are_rejected(setup, tmp_path):
    new, drive = setup
    first = new('first')
    node = add(first)
    (node / '.env').write_text('sensitive=hidden')
    assert '.env' not in [str(path.relative_to(node)) for path, _ in p.files(node)]
    (node / 'config.json').write_text('{"api_key": "secret123456789"}')
    with pytest.raises(RuntimeError, match='credential'):
        sync(first)
    (node / 'config.json').unlink()
    (node / 'link').symlink_to(tmp_path)
    # Directory links must fail, never silently omit code needed for restore.
    with pytest.raises(RuntimeError, match='symlink'):
        list(p.files(node))


@pytest.mark.parametrize('returncode,log', [(1, 'pip cannot resolve dependency'), (0, 'ERROR: Cannot install incompatible torch')])
def test_dependency_errors_report_plugins_and_do_not_stamp(setup, monkeypatch, returncode, log):
    new, drive = setup
    first = new('first')
    add(first)
    monkeypatch.setattr(p, 'manager_cli', lambda comfy: Path('/fake/cm-cli.py'))
    monkeypatch.setattr(p, 'protect_environment', lambda root: {'torch': '2.8'})
    monkeypatch.setattr(p.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=returncode, stdout=log, stderr=''))
    with pytest.raises(RuntimeError, match='Manager dependency restore failed'):
        first.dependencies()
    assert not (first.root / 'plugin-dependencies.json').exists()
    error = p.read(first.root / 'plugin-errors.json')
    assert error['plugins'] == ['demo']
    assert log in (first.root / 'plugin-dependencies.log').read_text()


def test_dependency_stamp_is_local_and_respects_gpu_version(setup, monkeypatch):
    new, drive = setup
    first = new('first')
    add(first)
    calls = []
    gpu = {'torch': '2.8'}
    monkeypatch.setattr(p, 'manager_cli', lambda comfy: Path('/fake/cm-cli.py'))
    monkeypatch.setattr(p, 'protect_environment', lambda root: dict(gpu))
    def run(args, **kw):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(p.subprocess, 'run', run)
    first.dependencies()
    first.dependencies()
    assert len(calls) == 2  # One Manager restore plus pip check.
    gpu['torch'] = '2.9'
    first.dependencies()
    assert len(calls) == 4
    assert all('restore-snapshot' not in c for c in calls)


def test_manager_exec_restart_verifies_dependencies_even_when_pid_is_unchanged(setup, monkeypatch):
    new, _ = setup
    instance = new('same-pid')
    log = instance.root / 'comfy.log'
    log.write_text('previous startup\n')
    p.atomic(instance.root / 'plugins-config.json', {
        'pid': p.os.getpid(), 'log': str(log), 'comfy': str(instance.comfy), 'user': str(instance.user),
    })
    calls = []
    monkeypatch.setattr(p, 'protect_environment', lambda root: {})
    monkeypatch.setattr(p.Persistence, 'dependencies', lambda self: calls.append(self.root))
    p.prestartup(instance.root)
    first = p.read(instance.root / 'plugins-config.json')['log_offset']
    log.write_text(log.read_text() + 'new process startup\n')
    p.prestartup(instance.root)
    assert len(calls) == 2
    assert p.read(instance.root / 'plugins-config.json')['log_offset'] > first


def test_vps_manifest_is_committed_in_existing_settings_only_after_drive_head(setup, monkeypatch):
    from app.db import Store
    from app.gpu import native
    from app.gpu.repository import Repository
    new, drive = setup
    first = new('first')
    add(first)
    status = sync(first)
    store = Store('sqlite:///:memory:')
    repo = Repository(store)
    monkeypatch.setattr(native, 'backend', lambda: ('https://internal.example', {}))
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={'plugins': status}))) as client:
        native.sync_plugins(repo, {'owner_id': 'owner'}, client)
    row = store.list('settings', 'owner')[0]
    assert row['manifest'] == json.loads((drive.path / 'head.json').read_text())


def test_modern_manager_disabled_git_repo_and_single_file_restore(setup):
    new, drive = setup
    first = new('first')
    node = add(first)
    subprocess.run(['git', 'init', str(node)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(node), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(node), '-c', 'user.name=Test', '-c', 'user.email=test@example.test',
                    'commit', '-m', 'initial'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(node), 'remote', 'add', 'origin', 'https://github.com/example/demo'], check=True)
    disabled = first.nodes / '.disabled'
    disabled.mkdir()
    node.rename(disabled / 'demo')
    (first.nodes / 'single.py.disabled').write_text('NODE_CLASS_MAPPINGS = {}\n')
    manifest = sync(first)['manifest']
    assert not any(item['enabled'] for item in manifest['plugins'])
    assert manifest['plugins'][0]['commit']
    second = new('second')
    second.restore()
    assert (second.nodes / '.disabled/demo/.git/shallow').exists()
    assert (second.nodes / 'single.py.disabled').is_file()


def test_official_cli_false_return_becomes_failure(tmp_path):
    fake = tmp_path / 'cm-cli.py'
    fake.write_text('''class Manager:
    def execute_install_script(self, *args, **kwargs):
        return False
unified_manager = Manager()
def app():
    unified_manager.execute_install_script('', '/custom_nodes/broken')
''')
    result = subprocess.run([__import__('sys').executable, '-c', p.DEPENDENCY_LAUNCHER, str(fake),
                             'restore-dependencies'], capture_output=True, text=True)
    assert result.returncode == 1
    assert '/custom_nodes/broken' in result.stdout


def test_corrupt_generation_can_recover_using_an_intact_historical_version(setup):
    new, drive = setup
    first = new('first')
    add(first)
    good = sync(first)['manifest']
    add(first, version=2)
    broken = sync(first)['manifest']
    (drive.path / 'objects' / (broken['plugins'][0]['archive'] + '.tar.gz')).write_bytes(b'broken')
    second = new('second')
    with pytest.raises(RuntimeError, match='Corrupt'):
        second.restore()
    second.restore(good['generation'])
    assert 'VERSION = 1' in (second.nodes / 'demo/__init__.py').read_text()


def test_archive_traversal_is_rejected_even_with_valid_object_hash(setup, tmp_path):
    import io
    import tarfile
    new, drive = setup
    first = new('first')
    archive = tmp_path / 'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as target:
        entry = tarfile.TarInfo('../../escaped')
        entry.size = 1
        target.addfile(entry, io.BytesIO(b'x'))
    item = {'path': 'demo', 'archive': p.file_hash(archive), 'checksum': 'fake', 'enabled': True}
    drive.put(archive, 'objects/' + item['archive'] + '.tar.gz')
    manifest = {'schema': 1, 'plugins': [item], 'generation': p.digest([item])}
    first.publish_json(manifest, 'pending.json')
    with pytest.raises(RuntimeError, match='Unsafe plugin archive'):
        first.restore()
    assert not (first.root / 'escaped').exists()


def test_pack_does_not_include_repository_history_or_git_credentials(setup):
    new, drive = setup
    first = new('first')
    node = add(first)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(node), '-c', 'user.name=Test',
                                        '-c', 'user.email=test@example.test', *args], stderr=subprocess.DEVNULL)
    git('init')
    (node / 'old.txt').write_text('refresh_token=historicalSecret12345')
    git('add', '.')
    git('commit', '-m', 'old')
    (node / 'old.txt').unlink()
    git('add', '.')
    git('commit', '-m', 'clean')
    git('remote', 'add', 'origin', 'https://github.com/example/demo.git')
    git('config', 'http.extraheader', 'Authorization: Bearer private-git-token')
    manifest = sync(first)['manifest']
    second = new('second')
    second.restore()
    config = (second.nodes / 'demo/.git/config').read_text()
    assert 'private-git-token' not in config
    result = subprocess.run(['git', '-C', str(second.nodes / 'demo'), 'rev-list', 'HEAD'], capture_output=True, text=True)
    assert result.stdout.splitlines() == [manifest['plugins'][0]['commit']]


def test_downloaded_csv_dictionary_has_bounded_larger_allowance(tmp_path):
    node = tmp_path / 'autocomplete'
    node.mkdir()
    dictionary = node / 'tags.csv'
    with dictionary.open('wb') as stream:
        stream.truncate(65 * 1024**2)
    assert [path for path, _ in p.files(node)] == [dictionary]
    dictionary.rename(node / 'weights.bin')
    with pytest.raises(RuntimeError, match='64 MiB'):
        list(p.files(node))
    (node / 'weights.bin').unlink()
    with dictionary.open('wb') as stream:
        stream.truncate(129 * 1024**2)
    with pytest.raises(RuntimeError, match='128 MiB'):
        list(p.files(node))


def test_inventory_error_is_reported_without_killing_status_polling(setup, monkeypatch):
    new, _ = setup
    obj = new('runtime')
    add(obj)
    original = p.inventory
    def broken(_):
        raise RuntimeError('Plugin source file exceeds limit')
    monkeypatch.setattr(p, 'inventory', broken)
    status = obj.status()
    assert status['durable'] is False
    assert status['error'] == 'Plugin source file exceeds limit'
    monkeypatch.setattr(p, 'inventory', original)
    assert obj.status()['error'] is None


def test_cached_scan_detects_secret_added_with_same_size_and_restored_mtime(tmp_path):
    import os
    node = tmp_path / 'node'
    node.mkdir()
    source = node / 'config.txt'
    source.write_bytes(b'x' * 64)
    list(p.files(node))
    before = source.stat()
    source.write_bytes(b'api_key="secret123456789"'.ljust(64, b' '))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(RuntimeError, match='credential'):
        list(p.files(node))


@pytest.mark.parametrize('source', [
    'api_key = settings.github_token\n',
    'headers = {"api_key": get_token()}\n',
    'client(api_key=settings.token)\n',
    'api_key: str = os.environ["TOKEN"]\n',
    'label = "中文"; api_key = settings.token\n',
    'class HF_TransformersClassifierProvider: pass\n',
    'nodes = {"Provider": HF_TransformersClassifierProvider}\n',
    'if proxy_password:\n    full_proxy_url = f"{proxy_password}@{host}"\n',
    'api_key = settings.token or None\n',
    'api_key = settings.token if enabled else None\n',
])
def test_python_dynamic_values_are_not_credentials(tmp_path, source):
    p.check_credentials(source.encode(), tmp_path / 'github.py')


@pytest.mark.parametrize('source', [
    'api_key = "secret123456789"\n',
    'headers = {"api_key": "secret123456789"}\n',
    'client(api_key="secret123456789")\n',
    'api_key = settings.token\nrefresh_token = "secret123456789"\n',
    'token = "ghp_12345678901234567890"\n',
    'token = "hf_12345678901234567890"\n',
    'API_KEY = "secret123456789"\n',
    '# api_key=secret123456789\n',
    'description = "api_key=settings.github_token"\n',
    'api_key = settings.token\nthis is invalid python!\n',
])
def test_literals_comments_and_parse_failures_remain_blocked(tmp_path, source):
    with pytest.raises(RuntimeError, match='credential'):
        p.check_credentials(source.encode(), tmp_path / 'github.py')


def test_dynamic_exemption_is_python_only(tmp_path):
    with pytest.raises(RuntimeError, match='credential'):
        p.check_credentials(b'api_key = settings.github_token', tmp_path / 'config.txt')


def test_git_archive_dynamic_source_restores_but_committed_secret_blocks(setup):
    new, drive = setup
    first = new('first')
    node = add(first)
    source = node / 'github.py'
    source.write_text('api_key = settings.github_token\n')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(node), '-c', 'user.name=Test',
                                        '-c', 'user.email=test@example.test', *args], stderr=subprocess.DEVNULL)
    git('init')
    git('add', '.')
    git('commit', '-m', 'dynamic')
    head = sync(first)['manifest']
    second = new('second')
    second.restore()
    assert (second.nodes / 'demo/github.py').read_bytes() == source.read_bytes()
    source.write_text('api_key = "secret123456789"\n')
    git('add', '.')
    git('commit', '-m', 'unsafe')
    source.write_text('api_key = settings.github_token\n')
    with pytest.raises(RuntimeError, match='credential'):
        sync(first)
    assert json.loads((drive.path / 'head.json').read_text()) == head


@pytest.mark.asyncio
async def test_boot_scan_failure_keeps_loop_alive_and_recovers_after_repair(setup, monkeypatch):
    import asyncio
    import io
    new, drive = setup
    obj = new('runtime')
    node = add(obj)
    source = node / 'github.py'
    source.write_text('api_key = "secret123456789"\n')
    log = obj.root / 'comfy.log'
    log.write_text('startup complete\n')
    p.atomic(obj.root / 'plugins-config.json', {
        'comfy': str(obj.comfy), 'user': str(obj.user), 'log': str(log), 'log_offset': 0,
    })
    monkeypatch.setattr(p, 'Remote', lambda root: drive)
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: io.BytesIO(b'{"system": {"os": "linux"}}'))
    clock = [0]
    monkeypatch.setattr(p.time, 'monotonic', lambda: clock[0])
    iteration = [0]
    def busy():
        return False
    async def sleep(seconds):
        assert seconds == 10
        iteration[0] += 1
        status = p.read(obj.root / 'plugins-status.json')
        if iteration[0] == 1:
            assert status['durable'] is False
            assert 'credential' in status['error']
            assert not (drive.path / 'head.json').exists()
            source.write_text('api_key = settings.github_token\n')
        if iteration[0] == 3:
            assert status['durable'] is True
            assert status['error'] is None
            assert status['restart_required'] is True
            assert (drive.path / 'pending.json').exists()
            assert not (drive.path / 'head.json').exists()
            raise asyncio.CancelledError
        clock[0] += 31
    monkeypatch.setattr(asyncio, 'sleep', sleep)
    tasks = []
    server = SimpleNamespace(loop=SimpleNamespace(create_task=lambda coro: tasks.append(asyncio.create_task(coro))))
    persistence = p.install(server, obj.root, busy)
    assert persistence is not None
    with pytest.raises(asyncio.CancelledError):
        await tasks[0]
    assert iteration[0] == 3


def test_small_file_validation_reuses_only_content_identical_scans(tmp_path, monkeypatch):
    source = tmp_path / 'asset.bin'
    source.write_bytes(b'x' * (512 * 1024))
    original = p.SECRET
    scans = []
    class Pattern:
        def search(self, data):
            scans.append(len(data))
            return original.search(data)
        def finditer(self, data):
            return original.finditer(data)
    monkeypatch.setattr(p, 'SECRET', Pattern())
    for _ in range(3):
        p.checked_source(source, p.source_stamp(source))
    assert len(scans) == 1
    before = source.stat()
    source.write_bytes(b'api_key="secret123456789"'.ljust(512 * 1024, b' '))
    p.os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(RuntimeError, match='credential'):
        p.checked_source(source, p.source_stamp(source))


def test_content_change_between_hash_and_validation_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / 'config.py'
    source.write_text('value = 1\n')
    original = p.file_hash
    def changed(path):
        checksum = original(path)
        path.write_text('value = 2\n')
        return checksum
    monkeypatch.setattr(p, 'file_hash', changed)
    with pytest.raises(RuntimeError, match='changed during scan'):
        p.checked_source(source, p.source_stamp(source))


def test_plugin_backup_captures_committed_wal_data_without_archiving_sidecars(setup):
    new, _ = setup
    instance = new('sqlite-wal')
    node = add(instance, 'weilin-demo')
    data_dir = node / 'user_data'
    data_dir.mkdir()
    db = data_dir / 'tags.db'
    with closing(sqlite3.connect(db)) as writer:
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('CREATE TABLE tags (name TEXT)')
        writer.commit()
        writer.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        original_main = db.read_bytes()
        original = sync(instance)['manifest']['generation']
        assert instance.status(use_cache=True)['durable'] is True

        writer.execute("INSERT INTO tags VALUES ('committed')")
        writer.commit()
        assert db.read_bytes() == original_main
        assert instance.status(use_cache=True)['durable'] is False
        writer.execute("INSERT INTO tags VALUES ('uncommitted')")
        latest = sync(instance)['manifest']['generation']
        assert latest != original

        restored = new('sqlite-wal-restored')
        restored.restore()
        restored_db = restored.nodes / 'weilin-demo/user_data/tags.db'
        assert [entry.name for entry in restored_db.parent.iterdir()] == ['tags.db']
        assert sync(restored)['manifest']['generation'] == latest
        with closing(sqlite3.connect(restored_db)) as reader:
            assert reader.execute('SELECT name FROM tags').fetchall() == [('committed',)]


def test_binary_magic_skips_regex_while_spoofed_png_secret_is_blocked(tmp_path):
    node = tmp_path / 'media-node'
    node.mkdir()
    genuine_png = node / 'preview.png'
    genuine_png.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 128 + b'api_key="not_a_text_file_123456"')
    genuine_ttf = node / 'font.ttf'
    genuine_ttf.write_bytes(b'\x00\x01\x00\x00' + b'\x00' * 128)
    assert p.is_verified_binary(genuine_png, genuine_png.read_bytes()[:64]) is True
    assert p.is_verified_binary(genuine_ttf, genuine_ttf.read_bytes()[:64]) is True
    entries = dict(p.files(node))
    assert genuine_png in entries and genuine_ttf in entries

    # Spoofed extension: plain text secret renamed to .png lacks PNG magic header and MUST be blocked!
    spoofed = node / 'spoofed.png'
    spoofed.write_bytes(b'api_key = "secret123456789"\n')
    assert p.is_verified_binary(spoofed, spoofed.read_bytes()[:64]) is False
    with pytest.raises(RuntimeError, match='credential'):
        list(p.files(node))


def test_restore_dependencies_prestartup_deduplicates_scans_and_rechecks_on_reboot(setup, monkeypatch):
    new, _ = setup
    first = new('first')
    add(first, 'node-a')
    add(first, 'node-b')
    sync(first)

    second = new('second')
    log = second.root / 'comfy.log'
    log.write_text('boot\n')
    p.atomic(second.root / 'plugins-config.json', {
        'comfy': str(second.comfy), 'user': str(second.user), 'log': str(log), 'log_offset': 0,
    })

    monkeypatch.setattr(p, 'manager_cli', lambda comfy: Path('/fake/cm-cli.py'))
    monkeypatch.setattr(p, 'protect_environment', lambda root: {'torch': '2.8'})
    monkeypatch.setattr(p.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout='', stderr=''))

    scan_calls = []
    orig_inventory = p.inventory

    def counting_inventory(*a, **kw):
        scan_calls.append((Path(a[0]).name, kw))
        return orig_inventory(*a, **kw)

    monkeypatch.setattr(p, 'inventory', counting_inventory)

    restored = second.restore()
    assert restored is not None and len(restored) == 2
    assert len(scan_calls) == 1  # Only stage hash verification in restore()

    dep_key = second.dependencies(plugins=restored)
    assert dep_key is not None
    assert len(scan_calls) == 1  # Zero redundant scan in dependencies()!

    p.mark_prestartup_verified(second.root, second.nodes, dep_key)
    p.prestartup(second.root)
    assert len(scan_calls) == 1  # Zero redundant scan on initial ComfyUI prestartup()!

    # Second prestartup() (e.g. Manager reboot) consumes the one-shot token and verifies again!
    p.prestartup(second.root)
    assert len(scan_calls) == 2


def test_parallel_restore_uses_rclone_files_from_and_verifies_sha256(setup, tmp_path, monkeypatch):
    new, drive = setup
    first = new('first')
    add(first, 'alpha', version=1)
    add(first, 'beta', version=2)
    manifest = sync(first)['manifest']
    assert len(manifest['plugins']) == 2

    target_vm = tmp_path / 'vm-rclone'
    comfy = target_vm / 'ComfyUI'
    (comfy / 'custom_nodes').mkdir(parents=True)
    (target_vm / 'rclone').write_text('#!/bin/sh\n')
    (target_vm / 'rclone.conf').write_text('[studio]\n')

    rclone_calls = []

    class BatchRemote(p.Remote):
        def get(self, key, target):
            return drive.get(key, target)

        def command(self, *args, timeout=600):
            rclone_calls.append(args)
            assert args[0] == 'copy' and '--files-from-raw' in args
            list_file = Path(args[args.index('--files-from-raw') + 1])
            dest_dir = Path(args[-1])
            for name in list_file.read_text().splitlines():
                if name.strip():
                    shutil.copyfile(drive.path / 'objects' / name.strip(), dest_dir / name.strip())
            return SimpleNamespace(returncode=0, stdout='', stderr='')

    pers = p.Persistence(target_vm, comfy, target_vm / 'user', BatchRemote(target_vm))
    restored = pers.restore()
    assert len(rclone_calls) == 1  # Single batch rclone copy for both plugins!
    assert [x['path'] for x in restored] == ['alpha', 'beta']

    # Corrupt remote archive still rejected even when returned by batch rclone copy
    bad_vm = tmp_path / 'vm-corrupt'
    (bad_vm / 'ComfyUI' / 'custom_nodes').mkdir(parents=True)
    (bad_vm / 'rclone').write_text('#!/bin/sh\n')
    (bad_vm / 'rclone.conf').write_text('[studio]\n')

    class CorruptBatchRemote(BatchRemote):
        def command(self, *args, timeout=600):
            res = super().command(*args, timeout=timeout)
            dest_dir = Path(args[-1])
            next(dest_dir.glob('*.tar.gz')).write_bytes(b'corrupted-bytes')
            return res

    bad_pers = p.Persistence(bad_vm, bad_vm / 'ComfyUI', bad_vm / 'user', CorruptBatchRemote(bad_vm))
    with pytest.raises(RuntimeError, match='Corrupt plugin cache'):
        bad_pers.restore()


def test_dependencies_batch_fast_path_skips_manager_runs_install_py_and_falls_back(setup, monkeypatch):
    new, _ = setup
    obj = new('batch-fast')
    node_a = add(obj, 'plugin-a')
    (node_a / 'requirements.txt').write_text('einops\n')
    (node_a / 'install.py').write_text('print("installed")\n')
    node_b = add(obj, 'plugin-b')
    (node_b / 'requirements.txt').write_text('requests\n')
    disabled_dir = obj.nodes / '.disabled'
    disabled_dir.mkdir()
    disabled_node = disabled_dir / 'plugin-off'
    disabled_node.mkdir()
    (disabled_node / '__init__.py').write_text('NODE_CLASS_MAPPINGS = {}\n')
    (disabled_node / 'install.py').write_text('raise SystemExit(1)\n')

    monkeypatch.setattr(p, 'manager_cli', lambda comfy: Path('/fake/cm-cli.py'))
    monkeypatch.setattr(p, 'protect_environment', lambda root: {'torch': '2.8'})
    monkeypatch.setattr(p.shutil, 'which', lambda name: '/usr/local/bin/uv' if name == 'uv' else None)

    calls = []
    batch_rc = [0]
    install_rc = [0]

    def fake_run(args, **kw):
        calls.append((args, kw.get('cwd')))
        if args[0] == '/usr/local/bin/uv':
            return SimpleNamespace(returncode=batch_rc[0], stdout='', stderr='')
        if len(args) >= 2 and str(args[1]).endswith('install.py'):
            return SimpleNamespace(returncode=install_rc[0], stdout='ok', stderr='bad' if install_rc[0] else '')
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(p.subprocess, 'run', fake_run)

    # 1. Batch uv succeeds -> skips cm-cli, runs ONLY enabled install.py in plugin cwd, then pip check
    key = obj.dependencies()
    assert key is not None
    assert (obj.root / 'plugin-dependencies.json').is_file()
    assert any(c[0][0] == '/usr/local/bin/uv' for c in calls)
    assert not any('restore-dependencies' in c[0] for c in calls)
    install_calls = [c for c in calls if len(c[0]) >= 2 and str(c[0][1]).endswith('install.py')]
    assert len(install_calls) == 1
    assert install_calls[0][1] == node_a
    assert any(c[0][1:4] == ['-m', 'pip', 'check'] for c in calls)

    # 2. install.py failure -> records plugin-errors.json & log, does not stamp or call cm-cli
    (obj.root / 'plugin-dependencies.json').unlink()
    calls.clear()
    install_rc[0] = 1
    with pytest.raises(RuntimeError, match='Plugin install.py failed'):
        obj.dependencies()
    assert not (obj.root / 'plugin-dependencies.json').exists()
    assert (obj.root / 'plugin-errors.json').is_file()
    assert (obj.root / 'plugin-dependencies.log').is_file()
    assert not any('restore-dependencies' in c[0] for c in calls)

    # 3. Batch uv failure -> falls back cleanly to cm-cli restore-dependencies
    calls.clear()
    batch_rc[0] = 1
    install_rc[0] = 0
    key2 = obj.dependencies()
    assert key2 == key
    assert any('restore-dependencies' in c[0] for c in calls)


def test_secret_prefilter_skips_benign_csv_and_hf_identifiers_without_false_negatives():
    assert p._may_contain_secret(b'password,1girl,solo,12345\n') is False
    assert p._may_contain_secret(b'password:1girl,solo\n') is True
    assert p._may_contain_secret(b'api_key = "secret123456789"\n') is True
    assert p._may_contain_secret(b'from huggingface_hub import hf_hub_download\n') is False
    assert p._may_contain_secret(b'token = "hf_1234567890abcdef12"\n') is True
    assert p._may_contain_secret(b'token = "ghp_1234567890abcdef12"\n') is True
    assert p._may_contain_secret(b'-----BEGIN RSA PRIVATE KEY-----\n') is True


@pytest.mark.asyncio
async def test_install_runs_boot_scan_asynchronously_without_blocking_import(setup, monkeypatch):
    import asyncio
    import io
    import threading

    new, drive = setup
    obj = new('async-boot')
    add(obj)
    log = obj.root / 'comfy.log'
    log.write_text('startup complete\n')
    p.atomic(obj.root / 'plugins-config.json', {
        'comfy': str(obj.comfy), 'user': str(obj.user), 'log': str(log), 'log_offset': 0,
    })

    gate = threading.Event()
    orig_inv = p.inventory

    def gated_inventory(nodes, **kw):
        gate.wait(timeout=5)
        return orig_inv(nodes, **kw)

    monkeypatch.setattr(p, 'inventory', gated_inventory)
    monkeypatch.setattr(p, 'Remote', lambda root: drive)
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: io.BytesIO(b'{"system": {"os": "linux"}}'))

    tasks = []
    server = SimpleNamespace(loop=SimpleNamespace(create_task=lambda coro: tasks.append(asyncio.create_task(coro))))
    persistence = p.install(server, obj.root, lambda: False)
    # install() returns immediately while boot scan is still blocked on gate!
    assert not persistence.boot_scan_done.is_set()
    assert persistence.boot is None

    gate.set()
    assert persistence.boot_scan_done.wait(timeout=5)
    tasks[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await tasks[0]

