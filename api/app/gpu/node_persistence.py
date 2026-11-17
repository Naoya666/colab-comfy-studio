"""Native nodes: local execution, immutable Drive objects, restart-verified manifests.

Only stdlib: shipped to Colab alongside native_runtime, never imports the VPS app.
Third-party code runs with the runtime's privileges, not in a security sandbox.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

DEPENDENCY_LAUNCHER = '''import runpy, sys
namespace = runpy.run_path(sys.argv[1], run_name='studio_manager_cli')
manager = namespace['unified_manager']
original = manager.execute_install_script
failed = []
def checked(*args, **kwargs):
    result = original(*args, **kwargs)
    if result is False:
        failed.append(str(args[1] if len(args) > 1 else args))
    return result
manager.execute_install_script = checked
sys.argv = sys.argv[1:]
try:
    namespace['app']()
finally:
    if failed:
        print('ERROR: Manager dependency installation failed: ' + ', '.join(failed))
        raise SystemExit(1)
'''

EXCLUDE = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.env', 'tokens', 'credentials',
           '.cache', '.pytest_cache', 'models', 'checkpoints', 'output', 'input'}
SECRET = re.compile(rb'(?i:(?:access_token|refresh_token|client_secret|api_key|password)["\s]*[:=]["\s]*(?P<assignment_value>[A-Za-z0-9_./+\-]{8,}))|(?:gh[pousr]_|hf_)[A-Za-z0-9]{16,}|-----BEGIN .*PRIVATE KEY-----')


def check_credentials(data, path):
    """Ignore proven Python dynamic values; keep literals and parse failures blocked."""
    dynamic_starts = None
    for match in SECRET.finditer(data):
        if match.group('assignment_value') is not None and path.suffix == '.py':
            if dynamic_starts is None:
                dynamic_starts = set()
                try:
                    tree = ast.parse(data)
                    offsets = [0]
                    for line in data.splitlines(keepends=True):
                        offsets.append(offsets[-1] + len(line))
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
                            dynamic_starts.add(offsets[node.lineno - 1] + node.col_offset)
                except (SyntaxError, ValueError, UnicodeError):
                    pass
            if match.start('assignment_value') in dynamic_starts:
                continue
        raise RuntimeError(f'Possible credential in plugin backup: {path}; remove it before syncing')


def file_hash(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True))
    temp.replace(path)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def safe_url(value):
    p = urlsplit(value)
    if p.scheme == 'https' and p.hostname and not p.username and not p.password and not p.query and not p.fragment:
        return value
    return None


def source_stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=4096)
def _cached_checked_source(path, stamp):
    """Reuse scans only while content metadata (including ctime) is unchanged."""
    checksum = hashlib.sha256()
    with path.open('rb') as source:
        previous = b''
        while block := source.read(1024**2):
            if SECRET.search(previous + block):
                check_credentials(path.read_bytes(), path)
            checksum.update(block)
            previous = block[-512:]
    if source_stamp(path) != stamp:
        raise RuntimeError(f'Plugin file changed during scan: {path}; retrying after settling')
    return checksum.hexdigest()


def checked_source(path, stamp):
    # Small code/config writes may share coarse filesystem timestamps. Always scan
    # them; reserve the metadata cache for large assets such as the CSV dictionary.
    scan = _cached_checked_source if stamp[2] >= 1024**2 else _cached_checked_source.__wrapped__
    return scan(path, stamp)


def files(folder):
    """Never follow links or archive credentials, environments, or Git auth config."""
    for base, dirs, names in os.walk(folder, followlinks=False):
        if any((Path(base) / d).is_symlink() for d in dirs if d not in EXCLUDE):
            raise RuntimeError(f'Plugin directory symlink is not supported: {base}')
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE)
        for name in sorted(names):
            path = Path(base) / name
            if name in EXCLUDE or name.startswith('.env') or name.endswith(('.pem', '.key', '.pyc')):
                continue
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f'Unsupported plugin file: {path}')
            limit = 128 if path.suffix.lower() == '.csv' else 64
            if path.stat().st_size > limit * 1024**2:
                raise RuntimeError(f'Plugin source file exceeds {limit} MiB: {path}; store weights under Drive models')
            yield path, checked_source(path, source_stamp(path))


def inventory(nodes):
    result = []
    candidates = []
    for path in sorted(nodes.iterdir()):
        if path.name in {'studio_native_state.py', 'studio_native_guard', '__pycache__'}:
            continue
        if path.name in {'.disabled', 'disabled'} and path.is_dir():
            candidates.extend((node, False) for node in sorted(path.iterdir()))
        elif not path.name.startswith('.'):
            candidates.append((path, not path.name.endswith('.disabled')))
    for node, enabled in candidates:
        if node.is_symlink():
            raise RuntimeError(f'Plugin symlink is not supported: {node}')
        if node.is_file() and not node.name.endswith(('.py', '.py.disabled')):
            continue
        entries = list(files(node)) if node.is_dir() else [(node, file_hash(node))]
        if node.is_file():
            check_credentials(node.read_bytes(), node)
        checks = [(str(path.relative_to(node)) if node.is_dir() else node.name, h) for path, h in entries]
        item = {'path': str(node.relative_to(nodes)), 'checksum': digest(checks),
                'enabled': enabled, 'repo': None, 'commit': None, 'version': None}
        if (node / '.git').exists():
            def git(*args):
                return subprocess.check_output(['git', '-C', str(node), *args], text=True,
                                               stderr=subprocess.DEVNULL).strip()
            item['commit'] = git('rev-parse', 'HEAD')
            try:
                item['repo'] = safe_url(git('config', '--get', 'remote.origin.url'))
            except subprocess.CalledProcessError:
                pass
            if any((node / '.git' / lock).exists() for lock in ('index.lock', 'HEAD.lock', 'shallow.lock')):
                raise RuntimeError(f'Git operation still in progress: {node.name}')
        project = node / 'pyproject.toml'
        if project.is_file():
            item['version'] = tomllib.loads(project.read_text()).get('project', {}).get('version')
        result.append(item)
    return sorted(result, key=lambda item: item['path'])


class Remote:
    """Direct rclone transfers, bypassing asynchronous FUSE writeback for commits."""
    def __init__(self, root):
        self.root = Path(root)

    def command(self, *args):
        return subprocess.run([str(self.root / 'rclone'), '--config', str(self.root / 'rclone.conf'), *args],
                              check=True, capture_output=True, text=True, timeout=600)

    def get(self, key, target):
        try:
            self.command('copyto', '--ignore-times', 'studio:native-comfy/plugins/' + key, str(target))
            return True
        except subprocess.CalledProcessError as exc:
            if exc.returncode in {3, 4}:  # rclone directory/file not found, not network failure
                return False
            raise RuntimeError('Drive download failed; inspect rclone connectivity (credentials omitted)') from None

    def put(self, source, key):
        try:
            self.command('copyto', '--ignore-times', str(source), 'studio:native-comfy/plugins/' + key)
            # Read after write ensures the object is in Drive, not merely queued locally.
            with tempfile.TemporaryDirectory() as temp:
                check = Path(temp) / 'verify'
                if not self.get(key, check) or file_hash(check) != file_hash(source):
                    raise RuntimeError('Drive read-after-write checksum mismatch: ' + key)
        except subprocess.CalledProcessError:
            raise RuntimeError('Drive upload failed (credentials omitted)') from None


def manager_cli(comfy):
    packaged = importlib.util.find_spec('cm_cli')
    if packaged and packaged.origin:
        return Path(packaged.origin).with_name('__main__.py')
    candidates = list((comfy / 'custom_nodes').glob('*/cm-cli.py'))
    spec = importlib.util.find_spec('comfyui_manager')
    if spec and spec.submodule_search_locations:
        for folder in spec.submodule_search_locations:
            candidates.extend(Path(folder).rglob('cm-cli.py'))
    if not candidates:
        raise RuntimeError('Installed Manager has no cm-cli.py; install a compatible pinned Manager release')
    return sorted(candidates)[0]


def protect_environment(root):
    """Pip constraints inherited by Manager and ordinary install.py subprocesses."""
    from importlib.metadata import distributions
    packages = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in distributions() if d.metadata['Name']}
    protected = {k: v for k, v in packages.items()
                 if k in {'torch', 'torchvision', 'torchaudio', 'triton'} or k.startswith('nvidia-')}
    constraints = root / 'gpu-constraints.txt'
    constraints.write_text(''.join(f'{k}=={v}\n' for k, v in sorted(protected.items())))
    os.environ['PIP_CONSTRAINT'] = str(constraints)
    os.environ['UV_CONSTRAINT'] = str(constraints)
    return protected


def prestartup(root):
    """Manager exec-restarts keep their PID; verify every startup, including those."""
    root = Path(root)
    config = read(root / 'plugins-config.json')
    if not config:
        return
    config['log_offset'] = Path(config['log']).stat().st_size
    config['pid'] = os.getpid()
    atomic(root / 'plugins-config.json', config)
    protect_environment(root)
    try:
        Persistence(root, Path(config['comfy']), Path(config['user'])).dependencies()
    except Exception as exc:
        atomic(root / 'plugin-errors.json', {'error': str(exc)})
        raise


class Persistence:
    def __init__(self, root, comfy, user, remote=None):
        self.root, self.comfy, self.user = Path(root), Path(comfy), Path(user)
        self.nodes = self.comfy / 'custom_nodes'
        self.remote = remote or Remote(root)
        self.cache = self.root / 'plugin-cache'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.previous = None
        self.stable_since = 0
        self.boot = None
        self.manifest = read(self.root / 'plugins-verified.json')
        self.error = None

    def download_json(self, key):
        path = self.cache / 'download.json'
        return read(path) if self.remote.get(key, path) else None

    def publish_json(self, value, key):
        path = self.cache / 'upload.json'
        atomic(path, value)
        self.remote.put(path, key)

    def restore(self, generation=None):
        # A candidate survives VM death but cannot replace the verified VPS list until it loads.
        manifest = self.download_json('history/' + generation + '.json') if generation else self.download_json('pending.json') or self.download_json('head.json')
        if not manifest:
            if generation:
                raise RuntimeError('Requested plugin generation not found: ' + generation)
            return
        if manifest.get('schema') != 1 or manifest.get('generation') != digest(manifest['plugins']):
            raise RuntimeError('Invalid plugin manifest')
        stage = self.root / 'plugins-restore'
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir()
        try:
            for item in manifest['plugins']:
                relative = Path(item['path'])
                if not relative.parts or relative.is_absolute() or '..' in relative.parts or len(relative.parts) > 2 or not re.fullmatch(r'[0-9a-f]{64}', item['archive']):
                    raise RuntimeError('Invalid plugin path')
                archive = self.cache / (item['archive'] + '.tar.gz')
                if not archive.exists() or file_hash(archive) != item['archive']:
                    if not self.remote.get('objects/' + archive.name, archive):
                        raise RuntimeError('Missing plugin cache: ' + item['path'])
                if file_hash(archive) != item['archive']:
                    archive.unlink(missing_ok=True)
                    raise RuntimeError('Corrupt plugin cache: ' + item['path'] + '; select a verified historical generation')
                with tarfile.open(archive) as source:
                    members = source.getmembers()
                    if sum(m.size for m in members) > 4 * 1024**3:
                        raise RuntimeError('Plugin archive exceeds 4 GiB')
                    if any(not (m.isfile() or m.isdir()) or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members):
                        raise RuntimeError('Unsafe plugin archive')
                    for member in members:
                        member.mode &= 0o777
                        member.uid = member.gid = 0
                    source.extractall(stage / relative if item.get('kind', 'directory') == 'directory' else stage / relative.parent, members=members)
                node = stage / relative
                bundle = node / '.studio-repo.pack'
                if bundle.is_file():
                    subprocess.run(['git', 'init', str(node)], check=True, capture_output=True)
                    with bundle.open('rb') as packed:
                        subprocess.run(['git', '-C', str(node), 'index-pack', '--stdin'], stdin=packed, check=True, capture_output=True)
                    (node / '.git' / 'shallow').write_text(item['commit'] + '\n')
                    subprocess.run(['git', '-C', str(node), 'reset', '--mixed', item['commit']], check=True, capture_output=True)
                    if item.get('repo'):
                        subprocess.run(['git', '-C', str(node), 'remote', 'add', 'origin', item['repo']], check=True, capture_output=True)
                    bundle.unlink()
                # Verify semantic source checksums as well as compressed object integrity.
            actual = inventory(stage)
            if [(p['path'], p['checksum']) for p in actual] != [(p['path'], p['checksum']) for p in manifest['plugins']]:
                raise RuntimeError('Restored plugin source checksum mismatch')
            # Replace the whole set; omissions are tombstones, preventing deleted nodes reappearing.
            backup = self.root / 'plugins-before-restore'
            shutil.rmtree(backup, ignore_errors=True)
            self.nodes.rename(backup)
            stage.rename(self.nodes)
            atomic(self.root / 'plugins-restored.json', manifest)
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def dependencies(self):
        plugins = inventory(self.nodes)
        if not plugins:
            return
        import platform
        protected = protect_environment(self.root)
        key = digest({'plugins': plugins, 'python': sys.version, 'platform': platform.platform(),
                      'gpu': protected, 'comfy': (self.comfy / '.git' / 'HEAD').read_text() if (self.comfy / '.git' / 'HEAD').exists() else ''})
        if read(self.root / 'plugin-dependencies.json', {}).get('key') == key:
            return
        env = dict(os.environ, COMFYUI_PATH=str(self.comfy))
        command = [sys.executable, '-c', DEPENDENCY_LAUNCHER, str(manager_cli(self.comfy)),
                   'restore-dependencies', '--user-directory', str(self.user)]
        notes = []
        for plugin in plugins:
            node = self.nodes / plugin['path']
            if not plugin['enabled'] or (node / 'requirements.txt').is_file():
                continue
            docs = [str(path.relative_to(node)) for path in node.glob('[Rr][Ee][Aa][Dd][Mm][Ee]*') if path.is_file()]
            notes.append({'plugin': plugin['path'], 'install_py': (node / 'install.py').is_file(),
                          'instructions': docs, 'message': 'Manager runs install.py when provided; inspect listed instructions for manual dependencies'})
            # Check installation instructions without executing arbitrary README shell commands.
            for name in docs:
                text = (node / name).read_text(errors='replace')
                if re.search(r'(?i)pip install|installation|安装', text):
                    print('[studio-native] Installation instructions: ' + str(node / name), flush=True)
        atomic(self.root / 'plugin-install-notes.json', notes)
        try:
            result = subprocess.run(command, cwd=self.comfy, env=env, capture_output=True, text=True, timeout=1800)
        except (subprocess.TimeoutExpired, OSError) as exc:
            atomic(self.root / 'plugin-errors.json', {'plugins': [p['path'] for p in plugins],
                                                     'error': 'Manager dependency restore could not finish: ' + type(exc).__name__})
            raise RuntimeError('Manager dependency restore could not finish: ' + type(exc).__name__) from None
        log = result.stdout + '\n' + result.stderr
        # Manager versions can swallow execute_install_script's False return value.
        if result.returncode or re.search(r'(?im)(?:^ERROR:|Traceback|installation failed|Failed to install|\[ERROR\])', log):
            error = {'plugins': [p['path'] for p in plugins], 'error': 'Manager dependency restore failed; see local plugin-dependencies.log',
                     'detail': SECRET.sub(b'[redacted]', log.encode()).decode(errors='replace')[-3000:]}
            atomic(self.root / 'plugin-errors.json', error)
            (self.root / 'plugin-dependencies.log').write_text(log)
            (self.root / 'plugin-dependencies.log').chmod(0o600)
            raise RuntimeError(error['error'] + ': ' + error['detail'])
        after = protect_environment(self.root)
        if protected != after:
            raise RuntimeError('Plugin installer changed protected PyTorch/CUDA packages; recreate runtime')
        check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
        if check.returncode:
            error = {'plugins': [p['path'] for p in plugins], 'error': 'Python dependency compatibility check failed',
                     'detail': SECRET.sub(b'[redacted]', (check.stdout + check.stderr).encode()).decode(errors='replace')[-3000:]}
            atomic(self.root / 'plugin-errors.json', error)
            raise RuntimeError(error['error'] + ': ' + error['detail'])
        atomic(self.root / 'plugin-dependencies.json', {'key': key})
        (self.root / 'plugin-errors.json').unlink(missing_ok=True)

    def snapshot(self):
        path = self.cache / 'manager-snapshot.json'
        path.unlink(missing_ok=True)
        result = subprocess.run([sys.executable, str(manager_cli(self.comfy)), 'save-snapshot', '--output', str(path),
                                 '--user-directory', str(self.user)], cwd=self.comfy,
                                env=dict(os.environ, COMFYUI_PATH=str(self.comfy)), capture_output=True, text=True, timeout=180)
        if result.returncode or not path.exists():
            raise RuntimeError('Manager save-snapshot failed')
        # Snapshot pip records can contain authenticated URLs; persist only node state.
        data = read(path)
        return {k: v for k, v in data.items() if k in {'comfyui', 'git_custom_nodes', 'cnr_custom_nodes', 'file_custom_nodes'}}

    def tick(self, busy=False, now=None):
        now = time.monotonic() if now is None else now
        plugins = inventory(self.nodes)
        fingerprint = digest(plugins)
        if busy or fingerprint != self.previous:
            self.previous, self.stable_since = fingerprint, now
            return False
        if now - self.stable_since < 30:
            return False
        restored = read(self.root / 'plugins-restored.json', {})
        old = {p['path']: p for p in restored.get('plugins', [])}
        for p in plugins:
            if p['path'] in old and p['checksum'] == old[p['path']]['checksum']:
                for k in ('repo', 'commit', 'version'):
                    p[k] = p[k] or old[p['path']].get(k)
        pending = read(self.root / 'plugins-pending.json', {})
        if pending.get('source') == fingerprint:
            manifest = pending['manifest']
        else:
            snapshot = self.snapshot()
            # Reject secrets in Manager metadata too, including URL credentials.
            raw = json.dumps(snapshot).encode()
            if SECRET.search(raw) or re.search(rb'https?://[^/\s"]+@', raw):
                raise RuntimeError('Manager snapshot contains credentials')
            known = {p['path']: p for p in pending.get('manifest', restored).get('plugins', [])}
            for p in plugins:
                prior = known.get(p['path'], {})
                if prior.get('checksum') == p['checksum'] and prior.get('commit') == p['commit'] and prior.get('archive'):
                    p['archive'] = prior['archive']
                    p['kind'] = prior.get('kind', 'directory')
                    continue
                node = self.nodes / p['path']
                with tempfile.NamedTemporaryFile(dir=self.cache, suffix='.tar.gz', delete=False) as temp:
                    archive = Path(temp.name)
                try:
                    p['kind'] = 'directory' if node.is_dir() else 'file'
                    with tarfile.open(archive, 'w:gz') as target:
                        if node.is_dir():
                            for path, _ in files(node):
                                target.add(path, arcname=str(path.relative_to(node)), recursive=False)
                            if (node / '.git').exists():
                                # Pack only this commit/tree, never repository history or Git config.
                                committed = subprocess.check_output(['git', '-C', str(node), 'archive', 'HEAD'])
                                commit_data = subprocess.check_output(['git', '-C', str(node), 'cat-file', 'commit', 'HEAD'])
                                if SECRET.search(commit_data) or re.search(rb'https?://[^/\s"]+@', committed):
                                    raise RuntimeError('Possible credential in committed plugin source: ' + p['path'])
                                with tarfile.open(fileobj=io.BytesIO(committed)) as source:
                                    for member in source:
                                        if member.isfile():
                                            check_credentials(source.extractfile(member).read(), Path(member.name))
                                objects = subprocess.check_output(['git', '-C', str(node), 'rev-list', '--objects', '--no-walk', 'HEAD'])
                                ids = b'\n'.join(line.split(b' ')[0] for line in objects.splitlines()) + b'\n'
                                bundle = self.cache / 'repo.pack'
                                with bundle.open('wb') as packed:
                                    subprocess.run(['git', '-C', str(node), 'pack-objects', '--stdout'], input=ids,
                                                   stdout=packed, stderr=subprocess.PIPE, check=True)
                                target.add(bundle, arcname='.studio-repo.pack', recursive=False)
                                bundle.unlink()
                        else:
                            # File nodes use a containing directory in the archive.
                            target.add(node, arcname=node.name, recursive=False)
                    p['archive'] = file_hash(archive)
                    cached = self.cache / (p['archive'] + '.tar.gz')
                    if not cached.exists():
                        self.remote.put(archive, 'objects/' + cached.name)
                        archive.replace(cached)
                finally:
                    archive.unlink(missing_ok=True)
            if digest(inventory(self.nodes)) != fingerprint:
                raise RuntimeError('Plugin files changed during backup; retrying after settling')
            manifest = {'schema': 1, 'plugins': plugins, 'snapshot': snapshot, 'backed_up_at': time.time()}
            manifest['generation'] = digest(plugins)
            self.publish_json(manifest, 'history/' + manifest['generation'] + '.json')
            self.publish_json(manifest, 'pending.json')
            atomic(self.root / 'plugins-pending.json', {'source': fingerprint, 'manifest': manifest})
        if self.boot == fingerprint and (not self.manifest or self.manifest['generation'] != manifest['generation']):
            self.publish_json(manifest, 'head.json')
            atomic(self.root / 'plugins-verified.json', manifest)
            self.manifest = manifest
        return True

    def status(self):
        pending = read(self.root / 'plugins-pending.json', {}).get('manifest')
        error = self.error or read(self.root / 'plugin-errors.json')
        try:
            durable = bool(read(self.root / 'plugins-pending.json', {}).get('source') == digest(inventory(self.nodes)))
        except Exception as exc:
            durable = False
            error = str(exc)
        return {'manifest': self.manifest, 'pending_generation': (pending or {}).get('generation'),
                'restart_required': bool(pending and (not self.manifest or pending['generation'] != self.manifest['generation'])),
                'durable': durable,
                'install_notes': read(self.root / 'plugin-install-notes.json', []),
                'error': error}


def install(server, root, manager_busy):
    """Called by the native extension. A restart captures a new immutable boot set."""
    import asyncio
    import logging

    root = Path(root)
    config = read(root / 'plugins-config.json')
    persistence = Persistence(root, config['comfy'], config['user'])
    # A rejected plugin must not prevent the status endpoint/loop from loading.
    # Leave boot unset on failure: later repairs require restart before head promotion.
    boot_fingerprint = None
    try:
        boot_fingerprint = digest(inventory(persistence.nodes))
    except Exception as exc:
        persistence.error = str(exc)

    def healthy():
        from urllib.request import urlopen
        try:
            with urlopen('http://127.0.0.1:8188/system_stats', timeout=3) as response:
                return isinstance(json.load(response).get('system'), dict)
        except Exception:
            return False

    async def loop():
        # PromptServer starts its event loop after node imports. The startup log
        # includes earlier nodes as well as later ones, unlike extension order hooks.
        while not await asyncio.to_thread(healthy):
            await asyncio.sleep(1)
        try:
            dependency_error = read(root / 'plugin-errors.json')
            if dependency_error:
                raise RuntimeError(dependency_error['error'])
            log = Path(config['log'])
            with log.open('rb') as source:
                source.seek(config['log_offset'])
                startup = source.read().decode(errors='replace')
            failed = [line for line in startup.splitlines() if '(IMPORT FAILED)' in line or 'Cannot import ' in line]
            if failed:
                raise RuntimeError('Plugin load verification failed: ' + '; '.join(failed)[-2000:])
            persistence.boot = boot_fingerprint
        except Exception as exc:
            persistence.error = str(exc)
            atomic(root / 'plugin-errors.json', {'error': str(exc)})
        # Scan failures are retried by tick; startup/dependency failures stay sticky.
        load_error = persistence.error if read(root / 'plugin-errors.json') else None
        while True:
            try:
                success = await asyncio.to_thread(persistence.tick, await asyncio.to_thread(manager_busy))
                if success:
                    persistence.error = load_error
            except Exception as exc:
                persistence.error = str(exc)
                logging.error('Plugin persistence: %s', exc)
            atomic(root / 'plugins-status.json', persistence.status())
            await asyncio.sleep(10)

    # Comfy imports nodes before run_until_complete; create_task on that loop works.
    server.loop.create_task(loop())
    return persistence
