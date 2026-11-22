"""Native nodes: local execution, immutable Drive objects, restart-verified manifests.

Only stdlib: shipped to Colab alongside native_runtime, never imports the VPS app.
Third-party code runs with the runtime's privileges, not in a security sandbox.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

NATIVE_FLAG_NAMES = (
    'NATIVE_FAST_SCAN',
    'NATIVE_PARALLEL_RESTORE',
    'NATIVE_USE_UV',
    'NATIVE_FAST_CONTROL_PLANE',
    'NATIVE_ENV_CACHE',
    'NATIVE_BATCH_PLUGIN_PIP',
    'NATIVE_PARALLEL_BOOTSTRAP',
)


def env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {'0', 'false', 'off', 'no'}


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
_DEFAULT_SECRET = SECRET
_SECRET_EXACT_HINTS = (b'ghp_', b'gho_', b'ghu_', b'ghs_', b'ghr_', b'hf_', b'-----BEGIN ')
_SECRET_LOWER_HINTS = (b'access_token', b'refresh_token', b'client_secret', b'api_key', b'password')
_SCAN_CACHE_LOCK = threading.Lock()
_MEM_SCAN_CACHE: dict[str, str] = {}
_LOADED_SCAN_CACHE_PATH: Path | None = None


def _may_contain_secret(chunk: bytes) -> bool:
    """Exact pre-filter using C memmem before running case-insensitive regex."""
    if SECRET is not _DEFAULT_SECRET or not env_flag('NATIVE_FAST_SCAN', True):
        return True
    if any(token in chunk for token in _SECRET_EXACT_HINTS):
        return True
    lowered = chunk.lower()
    return any(token in lowered for token in _SECRET_LOWER_HINTS)


def is_verified_binary(path: Path, header: bytes) -> bool:
    """Skip regex credential scans only when both extension and magic bytes match."""
    if SECRET is not _DEFAULT_SECRET or not env_flag('NATIVE_FAST_SCAN', True):
        return False
    if _may_contain_secret(header):
        return False
    ext = path.suffix.lower()
    if ext == '.png':
        return header.startswith(b'\x89PNG\r\n\x1a\n')
    if ext == '.gif':
        return header.startswith((b'GIF87a', b'GIF89a'))
    if ext in {'.jpg', '.jpeg'}:
        return header.startswith(b'\xff\xd8\xff')
    if ext == '.webp':
        return len(header) >= 12 and header[:4] == b'RIFF' and header[8:12] == b'WEBP'
    if ext == '.bmp':
        return len(header) >= 14 and header.startswith(b'BM')
    if ext == '.ico':
        return header.startswith((b'\x00\x00\x01\x00', b'\x00\x00\x02\x00'))
    if ext in {'.mp4', '.mov', '.m4v'}:
        return len(header) >= 12 and header[4:8] in {b'ftyp', b'moov', b'mdat', b'wide', b'free'}
    if ext in {'.webm', '.mkv'}:
        return header.startswith(b'\x1a\x45\xdf\xa3')
    if ext == '.wav':
        return len(header) >= 12 and header[:4] == b'RIFF' and header[8:12] == b'WAVE'
    if ext == '.ogg':
        return header.startswith(b'OggS')
    if ext == '.mp3':
        return header.startswith((b'ID3', b'\xff\xfb', b'\xff\xf3', b'\xff\xf2', b'\xff\xe3'))
    if ext == '.ttf':
        return header.startswith((b'\x00\x01\x00\x00', b'true'))
    if ext == '.otf':
        return header.startswith(b'OTTO')
    if ext == '.ttc':
        return header.startswith(b'ttcf')
    if ext == '.woff':
        return header.startswith(b'wOFF')
    if ext == '.woff2':
        return header.startswith(b'wOF2')
    if ext == '.eot':
        return len(header) >= 36 and header[34:36] == b'LP'
    if ext == '.so':
        return header.startswith(b'\x7fELF')
    return False


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
    temp = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
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


def _load_disk_scan_cache(cache_path: Path | None) -> None:
    global _LOADED_SCAN_CACHE_PATH
    if cache_path is None or not env_flag('NATIVE_FAST_SCAN', True) or SECRET is not _DEFAULT_SECRET:
        return
    with _SCAN_CACHE_LOCK:
        if _LOADED_SCAN_CACHE_PATH == cache_path and _MEM_SCAN_CACHE:
            return
        try:
            if cache_path.is_file():
                raw = json.loads(cache_path.read_text())
                if isinstance(raw, dict):
                    _MEM_SCAN_CACHE.update({str(k): str(v) for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)})
            _LOADED_SCAN_CACHE_PATH = cache_path
        except (OSError, ValueError, TypeError):
            pass


def _save_disk_scan_cache(cache_path: Path | None) -> None:
    if cache_path is None or not env_flag('NATIVE_FAST_SCAN', True) or SECRET is not _DEFAULT_SECRET:
        return
    with _SCAN_CACHE_LOCK:
        if len(_MEM_SCAN_CACHE) > 16384:
            keys = list(_MEM_SCAN_CACHE.keys())[-12000:]
            snapshot = {k: _MEM_SCAN_CACHE[k] for k in keys}
            _MEM_SCAN_CACHE.clear()
            _MEM_SCAN_CACHE.update(snapshot)
        else:
            snapshot = dict(_MEM_SCAN_CACHE)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp = cache_path.with_name(f'.scan-cache.{os.getpid()}.{threading.get_ident()}.tmp')
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as handle:
            json.dump(snapshot, handle, separators=(',', ':'))
        temp.replace(cache_path)
    except OSError:
        pass


@lru_cache(maxsize=8192)
def _cached_checked_source(path, stamp, expected_checksum=None):
    """Reuse scans only while content metadata (including ctime) is unchanged."""
    with path.open('rb') as source:
        header = source.read(64)
        source.seek(0)
        if is_verified_binary(path, header):
            result = expected_checksum if expected_checksum is not None else hashlib.file_digest(source, 'sha256').hexdigest()
        else:
            checksum = hashlib.sha256()
            previous = b''
            while block := source.read(1024**2):
                window = previous + block
                if _may_contain_secret(window) and SECRET.search(window):
                    check_credentials(path.read_bytes(), path)
                checksum.update(block)
                previous = block[-512:]
            result = checksum.hexdigest()
    if source_stamp(path) != stamp:
        raise RuntimeError(f'Plugin file changed during scan: {path}; retrying after settling')
    if expected_checksum is not None and result != expected_checksum:
        raise RuntimeError(f'Plugin file changed during scan: {path}; retrying after settling')
    return result


def checked_source(path, stamp, *, check_secrets=True, use_disk_cache=False):
    if not check_secrets:
        result = file_hash(path)
        if source_stamp(path) != stamp:
            raise RuntimeError(f'Plugin file changed during scan: {path}; retrying after settling')
        return result
    use_cache = use_disk_cache and env_flag('NATIVE_FAST_SCAN', True) and SECRET is _DEFAULT_SECRET
    # Small writes may share coarse timestamps. Hash their content on every read,
    # then reuse credential validation only for the exact same content/metadata.
    checksum = file_hash(path) if stamp[2] < 1024**2 else None
    cache_key = None
    if use_cache:
        if checksum is not None:
            cache_key = f's:{stamp[0]}:{stamp[1]}:{stamp[2]}:{stamp[3]}:{stamp[4]}:{checksum}'
        else:
            cache_key = f'l:{stamp[0]}:{stamp[1]}:{stamp[2]}:{stamp[3]}:{stamp[4]}'
        with _SCAN_CACHE_LOCK:
            cached = _MEM_SCAN_CACHE.get(cache_key)
        if cached:
            if source_stamp(path) != stamp:
                raise RuntimeError(f'Plugin file changed during scan: {path}; retrying after settling')
            return cached
    result = _cached_checked_source(path, stamp, checksum)
    if source_stamp(path) != stamp:
        raise RuntimeError(f'Plugin file changed during scan: {path}; retrying after settling')
    if use_cache and cache_key:
        with _SCAN_CACHE_LOCK:
            _MEM_SCAN_CACHE[cache_key] = result
    return result


@contextmanager
def sqlite_snapshot(path, limit):
    deadline = time.monotonic() + 5

    def check_progress(_status, _remaining, _total):
        if time.monotonic() >= deadline:
            raise RuntimeError(f'SQLite snapshot timed out: {path}; retrying after settling')

    with tempfile.TemporaryDirectory(prefix='studio-plugin-db-') as temp:
        snapshot = Path(temp) / path.name
        try:
            with closing(sqlite3.connect(path.absolute().as_uri() + '?mode=ro', uri=True, timeout=1)) as source:
                source.execute('BEGIN')
                page_size = source.execute('PRAGMA page_size').fetchone()[0]
                page_count = source.execute('PRAGMA page_count').fetchone()[0]
                if page_count * page_size > limit * 1024**2:
                    raise RuntimeError(f'Plugin SQLite snapshot exceeds {limit} MiB: {path}')
                with closing(sqlite3.connect(snapshot)) as target:
                    source.backup(target, pages=256, progress=check_progress)
        except sqlite3.Error as exc:
            raise RuntimeError(f'Cannot snapshot plugin SQLite database: {path}; retrying after settling') from exc
        snapshot.chmod(path.stat().st_mode & 0o777)
        yield snapshot


def files(folder, *, archive=None, snapshot_databases=True, check_secrets=True, use_disk_cache=False):
    """Never follow links or archive credentials, environments, or Git auth config."""
    for base, dirs, names in os.walk(folder, followlinks=False):
        if any((Path(base) / d).is_symlink() for d in dirs if d not in EXCLUDE):
            raise RuntimeError(f'Plugin directory symlink is not supported: {base}')
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE)
        for name in sorted(names):
            path = Path(base) / name
            if name in EXCLUDE or name.startswith('.env') or name.endswith(('.pem', '.key', '.pyc', '-wal', '-shm', '-journal')):
                continue
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f'Unsupported plugin file: {path}')
            limit = 128 if path.suffix.lower() == '.csv' else 64
            if path.stat().st_size > limit * 1024**2:
                raise RuntimeError(f'Plugin source file exceeds {limit} MiB: {path}; store weights under Drive models')
            with path.open('rb') as source:
                is_database = snapshot_databases and source.read(16) == b'SQLite format 3\x00'
            if is_database:
                with sqlite_snapshot(path, limit) as snapshot:
                    checksum = checked_source(snapshot, source_stamp(snapshot), check_secrets=check_secrets)
                    if archive is not None:
                        archive.add(snapshot, arcname=str(path.relative_to(folder)), recursive=False)
            else:
                checksum = checked_source(
                    path, source_stamp(path), check_secrets=check_secrets, use_disk_cache=use_disk_cache,
                )
                if archive is not None:
                    archive.add(path, arcname=str(path.relative_to(folder)), recursive=False)
            yield path, checksum


def _scan_cache_path_for_nodes(nodes: Path) -> Path | None:
    for candidate_root in (nodes.parent.parent, nodes.parent):
        if (candidate_root / 'plugin-cache').is_dir() or (candidate_root / 'plugins-config.json').is_file():
            return candidate_root / 'scan-cache.json'
    return None


def _scan_single_plugin(nodes: Path, node: Path, enabled: bool, *, snapshot_databases: bool, check_secrets: bool, use_disk_cache: bool) -> dict | None:
    if node.is_symlink():
        raise RuntimeError(f'Plugin symlink is not supported: {node}')
    if node.is_file() and not node.name.endswith(('.py', '.py.disabled')):
        return None
    entries = (
        list(files(node, snapshot_databases=snapshot_databases, check_secrets=check_secrets, use_disk_cache=use_disk_cache))
        if node.is_dir()
        else [(node, file_hash(node))]
    )
    if node.is_file() and check_secrets:
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
    return item


def inventory(nodes, *, snapshot_databases=True, check_secrets=True):
    nodes = Path(nodes)
    cache_path = _scan_cache_path_for_nodes(nodes) if check_secrets else None
    use_disk_cache = cache_path is not None
    if use_disk_cache:
        _load_disk_scan_cache(cache_path)
    before_cache_size = len(_MEM_SCAN_CACHE) if use_disk_cache else 0
    candidates = []
    for path in sorted(nodes.iterdir()):
        if path.name in {'studio_native_state.py', 'studio_native_guard', '__pycache__'}:
            continue
        if path.name in {'.disabled', 'disabled'} and path.is_dir():
            candidates.extend((node, False) for node in sorted(path.iterdir()))
        elif not path.name.startswith('.'):
            candidates.append((path, not path.name.endswith('.disabled')))

    def worker_fn(pair):
        return _scan_single_plugin(
            nodes, pair[0], pair[1],
            snapshot_databases=snapshot_databases,
            check_secrets=check_secrets,
            use_disk_cache=use_disk_cache,
        )

    if env_flag('NATIVE_FAST_SCAN', True) and len(candidates) > 1:
        max_workers = min(8, len(candidates), os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            scanned = list(pool.map(worker_fn, candidates))
    else:
        scanned = [worker_fn(pair) for pair in candidates]

    if use_disk_cache and len(_MEM_SCAN_CACHE) != before_cache_size:
        _save_disk_scan_cache(cache_path)
    result = [item for item in scanned if item is not None]
    return sorted(result, key=lambda item: item['path'])


def fast_tree_fingerprint(nodes: Path) -> str:
    """Lightweight directory & dependency-file metadata stamp (<2ms) for prestartup guard."""
    nodes = Path(nodes)
    if not nodes.is_dir():
        return 'missing'
    entries = []
    candidates = []
    for path in sorted(nodes.iterdir()):
        if path.name in {'studio_native_state.py', 'studio_native_guard', '__pycache__'}:
            continue
        if path.name in {'.disabled', 'disabled'} and path.is_dir():
            candidates.extend((node, False) for node in sorted(path.iterdir()))
        elif not path.name.startswith('.'):
            candidates.append((path, not path.name.endswith('.disabled')))
    for node, enabled in candidates:
        st = node.lstat()
        dep_meta = {}
        if node.is_dir() and not node.is_symlink():
            for fname in ('requirements.txt', 'install.py', 'pyproject.toml', '__init__.py'):
                fp = node / fname
                if fp.exists():
                    fst = fp.lstat()
                    dep_meta[fname] = [fst.st_size, fst.st_mtime_ns, fst.st_ctime_ns]
            entries.append([str(node.relative_to(nodes)), enabled, st.st_ino, dep_meta])
        else:
            entries.append([str(node.relative_to(nodes)), enabled, st.st_ino, st.st_size, st.st_mtime_ns, dep_meta])
    return digest(entries)


def mark_prestartup_verified(root: Path, nodes: Path, dep_key: str, *, ttl: float = 180.0) -> None:
    if not env_flag('NATIVE_FAST_SCAN', True) or not dep_key:
        return
    atomic(Path(root) / 'prestartup-verified.json', {
        'dep_key': dep_key,
        'tree_fp': fast_tree_fingerprint(Path(nodes)),
        'expires_at': time.time() + ttl,
    })


class Remote:
    """Direct rclone transfers, bypassing asynchronous FUSE writeback for commits."""
    def __init__(self, root):
        self.root = Path(root)

    def available(self) -> bool:
        return (self.root / 'rclone').is_file() and (self.root / 'rclone.conf').is_file()

    def command(self, *args, timeout=600):
        return subprocess.run([str(self.root / 'rclone'), '--config', str(self.root / 'rclone.conf'), *args],
                              check=True, capture_output=True, text=True, timeout=timeout)

    def get(self, key, target):
        try:
            self.command('copyto', '--ignore-times', 'studio:native-comfy/plugins/' + key, str(target))
            return True
        except subprocess.CalledProcessError as exc:
            if exc.returncode in {3, 4}:  # rclone directory/file not found, not network failure
                return False
            raise RuntimeError('Drive download failed; inspect rclone connectivity (credentials omitted)') from None

    def get_objects(self, names: list[str], target_dir: Path) -> None:
        if not names:
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile('w', dir=target_dir, prefix='.rclone-files-', delete=False) as handle:
            list_path = Path(handle.name)
            handle.write('\n'.join(sorted(set(names))) + '\n')
        try:
            self.command(
                'copy',
                '--files-from-raw', str(list_path),
                '--ignore-times',
                '--transfers', '8',
                '--checkers', '8',
                '--multi-thread-streams', '4',
                '--multi-thread-cutoff', '16M',
                '--buffer-size', '32M',
                '--retries', '3',
                '--low-level-retries', '10',
                'studio:native-comfy/plugins/objects',
                str(target_dir),
            )
        except subprocess.CalledProcessError as exc:
            if exc.returncode not in {3, 4}:
                raise RuntimeError('Drive download failed; inspect rclone connectivity (credentials omitted)') from None
        finally:
            list_path.unlink(missing_ok=True)

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

    def get_env_cache(self, name: str, target: Path) -> bool:
        if not self.available():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            extra = (
                ('--multi-thread-streams', '8', '--multi-thread-cutoff', '32M', '--buffer-size', '64M')
                if name.endswith('.tar.gz')
                else ()
            )
            self.command(
                'copyto', '--ignore-times', *extra, '--retries', '3', '--low-level-retries', '10',
                'studio:native-comfy/env-cache/' + name, str(target),
            )
            return target.is_file()
        except subprocess.CalledProcessError as exc:
            if exc.returncode in {3, 4}:
                return False
            return False

    def put_env_cache(self, source: Path, name: str, *, verify: bool = False) -> None:
        extra = ('--drive-chunk-size', '128M') if name.endswith('.tar.gz') else ()
        self.command(
            'copyto', '--ignore-times', *extra, '--retries', '3', '--low-level-retries', '10',
            str(source), 'studio:native-comfy/env-cache/' + name, timeout=900,
        )
        if verify:
            with tempfile.TemporaryDirectory() as temp:
                check = Path(temp) / 'verify'
                if not self.get_env_cache(name, check) or file_hash(check) != file_hash(source):
                    raise RuntimeError('Drive env-cache verification mismatch: ' + name)

    def prune_env_cache(self, keep: int = 3) -> None:
        if not self.available() or keep < 1:
            return
        try:
            res = self.command('lsf', '--format', 'tp', '--separator', '\t', 'studio:native-comfy/env-cache')
        except subprocess.CalledProcessError:
            return
        metas = []
        for line in res.stdout.splitlines():
            parts = line.strip().split('\t', 1)
            if len(parts) == 2 and parts[1].endswith('.meta.json'):
                metas.append((parts[0], parts[1][:-len('.meta.json')]))
        metas.sort(key=lambda item: item[0], reverse=True)
        for _, stale_key in metas[keep:]:
            for suffix in ('.meta.json', '.tar.gz'):
                try:
                    self.command('deletefile', f'studio:native-comfy/env-cache/{stale_key}{suffix}', timeout=120)
                except subprocess.CalledProcessError:
                    pass


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


def _detected_cuda_arch_list() -> str | None:
    smi = shutil.which('nvidia-smi')
    if not smi:
        return None
    try:
        proc = subprocess.Popen(
            [smi, '--query-gpu=compute_cap', '--format=csv,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        out, _ = proc.communicate(timeout=5)
        if proc.returncode != 0:
            return None
    except (OSError, subprocess.SubprocessError):
        return None
    caps = sorted({line.strip() for line in out.splitlines() if re.fullmatch(r'\d+\.\d+', line.strip())})
    return ';'.join(caps) if caps else None


def prestartup(root):
    """Manager exec-restarts keep their PID; verify every startup, including those."""
    root = Path(root)
    config = read(root / 'plugins-config.json')
    if not config:
        return
    for flag_name, flag_val in (config.get('flags') or {}).items():
        if flag_name in NATIVE_FLAG_NAMES:
            os.environ.setdefault(flag_name, str(flag_val))
    config['log_offset'] = Path(config['log']).stat().st_size
    config['pid'] = os.getpid()
    atomic(root / 'plugins-config.json', config)
    protect_environment(root)
    verified_marker = root / 'prestartup-verified.json'
    if verified_marker.is_file():
        verified = read(verified_marker)
        verified_marker.unlink(missing_ok=True)
        if (
            env_flag('NATIVE_FAST_SCAN', True)
            and isinstance(verified, dict)
            and time.time() <= float(verified.get('expires_at') or 0)
            and read(root / 'plugin-dependencies.json', {}).get('key') == verified.get('dep_key')
            and fast_tree_fingerprint(Path(config['comfy']) / 'custom_nodes') == verified.get('tree_fp')
        ):
            print('[studio-native] Prestartup fast-path verified; skipping redundant plugin scan', flush=True)
            return
    try:
        Persistence(root, Path(config['comfy']), Path(config['user'])).dependencies()
    except Exception as exc:
        atomic(root / 'plugin-errors.json', {'error': str(exc)})
        raise


def _extract_plugin_archive(archive: Path, target_dir: Path, *, allow_pigz: bool = True) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    use_pigz = allow_pigz and archive.stat().st_size >= 16 * 1024**2 and bool(shutil.which('pigz'))
    if use_pigz:
        proc = subprocess.Popen(['pigz', '-dc', str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        total = 0
        try:
            assert proc.stdout is not None
            with tarfile.open(fileobj=proc.stdout, mode='r|') as source:
                for member in source:
                    if not (member.isfile() or member.isdir()) or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                        raise RuntimeError('Unsafe plugin archive')
                    total += member.size
                    if total > 4 * 1024**3:
                        raise RuntimeError('Plugin archive exceeds 4 GiB')
                    member.mode &= 0o777
                    member.uid = member.gid = 0
                    source.extract(member, target_dir, set_attrs=True)
            _, stderr_bytes = proc.communicate(timeout=300)
            if proc.returncode != 0:
                raise RuntimeError('pigz decompression failed: ' + stderr_bytes.decode(errors='replace')[-200:])
            return
        except BaseException:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            raise
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if sum(m.size for m in members) > 4 * 1024**3:
            raise RuntimeError('Plugin archive exceeds 4 GiB')
        if any(not (m.isfile() or m.isdir()) or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members):
            raise RuntimeError('Unsafe plugin archive')
        for member in members:
            member.mode &= 0o777
            member.uid = member.gid = 0
        source.extractall(target_dir, members=members)


def _restore_git_bundle(node: Path, item: dict) -> None:
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
            return None
        if manifest.get('schema') != 1 or manifest.get('generation') != digest(manifest['plugins']):
            raise RuntimeError('Invalid plugin manifest')
        plugins = manifest['plugins']
        for item in plugins:
            relative = Path(item['path'])
            if not relative.parts or relative.is_absolute() or '..' in relative.parts or len(relative.parts) > 2 or not re.fullmatch(r'[0-9a-f]{64}', item['archive']):
                raise RuntimeError('Invalid plugin path')
        parallel = env_flag('NATIVE_PARALLEL_RESTORE', True)
        stage = self.root / 'plugins-restore'
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir()
        try:
            if parallel and plugins:
                max_workers = min(8, max(1, len(plugins)), os.cpu_count() or 4)

                def check_cached(item):
                    archive = self.cache / (item['archive'] + '.tar.gz')
                    if archive.is_file() and file_hash(archive) == item['archive']:
                        return None
                    archive.unlink(missing_ok=True)
                    return item

                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    missing_items = [m for m in pool.map(check_cached, plugins) if m is not None]

                if missing_items:
                    missing_names = sorted({m['archive'] + '.tar.gz' for m in missing_items})
                    if hasattr(self.remote, 'get_objects') and callable(getattr(self.remote, 'available', None)) and self.remote.available():
                        self.remote.get_objects(missing_names, self.cache)
                    else:
                        by_archive = {m['archive']: m for m in missing_items}

                        def fetch_one(sha):
                            target = self.cache / (sha + '.tar.gz')
                            if not self.remote.get('objects/' + target.name, target):
                                raise RuntimeError('Missing plugin cache: ' + by_archive[sha]['path'])

                        with ThreadPoolExecutor(max_workers=min(8, len(by_archive))) as pool:
                            list(pool.map(fetch_one, sorted(by_archive)))

                def verify_and_extract(item):
                    relative = Path(item['path'])
                    archive = self.cache / (item['archive'] + '.tar.gz')
                    if not archive.is_file():
                        raise RuntimeError('Missing plugin cache: ' + item['path'])
                    if file_hash(archive) != item['archive']:
                        archive.unlink(missing_ok=True)
                        raise RuntimeError('Corrupt plugin cache: ' + item['path'] + '; select a verified historical generation')
                    dest = stage / relative if item.get('kind', 'directory') == 'directory' else stage / relative.parent
                    _extract_plugin_archive(archive, dest, allow_pigz=True)
                    _restore_git_bundle(stage / relative, item)

                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    list(pool.map(verify_and_extract, plugins))
            else:
                for item in plugins:
                    relative = Path(item['path'])
                    archive = self.cache / (item['archive'] + '.tar.gz')
                    if not archive.exists() or file_hash(archive) != item['archive']:
                        if not self.remote.get('objects/' + archive.name, archive):
                            raise RuntimeError('Missing plugin cache: ' + item['path'])
                    if file_hash(archive) != item['archive']:
                        archive.unlink(missing_ok=True)
                        raise RuntimeError('Corrupt plugin cache: ' + item['path'] + '; select a verified historical generation')
                    dest = stage / relative if item.get('kind', 'directory') == 'directory' else stage / relative.parent
                    _extract_plugin_archive(archive, dest, allow_pigz=False)
                    _restore_git_bundle(stage / relative, item)
            # Verify semantic source checksums as well as compressed object integrity.
            # Archives already contain standalone database files. Hash their exact bytes,
            # including in older archives that captured the original main database.
            actual = inventory(
                stage,
                snapshot_databases=False,
                check_secrets=not env_flag('NATIVE_FAST_SCAN', True),
            )
            if [(p['path'], p['checksum']) for p in actual] != [(p['path'], p['checksum']) for p in plugins]:
                raise RuntimeError('Restored plugin source checksum mismatch')
            # Replace the whole set; omissions are tombstones, preventing deleted nodes reappearing.
            backup = self.root / 'plugins-before-restore'
            shutil.rmtree(backup, ignore_errors=True)
            self.nodes.rename(backup)
            stage.rename(self.nodes)
            atomic(self.root / 'plugins-restored.json', manifest)
            return actual
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def _dependency_key(self, plugins: list[dict], protected: dict) -> str:
        import platform
        return digest({
            'plugins': plugins,
            'python': sys.version,
            'platform': platform.platform(),
            'gpu': protected,
            'comfy': (self.comfy / '.git' / 'HEAD').read_text() if (self.comfy / '.git' / 'HEAD').exists() else '',
        })

    def _write_install_notes(self, plugins: list[dict]) -> None:
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

    def stamp_cached_dependencies(self, plugins: list[dict] | None = None) -> str | None:
        if plugins is None or not env_flag('NATIVE_FAST_SCAN', True):
            plugins = inventory(self.nodes)
        if not plugins:
            return None
        protected = protect_environment(self.root)
        key = self._dependency_key(plugins, protected)
        self._write_install_notes(plugins)
        atomic(self.root / 'plugin-dependencies.json', {'key': key})
        (self.root / 'plugin-errors.json').unlink(missing_ok=True)
        return key

    def dependencies(self, plugins: list[dict] | None = None) -> str | None:
        if plugins is None or not env_flag('NATIVE_FAST_SCAN', True):
            plugins = inventory(self.nodes)
        if not plugins:
            return None
        protected = protect_environment(self.root)
        key = self._dependency_key(plugins, protected)
        if read(self.root / 'plugin-dependencies.json', {}).get('key') == key:
            return key
        env = dict(os.environ, COMFYUI_PATH=str(self.comfy))
        if env_flag('NATIVE_BATCH_PLUGIN_PIP', True):
            arch_list = _detected_cuda_arch_list()
            if arch_list:
                env.setdefault('TORCH_CUDA_ARCH_LIST', arch_list)
            env.setdefault('MAX_JOBS', str(min(8, os.cpu_count() or 4)))
        self._write_install_notes(plugins)
        if env_flag('NATIVE_BATCH_PLUGIN_PIP', True):
            req_files = [
                self.nodes / p['path'] / 'requirements.txt'
                for p in plugins
                if p.get('enabled') and (self.nodes / p['path'] / 'requirements.txt').is_file()
            ]
            if req_files:
                constraints = self.root / 'gpu-constraints.txt'
                req_args = [arg for req in req_files for arg in ('-r', str(req))]
                uv_bin = shutil.which('uv') if env_flag('NATIVE_USE_UV', True) else None
                batch_cmd = (
                    [uv_bin, 'pip', 'install', '--system', '--python', sys.executable, '--constraint', str(constraints), '-q', *req_args]
                    if uv_bin
                    else [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-q', '-c', str(constraints), *req_args]
                )
                try:
                    batch_res = subprocess.run(batch_cmd, cwd=self.comfy, env=env, capture_output=True, text=True, timeout=1500)
                    if batch_res.returncode != 0:
                        print('[studio-native] Batch plugin requirements install exited non-zero; Manager restore-dependencies will reconcile', flush=True)
                except (subprocess.SubprocessError, OSError):
                    print('[studio-native] Batch plugin requirements install skipped; falling back to Manager restore-dependencies', flush=True)
        command = [sys.executable, '-c', DEPENDENCY_LAUNCHER, str(manager_cli(self.comfy)),
                   'restore-dependencies', '--user-directory', str(self.user)]
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
        return key

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
                            checks = [(str(path.relative_to(node)), checksum)
                                      for path, checksum in files(node, archive=target)]
                            if digest(checks) != p['checksum']:
                                raise RuntimeError('Plugin files changed during backup; retrying after settling')
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


# --- Phase 2: Incremental Native Environment Cache (site-packages delta) ---

def plugin_dep_fingerprint(nodes: Path, plugins: list[dict]) -> str:
    """Hash only dependency-relevant plugin inputs so runtime SQLite/config writes do not bust env-cache."""
    nodes = Path(nodes)
    entries = []
    for item in sorted(plugins, key=lambda x: x['path']):
        if not item.get('enabled', True):
            continue
        node = nodes / item['path']
        dep_files = {}
        if node.is_dir() and not node.is_symlink():
            for fname in ('requirements.txt', 'install.py', 'pyproject.toml', 'setup.py', 'setup.cfg'):
                fp = node / fname
                if fp.is_file() and not fp.is_symlink():
                    dep_files[fname] = file_hash(fp)
            if 'install.py' in dep_files and not item.get('commit'):
                for py in sorted(node.glob('*.py')):
                    if py.is_file() and not py.is_symlink():
                        dep_files[py.name] = file_hash(py)
        entries.append({
            'path': item['path'],
            'commit': item.get('commit') if 'install.py' in dep_files else None,
            'files': dep_files,
        })
    return digest(entries)


def compute_env_cache_key(comfy: Path, commit: str, gpu_tag: str, nodes: Path, plugins: list[dict]) -> dict:
    dists = sorted(
        f"{(d.metadata.get('Name') or '').lower().replace('_', '-')}=={d.version}"
        for d in importlib.metadata.distributions()
        if d.metadata.get('Name')
    )
    base_pkg_sha = hashlib.sha256('\n'.join(dists).encode()).hexdigest()
    comfy_req_sha = file_hash(comfy / 'requirements.txt')
    manager_req_sha = file_hash(comfy / 'manager_requirements.txt')
    plugin_deps_sha = plugin_dep_fingerprint(nodes, plugins)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX') or ''
    spec = {
        'schema': 1,
        'comfy_commit': commit,
        'python': sys.version,
        'ext_suffix': ext_suffix,
        'base_pkg_sha': base_pkg_sha,
        'gpu_tag': gpu_tag,
        'comfy_req_sha': comfy_req_sha,
        'manager_req_sha': manager_req_sha,
        'plugin_deps_sha': plugin_deps_sha,
    }
    spec_sha = digest(spec)
    py_tag = f'py{sys.version_info.major}{sys.version_info.minor}'
    key = f'native-env-v1-{commit[:12]}-{py_tag}-img{base_pkg_sha[:12]}-gpu{gpu_tag[:12]}-deps{spec_sha[:16]}'
    return {'key': key, 'spec': spec}


def site_snapshot(site: Path) -> dict[str, list[int]]:
    if not site.is_dir():
        return {}
    state = {}
    for entry in site.iterdir():
        if entry.name == '__pycache__' or entry.name.startswith('.studio-native-'):
            continue
        st = entry.lstat()
        state[entry.name] = [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
    return state


def _remove_entry(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def _safe_extract_env_archive(archive: Path, stage_dir: Path) -> None:
    stage_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    stage_resolved = stage_dir.resolve()

    def check_member(member: tarfile.TarInfo) -> None:
        rel = Path(member.name)
        if not rel.parts or rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError('Unsafe env-cache member path')
        if not (member.isfile() or member.isdir() or member.issym()):
            raise RuntimeError('Unsupported env-cache member type')
        if member.issym():
            link_target = (stage_dir / rel.parent / member.linkname).resolve()
            if stage_resolved != link_target and stage_resolved not in link_target.parents:
                raise RuntimeError('Unsafe env-cache symlink target')
        member.uid = member.gid = 0
        if not member.issym():
            member.mode &= 0o777

    if shutil.which('pigz'):
        proc = subprocess.Popen(['pigz', '-dc', str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            assert proc.stdout is not None
            with tarfile.open(fileobj=proc.stdout, mode='r|') as tar:
                for member in tar:
                    check_member(member)
                    tar.extract(member, stage_dir, set_attrs=not member.issym())
            _, err = proc.communicate(timeout=300)
            if proc.returncode != 0:
                raise RuntimeError('pigz env-cache extract failed: ' + err.decode(errors='replace')[-200:])
            return
        except BaseException:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            raise
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        for member in members:
            check_member(member)
        tar.extractall(stage_dir, members=members)


def try_restore_env_cache(root: Path, site: Path, key_info: dict, baseline_protected: dict, remote: Remote) -> bool:
    """Restore incremental site-packages cache atomically; return False cleanly on miss or validation failure."""
    if not env_flag('NATIVE_ENV_CACHE', True) or not site.is_dir():
        return False
    if not (hasattr(remote, 'get_env_cache') and callable(getattr(remote, 'available', None)) and remote.available()):
        return False
    key = key_info['key']
    cache_dir = Path(root) / 'env-cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f'{key}.meta.json'
    tar_path = cache_dir / f'{key}.tar.gz'
    try:
        if not remote.get_env_cache(f'{key}.meta.json', meta_path):
            return False
        meta = read(meta_path)
        if (
            not isinstance(meta, dict)
            or meta.get('schema') != 1
            or meta.get('key') != key
            or meta.get('spec') != key_info['spec']
            or not re.fullmatch(r'[0-9a-f]{64}', str(meta.get('archive_sha256') or ''))
        ):
            return False
        expected_sha = meta['archive_sha256']
        if not tar_path.is_file() or file_hash(tar_path) != expected_sha:
            tar_path.unlink(missing_ok=True)
            if not remote.get_env_cache(f'{key}.tar.gz', tar_path):
                return False
            if file_hash(tar_path) != expected_sha:
                tar_path.unlink(missing_ok=True)
                return False
    except Exception as exc:
        print(f'[studio-native] env-cache fetch skipped ({type(exc).__name__}); falling back to clean install', flush=True)
        return False

    stage = site.parent / f'.studio-native-env-stage-{os.getpid()}'
    rollback = site.parent / f'.studio-native-env-rollback-{os.getpid()}'
    shutil.rmtree(stage, ignore_errors=True)
    shutil.rmtree(rollback, ignore_errors=True)
    moved_to_rollback: list[str] = []
    installed_names: list[str] = []
    try:
        _safe_extract_env_archive(tar_path, stage)
        staged_site = stage / 'site'
        removed = [str(x) for x in (meta.get('removed') or []) if isinstance(x, str) and '/' not in x and x not in {'.', '..'}]
        incoming = [entry.name for entry in staged_site.iterdir()] if staged_site.is_dir() else []
        rollback.mkdir(mode=0o700, parents=True, exist_ok=True)
        for name in sorted(set(removed) | set(incoming)):
            target = site / name
            if target.exists() or target.is_symlink():
                shutil.move(str(target), str(rollback / name))
                moved_to_rollback.append(name)
        for name in incoming:
            shutil.move(str(staged_site / name), str(site / name))
            installed_names.append(name)
        importlib.invalidate_caches()
        after_protected = protect_environment(Path(root))
        if after_protected != baseline_protected:
            raise RuntimeError('env-cache altered protected PyTorch/CUDA packages')
        check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True, timeout=60)
        if check.returncode != 0:
            raise RuntimeError('env-cache failed pip check')
        print(f'[studio-native] Restored native env-cache: {key} ({len(installed_names)} site entries)', flush=True)
        return True
    except Exception as exc:
        print(f'[studio-native] env-cache restore rejected ({exc}); rolling back to clean base image', flush=True)
        for name in installed_names:
            _remove_entry(site / name)
        for name in moved_to_rollback:
            src = rollback / name
            if src.exists() or src.is_symlink():
                _remove_entry(site / name)
                shutil.move(str(src), str(site / name))
        importlib.invalidate_caches()
        return False
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(rollback, ignore_errors=True)
        tar_path.unlink(missing_ok=True)


def prepare_env_cache_plan(root: Path, site: Path, before_site: dict[str, list[int]], key_info: dict) -> Path | None:
    if not env_flag('NATIVE_ENV_CACHE', True) or not site.is_dir():
        return None
    after_site = site_snapshot(site)
    changed = {name for name, st in after_site.items() if before_site.get(name) != st}
    # Include top-level package dirs referenced by newly installed dist-info RECORDs (e.g. namespace packages).
    for name in list(changed):
        if name.endswith('.dist-info'):
            record = site / name / 'RECORD'
            if record.is_file():
                for line in record.read_text(errors='replace').splitlines():
                    top = line.split(',', 1)[0].split('/', 1)[0].strip()
                    if top and top not in {'.', '..', '__pycache__'} and ((site / top).exists() or (site / top).is_symlink()):
                        changed.add(top)
    site_resolved = site.resolve()
    native_resolved = Path(root).resolve()
    for name in changed:
        if name.endswith('.pth') and (site / name).is_file():
            for line in (site / name).read_text(errors='replace').splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(('#', 'import ')):
                    continue
                candidate = Path(stripped)
                if candidate.is_absolute():
                    resolved = candidate.resolve()
                    if (
                        resolved != site_resolved
                        and site_resolved not in resolved.parents
                        and resolved != native_resolved
                        and native_resolved not in resolved.parents
                    ):
                        print(f'[studio-native] Skipping env-cache build due to external .pth entry in {name}', flush=True)
                        return None
    removed = sorted(name for name in before_site if name not in after_site)
    plan_path = Path(root) / 'env-cache-plan.json'
    atomic(plan_path, {
        'schema': 1,
        'key': key_info['key'],
        'spec': key_info['spec'],
        'site': str(site),
        'added': sorted(changed),
        'removed': removed,
    })
    return plan_path


def upload_env_cache_from_plan(root: Path, plan_path: Path, remote: Remote | None = None) -> bool:
    """Pack and upload site-packages delta in background; .meta.json is published last for atomicity."""
    import fcntl

    root = Path(root)
    remote = remote or Remote(root)
    lock_file = root / '.env-cache-upload.lock'
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open('a+') as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        plan = read(Path(plan_path))
        if not isinstance(plan, dict) or plan.get('schema') != 1:
            return False
        key = plan['key']
        site = Path(plan['site'])
        added = [str(x) for x in (plan.get('added') or [])]
        removed = [str(x) for x in (plan.get('removed') or [])]
        cache_dir = root / 'env-cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        existing_meta = cache_dir / f'{key}.remote.meta.json'
        if remote.get_env_cache(f'{key}.meta.json', existing_meta):
            existing_meta.unlink(missing_ok=True)
            return True
        if any(not ((site / name).exists() or (site / name).is_symlink()) for name in added):
            print('[studio-native] Aborting env-cache pack: site-packages changed before archive build', flush=True)
            return False
        tar_tmp = cache_dir / f'{key}.tar.gz.tmp'
        tar_final = cache_dir / f'{key}.tar.gz'
        meta_file = cache_dir / f'{key}.meta.json'
        tar_tmp.unlink(missing_ok=True)
        try:
            if shutil.which('pigz'):
                with tar_tmp.open('wb') as out_stream:
                    pigz = subprocess.Popen(['pigz', '-1', '-c'], stdin=subprocess.PIPE, stdout=out_stream, stderr=subprocess.PIPE)
                    try:
                        assert pigz.stdin is not None
                        with tarfile.open(fileobj=pigz.stdin, mode='w|') as tar:
                            for name in added:
                                tar.add(site / name, arcname=f'site/{name}')
                        pigz.stdin.close()
                        pigz.stdin = None
                        _, err = pigz.communicate(timeout=600)
                        if pigz.returncode != 0:
                            raise RuntimeError('pigz pack failed: ' + err.decode(errors='replace')[-200:])
                    finally:
                        if pigz.poll() is None:
                            pigz.kill()
                            pigz.wait(timeout=5)
            else:
                with tarfile.open(tar_tmp, 'w:gz', compresslevel=1) as tar:
                    for name in added:
                        tar.add(site / name, arcname=f'site/{name}')
            tar_tmp.replace(tar_final)
            archive_sha = file_hash(tar_final)
            size = tar_final.stat().st_size
            remote.put_env_cache(tar_final, f'{key}.tar.gz', verify=False)
            atomic(meta_file, {
                'schema': 1,
                'key': key,
                'spec': plan['spec'],
                'archive_sha256': archive_sha,
                'size': size,
                'added_count': len(added),
                'removed': removed,
                'created_at': time.time(),
            })
            remote.put_env_cache(meta_file, f'{key}.meta.json', verify=True)
            remote.prune_env_cache(keep=3)
            print(f'[studio-native] Uploaded env-cache {key} ({size / 1024**2:.1f} MiB)', flush=True)
            return True
        finally:
            tar_tmp.unlink(missing_ok=True)
            tar_final.unlink(missing_ok=True)


def install(server, root, manager_busy):
    """Called by the native extension. A restart captures a new immutable boot set."""
    import asyncio
    import logging

    root = Path(root)
    config = read(root / 'plugins-config.json')
    for flag_name, flag_val in ((config or {}).get('flags') or {}).items():
        if flag_name in NATIVE_FLAG_NAMES:
            os.environ.setdefault(flag_name, str(flag_val))
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


def _main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == 'upload-env-cache':
        plan_path = Path(argv[1]).resolve()
        try:
            upload_env_cache_from_plan(plan_path.parent, plan_path)
            return 0
        except Exception as exc:
            print(f'[studio-native] Background env-cache upload failed: {exc}', flush=True)
            return 1
    return 2


if __name__ == '__main__':
    raise SystemExit(_main(sys.argv[1:]))
