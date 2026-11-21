from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Barrier

import pytest
from app import db
from app.gpu import legacy_migration
from app.gpu.repository import Repository
from sqlalchemy import create_engine, event


def test_concurrent_repository_initialization_uses_one_connection(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "create_engine", partial(create_engine, pool_size=1, max_overflow=0, pool_timeout=3))
    store = db.Store(f"sqlite:///{tmp_path / 'studio.db'}")
    upgrade = legacy_migration.upgrade_legacy_identity
    checkouts = []
    upgrades = []

    def count_checkout(connection, _record, _proxy):
        checkouts.append(connection)

    def tracked_upgrade(connection):
        upgrades.append(len(checkouts))
        upgrade(connection)

    event.listen(store.engine, "checkout", count_checkout)
    monkeypatch.setattr(legacy_migration, "upgrade_legacy_identity", tracked_upgrade)
    workers = 24
    barrier = Barrier(workers, timeout=10)

    def write_and_read(index):
        barrier.wait()
        repo = Repository(store)
        job = repo.create("jobs", "owner", {"state": "queued", "warmup": True, "index": index})
        return Repository(store).get("jobs", job["id"], "owner")["index"]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        assert list(executor.map(write_and_read, range(workers))) == list(range(workers))
    assert upgrades == [1]
    assert len(checkouts) == 1 + 2 * workers


def test_repository_retries_schema_initialization_after_failure(tmp_path, monkeypatch):
    store = db.Store(f"sqlite:///{tmp_path / 'studio.db'}")
    upgrade = legacy_migration.upgrade_legacy_identity

    def fail_upgrade(_connection):
        raise RuntimeError("migration failed")

    monkeypatch.setattr(legacy_migration, "upgrade_legacy_identity", fail_upgrade)
    with pytest.raises(RuntimeError, match="migration failed"):
        Repository(store)

    monkeypatch.setattr(legacy_migration, "upgrade_legacy_identity", upgrade)
    assert Repository(store).list("jobs", "owner") == []
    assert store._gpu_schema_ready
