from app.job_history import job_timing_details, job_timings, safe_history_value


def test_job_timings_separates_persisted_worker_stages_from_event_stages_without_fabricated_zeroes():
    job = {
        "state": "completed",
        "worker_timings": {"restore_weights": 91.2, "sampling": 8.4},
    }
    events = [
        {"state": "queued", "created_at": "2026-11-08T00:00:00+00:00"},
        {"state": "waiting_for_capacity", "created_at": "2026-11-08T00:00:12+00:00"},
        {"state": "allocating", "created_at": "2026-11-08T00:00:20+00:00"},
        {"state": "loading", "created_at": "2026-11-08T00:00:25+00:00"},
        {"state": "running", "created_at": "2026-11-08T00:01:00+00:00"},
        {"state": "uploading", "created_at": "2026-11-08T00:01:20+00:00"},
        {"state": "cleaning_up", "created_at": "2026-11-08T00:01:25+00:00"},
        {"state": "completed", "created_at": "2026-11-08T00:01:32+00:00"},
    ]

    timings = job_timings(job, events)

    assert timings["queue"] == 20.0
    assert timings["allocate"] == 5.0
    assert timings["upload"] == 5.0
    assert timings["cleanup"] == 7.0
    assert timings["restore_weights"] == 91.2
    assert timings["sampling"] == 8.4
    assert "restore_env" not in timings
    assert "delete" not in timings


def test_timing_details_mark_an_active_stage_and_ignore_events_without_timestamps():
    job = {"state": "running", "stage": "sampling", "worker_timings": {}}
    events = [
        {"state": "queued", "created_at": "2026-11-08T00:00:00Z"},
        {"state": "running", "created_at": "2026-11-08T00:00:10Z"},
        {"state": "running", "created_at": None},
    ]

    details = job_timing_details(job, events, now=25.0)

    assert details["queue"]["seconds"] == 10.0
    assert details["queue"]["source"] == "events"
    assert details["sampling"]["status"] == "in_progress"
    assert details["sampling"]["seconds"] is None
    assert "cleanup" not in details


def test_timing_details_derive_active_lifecycle_stage_from_persisted_events():
    details = job_timing_details(
        {"state": "uploading", "worker_timings": {}},
        [
            {"state": "queued", "created_at": 0},
            {"state": "uploading", "created_at": 10},
        ],
        now=25.0,
    )

    assert details["upload"] == {
        "seconds": 15.0,
        "source": "events",
        "status": "in_progress",
    }


def test_timing_details_never_turn_missing_active_event_time_into_zero():
    details = job_timing_details(
        {"state": "uploading", "worker_timings": {}},
        [{"state": "uploading", "created_at": None}],
    )

    assert details["upload"] == {
        "seconds": None,
        "source": "events",
        "status": "in_progress",
    }


def test_active_event_clock_spans_repeated_states_and_queue_substates():
    details = job_timing_details(
        {"state": "waiting_for_capacity", "stage": "untrusted_stage"},
        [
            {"state": "queued", "created_at": 0},
            {"state": "validating", "created_at": 5},
            {"state": "waiting_for_capacity", "created_at": 10},
            {"state": "waiting_for_capacity", "created_at": 15},
        ],
        now=25.0,
    )

    assert details["queue"]["seconds"] == 25.0
    assert "untrusted_stage" not in details


def test_timing_details_preserve_zero_only_when_worker_explicitly_persisted_it():
    details = job_timing_details(
        {"state": "completed", "worker_timings": {"warm_reuse": 0.0}},
        [],
    )
    assert details["warm_reuse"] == {
        "seconds": 0.0,
        "source": "worker",
        "status": "completed",
    }


def test_native_split_worker_stages_are_preserved_in_job_timings():
    timings = job_timings(
        {
            "state": "completed",
            "worker_timings": {
                "native_system": 14.2,
                "native_drive": 0.6,
                "native_models": 3.1,
                "native_plugins_restore": 9.4,
                "native_env_cache": 11.2,
                "comfy_start": 51.0,
                "native_tunnel": 1.1,
            },
        },
        [],
    )
    assert timings["native_system"] == 14.2
    assert timings["native_plugins_restore"] == 9.4
    assert timings["native_env_cache"] == 11.2
    assert timings["native_tunnel"] == 1.1


def test_timing_details_computes_live_worker_stage_elapsed_when_stage_started_present():
    job = {
        "state": "running",
        "stage": "native_env_cache",
        "stage_started": 100.0,
        "worker_timings": {"native_system": 5.4},
    }
    details = job_timing_details(job, [], now=118.6)
    assert details["native_system"] == {"seconds": 5.4, "source": "worker", "status": "completed"}
    assert details["native_env_cache"] == {"seconds": 18.6, "source": "worker", "status": "in_progress"}


def test_timing_details_tracks_native_proxy_wait_as_in_progress_tunnel():
    job = {
        "state": "running",
        "stage": "native_tunnel",
        "native_proxy_wait_started": 200.0,
        "worker_timings": {"comfy_start": 22.0, "native_tunnel": 1.2},
    }
    details = job_timing_details(job, [], now=206.5)
    assert details["comfy_start"] == {"seconds": 22.0, "source": "worker", "status": "completed"}
    assert details["native_tunnel"] == {"seconds": 7.7, "source": "events", "status": "in_progress"}


def test_safe_history_value_drops_credentials_inside_nested_metadata():
    value = safe_history_value({
        "api_key": "api-secret",
        "client_secret": "client-secret",
        "private_key": "private-secret",
        "metadata": {
            "safe": "kept",
            "credentials": {
                "api_key": "nested-api-secret",
                "private_key": "nested-private-secret",
            },
        },
    })

    assert value == {"metadata": {"safe": "kept"}}
    assert "api-secret" not in repr(value)
    assert "nested-api-secret" not in repr(value)
