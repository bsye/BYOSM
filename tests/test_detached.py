import time

from byosm.app.detached import EXIT_SUFFIX, LOG_SUFFIX, exit_code, job_file, spawn, wrapped

SLOW_HOOK = "import sys, time\nsys.excepthook = lambda *a: (time.sleep(1), sys.__excepthook__(*a))\n"


def test_a_crash_is_logged_before_its_exit_code(tmp_path):
    command = wrapped(job_file(tmp_path, "j", EXIT_SUFFIX), SLOW_HOOK + "raise RuntimeError('boom')", [])
    proc = spawn(tmp_path, "j", command, {})
    deadline = time.time() + 10
    while exit_code(tmp_path, "j") is None and time.time() < deadline:
        time.sleep(0.001)
    log = job_file(tmp_path, "j", LOG_SUFFIX).read_text()
    proc.wait()
    assert exit_code(tmp_path, "j") == 1
    assert "RuntimeError: boom" in log


def test_a_job_stopped_from_outside_is_interrupted_not_failed():
    import signal

    from byosm.app.detached import ending

    assert ending(None, cancelled=False, stalled=False) == "interrupted"
    assert ending(-signal.SIGTERM, cancelled=False, stalled=False) == "interrupted"
    assert ending(-signal.SIGTERM, cancelled=False, stalled=True) == "failed"
    assert ending(-signal.SIGTERM, cancelled=True, stalled=False) == "cancelled"
    assert ending(1, cancelled=False, stalled=False) == "failed" and ending(0, False, False) == "done"


def test_a_job_runs_from_a_frozen_copy_of_the_code_so_later_edits_cannot_reach_it(ws):
    import subprocess
    import sys

    from byosm.app.snapshot import frozen

    code = frozen()
    assert frozen() == code and (code / "byosm" / "__init__.py").exists()
    found = subprocess.run(
        [sys.executable, "-P", "-c", "import byosm; print(byosm.__file__)"],
        env={"PYTHONPATH": str(code)},
        capture_output=True,
        text=True,
        cwd=ws,
    ).stdout
    assert found.startswith(str(code))
