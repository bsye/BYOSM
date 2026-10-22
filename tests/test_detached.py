import time

from byosm.app.detached import LOG_SUFFIX, exit_code, job_file, spawn

SLOW_HOOK = "import sys, time\nsys.excepthook = lambda *a: (time.sleep(1), sys.__excepthook__(*a))\n"


def test_a_crash_is_logged_before_its_exit_code(tmp_path):
    proc = spawn(tmp_path, "j", SLOW_HOOK + "raise RuntimeError('boom')", [], {})
    deadline = time.time() + 10
    while exit_code(tmp_path, "j") is None and time.time() < deadline:
        time.sleep(0.001)
    log = job_file(tmp_path, "j", LOG_SUFFIX).read_text()
    proc.wait()
    assert exit_code(tmp_path, "j") == 1
    assert "RuntimeError: boom" in log
