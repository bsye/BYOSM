import signal

from byosm.app.detached import INTERRUPTED, ending


def test_stalled_job_killed_without_exit_code_is_failed():
    assert ending(None, cancelled=False, stalled=True) == "failed"
    assert ending(-signal.SIGTERM, cancelled=False, stalled=True) == "failed"


def test_stalled_job_that_finished_in_grace_window_is_done():
    assert ending(0, cancelled=False, stalled=True) == "done"
    assert ending(None, cancelled=True, stalled=True) == "cancelled"


def test_unstalled_job_without_exit_code_is_interrupted():
    assert ending(None, cancelled=False, stalled=False) == INTERRUPTED
    assert ending(-signal.SIGTERM, cancelled=False, stalled=False) == INTERRUPTED
    assert ending(1, cancelled=False, stalled=False) == "failed"
