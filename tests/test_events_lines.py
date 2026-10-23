import threading

from byosm import events


def test_events_from_many_threads_each_get_their_own_line(monkeypatch, capsys):
    monkeypatch.setenv(events.EVENTS_ENV, "1")
    threads = [
        threading.Thread(target=lambda i=i: [events.emit("llm", call=f"{i}-{k}") for k in range(50)]) for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 400 and all(events.parse(line) for line in lines)
