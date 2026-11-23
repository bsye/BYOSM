import threading

from byosm.locks import exclusive


def test_a_second_holder_says_it_waits_and_enters_only_after_the_first_leaves(tmp_path):
    path = tmp_path / "state" / "turn"
    order = []
    first_in, release = threading.Event(), threading.Event()

    def first():
        with exclusive(path):
            first_in.set()
            release.wait(5)
            order.append("first leaves")

    def second():
        with exclusive(path, lambda: (order.append("second waits"), release.set())):
            order.append("second enters")

    holder = threading.Thread(target=first)
    holder.start()
    first_in.wait(5)
    waiter = threading.Thread(target=second)
    waiter.start()
    holder.join(5)
    waiter.join(5)
    assert order == ["second waits", "first leaves", "second enters"]
