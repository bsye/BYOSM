import os
import socket

import pytest

from byosm.benchmarks import bfcl, when2call
from byosm.benchmarks.catalog import BENCHMARKS
from byosm.data.hf_source.cards import UNKNOWN_LICENCE, licence

HUB = ("huggingface.co", 443)
TIMEOUT_S = 3


def online() -> bool:
    if os.environ.get("HF_HUB_OFFLINE"):
        return False
    try:
        socket.create_connection(HUB, timeout=TIMEOUT_S).close()
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not online(), reason="reads the Hugging Face Hub")


def files(repo):
    from huggingface_hub import HfApi

    return set(HfApi().list_repo_files(repo, repo_type="dataset"))


def test_bfcl_has_a_question_file_per_category_and_an_answer_file_where_a_tool_fits():
    found = files(bfcl.REPO)
    for category in bfcl.NON_LIVE + bfcl.LIVE:
        assert bfcl.QUESTIONS.format(version=bfcl.VERSION, category=category) in found
        answers = bfcl.ANSWERS.format(version=bfcl.VERSION, category=category) in found
        assert answers == (category not in bfcl.NO_TOOL)


def test_when2call_has_its_multiple_choice_test_file():
    assert when2call.TEST_FILE in files(when2call.REPO)


def test_every_public_benchmark_states_its_licence_on_its_card():
    assert all(licence(b.dataset) != UNKNOWN_LICENCE for b in BENCHMARKS.values())
