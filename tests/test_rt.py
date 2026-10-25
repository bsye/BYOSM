def test_head_and_tail_truncation_keeps_the_start_and_the_end_of_a_long_input():
    from byosm.rt import HEAD_SHARE, head_tail

    ids = list(range(1000))
    kept = head_tail(ids, 512)
    head = int(512 * HEAD_SHARE)
    assert len(kept) == 512 and kept[:head] == ids[:head] and kept[-1] == 999 and kept[head] == 1000 - (512 - head)
    assert head_tail([1, 2, 3], 512) == [1, 2, 3]


def test_new_text_specialists_read_long_inputs_head_and_tail_while_older_ones_keep_their_reading():
    from byosm.spec.compiler import train_cfg
    from byosm.spec.sections import TrainCfg

    assert train_cfg("text", "hf:bert-base-uncased").truncation == "head_tail"
    assert train_cfg("audio", "tiny-crnn").truncation == "right" and TrainCfg().truncation == "right"
