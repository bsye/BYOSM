from byosm.ideas import IDEAS, save, saved


def test_ideas_are_kept_for_the_workspace_and_replaced_by_the_next_research(ws):
    assert saved(ws)[IDEAS] == []
    first = [{"title": "Toxicity", "prompt": "flag toxic prompts", "dataset": "lmsys/toxic-chat"}]
    save(ws, first)
    assert saved(ws)[IDEAS] == first and saved(ws)["at"]
    save(ws, [])
    assert saved(ws)[IDEAS] == []
