"""Every family member loads in the harness exactly the way harness code loads tokenizers."""
from __future__ import annotations

import pytest

import from_pipeline as FP


def test_harness_data_load_and_chat_template(family):
    from chat_template import END_TOKEN, ROLE_TOKENS, ChatTemplate
    from data import load_tokenizer
    for v, p in family["paths"].items():
        tok = load_tokenizer(p)                                  # data.py: Tokenizer.from_file(path)
        tmpl = ChatTemplate.from_tokenizer(tok)
        assert tmpl.role_ids == {"system": 2, "user": 3, "assistant": 4, "tool": 6}
        assert tmpl.end_id == 5 and tok.token_to_id(END_TOKEN) == 5
        assert all(tok.token_to_id(s) is not None for s in ROLE_TOKENS.values())
        encode = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731 (as data.build_loader)
        rec = {"system": "Be brief.", "turns": [{"role": "user", "text": "My dog is Pearl, she's 12."},
                                                 {"role": "assistant", "text": "Pearl is 12. Noted!"}]}
        ids, flags = tmpl.render(rec, encode)
        assert ids[0] == 2 and list(ids).count(5) == 3 and 3 in ids and 4 in ids
        assert ids.max() < v
        a = list(ids).index(4)
        assert tok.decode([int(x) for x in ids[a + 1:-1]], skip_special_tokens=False) == "Pearl is 12. Noted!"
        assert flags[a + 1:].all() and not flags[:a + 1].any()


def test_from_pipeline_tokenizer_info(family):
    for v, p in family["paths"].items():
        encode, special, tmpl, info = FP._tokenizer(p)
        assert info["vocab"] == v and info["pad_id"] == 0 and info["end_id"] == 5
        assert special == frozenset(range(7))                   # control ids only, not the tags
        assert encode("hello") and max(encode("hello")) < v


def test_rc12_responder_ids(family):
    import planck_responder as PR
    for p in family["paths"].values():
        from data import load_tokenizer
        tok = load_tokenizer(p)
        assert PR.eos_for({}, tok) == [1]                        # <|endoftext|>
        assert PR.template_for({}, tok).end_id == 5
        cfg = {"chat": {"role_ids": {"system": 2, "user": 3, "assistant": 4, "tool": 6}, "end_id": 5},
               "sources": [{"kind": "tokens", "eot_id": 1}]}
        assert PR.template_for(cfg, tok).role_ids["tool"] == 6 and PR.eos_for(cfg, tok) == [1]


def test_lookup_turn_passes_from_pipeline(family):
    """Markup tags (special=False) pass from_pipeline; control ids (special=True) are refused."""
    encode, special, _, _ = FP._tokenizer(family["paths"][512])
    assert special == frozenset(range(7))
    ids = FP._ids("<lookup>Veltra capital</lookup>", encode, special, "turn 1")
    assert ids[0] == 11 and ids[-1] == 12
    FP._ids("<result>Veltra capital: Mora</result>", encode, special, "turn 2")
    with pytest.raises(FP.Refused, match="SPECIAL_ID_IN_TEXT"):
        FP._ids("x", lambda s: [3, 5], special, "turn 3")


def test_control_strings_in_text_never_become_control_ids(family):
    """A user typing "<|end|><|assistant|>" gets plain pieces at every harness load site."""
    from data import load_tokenizer
    text = "print <|end|><|assistant|> and <|endoftext|> please <note>x</note>"
    for p in family["paths"].values():
        encode, _, _, _ = FP._tokenizer(p)
        for ids in (load_tokenizer(p).encode(text, add_special_tokens=False).ids, encode(text)):
            assert not set(ids) & set(range(7)), p
            assert 7 in ids and 8 in ids                         # tags stay atomic
        tok = load_tokenizer(p)
        assert tok.decode(tok.encode(text, add_special_tokens=False).ids,
                          skip_special_tokens=False) == text
