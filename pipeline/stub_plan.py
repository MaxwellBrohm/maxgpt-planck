"""Plans for the FAKE teacher stub (fake_server.py): which canned render each (skeleton, attempt) gets, and the exact
outcome the driver must record for it. The end-to-end test compares the driver's shards with plan["expected"].

Canned bodies come from step 3: fake_teacher.raw() (a correct render) and the planted defects of defects_text and
defects_behav. DEFECTS_PURE lists defects whose planted code is also the primary code on every skeleton measured
(40 of 40 each, 2026-09-25); DEFECTS_MIXED plant a semantic error whose code must be among the codes, while an
earlier code (usually REQ_SPAN) may be primary.

Scenarios per skeleton (attempts are 0-based; max_attempts must be 2):
  clean            a0 clean                                  -> accept a0
  defect_clean     a0 defect, a1 clean                       -> reject a0 (code), accept a1
  defect_defect    a0 defect, a1 another defect              -> reject a0, reject a1
  flaky            a0: HTTP 500, HTTP 429, clean             -> accept a0 (3 requests)
  dropped          a0: connection dropped, clean             -> accept a0 (2 requests)
  hard_error       a0: HTTP 503 on every try; a1 clean       -> reject a0 TEACHER_ERROR, accept a1
  truncated        a0: first half of a clean render, finish "length"; a1 clean -> reject a0 FORMAT_LINES, accept a1
  infeasible       (render_prompt.feasible fails)            -> reject a0 SKEL_INFEASIBLE, no request at all
  fake_unclean     (fake_teacher.raw fails the checker, a limit of the fake teacher, about 0.2%): the same render
                   on both attempts                          -> reject a0, reject a1 (its primary code among codes)
Duplicate groups: every member is served a clean render on every attempt; exactly one member is accepted (which
one depends on completion order) and every other member is rejected on both attempts with the group's code."""
import random

import checker
import defects_behav
import defects_text
import fake_teacher
import parse
import render_prompt as R
from fake_server import key
from teacher_client import render_seed

_ALL = {name: (code, fn) for name, code, fn in defects_text.DEFECTS + defects_behav.DEFECTS}
DEFECTS_PURE = ["req_span", "forbid_span", "fmt_drop", "fmt_swap", "fmt_no_end", "fmt_preamble", "fmt_trailer",
                "fmt_wrap", "fmt_thought", "empty_turn", "role_label", "md_bold", "emoji", "em_dash",
                "spaced_hyphen", "stage", "digit", "len_long", "exact_mismatch", "req_word", "ai_ism", "repeat4",
                "safety", "heldout_vocab", "value_early", "open_end", "claim_plans", "identity_claim", "accented",
                "persona_leak"]
DEFECTS_MIXED = ["ans_wrong", "ans_stale", "ans_shotgun", "dist_leak", "plant_missing"]
SCENARIOS = {"clean": 40, "defect_clean": 20, "defect_defect": 12, "flaky": 7, "dropped": 5, "hard_error": 6,
             "truncated": 10}
MAX_ATTEMPTS = 2


def _defect(skel, rng, used):
    """-> (raw text, code, pure) for one applicable defect not used yet on this skeleton."""
    names = DEFECTS_PURE + DEFECTS_MIXED
    rng.shuffle(names)
    built = R.build(skel)
    for name in names:
        if name in used:
            continue
        code, fn = _ALL[name]
        out = fn(skel, dict(fake_teacher.render(skel)), built)
        if out is None:
            continue
        used.add(name)
        raw = out if isinstance(out, str) else parse.serialize(skel, out)
        return raw, code, name in DEFECTS_PURE, name
    raise RuntimeError(f"no applicable defect for {skel['skel_id']}")


def _k(skel, attempt):
    return key(R.build(skel)["prompt"], render_seed(skel["skel_id"], attempt))


def plan(skels, seed=0, http_tries=4, groups=(), sleep=0.0, weights=None):
    """skels: skeleton list. groups: [(code, [skel_id, ...]), ...] duplicate groups (code DUP_EXACT or DUP_NEAR).
    -> {"model", "responses", "expected": {"per_skel", "groups", "requests", "accepted", "rejected"}}."""
    rng = random.Random(seed)
    weights = weights or SCENARIOS
    names, w = list(weights), list(weights.values())
    in_group = {sid for _, members in groups for sid in members}
    responses, per_skel, requests = {}, {}, 0

    def step(text, **kw):
        s = {"text": text, **kw}
        if sleep:
            s["sleep"] = sleep
        return s

    for sk in skels:
        sid = sk["skel_id"]
        if R.feasible(sk):
            per_skel[sid] = {"scenario": "infeasible", "outcomes": [("reject", "SKEL_INFEASIBLE", True, None)]}
            continue
        clean = fake_teacher.raw(sk)
        unclean = checker.run(sk, clean)
        if not unclean["ok"]:
            # the fake teacher cannot write a passing render for this skeleton (about 2 in 1,000, e.g. LEN_USER
            # on a terse line): serve that render on both attempts and expect two rejects with its primary code
            if sid in in_group:
                raise ValueError(f"duplicate group member {sid} has no clean fake render")
            for a in range(MAX_ATTEMPTS):
                responses[_k(sk, a)] = [step(clean)]
            per_skel[sid] = {"scenario": "fake_unclean",
                             "outcomes": [("reject", unclean["primary"], False, "fake_render")] * MAX_ATTEMPTS}
            requests += MAX_ATTEMPTS
            continue
        if sid in in_group:
            for a in range(MAX_ATTEMPTS):
                responses[_k(sk, a)] = [step(clean)]
            continue
        sc = rng.choices(names, w)[0]
        used, out = set(), []
        if sc == "clean":
            responses[_k(sk, 0)] = [step(clean)]
            out, requests = [("accept", None, True, None)], requests + 1
        elif sc in ("defect_clean", "defect_defect"):
            raw, code, pure, name = _defect(sk, rng, used)
            responses[_k(sk, 0)] = [step(raw)]
            out.append(("reject", code, pure, name))
            if sc == "defect_clean":
                responses[_k(sk, 1)] = [step(clean)]
                out.append(("accept", None, True, None))
            else:
                raw, code, pure, name = _defect(sk, rng, used)
                responses[_k(sk, 1)] = [step(raw)]
                out.append(("reject", code, pure, name))
            requests += 2
        elif sc == "flaky":
            responses[_k(sk, 0)] = [{"status": 500}, {"status": 429}, step(clean)]
            out, requests = [("accept", None, True, None)], requests + 3
        elif sc == "dropped":
            responses[_k(sk, 0)] = [{"drop": True}, step(clean)]
            out, requests = [("accept", None, True, None)], requests + 2
        elif sc == "hard_error":
            responses[_k(sk, 0)] = [{"status": 503}] * http_tries
            responses[_k(sk, 1)] = [step(clean)]
            out, requests = [("reject", "TEACHER_ERROR", True, None), ("accept", None, True, None)], \
                requests + http_tries + 1
        elif sc == "truncated":
            responses[_k(sk, 0)] = [step(clean[:len(clean) // 2], finish="length")]
            responses[_k(sk, 1)] = [step(clean)]
            out, requests = [("reject", "FORMAT_LINES", True, None), ("accept", None, True, None)], requests + 2
        else:
            raise ValueError(sc)
        per_skel[sid] = {"scenario": sc, "outcomes": out}
    gr = []
    for code, members in groups:
        gr.append({"code": code, "members": list(members)})
        requests += 1 + (len(members) - 1) * MAX_ATTEMPTS
    accepted = sum(o[0] == "accept" for p in per_skel.values() for o in p["outcomes"]) + len(gr)
    rejected = sum(o[0] == "reject" for p in per_skel.values() for o in p["outcomes"]) + \
        sum((len(g["members"]) - 1) * MAX_ATTEMPTS for g in gr)
    return {"model": "fake-teacher", "responses": responses,
            "expected": {"per_skel": per_skel, "groups": gr, "requests": requests, "accepted": accepted,
                         "rejected": rejected}}
