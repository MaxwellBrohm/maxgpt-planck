"""The bank pass teacher server (PC, inside one gpu.lock hold): teachers/serve_http.py unchanged, with one request
field added for bank calls. serve_http builds its regex only from chat label lines (decode.label_regex); a bank call
carries its own regex, built by prompts.decode_spec / prompts3.spec on the client side, so this wrapper accepts
"bank_regex" and checks it the way serve_http checks a constraint: a non-empty string, and only from a server
loaded with a structured-output backend. Everything else (load, the dash ban, the round 4 phrase ban, presets,
--sampling-json, idle exit, stats, unload) is serve_http's own code path.

    flock -w 7200 ~/planck/locks/gpu.lock timeout -k 30 1980 python bankserve.py --teacher qwen3.5-9b --port 18851 \
        --dash-ban --gpu-util 0.86 --sampling-json '{"temperature": 0.0}' [serve_http flags]

The row's "decode" record carries regex_sha256 (serve.sampling_record), so every call record shows the regex that
was applied. DRY marker: serve_http's identity says status "dry"; the bank pass records are marked by gen.py's mode
and max_ok instead."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "teachers"))
import serve_http as SH  # noqa: E402

_ORIG = SH.decode_request
BANK_RULE = "bank-regex-v1"


def decode_request(body, info):
    """serve_http.decode_request, plus "bank_regex" (str): a bank call's own structured-output regex."""
    rx = body.get("bank_regex")
    rest = {k: v for k, v in body.items() if k != "bank_regex"}
    if rx is not None and rest.get("constraint") is not None:
        raise ValueError("send either a chat constraint or a bank_regex, not both")
    req = _ORIG(rest, info)
    if rx is not None:
        if not isinstance(rx, str) or not rx:
            raise ValueError("bank_regex must be a non-empty string")
        if not info.get("structured_backend"):
            raise ValueError("this server was loaded without a structured-output backend")
        req.update(regex=rx, label_rule=BANK_RULE)
    return req


def install():
    SH.decode_request = decode_request
    return SH


if __name__ == "__main__":
    install()
    sys.exit(SH.main())
