"""Shard iteration over skeleton.build (moved out of skeleton.py unchanged on 2026-09-28): personas and topics drawn
without replacement, no repeated (topic path, persona, kind set) triple, resumable."""
import pools as P


def triple(sk):
    """the (topic path, persona, kind set) triple that a shard never repeats."""
    return tuple(sk["topic_path"]), sk["user"]["persona_seed"], tuple(sorted(e["kind"] for e in sk["events"]))


def iter_shard(shard_seed, register="RM", start_j=0, seen=None):
    """endless generator of (j, skeleton) in shard order; shard() takes the first n. Resumable: pass the j of the
    last skeleton already produced and the triples seen so far, and the stream continues exactly where it stopped."""
    import skeleton   # late: skeleton imports this module at its end
    rng = P.seeded("planck-shard", skeleton.RNG_VERSION, shard_seed)
    personas = [f"np{i:05d}" for i in range(skeleton.N_PERSONAS)]
    tops = skeleton.topic_ids()
    rng.shuffle(personas)
    rng.shuffle(tops)
    seen, j = set() if seen is None else set(seen), start_j
    while True:
        persona = personas[j % len(personas)]
        three = [tops[(3 * j + m) % len(tops)] for m in range(3)]
        j += 1
        sk = skeleton.build(f"{shard_seed}:{j}", register, persona=persona, topics=three)
        tr = triple(sk)
        if tr in seen:
            continue
        seen.add(tr)
        yield j, sk


def shard(shard_seed, n, register="RM"):
    """n skeletons with personas and topics drawn without replacement (cycling when exhausted) and no repeated
    (topic path, persona, kind set) triple."""
    out = []
    if n <= 0:
        return out
    for _, sk in iter_shard(shard_seed, register):
        out.append(sk)
        if len(out) >= n:
            break
    return out
