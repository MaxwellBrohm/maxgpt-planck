#!/bin/bash
# Read-only on the PC: sha256 of every E006 output file, the guard verdicts, the crashed C2 manifest,
# and the sha256 of every trained or reference weights file used by E006 (nothing is written).
cd ~/planck/e006_run || exit 1
echo "== files"
find out logs transcripts -type f -print0 | sort -z | xargs -0 sha256sum
echo "== crashed"
ls -la crashed 2>/dev/null; find crashed -type f | sort
cat crashed/*/MANIFEST_sha256.txt 2>/dev/null | head -60
echo "== weights"
for d in weights/*; do f="$d/model.safetensors"; [ -f "$f" ] && echo "$(sha256sum "$f" | cut -c1-64) $f"; done
for d in ~/planck/dev/e006_refs_e005/weights/* ~/planck/dev/e006_refs_e004/weights/*; do f="$d/model.safetensors"; [ -f "$f" ] && echo "$(sha256sum "$f" | cut -c1-64) $f"; done
ls -d ~/planck/dev/e006_refs_e005/* ~/planck/dev/e006_refs_e004/* 2>/dev/null | head
echo "== done"
