#!/usr/bin/env bash
set -euo pipefail

# High-confidence patterns only. External secret scanning remains a CI/release gate.
pattern='(AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9_]{30,}|xox[baprs]-[A-Za-z0-9-]{20,}|cvp_[A-Za-z0-9]{8,}\.[A-Za-z0-9]{20,}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|Huyquan1607)'

matches="$(git grep -IlE "$pattern" -- . || true)"
if [[ -n "$matches" ]]; then
  echo "Potential committed secret found in:"
  echo "$matches"
  exit 1
fi

echo "No high-confidence committed secrets detected."
