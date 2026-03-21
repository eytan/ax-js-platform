#!/usr/bin/env bash
# verify-docs.sh — Detect structural drift between plans/ docs and codebase
#
# Run periodically or before doc-related PRs:
#   bash scripts/verify-docs.sh
#
# Checks only things that break silently: dead links, missing files,
# stale references. Does NOT enforce counts or duplicate test-suite checks.
#
# Exit code 0 = all checks pass, 1 = drift detected

set -euo pipefail
cd "$(dirname "$0")/.."

FAIL=0
WARN=0

fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }
warn() { echo "  WARN: $1"; WARN=$((WARN + 1)); }
pass() { echo "  OK:   $1"; }

echo "=== ax-js Documentation Verification ==="
echo ""

# ─── 1. File paths referenced in docs still exist ────────────────────────

echo "1. Checking file paths referenced in docs..."

# Extract backtick-wrapped src/ test/ python/ paths, skip templates (<name>, my_)
REFERENCED_PATHS=$(grep -roh '`\(src/[^`]*\.ts\|test/[^`]*\.ts\|python/[^`]*\.py\)' plans/ docs/developer-guide.md \
  | sed 's/^`//' | grep -v '<' | grep -v 'my_' | sort -u)

MISSING=0
for p in $REFERENCED_PATHS; do
  if [ ! -f "$p" ]; then
    fail "Referenced file does not exist: $p"
    MISSING=$((MISSING + 1))
  fi
done
if [ "$MISSING" -eq 0 ]; then
  pass "All $(echo "$REFERENCED_PATHS" | wc -l | tr -d ' ') referenced file paths exist"
fi

echo ""

# ─── 2. Cross-references between plans/ docs resolve ─────────────────────

echo "2. Checking cross-references between docs..."

BROKEN_REFS=0
for doc in plans/*.md; do
  REFS=$(grep -oE '\]\([^)]+\.md' "$doc" 2>/dev/null | sed 's/\](//' || true)
  for ref in $REFS; do
    resolved="plans/$ref"
    if [[ "$ref" == ../* ]]; then
      resolved=$(echo "$ref" | sed 's|^\.\./||')
    fi
    if [ ! -f "$resolved" ]; then
      fail "Broken cross-reference in $doc: $ref -> $resolved"
      BROKEN_REFS=$((BROKEN_REFS + 1))
    fi
  done
done
if [ "$BROKEN_REFS" -eq 0 ]; then
  pass "All cross-references resolve"
fi

echo ""

# ─── 3. No active references to deleted docs ─────────────────────────────

echo "3. Checking for references to deleted docs..."

DELETED_DOCS=(
  "docs/internal/observations.md"
  "docs/internal/serialization-contract.md"
)

for deleted in "${DELETED_DOCS[@]}"; do
  REFS=$(grep -rn "\]($deleted\|$(basename "$deleted"))" plans/ docs/ CLAUDE.md 2>/dev/null \
    | grep -v "Subsumes" || true)
  if [ -n "$REFS" ]; then
    fail "Active reference to deleted file $deleted:"
    echo "$REFS" | head -3
  fi
done
pass "No active references to deleted docs"

echo ""

# ─── 4. CLAUDE.md references plans/ ──────────────────────────────────────

echo "4. Checking CLAUDE.md discoverability..."

if grep -q 'plans/' CLAUDE.md 2>/dev/null; then
  pass "CLAUDE.md references plans/ directory"
else
  fail "CLAUDE.md does not reference plans/ — agents won't discover the guides"
fi

echo ""

# ─── Summary ─────────────────────────────────────────────────────────────

echo "=== Summary ==="
echo "  Failures: $FAIL"
echo "  Warnings: $WARN"

if [ "$FAIL" -gt 0 ]; then
  echo ""
  echo "Documentation has drifted from codebase. Fix the FAILs above."
  exit 1
else
  echo ""
  echo "All checks passed. Docs are in sync."
  exit 0
fi
