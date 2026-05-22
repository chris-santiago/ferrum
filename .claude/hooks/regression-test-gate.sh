#!/usr/bin/env bash
# PreToolUse hook: gates git-commit when source files are staged without tests.
# fix/ branches: hard block (exit 2). Other branches: soft reminder.
set -euo pipefail

INPUT=$(cat) || exit 0

CWD=$(printf '%s' "$INPUT" | python3 -c "import sys,json;print(json.load(sys.stdin).get('cwd',''))" 2>/dev/null) || exit 0
CMD=$(printf '%s' "$INPUT" | python3 -c "import sys,json;print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null) || exit 0

[[ "$CMD" == git\ commit* ]] || exit 0

cd "$CWD" 2>/dev/null || exit 0
STAGED=$(git diff --cached --name-only 2>/dev/null) || exit 0
[ -n "$STAGED" ] || exit 0

HAS_SOURCE=false
HAS_TESTS=false

if printf '%s' "$STAGED" | grep -qE '^(src/ferrum/|crates/ferrum-(core|wasm)/src/).*\.(py|rs)$'; then
  HAS_SOURCE=true
fi
if printf '%s' "$STAGED" | grep -qE '(^tests/.*\.py$|^crates/.*/tests/)'; then
  HAS_TESTS=true
fi

$HAS_SOURCE || exit 0
$HAS_TESTS && exit 0

# Source without tests — severity depends on branch
BRANCH=$(git branch --show-current 2>/dev/null || echo "")
case "$BRANCH" in
  fix/*|bugfix/*|hotfix/*)
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"REGRESSION TEST GATE: Source files staged on a fix/ branch with no test files. Invoke /regression-test before committing."}}\n'
    exit 2 ;;
  *)
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","additionalContext":"REMINDER: Source files staged without test files. If this includes a bug fix, invoke /regression-test first."}}\n'
    exit 0 ;;
esac
