#!/bin/bash

# Adjust this path to your claude installation
CLAUDE_PATH="${CLAUDE_PATH:-/Users/johan/.claude/local/claude}"

MAX_ITER=${1:-100}
ITER=0

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  RALPH RUST BACKTEST - OPEN ENDED                        ║"
echo "║  Max iterations: $MAX_ITER                                       ║"
echo "║  Claude: $CLAUDE_PATH"
echo "╚══════════════════════════════════════════════════════════╝"

# Verify claude exists
if [ ! -f "$CLAUDE_PATH" ]; then
    echo "❌ Claude not found at $CLAUDE_PATH"
    echo "   Set CLAUDE_PATH environment variable or edit this script"
    exit 1
fi

while [ $ITER -lt $MAX_ITER ]; do
    ITER=$((ITER + 1))
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ITERATION $ITER / $MAX_ITER  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check if DONE file exists
    if [ -f "DONE" ]; then
        echo ""
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "║  ✅ DONE FILE CREATED - EXPERIMENT COMPLETE              ║"
        echo "╚══════════════════════════════════════════════════════════╝"
        echo ""
        echo "📄 DONE file contents:"
        echo "────────────────────────────────────────────────────────────"
        cat DONE
        echo "────────────────────────────────────────────────────────────"
        echo ""
        echo "📊 Final statistics:"
        echo "   Iterations: $ITER"
        echo "   Lines of Rust: $(find . -name '*.rs' 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 || echo 'N/A')"
        echo "   Test count: $(grep -r '#\[test\]' --include='*.rs' 2>/dev/null | wc -l || echo 'N/A')"
        break
    fi
    
    # Run Claude
    cat PROMPT.md | "$CLAUDE_PATH" -p --dangerously-skip-permissions
    
    # Commit whatever changed
    git add -A
    git commit -m "ralph: iteration $ITER" 2>/dev/null || echo "(no changes to commit)"
    
    # Brief pause
    sleep 1
done

if [ ! -f "DONE" ]; then
    echo ""
    echo "⚠️  Reached max iterations ($MAX_ITER) without DONE file"
    echo "   The experiment may need more iterations or manual review"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXPERIMENT ENDED after $ITER iterations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
