#!/bin/bash
# Quick eval progress checker for R100
echo "=== R100 Flash Eval Progress ==="
echo "Completed: $(ls tests/evaluation/r100-flash-streaming/*.json 2>/dev/null | wc -l) / 250 scenarios"
echo ""
echo "Latest eval:"
tail -1 /tmp/r100-flash-eval.log 2>/dev/null
echo ""
echo "Latest judge scores:"
grep "DASHBOARD" /tmp/r100-flash-judge.log 2>/dev/null | tail -1
echo ""
echo "Errors:"
grep -c "ERROR\|error_count" /tmp/r100-flash-eval.log 2>/dev/null
echo ""
echo "Processes alive:"
ps aux | grep -E "run_live_eval|streaming_judge" | grep -v grep | awk '{print $2, $11, $12, $13}' 2>/dev/null
