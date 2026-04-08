#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${SELDON_BIN:-$ROOT/build/seldon}"
FIXTURES="$ROOT/tests/security/fixtures"
RUN_DIR="${SELDON_SECURITY_RUN_DIR:-$ROOT/build/security_regression}"

mkdir -p "$RUN_DIR"
rm -rf "$RUN_DIR"/*

info() {
    printf '[security-regression] %s\n' "$*"
}

fail() {
    printf '[security-regression][FAIL] %s\n' "$*" >&2
    exit 1
}

require_file() {
    local path="$1"
    [[ -f "$path" ]] || fail "Required file not found: $path"
}

count_tmp_inputs() {
    shopt -s nullglob
    local files=(/tmp/seldon_input_*)
    local count="${#files[@]}"
    shopt -u nullglob
    printf '%s' "$count"
}

run_case() {
    local name="$1"
    shift
    local out="$RUN_DIR/${name}.stdout"
    local err="$RUN_DIR/${name}.stderr"

    set +e
    "$@" >"$out" 2>"$err"
    local rc=$?
    set -e

    printf '%s\n' "$rc" >"$RUN_DIR/${name}.rc"
    return "$rc"
}

expect_fail_contains() {
    local name="$1"
    local needle="$2"
    shift 2

    if run_case "$name" "$@"; then
        fail "Case '$name' unexpectedly succeeded"
    fi

    if ! grep -Fqi "$needle" "$RUN_DIR/${name}.stderr"; then
        fail "Case '$name' failed without expected diagnostic substring: $needle"
    fi
    info "Case '$name' correctly failed with expected diagnostic"
}

expect_success() {
    local name="$1"
    shift

    if ! run_case "$name" "$@"; then
        tail -n 60 "$RUN_DIR/${name}.stderr" >&2 || true
        fail "Case '$name' failed unexpectedly"
    fi
    info "Case '$name' succeeded"
}

require_file "$BIN"
require_file "$FIXTURES/security_test.csv"
require_file "$FIXTURES/security_xss.csv"
require_file "$FIXTURES/security_xss_edges.csv"

info "Running secure-path rejection checks"
expect_fail_contains path_traversal "must not contain traversal segments" \
    "$BIN" --cli "$FIXTURES/security_test.csv" \
    --output-dir "$RUN_DIR/out_traversal" \
    --assets-dir ../escape \
    --report "$RUN_DIR/out_traversal/report.html"

expect_fail_contains path_outside_workspace "must resolve inside workspace" \
    "$BIN" --cli "$FIXTURES/security_test.csv" \
    --output-dir "$RUN_DIR/out_outside" \
    --assets-dir /tmp/seldon_outside \
    --report "$RUN_DIR/out_outside/report.html"

expect_fail_contains path_assets_root "must not be a filesystem root path" \
    "$BIN" --cli "$FIXTURES/security_test.csv" \
    --output-dir "$RUN_DIR/out_assets_root" \
    --assets-dir / \
    --report "$RUN_DIR/out_assets_root/report.html"

expect_fail_contains cleanup_guard "Refusing to clean workspace/current directory" \
    "$BIN" --cli "$FIXTURES/security_test.csv" \
    --output-dir "$ROOT" \
    --assets-dir "$RUN_DIR/out_cleanup/assets" \
    --report "$RUN_DIR/out_cleanup/report.html"

expect_fail_contains empty_report "report must not be empty" \
    "$BIN" --cli "$FIXTURES/security_test.csv" \
    --output-dir "$RUN_DIR/out_empty_report" \
    --assets-dir "$RUN_DIR/out_empty_report/assets" \
    --report "   "

info "Running positive safe-path check"
expect_success safe_paths \
    "$BIN" --cli "$FIXTURES/security_test.csv" \
    --output-dir "$RUN_DIR/out_safe" \
    --assets-dir "$RUN_DIR/out_safe/assets" \
    --report "$RUN_DIR/out_safe/report.html" \
    --fast true
require_file "$RUN_DIR/out_safe/report.html"

info "Running report XSS-safety checks"
expect_success xss_report \
    "$BIN" --cli "$FIXTURES/security_xss.csv" \
    --target normal_target \
    --output-dir "$RUN_DIR/out_xss" \
    --assets-dir "$RUN_DIR/out_xss/assets" \
    --report "$RUN_DIR/out_xss/report.html" \
    --fast true
require_file "$RUN_DIR/out_xss/report.html"

if grep -Fqi 'x<script>alert(1)</script>' "$RUN_DIR/out_xss/report.html"; then
    fail "Raw script payload was not escaped in xss report"
fi
if ! grep -Fqi 'x&lt;script&gt;alert(1)&lt;/script&gt;' "$RUN_DIR/out_xss/report.html"; then
    fail "Escaped script payload was not found in xss report"
fi

expect_success xss_edges \
    "$BIN" --cli "$FIXTURES/security_xss_edges.csv" \
    --target target \
    --output-dir "$RUN_DIR/out_xss_edges" \
    --assets-dir "$RUN_DIR/out_xss_edges/assets" \
    --report "$RUN_DIR/out_xss_edges/report.html" \
    --plot-modes-explicit true \
    --plot-univariate false \
    --plot-bivariate-significant false \
    --plot-overall false
require_file "$RUN_DIR/out_xss_edges/report.html"
if grep -Fqi 'x<script>alert(1)</script>' "$RUN_DIR/out_xss_edges/report.html"; then
    fail "Raw script payload was not escaped in xss edge report"
fi

info "Running temp-file and subprocess-isolation checks"
command -v gzip >/dev/null 2>&1 || fail "gzip is required for subprocess isolation test"

before_tmp="$(count_tmp_inputs)"
rm -f "$RUN_DIR/PWNED"
MAL_GZ="$RUN_DIR/security_;touch PWNED;_.csv.gz"
gzip -c "$FIXTURES/security_test.csv" >"$MAL_GZ"

expect_success gz_subprocess_isolation \
    "$BIN" --cli "$MAL_GZ" \
    --output-dir "$RUN_DIR/out_gz" \
    --assets-dir "$RUN_DIR/out_gz/assets" \
    --report "$RUN_DIR/out_gz/report.html" \
    --fast true
require_file "$RUN_DIR/out_gz/report.html"

after_tmp="$(count_tmp_inputs)"
if [[ "$before_tmp" != "$after_tmp" ]]; then
    fail "Temporary conversion files leaked: before=$before_tmp after=$after_tmp"
fi
if [[ -f "$RUN_DIR/PWNED" ]]; then
    fail "Subprocess isolation failed: malicious filename created sentinel file"
fi

info "All security regression checks passed"
