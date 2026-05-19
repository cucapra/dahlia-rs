#!/usr/bin/env bash
set -u
set -o pipefail

cd "$(dirname "$0")"

if [[ -z "${CALYX_ROOT:-}" ]]; then
    echo "error: CALYX_ROOT is not set" >&2
    exit 2
fi

if ! command -v fud2 >/dev/null 2>&1; then
    echo "error: fud2 not found on PATH" >&2
    exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "error: jq not found on PATH" >&2
    exit 2
fi

test_dir="emit-tests"
out_dir="$test_dir/out"
mkdir -p "$out_dir"

passed=()
failed=()

shopt -s nullglob
fuse_files=("$test_dir"/*.fuse)
shopt -u nullglob

for fuse in "${fuse_files[@]}"; do
    name="$(basename "$fuse" .fuse)"
    data="$test_dir/$name.fuse.data.json"
    expect="$test_dir/$name.expect"
    futil="$out_dir/$name.futil"
    out="$out_dir/$name.out"

    echo "==> $name"

    if [[ ! -f "$data" ]]; then
        echo "    skip: missing $data"
        failed+=("$name (no data file)")
        continue
    fi

    if [[ ! -f "$expect" ]]; then
        echo "    skip: missing $expect"
        failed+=("$name (no expect file)")
        continue
    fi

    if ! cargo run --quiet <"$fuse" >"$futil" 2>"$out_dir/$name.emit.log"; then
        echo "    fail: emit (see $out_dir/$name.emit.log)"
        failed+=("$name (emit)")
        continue
    fi

    if ! fud2 "$futil" --to jq --through verilator \
        -s sim.data="$data" \
        -s jq.expr=".memories" \
        >"$out" 2>"$out_dir/$name.sim.log"; then
        echo "    fail: simulate (see $out_dir/$name.sim.log)"
        failed+=("$name (sim)")
        continue
    fi

    diff_out=$(diff -u \
        <(jq -S . "$expect") \
        <(jq -S . "$out"))
    if [[ -n "$diff_out" ]]; then
        echo "    fail: diff"
        printf '%s\n' "$diff_out" | sed 's/^/      /'
        failed+=("$name (diff)")
        continue
    fi

    echo "    ok"
    passed+=("$name")
done

echo
echo "==> summary: ${#passed[@]} passed, ${#failed[@]} failed"
for n in "${passed[@]}"; do echo "    ok   $n"; done
for n in "${failed[@]}"; do echo "    FAIL $n"; done

[[ ${#failed[@]} -eq 0 ]]
