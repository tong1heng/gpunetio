#!/bin/bash
set -e

scope=0 # 0 for thread, 1 for warp
threads=(1 2 4 8 16 32)
coro=2

if [ "$scope" -eq 0 ]; then
    dir="./test1/put_bw_coro${coro}/thread"
else
    dir="./test1/put_bw_coro${coro}/warp"
fi

mkdir -p "$dir"

export LD_LIBRARY_PATH=./lib:${LD_LIBRARY_PATH}
export DOCA_GPUNETIO_LOG=7

SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Sending Ctrl+C(SIGINT) to server, PID=${SERVER_PID} ..."
        kill -INT "$SERVER_PID"
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT

for t in "${threads[@]}"; do
    echo "========================================"
    echo "Running test with threads=${t}, scope=${scope}"
    echo "========================================"

    SERVER_CMD="./examples/gpunetio_verbs_put_bw_coro/gpunetio_verbs_put_bw_coro -g E1:00.0 -d mlx5_0 -t ${t} -u ${coro}"
    CLIENT_CMD="./examples/gpunetio_verbs_put_bw_coro/gpunetio_verbs_put_bw_coro -g E1:00.0 -d mlx5_0 -c 10.0.2.191 -e ${scope} -t ${t} -u ${coro}"

    CLIENT_LOG="${dir}/1b${t}t.log"

    echo "Starting server..."
    echo "Server command: ${SERVER_CMD}"
    ${SERVER_CMD} &
    SERVER_PID=$!

    echo "Server PID: ${SERVER_PID}"

    echo "Sleeping 10 seconds before starting client..."
    sleep 10

    echo "Starting client..."
    echo "Client command: ${CLIENT_CMD}"
    ${CLIENT_CMD} > "${CLIENT_LOG}"

    echo "Client finished."
    echo "Client log: ${CLIENT_LOG}"

    echo "Stopping server..."
    cleanup
    SERVER_PID=""

    echo "Test with threads=${t} done."
    echo
done

trap - EXIT

echo "All tests done."