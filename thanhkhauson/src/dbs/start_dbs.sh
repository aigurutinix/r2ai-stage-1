#!/usr/bin/env bash
set -euo pipefail

DBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ES_DIR="${DBS_DIR}/elasticsearch-9.4.2"
QDRANT_BIN="${DBS_DIR}/qdrant"

PID_DIR="${DBS_DIR}/.pids"
LOG_DIR="${DBS_DIR}/runtime_logs"
mkdir -p "${PID_DIR}" "${LOG_DIR}"

ES_HTTP_PORT="${ES_HTTP_PORT:-9201}"
ES_TRANSPORT_PORT="${ES_TRANSPORT_PORT:-9301}"
ES_JAVA_OPTS="${ES_JAVA_OPTS:--Xms4g -Xmx4g}"

QDRANT_HTTP_PORT="${QDRANT_HTTP_PORT:-6333}"

is_port_open() {
    local port="$1"
    timeout 1 bash -c "</dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1
}

wait_for_port() {
    local name="$1"
    local port="$2"
    local timeout_seconds="${3:-60}"
    local elapsed=0

    while ! is_port_open "${port}"; do
        if (( elapsed >= timeout_seconds )); then
            echo "[WARN] ${name} chưa mở port ${port} sau ${timeout_seconds}s."
            return 1
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    echo "[OK] ${name} đang lắng nghe trên port ${port}."
}

start_elasticsearch() {
    if is_port_open "${ES_HTTP_PORT}"; then
        echo "[SKIP] Elasticsearch đã chạy trên port ${ES_HTTP_PORT}."
        return 0
    fi

    if [[ ! -x "${ES_DIR}/bin/elasticsearch" ]]; then
        echo "[ERROR] Không thấy executable: ${ES_DIR}/bin/elasticsearch" >&2
        return 1
    fi

    echo "[START] Elasticsearch local: http://localhost:${ES_HTTP_PORT}"
    (
        cd "${ES_DIR}"
        ES_JAVA_OPTS="${ES_JAVA_OPTS}" ./bin/elasticsearch \
            -d \
            -p "${PID_DIR}/elasticsearch.pid" \
            -Ehttp.port="${ES_HTTP_PORT}" \
            -Etransport.port="${ES_TRANSPORT_PORT}" \
            -Expack.security.enabled=false \
            >"${LOG_DIR}/elasticsearch.stdout.log" 2>&1
    )

    wait_for_port "Elasticsearch" "${ES_HTTP_PORT}" 90 || {
        echo "[HINT] Xem log: ${ES_DIR}/logs/elasticsearch.log"
        echo "[HINT] Xem stdout: ${LOG_DIR}/elasticsearch.stdout.log"
        return 1
    }
}

start_qdrant() {
    if is_port_open "${QDRANT_HTTP_PORT}"; then
        echo "[SKIP] Qdrant đã chạy trên port ${QDRANT_HTTP_PORT}."
        return 0
    fi

    if [[ ! -x "${QDRANT_BIN}" ]]; then
        echo "[ERROR] Không thấy executable: ${QDRANT_BIN}" >&2
        return 1
    fi

    echo "[START] Qdrant: http://localhost:${QDRANT_HTTP_PORT}"
    (
        cd "${DBS_DIR}"
        nohup "${QDRANT_BIN}" >"${LOG_DIR}/qdrant.log" 2>&1 &
        echo "$!" >"${PID_DIR}/qdrant.pid"
    )

    wait_for_port "Qdrant" "${QDRANT_HTTP_PORT}" 45 || {
        echo "[HINT] Xem log: ${LOG_DIR}/qdrant.log"
        return 1
    }
}

start_elasticsearch
start_qdrant

echo
echo "DB services ready:"
echo "  Elasticsearch: http://localhost:${ES_HTTP_PORT}"
echo "  Qdrant:        http://localhost:${QDRANT_HTTP_PORT}"
