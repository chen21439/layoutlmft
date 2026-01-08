#!/bin/bash
APP_NAME="layoutxlm"
PORT=9197

# 自动检测项目目录（脚本所在目录的父目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

LOG_DIR="${PROJECT_DIR}/api/log"
mkdir -p "$LOG_DIR"
TODAY=$(date +%Y-%m-%d)
LOG_FILE="${LOG_DIR}/uvicorn_${PORT}_${TODAY}.log"
PID_FILE="${PROJECT_DIR}/.pid_${PORT}"

export HF_HOME=/data/LLM_group/HuggingFace
export HF_HUB_CACHE=/data/LLM_group/HuggingFace/Hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1
export ENV=test

cd "$PROJECT_DIR" || {
    echo "[${APP_NAME}] 无法进入目录: $PROJECT_DIR"
    exit 1
}

find_pid() {
    ps aux | grep "api.app.main" | grep -v grep | awk '{print $2}'
}

stop_service() {
    PORT_PID=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$PORT_PID" ]; then
        echo "[${APP_NAME}] 发现端口 $PORT 被占用，PID: $PORT_PID"
        kill $PORT_PID 2>/dev/null
        sleep 1
        if lsof -ti:$PORT > /dev/null 2>&1; then
            kill -9 $(lsof -ti:$PORT) 2>/dev/null
        fi
    fi

    PID=$(find_pid)
    if [ -n "$PID" ]; then
        echo "[${APP_NAME}] 正在停止进程 PID: $PID ..."
        kill $PID 2>/dev/null
        sleep 2
        if ps -p $PID > /dev/null 2>&1; then
            kill -9 $PID 2>/dev/null
        fi
        echo "[${APP_NAME}] 已停止"
    else
        echo "[${APP_NAME}] uvicorn 进程未运行"
    fi

    sleep 1
    if lsof -ti:$PORT > /dev/null 2>&1; then
        echo "[${APP_NAME}] 警告: 端口 $PORT 仍被占用，强制释放..."
        kill -9 $(lsof -ti:$PORT) 2>/dev/null
        sleep 1
    fi
}

start_service() {
    echo "[${APP_NAME}] 启动服务..."
    echo "[${APP_NAME}] 环境: $ENV"
    echo "[${APP_NAME}] 端口: $PORT"
    echo "[${APP_NAME}] 项目目录: $PROJECT_DIR"

    nohup python -m api.app.main --env "$ENV" \
        >> "$LOG_FILE" 2>&1 &

    NEW_PID=$!
    echo "$NEW_PID" > "$PID_FILE"

    sleep 3
    if ps -p "$NEW_PID" > /dev/null 2>&1; then
        echo "[${APP_NAME}] 启动成功 PID: $NEW_PID"
        echo "[${APP_NAME}] 日志文件: $LOG_FILE"
        echo "[${APP_NAME}] 查看日志: tail -f $LOG_FILE"
    else
        echo "[${APP_NAME}] 启动失败，请检查日志: $LOG_FILE"
        return 1
    fi
}

status_service() {
    PID=$(find_pid)
    if [ -n "$PID" ]; then
        echo "[${APP_NAME}] 运行中 PID: $PID"
        echo "[${APP_NAME}] 端口: $PORT"
        echo "[${APP_NAME}] 环境: $ENV"
    else
        echo "[${APP_NAME}] 未运行"
    fi
}

# 默认启动，stop/status 为可选参数
case "${1:-start}" in
    start)
        PID=$(find_pid)
        if [ -n "$PID" ]; then
            echo "[${APP_NAME}] 检测到服务已在运行 PID: $PID，先停止再重启..."
            stop_service
            sleep 1
        else
            echo "[${APP_NAME}] 服务未运行，直接启动..."
        fi
        start_service
        ;;
    stop)
        stop_service
        ;;
    status)
        status_service
        ;;
    *)
        echo "用法: $0 [stop|status]"
        exit 1
        ;;
esac
