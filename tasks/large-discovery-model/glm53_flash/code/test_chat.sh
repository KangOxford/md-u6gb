#!/bin/bash
# 部署冒烟:OpenAI 兼容接口发一条中文对话,测通即算部署可用
# 用法: bash test_chat.sh [host] [port]
set -u
HOST=${1:-nid010815}
PORT=${2:-8383}
curl -s --max-time 300 "http://$HOST:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "GLM-5.3-Flash",
        "messages": [{"role": "user", "content": "用一句话介绍你自己,并算一下 37*43 等于多少。"}],
        "max_tokens": 512,
        "temperature": 0.6
    }' | python3 -m json.tool
