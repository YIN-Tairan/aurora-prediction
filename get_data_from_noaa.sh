#!/bin/bash

# 动态生成 vars=1 到 vars=56 的字符串
# (注：高精度表单的变量定义可能与小时精度不同，若需全选请确保 56 是准确的变量上限)
VARS_STRING=$(for i in {1..37}; do echo -n "&vars=$i"; done)

FINAL_FILE="omni_1min_data_1964_2025.txt"
> "$FINAL_FILE"

# 设置一个标记，确保表头只被提取一次
HEADER_EXTRACTED=0

echo "开始获取 1981-2025 年的 1 分钟精度数据..."
echo "注：OMNI 的 1 分钟精度数据官方从 1981 年才开始提供，早期年份将自动跳过。"

# 按年份循环请求
for YEAR in {1981..2025}; do
    # 高精度数据的起止时间要求 YYYYMMDDHH (精确到小时)
    START_DATE="${YEAR}010100"
    END_DATE="${YEAR}123123"
    TEMP_FILE="omni_temp_${YEAR}.txt"

    echo "正在请求 ${YEAR} 年的数据..."
    
    # 核心变更：res=min 和 spacecraft=omni_min 是 1 分钟精度的专属参数
    wget -q --post-data "activity=retrieve&res=min&spacecraft=omni_min&start_date=${START_DATE}&end_date=${END_DATE}${VARS_STRING}&scale=Linear&view=0" \
         "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi" -O "$TEMP_FILE"

    # 检查文件是否包含有效的数据行（以年份开头，例如 1981 或 2005）
    if grep -q -E '^(19|20)[0-9]{2}[[:space:]]+' "$TEMP_FILE"; then
        if [ "$HEADER_EXTRACTED" -eq 0 ]; then
            # 第一次遇到有效数据时，提取唯一的表头（从文件开头打印直到遇到第一行有效数据为止）
            awk '/^(19|20)[0-9]{2}[[:space:]]+/{exit} {print}' "$TEMP_FILE" > "$FINAL_FILE"
            HEADER_EXTRACTED=1
            echo "  -> 成功提取表头"
        fi
        
        # 丢弃当前文件的 HTML 头尾，只提取纯数据行并追加到最终文件
        grep -E '^(19|20)[0-9]{2}[[:space:]]+' "$TEMP_FILE" >> "$FINAL_FILE"
        echo "  -> 数据已追加"
    else
        echo "  -> 警告：${YEAR} 年未返回有效数据 (这是 1981 年前的正常现象)。"
    fi

    # 清理临时文件
    rm -f "$TEMP_FILE"
    
    # 暂停 2 秒，防止高频大量请求触发 NASA 服务器的反爬拦截
    sleep 2
done

echo "所有数据下载并拼接完成！最终合并文件保存在：$FINAL_FILE"
