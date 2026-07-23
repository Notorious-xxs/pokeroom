FROM python:3.11-slim

# 安装构建依赖（YOLO/pytorch 需要 C++ 编译器）
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    make \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 拷贝项目文件
COPY requirements.txt .
COPY templates/ templates/
COPY run.py .
COPY run_entry.py .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口（必须与服务设置中的端口一致）
EXPOSE 8080

# 启动入口
CMD ["python", "run_entry.py", "0.0.0.0", "8080"]
