# 腾讯云托管启动入口
import sys

# 从当前目录导入 Flask 应用实例
from run import app

# 启动Flask Web服务（端口从命令行参数读取，默认8080）
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
print(f"🚀 Starting Flask on port {port}")
app.run(host=sys.argv[1] if len(sys.argv) > 1 else '0.0.0.0', port=port, debug=False, threaded=True)
