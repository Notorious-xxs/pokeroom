# 腾讯云托管启动入口
import sys
import os

# 从 app.py 导入 Flask 应用实例
sys.path.insert(0, os.path.dirname(__file__))
from app import app as flask_app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"🚀 Starting Flask on port {port}")
    flask_app.run(host='0.0.0.0', port=port, debug=False)
