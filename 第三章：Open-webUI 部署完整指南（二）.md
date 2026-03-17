# Open-webUI 部署完整指南

Open-webUI 是大模型可视化交互部署的核心方案，基于 Web 端呈现，支持模型调用、参数配置、流式交互等核心功能，无需额外开发前端界面，适配 OpenAI 兼容服务器、自定义 HTTP 服务器等多种后端部署模式，兼顾易用性与灵活性，适合个人测试、企业内部使用及小型在线服务场景。

```bash
root@1a9ad6aef4f4:/data# conda create -p openwebui python==3.11

root@1a9ad6aef4f4:/data# conda activate /data/openwebui/

root@1a9ad6aef4f4:/data# pip install open-webui

(/data/openwebui) root@1a9ad6aef4f4:/data# open-webui serve --port 8080

####服务启动后默认监听的端口是8080端口

在本地开启ssh 转发
xuhlv@xuhlv-mac Downloads % ssh  -CNg -L 8080:127.0.0.1:8080 root+vm-vGe4p6ezAhVkQWbH@140.207.205.81 -p 32222
root+vm-vGe4p6ezAhVkQWbH@140.207.205.81's password: 
client_global_hostkeys_prove_confirm: server gave bad signature for RSA key 0: incorrect signature


启动模型！
(base) root@1a9ad6aef4f4:/data# vllm serve /mnt/moark-models/Qwen3-8B --host 0.0.0.0 --port 8000 -
-trust-remote-code

访问webui 接入模型
http://localhost:8080/auth?redirect=%2F
```

![image-20260304104715035](http://www.410166399.xyz/image-20260304104715035.png)

![image-20260304104931750](http://www.410166399.xyz/image-20260304104931750.png)

![image-20260304105038961](http://www.410166399.xyz/image-20260304105038961.png)

![image-20260304122252400](http://www.410166399.xyz/image-20260304122252400.png)