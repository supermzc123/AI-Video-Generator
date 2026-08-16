# ComfyUI 节点安装

项目节点提供：

- `AVG Save H3 Conditioning`：把H3 `CONDITIONING`保存为安全safetensors和JSON manifest。
- `AVG Load H3 Conditioning`：校验哈希、格式和结构后加载缓存，不加载文本编码器。
- `AVG Save H3 Static Bundle`：原子保存H3节点的`CONDITIONING`和初始AV `LATENT`。
- `AVG Load H3 Static Bundle`：同时恢复两个静态输出，避免扩散阶段因latent连线再次执行H3文本编码节点。

节点代码位于`comfyui_nodes/ai_video_generator_nodes`。使用安装脚本将项目以不修改ComfyUI既有依赖的方式安装，并建立节点目录junction：

```powershell
.\scripts\install-avg-comfyui-nodes.ps1 -ComfyUIRoot D:\Comfy_new\ComfyUI
```

脚本完成后需正常重启ComfyUI一次以注册节点。脚本不会自行停止或重启ComfyUI。

节点缓存固定写入ComfyUI输出目录下的`ai-video-generator/conditioning`，不接受任意文件路径。加载时会验证fingerprint、文件名、大小、SHA-256、tensor数量和JSON结构；不使用pickle或`torch.load`。

H3执行必须使用静态bundle编译接口：编码工作流按保存节点的依赖闭包裁剪，不包含UNet；扩散工作流同时替换conditioning和latent输入，并按显式输出节点裁剪，不包含CLIP、VAE或原始H3 conditioning节点。Motion Context是前序片段完成后产生的运行时输入，不进入静态bundle。

生产Worker只能保留一个Motion Context patch owner。不要在同一实例并装独立`ComfyUI-H3-Motion-Context`或`Contex Loop`。
