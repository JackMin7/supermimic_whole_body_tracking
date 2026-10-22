# GitHub 发布

仓库：https://github.com/JackMin7/supermimic_whole_body_tracking

本目录即仓库根目录。README、Python 包、脚本、实验结果、图表及视频均包含在发布包中；SuperMimic 的方法归属见 NOTICE.md。大体积权重、原始运动库、私人机器配置和第三方资产不在发布包内。

新代码使用仓库原有的 MIT 许可证，见 LICENSE。第三方材料的权利与授权范围见 NOTICE.md。

更新后在本目录执行：

```bash
git add .
git commit -m "Update humanoid control experiment"
git push origin main
```

MP4 为较小的演示文件，直接包含在仓库中。README 使用相对链接，GitHub 可能将 MP4 显示为文件链接。不要提交 `runs/` 或完整训练环境。
