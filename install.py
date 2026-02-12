#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANT 项目安装脚本

一键安装依赖，支持优雅降级
"""

import os
import sys
import subprocess
import platform
from argparse import ArgumentParser


def run_command(cmd, shell=True, check=False):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {cmd}")
        print(f"错误输出: {e.stderr}")
        return e


def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or version.minor < 8:
        print("警告: Python 版本低于 3.8，可能会导致某些功能异常")
    return version


def install_package(package, extra_index_url=None):
    """安装单个包，支持优雅降级"""
    cmd = [sys.executable, "-m", "pip", "install", "-U"]
    if extra_index_url:
        cmd.extend(["--extra-index-url", extra_index_url])
    cmd.append(package)
    
    print(f"安装: {package}")
    result = run_command(cmd, shell=False)
    return result.returncode == 0 if hasattr(result, 'returncode') else False


def install_optional_package(package, fallback=None):
    """安装可选包，失败时尝试降级"""
    print(f"尝试安装可选包: {package}")
    if install_package(package):
        print(f"✓ {package} 安装成功")
        return True
    else:
        if fallback:
            print(f"尝试降级安装: {fallback}")
            if install_package(fallback):
                print(f"✓ {fallback} 安装成功")
                return True
        print(f"✗ {package} 安装失败，将跳过此包")
        return False


def main():
    """主安装函数"""
    parser = ArgumentParser(description="ANT 项目安装脚本")
    parser.add_argument("--no-optional", action="store_true", help="跳过可选包安装")
    parser.add_argument("--only-asr", action="store_true", help="仅安装 ASR 相关依赖")
    parser.add_argument("--only-tts", action="store_true", help="仅安装 TTS 相关依赖")
    parser.add_argument("--only-nlp", action="store_true", help="仅安装 NLP 相关依赖")
    args = parser.parse_args()

    print("=== ANT 项目安装脚本 ===")
    print(f"系统: {platform.system()} {platform.release()}")
    check_python_version()

    # 升级 pip
    print("\n1. 升级 pip")
    install_package("pip")

    # 安装必要依赖
    print("\n2. 安装必要依赖")
    base_packages = ["numpy", "torch", "transformers"]
    for pkg in base_packages:
        install_package(pkg)

    # 安装 ASR 相关依赖
    if not args.only_tts and not args.only_nlp:
        print("\n3. 安装 ASR 相关依赖")
        install_package("qwen-asr")
        if not args.no_optional:
            install_package("qwen-asr[vllm]")
            install_optional_package("flash-attn --no-build-isolation")

    # 安装 TTS 相关依赖
    if not args.only_asr and not args.only_nlp:
        print("\n4. 安装 TTS 相关依赖")
        install_package("qwen-tts")
        if not args.no_optional:
            install_optional_package("flash-attn --no-build-isolation")

    # 安装 NLP 相关依赖
    if not args.only_asr and not args.only_tts:
        print("\n5. 安装 NLP 相关依赖")
        install_package("openai")
        install_package("transformers")

    # 验证安装
    print("\n6. 验证安装结果")
    packages_to_check = []
    if not args.only_tts and not args.only_nlp:
        packages_to_check.append("qwen-asr")
    if not args.only_asr and not args.only_nlp:
        packages_to_check.append("qwen-tts")
    if not args.only_asr and not args.only_tts:
        packages_to_check.extend(["openai", "transformers"])

    for pkg in packages_to_check:
        result = run_command([sys.executable, "-m", "pip", "show", pkg], shell=False)
        if result.returncode == 0:
            print(f"✓ {pkg} 已安装")
        else:
            print(f"✗ {pkg} 未安装")

    print("\n=== 安装完成 ===")
    print("可以通过以下命令启动 ANT 项目:")
    print("python ANT.py")


if __name__ == "__main__":
    main()
