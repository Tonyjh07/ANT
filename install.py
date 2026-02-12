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


def install_package(package, python_path=None, extra_index_url=None):
    """安装单个包，支持优雅降级"""
    python = python_path or sys.executable
    cmd = [python, "-m", "pip", "install", "-U"]
    if extra_index_url:
        cmd.extend(["--extra-index-url", extra_index_url])
    cmd.append(package)
    
    print(f"安装: {package}")
    result = run_command(cmd, shell=False)
    return result.returncode == 0 if hasattr(result, 'returncode') else False


def install_optional_package(package, python_path=None, fallback=None):
    """安装可选包，失败时尝试降级"""
    print(f"尝试安装可选包: {package}")
    if install_package(package, python_path):
        print(f"✓ {package} 安装成功")
        return True
    else:
        if fallback:
            print(f"尝试降级安装: {fallback}")
            if install_package(fallback, python_path):
                print(f"✓ {fallback} 安装成功")
                return True
        print(f"✗ {package} 安装失败，将跳过此包")
        return False


def create_venv(venv_name="venv"):
    """创建虚拟环境"""
    venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), venv_name)
    
    if os.path.exists(venv_path):
        print(f"虚拟环境已存在: {venv_path}")
        return venv_path
    
    print(f"创建虚拟环境: {venv_path}")
    result = run_command([sys.executable, "-m", "venv", venv_path], shell=False)
    
    if result.returncode == 0:
        print(f"✓ 虚拟环境创建成功: {venv_path}")
        return venv_path
    else:
        print(f"✗ 虚拟环境创建失败")
        return None


def get_venv_python(venv_path):
    """获取虚拟环境中的Python路径"""
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:
        return os.path.join(venv_path, "bin", "python")


def is_in_venv():
    """检查是否在虚拟环境中"""
    return hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix


def main():
    """主安装函数"""
    parser = ArgumentParser(description="ANT 项目安装脚本")
    parser.add_argument("--no-optional", action="store_true", help="跳过可选包安装")
    parser.add_argument("--only-asr", action="store_true", help="仅安装 ASR 相关依赖")
    parser.add_argument("--only-tts", action="store_true", help="仅安装 TTS 相关依赖")
    parser.add_argument("--only-nlp", action="store_true", help="仅安装 NLP 相关依赖")
    parser.add_argument("--no-venv", action="store_true", help="不使用虚拟环境")
    parser.add_argument("--venv-name", default="venv", help="虚拟环境名称 (默认: venv)")
    args = parser.parse_args()

    print("=== ANT 项目安装脚本 ===")
    print(f"系统: {platform.system()} {platform.release()}")
    check_python_version()

    # 检查是否在虚拟环境中
    in_venv = is_in_venv()
    if in_venv:
        print("\n在虚拟环境中运行，将直接安装依赖")
        venv_python = sys.executable
    else:
        # 创建虚拟环境
        if not args.no_venv:
            print("\n0. 创建虚拟环境")
            venv_path = create_venv(args.venv_name)
            if venv_path:
                venv_python = get_venv_python(venv_path)
                print(f"虚拟环境Python路径: {venv_python}")
                print("\n请先激活虚拟环境:")
                if platform.system() == "Windows":
                    print(f"  {args.venv_name}\\Scripts\\activate")
                else:
                    print(f"  source {args.venv_name}/bin/activate")
                print("然后重新运行此脚本")
                return
            else:
                print("虚拟环境创建失败，继续在当前环境安装")
                venv_python = sys.executable
        else:
            venv_python = sys.executable

    # 升级 pip
    print("\n1. 升级 pip")
    install_package("pip", venv_python)

    # 安装必要依赖
    print("\n2. 安装必要依赖")
    base_packages = ["numpy", "torch", "transformers"]
    for pkg in base_packages:
        install_package(pkg, venv_python)

    # 安装 ASR 相关依赖
    if not args.only_tts and not args.only_nlp:
        print("\n3. 安装 ASR 相关依赖")
        install_package("qwen-asr", venv_python)
        if not args.no_optional:
            install_package("qwen-asr[vllm]", venv_python)
            install_optional_package("flash-attn --no-build-isolation", venv_python)

    # 安装 TTS 相关依赖
    if not args.only_asr and not args.only_nlp:
        print("\n4. 安装 TTS 相关依赖")
        install_package("qwen-tts", venv_python)
        if not args.no_optional:
            install_optional_package("flash-attn --no-build-isolation", venv_python)

    # 安装 NLP 相关依赖
    if not args.only_asr and not args.only_tts:
        print("\n5. 安装 NLP 相关依赖")
        install_package("openai", venv_python)
        install_package("transformers", venv_python)
        install_package("python-dotenv", venv_python)

    # 安装其他必要依赖
    print("\n6. 安装其他必要依赖")
    install_package("python-dotenv", venv_python)
    install_package("soundfile", venv_python)

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
        result = run_command([venv_python, "-m", "pip", "show", pkg], shell=False)
        if result.returncode == 0:
            print(f"✓ {pkg} 已安装")
        else:
            print(f"✗ {pkg} 未安装")

    print("\n=== 安装完成 ===")
    print("可以通过以下命令启动 ANT 项目:")
    print("python ANT.py")


if __name__ == "__main__":
    main()
