import os
import sys
import subprocess
from typing import List, Optional
from NLP.LLM import LLMInterface, AgentManager


def is_in_venv():
    """检查是否在虚拟环境中"""
    return hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix


def get_venv_python(venv_name="venv"):
    """获取虚拟环境中的Python路径"""
    venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), venv_name)
    if os.name == 'nt':  # Windows
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:  # Linux/Mac
        return os.path.join(venv_path, "bin", "python")


def venv_exists(venv_name="venv"):
    """检查虚拟环境是否存在"""
    venv_python = get_venv_python(venv_name)
    return os.path.exists(venv_python)

class ANT:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """初始化ANT智能体系统"""
        # 初始化NLP模块
        self.llm = LLMInterface(api_key=api_key, base_url=base_url)
        self.agent_manager = AgentManager(self.llm)
        
        # 初始化ASR和TTS模块（预留接口）
        self.asr_available = False
        self.tts_available = False
        
        # 尝试导入ASR模块
        try:
            from qwen_asr.inference.qwen3_asr import Qwen3ASR
            self.asr = Qwen3ASR()
            self.asr_available = True
        except ImportError:
            print("ASR module not available. Please install ASR dependencies.")
        except Exception as e:
            print(f"Error initializing ASR: {str(e)}")
        
        # 尝试导入TTS模块
        try:
            from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
            from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
            self.tts_tokenizer = Qwen3TTSTokenizer()
            self.tts_model = Qwen3TTSModel()
            self.tts_available = True
        except ImportError:
            print("TTS module not available. Please install TTS dependencies.")
        except Exception as e:
            print(f"Error initializing TTS: {str(e)}")
    
    def set_agent(self, agent_name: str):
        """设置智能体角色"""
        return self.agent_manager.set_agent(agent_name)
    
    def list_agents(self) -> List[str]:
        """列出可用的智能体角色"""
        return self.agent_manager.list_agents()
    
    def process_text(self, text: str) -> str:
        """处理文本输入"""
        return self.llm.generate_response(text)
    
    def process_audio(self, audio_path: str) -> Optional[str]:
        """处理音频输入"""
        if not self.asr_available:
            return "ASR module not available"
        
        try:
            # 使用ASR识别音频
            text = self.asr(audio_path)
            
            # 处理识别的文本
            response = self.process_text(text)
            
            # 使用TTS合成语音（如果可用）
            if self.tts_available:
                self.synthesize_speech(response, "output.wav")
            
            return response
        except Exception as e:
            return f"Error processing audio: {str(e)}"
    
    def synthesize_speech(self, text: str, output_path: str):
        """合成语音"""
        if not self.tts_available:
            print("TTS module not available")
            return
        
        try:
            # 使用TTS合成语音
            tokens = self.tts_tokenizer(text)
            audio = self.tts_model(tokens)
            
            # 保存音频到文件
            import soundfile as sf
            sf.write(output_path, audio, 24000)
            print(f"Speech synthesized to {output_path}")
        except Exception as e:
            print(f"Error synthesizing speech: {str(e)}")
    
    def chat(self, message: str) -> str:
        """聊天功能"""
        return self.llm.generate_response(message)
    
    def clear_history(self):
        """清除对话历史"""
        self.llm.clear_history()
        print("Conversation history cleared")

def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="ANT 智能体系统")
    parser.add_argument("--no-venv", action="store_true", help="强制不使用虚拟环境")
    parser.add_argument("--venv-name", default="venv", help="虚拟环境名称 (默认: venv)")
    args = parser.parse_args()
    
    # 检查是否在虚拟环境中
    in_venv = is_in_venv()
    
    # 如果不在虚拟环境中，且没有强制不使用虚拟环境，且虚拟环境存在
    if not in_venv and not args.no_venv and venv_exists(args.venv_name):
        venv_python = get_venv_python(args.venv_name)
        print(f"检测到虚拟环境，将在虚拟环境中运行 ANT 智能体系统...")
        print(f"虚拟环境路径: {venv_python}")
        
        # 重新在虚拟环境中运行自身
        subprocess.run([venv_python] + sys.argv, check=False)
        return
    
    # 如果不在虚拟环境中，且虚拟环境不存在
    if not in_venv and not args.no_venv and not venv_exists(args.venv_name):
        print("未检测到虚拟环境，将在当前环境运行 ANT 智能体系统")
        print("提示: 建议使用虚拟环境运行，可通过以下命令创建:")
        print("  python install.py")
    
    # 如果在虚拟环境中
    if in_venv:
        print(f"在虚拟环境中运行 ANT 智能体系统")
        print(f"虚拟环境路径: {sys.prefix}")
    
    # 从环境变量获取API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # 初始化ANT
    ant = ANT(api_key=api_key, base_url=base_url)
    
    # 示例：列出可用的智能体角色
    print("Available agents:", ant.list_agents())
    
    # 示例：设置智能体角色
    ant.set_agent("friend")
    print("Agent set to: friend")
    
    # 示例：聊天
    print("\nChat example:")
    response = ant.chat("Hello, how are you?")
    print(f"Response: {response}")
    
    # 示例：继续聊天
    response = ant.chat("What can you do?")
    print(f"Response: {response}")
    
    # 示例：清除对话历史
    ant.clear_history()
    
    # 示例：使用不同的智能体角色
    ant.set_agent("professional")
    print("\nAgent set to: professional")
    response = ant.chat("Explain quantum computing in simple terms.")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
