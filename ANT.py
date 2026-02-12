import os
import sys
from typing import List, Optional
from NLP.LLM import LLMInterface, AgentManager

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

if __name__ == "__main__":
    """示例用法"""
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
