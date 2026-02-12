import os
import openai
from typing import List, Dict, Optional, Any

class LLMInterface:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """初始化LLM接口"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        if not self.api_key:
            raise ValueError("API key is required. Please set OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = "You are a helpful assistant."
    
    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.system_prompt = prompt
        # 重置对话历史，应用新的系统提示词
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
    
    def add_message(self, role: str, content: str):
        """添加消息到对话历史"""
        self.conversation_history.append({"role": role, "content": content})
    
    def generate_response(self, user_input: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 500) -> str:
        """生成响应"""
        # 确保系统提示词在对话历史的开头
        if not self.conversation_history or self.conversation_history[0]["role"] != "system":
            self.conversation_history = [{"role": "system", "content": self.system_prompt}]
        
        # 添加用户输入
        self.add_message("user", user_input)
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            assistant_response = response.choices[0].message.content
            
            # 添加助手响应到对话历史
            self.add_message("assistant", assistant_response)
            
            return assistant_response
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
    
    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history

class AgentManager:
    def __init__(self, llm_interface: LLMInterface):
        """初始化智能体管理器"""
        self.llm = llm_interface
        self.agents = {
            "default": {
                "system_prompt": "你是一个有帮助的助手。使用中文回答。",
                "voice": "default"
            }
        }
    
    def set_agent(self, agent_name: str):
        """设置智能体角色"""
        if agent_name in self.agents:
            agent = self.agents[agent_name]
            self.llm.set_system_prompt(agent["system_prompt"])
            return True
        return False
    
    def list_agents(self) -> List[str]:
        """列出可用的智能体角色"""
        return list(self.agents.keys())
    
    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, str]]:
        """获取智能体角色信息"""
        return self.agents.get(agent_name)
