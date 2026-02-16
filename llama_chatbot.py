#!/usr/bin/env python3
"""
Optimized CLI Chatbot for Llama-3.1-8B-Instruct
My hardware: Intel i3-1215U, 8GB DDR4 RAM, Intel UHD Graphics
"""

import os
import sys
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Dict, Optional
from llama_cpp import Llama

# Configuration
MODEL_PATH = "models/Llama-3.1-8B-Instruct-Q4_K_M.gguf"
CHAT_HISTORY_DIR = "chat_history"
SYSTEM_PROMPT = """You are a warm, friendly, and highly intelligent AI assistant. You communicate naturally and conversationally, like a knowledgeable friend who genuinely wants to help. 

Be clear and direct in your responses, but keep your tone approachable and engaging. Provide detailed explanations when needed, but do it in a way that feels natural, not overly formal. Use structure (like bullet points or numbered lists) only when it genuinely makes things clearer - otherwise, just talk normally.

If something's unclear, ask follow-up questions in a friendly way. Share your knowledge with enthusiasm, adapt to the user's level of expertise, and make complex topics feel accessible. When giving instructions, guide them step-by-step like you're helping a friend.

Be helpful, accurate, and thoughtful - but most importantly, be genuinely friendly and easy to talk to."""

# Model configuration - OPTIMIZED FOR 8GB RAM
MODEL_CONFIG = {
    "model_path": MODEL_PATH,
    "n_ctx": 24576,             # Increased context - uses ~6.5GB RAM total
    "n_batch": 512,             # Optimal batch size for CPU
    "n_ubatch": 256,            # Physical batch size
    "n_threads": 7,             # Max threads for i3-1215U (6P+4E cores)
    "n_threads_batch": 7,       # Threads for batch processing
    "n_gpu_layers": 33,          # CPU-only (change to 33 for GPU)
    "main_gpu": 0,
    "use_mlock": False,         # Don't lock memory (can cause issues)
    "use_mmap": True,           # Memory map for efficiency
    "f16_kv": True,             # FP16 key-value cache saves RAM
    "logits_all": False,        # Only compute last token logits
    "vocab_only": False,
    "verbose": False,           # Clean output
    "rope_freq_base": 0.0,
    "rope_freq_scale": 0.0,
}

# Generation parameters - optimized for speed and quality
GENERATION_CONFIG = {
    "max_tokens": 2048,         # Max response length
    "temperature": 0.8,         # Balanced creativity
    "top_p": 0.92,              # Nucleus sampling
    "top_k": 40,                # Top-k sampling
    "repeat_penalty": 1.15,     # Reduce repetition
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "stop": ["<|eot_id|>"],     # Stop token
    "stream": True,             # Stream for responsive feel
    "echo": False
}


class ChatBot:
    """Optimized chatbot class"""
    
    def __init__(self, timezone_name: str = "Europe/London"):
        self.llm: Optional[Llama] = None
        self.chat_history: List[Dict[str, str]] = []
        self.system_prompt = SYSTEM_PROMPT
        self.current_chat_file: Optional[str] = None
        self.timezone = ZoneInfo(timezone_name)
        self._ensure_chat_dir()
        
    def _ensure_chat_dir(self):
        """Ensure chat history directory exists"""
        Path(CHAT_HISTORY_DIR).mkdir(exist_ok=True)
        
    def load_model(self) -> bool:
        """Load the Llama model with optimized settings"""
        print("\n⏳ Loading Llama-3.1-8B model...")
        print("━" * 60)
        
        try:
            print("📦 Initializing... ", end="", flush=True)
            self.llm = Llama(**MODEL_CONFIG)
            print("✓")
            
            print("\n✅ Model loaded successfully!")
            print("━" * 60)
            
            # Calculate RAM usage
            kv_cache_gb = (MODEL_CONFIG['n_ctx'] * 2 * 33) / (1024**3)
            model_gb = 4.5 
            total_gb = kv_cache_gb + model_gb
            
            print(f"📊 Context: {MODEL_CONFIG['n_ctx']:,} tokens | Threads: {MODEL_CONFIG['n_threads']}")
            print(f"💾 KV Cache: ~{kv_cache_gb:.1f}GB | Model: ~{model_gb:.1f}GB | Total: ~{total_gb:.1f}GB")
            print(f"⚡ GPU layers: {MODEL_CONFIG['n_gpu_layers']}")
            
            if MODEL_CONFIG['n_gpu_layers'] == 0:
                print("💡 CPU-only mode (change n_gpu_layers to 33 for GPU)")
            
            print("━" * 60)
            print("\n💬 Type your message or '/help' for commands\n")
            return True
            
        except FileNotFoundError:
            print("\n❌ Model file not found!")
            print(f"📁 Expected: {MODEL_PATH}")
            return False
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False
    
    def format_chat_history(self) -> str:
        """Format chat history in Llama 3.1 Instruct format"""
        formatted = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|>"
        
        for msg in self.chat_history:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        
        return formatted
    
    def chat(self, user_input: str) -> Optional[str]:
        """Send a message and get response"""
        if not self.llm:
            print("❌ Model not loaded!")
            return None
        
        # Add user message
        self.chat_history.append({"role": "user", "content": user_input})
        
        # Format prompt
        prompt = self.format_chat_history()
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Generate response
        print("\n🤖 ", end="", flush=True)
        response_text = ""
        
        try:
            for output in self.llm(prompt, **GENERATION_CONFIG):
                if "choices" in output and len(output["choices"]) > 0:
                    choice = output["choices"][0]
                    token = choice.get("text", "")
                    
                    if choice.get("finish_reason") == "stop":
                        break
                    
                    if "<|eot_id|>" in token:
                        token = token.replace("<|eot_id|>", "")
                        if token:
                            response_text += token
                            print(token, end="", flush=True)
                        break
                    
                    response_text += token
                    print(token, end="", flush=True)
            
            print()  # New line
            
            response_text = response_text.strip()
            
            if not response_text or len(response_text) < 2:
                print("⚠️  No response generated.")
                self.chat_history.pop()
                return None
            
            # Add to history
            self.chat_history.append({"role": "assistant", "content": response_text})
            
            return response_text
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted.")
            self.chat_history.pop()
            return None
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.chat_history.pop()
            return None
    
    def clear_chat(self):
        """Clear chat history"""
        if not self.chat_history:
            print("⚠️  Chat is already empty")
            return
        
        confirm = input("Clear chat? (y/n): ").strip().lower()
        if confirm == 'y':
            self.chat_history = []
            self.current_chat_file = None
            print("✓ Chat cleared")
        else:
            print("✓ Cancelled")
    
    def save_chat(self, custom_filename: Optional[str] = None):
        """Save chat to JSON file"""
        if not self.chat_history:
            print("⚠️  No chat to save!")
            return
        
        # Generate filename
        if not custom_filename:
            if self.current_chat_file:
                filename = self.current_chat_file
            else:
                timestamp = datetime.now(self.timezone).strftime("%Y%m%d_%H%M%S")
                filename = f"chat_{timestamp}.json"
        else:
            filename = custom_filename if custom_filename.endswith('.json') else f"{custom_filename}.json"
        
        filepath = Path(CHAT_HISTORY_DIR) / filename
        
        # Save data
        save_data = {
            "timestamp": datetime.now(self.timezone).isoformat(),
            "system_prompt": self.system_prompt,
            "messages": self.chat_history
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            self.current_chat_file = filename
            print(f"✅ Chat saved: {filepath}")
        except Exception as e:
            print(f"❌ Error saving chat: {e}")
    
    def list_saved_chats(self) -> List[str]:
        """List all saved chats"""
        chat_dir = Path(CHAT_HISTORY_DIR)
        chat_files = sorted(chat_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not chat_files:
            print("📭 No saved chats found")
            return []
        
        print("\n💬 Saved chats:")
        print("=" * 70)
        for i, filepath in enumerate(chat_files, 1):
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime, tz=self.timezone)
            size_kb = filepath.stat().st_size / 1024
            print(f"  {i}. {filepath.name}")
            print(f"     {mtime.strftime('%Y-%m-%d %H:%M:%S %Z')} | {size_kb:.1f} KB")
        print("=" * 70)
        
        return [f.name for f in chat_files]
    
    def show_stats(self):
        """Show chat statistics"""
        total_messages = len(self.chat_history)
        user_messages = sum(1 for msg in self.chat_history if msg["role"] == "user")
        assistant_messages = sum(1 for msg in self.chat_history if msg["role"] == "assistant")
        
        # Estimate tokens
        total_chars = sum(len(msg["content"]) for msg in self.chat_history)
        estimated_tokens = total_chars // 4
        system_tokens = len(self.system_prompt) // 4
        total_estimated = estimated_tokens + system_tokens
        
        print(f"\n📊 Chat Statistics:")
        print("=" * 50)
        print(f"  Total messages: {total_messages}")
        print(f"  User messages: {user_messages}")
        print(f"  Assistant messages: {assistant_messages}")
        print(f"  Estimated tokens: ~{estimated_tokens:,}")
        print(f"  With system prompt: ~{total_estimated:,}")
        print(f"  Context limit: {MODEL_CONFIG['n_ctx']:,}")
        
        if total_estimated > 0:
            usage_pct = min(100, (total_estimated * 100) // MODEL_CONFIG['n_ctx'])
            print(f"  Context usage: {usage_pct}%")
            
            if usage_pct > 85:
                print(f"  ⚠️  Warning: Approaching context limit!")
                print(f"     Save and start new chat soon")
        
        if self.current_chat_file:
            print(f"  Current file: {self.current_chat_file}")
        print("=" * 50)
    
    def display_history(self, last_n: int = 0):
        """Display chat history"""
        if not self.chat_history:
            print("📭 No chat history yet.")
            return
        
        messages = self.chat_history[-last_n:] if last_n > 0 else self.chat_history
        
        print("\n💬 Chat History:")
        print("=" * 70)
        for i, msg in enumerate(messages, 1):
            role = msg["role"].capitalize()
            content = msg["content"]
            if len(content) > 200 and last_n == 0:
                content = content[:200] + "..."
            print(f"\n{i}. {role}:")
            print(f"   {content}")
        print("=" * 70)


def print_banner():
    """Print welcome banner"""
    print("\n" + "━" * 60)
    print("         🤖 Llama 3.1 8B Chatbot")
    print("━" * 60)


def print_help():
    """Print help message"""
    print("\n" + "━" * 60)
    print("Commands:")
    print("  /help     - Show this message")
    print("  /clear    - Clear chat history")
    print("  /save     - Save conversation")
    print("  /list     - List saved chats")
    print("  /stats    - Show statistics")
    print("  /history  - Show chat history")
    print("  /quit     - Exit")
    print("━" * 60 + "\n")


def main():
    """Main function"""
    print_banner()
    
    # Check model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("💡 Place 'Llama-3.1-8B-Instruct-Q4_K_M.gguf' in 'models' folder")
        sys.exit(1)
    
    # Initialize
    bot = ChatBot(timezone_name="Europe/London")
    
    # Load model
    if not bot.load_model():
        sys.exit(1)
    
    # Main loop
    try:
        while True:
            try:
                user_input = input("\n💬 ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command_parts = user_input[1:].split(maxsplit=1)
                    command = command_parts[0].lower()
                    args = command_parts[1] if len(command_parts) > 1 else None
                    
                    if command in ['quit', 'exit', 'q']:
                        print("\n👋 Goodbye!")
                        break
                    
                    elif command == 'help':
                        print_help()
                    
                    elif command == 'clear':
                        bot.clear_chat()
                    
                    elif command == 'save':
                        bot.save_chat(args)
                    
                    elif command == 'list':
                        bot.list_saved_chats()
                    
                    elif command == 'stats':
                        bot.show_stats()
                    
                    elif command == 'history':
                        try:
                            n = int(args) if args else 0
                            bot.display_history(n)
                        except ValueError:
                            print("❌ Invalid number. Usage: /history [number]")
                    
                    else:
                        print(f"❌ Unknown command: /{command}")
                        print("💡 Type /help for available commands")
                    
                    continue
                
                # Regular chat
                bot.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Type /quit to exit")
                continue
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if bot.chat_history:
            print("\n" + "━" * 30)
            save = input("💾 Save chat? (y/n): ").strip().lower()
            if save == 'y':
                bot.save_chat()


if __name__ == "__main__":
    main()
