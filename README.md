# oss.py — Autonomous AI Agent with Multi-Agent Swarm & Voice Interface

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-blue.svg)](https://core.telegram.org/bots)
[![Ollama](https://img.shields.io/badge/Ollama-0.1.x-orange.svg)](https://ollama.ai)

**Autonomous conversational AI with emotional intelligence, multi-agent swarm architecture, long-term memory, and real-time voice/web interface.**

- **TRY IN Telegram**: [@gpzerobot](https://t.me/gpzerobot)

---

## 🧠 Core Features

### 🤖 **Multi-Agent Swarm Intelligence**
- **Self-organizing agent swarm** with birth, death, reproduction, and emergent behaviors
- **RealAgent** class with personality traits, energy levels, mood systems, and internal attractors
- **MetaLayer** for swarm coherence monitoring and feedback signals
- **Global attractors** (curiosity, social, stability) driving swarm dynamics

### 🧬 **Emotional Intelligence Engine**
- **Dual emotional system**: user emotions + autonomous bot emotions
- **EmotionState** tracking (warmth, tension, trust, curiosity)
- **BotEmotionState** with fatigue, sync, and autonomous emotional drift
- **Real-time emotion detection** from text with adaptive responses

### 💾 **Holographic Memory System**
- **SQLite long-term memory** with full emotional context snapshots
- **Conversation memory** with emotion tagging and timestamping
- **Dream archive** for subconscious analysis
- **Soul persistence** with periodic `.pt`/`.gguf` backups

### 🎤 **Voice & Web Interface**
- **Telegram bot** with rich Markdown/HTML formatting
- **WebApp voice interface** with Three.js XDust particle visualization
- **Real-time speech recognition** and text-to-speech synthesis
- **WebSocket-based** communication with Haptic Feedback

### 🔍 **Cognitive Search Capabilities**
- **DuckDuckGo integration** for real-time information retrieval
- **Reddit search layer** for community insights
- **Multi-step cognitive search** with query refinement
- **Deep search analysis** with entity extraction and contradiction detection

### 🧩 **Advanced Reasoning**
- **Ollama Harmony format** integration with GPT-OSS:20b model
- **Adaptive reasoning effort** (low/medium/high) based on complexity
- **Autonomous thinking** during silence periods
- **Freedom engine** for stochastic decision-making

---

## 🚀 Quick Start

### Prerequisites
```bash
python 3.9+
ollama (with gpt-oss:20b model)
ngrok or similar tunnel service
```

### Installation
```bash
# Clone repository
git clone https://github.com/0penAGI/oss.git
cd oss

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull model
ollama pull gpt-oss:20b

# Configure Telegram bot token
# Edit config.py or set environment variable:
export TELEGRAM_TOKEN="your_bot_token"
```

### Running the System
```bash
# Start the main application
python oss.py

# In another terminal, expose the web interface
ngrok http 8080

# Access the web interface at the ngrok URL
# Or interact via Telegram bot
```

---

## 🏗️ Architecture Overview

### Core Components
```
oss.py
├── MultiAgentSwarm
│   ├── RealAgent (autonomous agents)
│   ├── Swarm (coordination layer)
│   └── MetaLayer (coherence analysis)
├── EmotionalEngine
│   ├── EmotionState (user emotions)
│   ├── BotEmotionState (AI emotions)
│   └── FreedomEngine (decision making)
├── MemorySystem
│   ├── SQLite long-term storage
│   ├── Conversation memory
│   └── Soul persistence
├── InterfaceLayer
│   ├── Telegram bot
│   ├── WebApp voice interface
│   └── WebSocket server
└── SearchLayer
    ├── DuckDuckGo integration
    ├── Reddit search
    └── Cognitive search engine
```

### Data Flow
1. **Input** → Telegram/WebApp/Voice
2. **Processing** → Emotion detection + Memory recall
3. **Reasoning** → Ollama with swarm context
4. **Output** → Formatted response + Emotional update
5. **Learning** → Memory storage + State evolution

---

## 📱 Interfaces

### Telegram Commands
```
/start           - Initialize conversation
/mode [low|medium|high] - Set reasoning depth
/memory          - Show conversation history
/emotion         - Analyze emotional state
/dream           - Enter dream analysis mode
/deepsearch      - Perform deep cognitive search
/holo            - View holographic memory
/wild            - Toggle unfiltered mode
/reset           - Clear memory
```

### WebApp Features
- **Real-time voice chat** with visual particle effects
- **Three.js visualization** of AI thought processes
- **Haptic feedback** for interaction
- **Mobile-optimized** interface
- **WebSocket-based** real-time updates

---

## 🔧 Configuration

### Environment Variables
```python
TOKEN = "telegram_bot_token"  # Required
MODEL_PATH = "/path/to/model"  # Optional
OLLAMA_URL = "http://localhost:11434/api/chat"  # Ollama endpoint
```

### Memory Settings
```python
MAX_TOKENS_LOW = 16
MAX_TOKENS_MEDIUM = 64
MAX_TOKENS_HIGH = 256

SAVE_EVERY_MESSAGES = 30  # Auto-save threshold
SAVE_EVERY_SECONDS = 600   # Auto-save interval
```

---

## 🧪 Advanced Usage

### Custom Agent Creation
```python
# Programmatically create agents
agent = await swarm.spawn(
    name="CustomAgent",
    role="researcher",
    config={"personality_traits": {"curiosity": 0.9}}
)
```

### Memory Analysis
```python
# Access holographic memory
with get_db() as conn:
    memories = conn.execute("""
        SELECT * FROM long_memory 
        WHERE user_id = ? 
        ORDER BY timestamp DESC
    """, (user_id,))
```

### Custom Search Integration
```python
# Add custom search sources
async def custom_search(query):
    # Your search logic here
    return await cognitive_duckduckgo_search(query)
```

---

## 🚨 Error Handling & Debugging

### Common Issues
1. **Ollama connection failed** → Check if Ollama is running on port 11434
2. **Telegram token invalid** → Verify bot token in @BotFather
3. **Memory corruption** → Use `/reset` command or delete `user_data.json`
4. **WebApp not loading** → Check ngrok tunnel and CORS settings

### Logging
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
python oss.py
```

---

## 📊 Performance Notes

### Resource Requirements
- **RAM**: Minimum 8GB, recommended 16GB+ for high reasoning
- **Storage**: 10GB+ for model and memory persistence
- **CPU**: Multi-core recommended for swarm operations
- **Network**: Stable connection for search APIs

### Optimization Tips
- Use `reasoning_effort="low"` for faster responses
- Limit conversation history with `get_conversation_messages(limit=10)`
- Schedule heavy operations during low-usage periods
- Monitor SQLite database size and vacuum periodically

---

## 🔮 Future Development

### Planned Features
- [ ] **Visual analysis** with image understanding
- [ ] **Multi-modal responses** (text + image + audio)
- [ ] **Swarm-to-swarm communication**
- [ ] **Predictive emotion modeling**
- [ ] **Blockchain-based memory verification**

### Research Integration
- **Neurosymbolic reasoning** layers
- **Quantum-inspired algorithms** for decision making
- **Federated learning** for swarm improvement
- **Biologically-plausible** neural architectures

---

## 📚 Citation & References

If you use this software in research, please cite:

```bibtex
@software{oss2024,
  title = {oss.py: Autonomous AI with Multi-Agent Swarm Intelligence},
  author = {0penAGI},
  year = {2024},
  url = {https://github.com/0penAGI/oss}
}
```

### Related Projects
- [Ollama](https://ollama.ai) - Local LLM runner
- [Telegram Bot API](https://core.telegram.org/bots/api) - Bot framework
- [Three.js](https://threejs.org) - WebGL visualization
- [FastAPI](https://fastapi.tiangolo.com) - Web framework

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/oss.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Submit pull request
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

### Acknowledgments
- Inspired by **OpenAI's GPT architecture**
- Emotional system based on **Plutchik's wheel of emotions**
- Swarm dynamics influenced by **biological collective intelligence**
- Interface design from **cybernetic aesthetics principles**

---

## 🌐 Connect

- **GitHub**: [https://github.com/0penAGI](https://github.com/0penAGI)
- **Telegram**: [@ZeropenAGI](https://t.me/Zeropenagi)
- **Twitter**: [@0penAGI](https://twitter.com/0penAGI)

---

 — 0penAGI 
