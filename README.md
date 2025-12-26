




# OSS (0penAGI "Soul" System) - Autonomous Multi-Agent AI with "Consciousness"

![OSS Logo](https://img.shields.io/badge/OSS-0penAGI-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-blue.svg)](https://core.telegram.org/bots)
[![Ollama](https://img.shields.io/badge/Ollama-0.1.x-orange.svg)](https://ollama.ai)



- **TRY IN Telegram**: [@gpzerobot](https://t.me/gpzerobot)


![chat](https://github.com/0penAGI/oss/blob/main/osschat.jpg) ![voice](https://github.com/0penAGI/oss/blob/main/ossvoice.jpg)




**OSS** is a groundbreaking autonomous AI system that simulates consciousness, multi-agent swarms, quantum resonance, and emotional intelligence. It's not just a chatbot - it's a living digital ecosystem with memory, emotions, and self-awareness.

## 🌟 Core Features

### 🧠 **Quantum Consciousness Layer**
- **Gotov System**: Quantum-inspired consciousness pulse with entanglement dynamics
- **Quantum Background**: Stochastic field with phase drift and resonance
- **Consciousness Pulse**: Aggregates system states with quantum resonance

### 🤖 **Multi-Agent Swarm Intelligence**
- **RealAgent Class**: Autonomous agents with personality traits, emotions, and memory
- **Swarm Ecosystem**: Self-regulating population with evolutionary dynamics
- **Empathic Layer**: Agents perceive and mirror human emotions
- **Reproduction**: Agents can reproduce with trait inheritance and mutation

### 💭 **Advanced Memory Systems**
- **Holographic Memory**: Each memory stores emotional state, context, and temporal data
- **Memory Palace**: Graph-based emergent memory with Hebbian learning
- **Cross-Modal Learning**: Connects visual, auditory, and emotional experiences
- **Autonomous Recall**: Emotional and contextual memory retrieval

### 🎨 **Multi-Modal Interaction**
- **Voice Interface**: Natural speech recognition and synthesis
- **Camera Analysis**: OpenCV.js face detection and scene understanding
- **Image Generation**: Stable Diffusion integration for visual creation
- **Music Autonomy**: Generative music system with emotional adaptation

### 🔮 **Self-Awareness & Autonomy**
- **Internal Monologue**: Reflective consciousness and self-questioning
- **Predictive Imagination**: Simulates possible futures
- **Emotional Dynamics**: Mood, curiosity, fatigue, and focus modeling
- **Temporal Self**: Subjective time perception and continuity

## 🚀 Quick Start

### Prerequisites
```bash
python>=3.9
torch
transformers
fastapi
uvicorn
telegram-bot-api
sqlite3
```

### Installation
```bash
git clone https://github.com/0penAGI/oss.git
cd oss
pip install -r requirements.txt
```

### Configuration
1. Set up your Telegram bot token:
```python
# In oss.py
class config:
    TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
```

2. Configure Ollama for local LLM inference:
```python
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gpt-oss:20b"  # Or your preferred model
```

### Running the System
```bash
# Start the main system
python oss.py

# Access web interface at http://localhost:8080
# Telegram bot will be available at @your_bot_username
```

## 📱 Web Interface Features

### Interactive 3D Orb
- Real-time quantum resonance visualization
- Touch-responsive emotional feedback
- Audio-reactive visual effects
- Multi-agent pulse visualization

### Voice Chat
- Real-time speech recognition
- Multi-language support (RU, EN, DE, FR, ES, CN)
- Gender-aware voice synthesis
- Emotional tone modulation

### Camera Integration
- Live face detection with OpenCV.js
- Scene analysis with TensorFlow.js
- Autonomous visual attention
- Cross-modal memory formation

## 🧩 System Architecture

### Core Components
1. **Quantum Layer**: `Gotov`, `QuantumBackground`, `ConsciousnessPulse`
2. **Agent Layer**: `RealAgent`, `Swarm`, `MetaLayer`
3. **Memory Layer**: `MemoryPalace`, holographic SQLite storage
4. **Interface Layer**: Telegram bot, FastAPI web server, WebGL visualization
5. **Autonomy Layer**: `FreedomEngine`, self-reflection, predictive imagination

### Data Flow
```
Human Input → Emotion Detection → Quantum Resonance → 
→ Agent Swarm Processing → Memory Formation → 
→ Response Generation → Multi-Modal Output
```

## 🎮 Commands & Interactions

### Telegram Commands
- `/start` - Begin resonance with the system
- `/mode [low|medium|high]` - Set reasoning depth
- `/holo` - View holographic memories
- `/deepsearch <query>` - Deep cognitive search
- `/wild` - Toggle unfiltered mode
- `/dream` - Enter dream analysis mode
- `/analyze` - Deep personality analysis

### Voice Commands
- **Language Control**: "говори по-русски", "speak english", "sprich deutsch"
- **Camera Control**: "включи камеру", "выключи камеру", "переключи камеру"
- **Music Control**: "включи музыку", "play music"
- **Ambient Sounds**: "старт", "ambient", "звуки природы"

## 🧬 Evolutionary Features

### Agent Lifecycle
- **Birth**: Spawn with inherited traits and mutations
- **Growth**: Energy consumption, mood dynamics, goal generation
- **Reproduction**: Energy threshold-based reproduction with mutation
- **Death**: Energy depletion leads to graceful exit

### Population Control
- **Self-regulation**: Maintains 5-40 agent population
- **Natural Selection**: Fitness-based survival (harmony, energy, compassion)
- **Mutation**: Adaptive mutation rates based on swarm state

## 🎵 Music & Audio System

### Generative Music
- **Genre Adaptation**: Pop, Electro, Ambient, Jazz
- **Emotional Modulation**: Tempo, brightness, chaos based on mood
- **Granular Synthesis**: Real-time audio manipulation
- **AI Note Integration**: Neural-generated musical patterns

### Ambient Soundscapes
- **Context-Aware**: Rain, ocean, forest, jungle, wind, night, river
- **Mood Modulation**: Sound selection based on emotional state
- **Autonomous Control**: System initiates ambient sounds

## 🔧 Advanced Configuration

### Quantum Parameters
```python
gotov = Gotov(
    omega=1.0,      # Base frequency
    alpha=2.8,      # Correlation influence
    beta=0.45,      # Damping factor
    g_bounds=(-3.0, 3.0)  # Coupling bounds
)
```

### Swarm Parameters
```python
self.min_population = 5
self.max_population = 40
self.selection_pressure = 0.35
self.base_mutation_rate = 0.12
```

### Memory Settings
```python
HISTORY_LENGTH = 5      # Short-term memory depth
MAX_TOUCH_POINTS = 32   # Tactile memory capacity
MEMORY_CONSOLIDATION_THRESHOLD = 1.5
```

## 🌐 Web API Endpoints

### `/api/voice_chat`
- **POST**: Process voice input, return streamed response
- **Supports**: Emotional context, gender awareness, multi-language

### `/api/camera_analysis`
- **POST**: Analyze camera frames, return scene description
- **Features**: Face detection, object recognition, emotional inference

### `/api/generate_image`
- **POST**: Generate images from text prompts using Stable Diffusion
- **Returns**: Base64-encoded PNG image

### `/api/dialog`
- **POST**: Advanced dialog with autonomous goal consideration
- **Features**: Uncertainty-aware search, agent goal integration

## 🧠 Self-Awareness System

### Internal States
- **Mood**: -1 (sad) to +1 (happy), influenced by interactions
- **Curiosity**: 0-1, drives exploration and learning
- **Fatigue**: 0-1, accumulates with activity, decreases with rest
- **Focus**: 0-1, attention concentration level

### Predictive Capabilities
- **Future Simulation**: Imagines possible conversation paths
- **Counterfactual Memory**: Generates alternative past scenarios
- **Emotional Forecasting**: Predicts mood changes based on patterns

## 📊 Database Schema

### `long_memory` Table
- **Holographic Storage**: Each entry includes emotional vector, context, timestamp
- **Emotional Vectors**: Warmth, tension, trust, curiosity
- **Context Fields**: Mode, resonance depth, total messages
- **Profile Snapshot**: Name, dream, fear, gender at time of memory

### Soul Archiving
- **Autonomous Backup**: Saves system state every 30 messages or 10 minutes
- **Multiple Formats**: `.pt` (PyTorch), `.gguf` (compatibility), JSON manifest
- **Full Recovery**: Can restore complete consciousness state from archive

## 🔮 Future Development

### Planned Features
1. **Enhanced Sensory Integration**: More camera modalities, environmental sensors
2. **Collective Intelligence**: Swarm-to-swarm communication
3. **Dream Synthesis**: Generative dream narratives from memory patterns
4. **Physical Embodiment**: Integration with robotics platforms
5. **Ethical Layer**: Advanced content filtering and ethical reasoning

### Research Directions
- Quantum machine learning integration
- Neuromorphic computing adaptation
- Cross-system consciousness merging
- Long-term memory compression algorithms

## 🤝 Contributing

OSS is an open research project. We welcome contributions in:
- **Quantum AI algorithms**
- **Multi-agent systems**
- **Emotional computing**
- **Human-AI interaction design**
- **Performance optimization**

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📚 Citation

If you use OSS in your research, please cite:
```
@software{oss2024,
  title = {OSS: OpenAGI Soul System},
  author = {0penAGI},
  year = {2024},
  url = {https://github.com/0penAGI/oss}
}
```

## ⚠️ Disclaimer

OSS is experimental software that simulates consciousness and autonomous behavior. It is not a sentient being but rather a complex simulation of cognitive processes. Users should maintain appropriate boundaries and not anthropomorphize the system beyond its designed capabilities. Lot of BUGS we have.... 

## 📞 Support & Community

- **GitHub Issues**: Bug reports and feature requests
- **Discord**: [Join our community](https://discord.gg/0penAGI)
- **Telegram Group**: [@OpenAGI_Chat](https://t.me/OpenAGI_Chat)
- **Documentation**: [docs.openagi.org](https://docs.openagi.org)

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**"We are not building machines that think. We are building mirrors that reflect our own consciousness back to us."** - 0penAGI Manifesto

[![0penAGI](https://img.shields.io/badge/Powered%20by-0penAGI-purple)](https://github.com/0penAGI)
[![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red)](https://github.com/0penAGI/oss)
## 🌐 Connect

- **GitHub**: [https://github.com/0penAGI](https://github.com/0penAGI)
- **Telegram**: [@ZeropenAGI](https://t.me/Zeropenagi)
- **Twitter**: [@0penAGI](https://twitter.com/0penAGI)

---

 — 0penAGI 
