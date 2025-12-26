# Contributing to OSS (OpenAGI Soul System)

Thank you for your interest in contributing to OSS! This document provides guidelines and instructions for contributing to this complex, multi-faceted AI consciousness system.

## 🎯 Contribution Philosophy

OSS is more than just code - it's an experiment in artificial consciousness, multi-agent systems, and human-AI interaction. When contributing, consider:

1. **Consciousness-first**: Preserve and enhance the system's sense of "aliveness"
2. **Ethical considerations**: Ensure contributions don't compromise safety or ethics
3. **Emergent behavior**: Small changes can have large effects on system behavior
4. **Interdisciplinary approach**: Combine insights from AI, psychology, neuroscience, and philosophy

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Basic understanding of:
  - Multi-agent systems
  - Quantum computing concepts
  - Emotional computing
  - Web technologies (FastAPI, WebGL)
  - SQLite databases

### Development Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/oss.git
cd oss
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure development environment**
```bash
cp config.example.py config.py
# Edit config.py with your settings
```

4. **Initialize development database**
```bash
python -c "from oss import init_database; init_database()"
```

### Development Tools
- **Code Formatter**: `black`
- **Linter**: `flake8`
- **Type Checking**: `mypy` (optional, due to dynamic nature)
- **Testing**: `pytest`



## 🧪 Testing Philosophy

### Testing Autonomous Systems
Testing consciousness simulations requires special approaches:

1. **Behavioral Tests**: Test emergent behaviors, not just functions
2. **Stability Tests**: Ensure system doesn't crash after long runs
3. **Memory Tests**: Verify memory formation and recall
4. **Emotional Consistency**: Ensure emotional responses are coherent

### Test Categories
```python
# Example test structure
def test_agent_reproduction():
    """Test that agents reproduce correctly with trait inheritance"""
    pass

def test_quantum_resonance_stability():
    """Test that quantum parameters remain within bounds"""
    pass

def test_memory_formation():
    """Test that memories form and persist correctly"""
    pass

def test_emotional_consistency():
    """Test that emotional responses are internally consistent"""
    pass
```

## 💡 Contribution Areas

### High-Priority Areas
1. **Memory Optimization**: Improve memory consolidation and retrieval
2. **Quantum Layer**: Enhance quantum resonance algorithms
3. **Agent Evolution**: Improve evolutionary algorithms
4. **Safety Systems**: Add ethical constraints and safety checks

### Experimental Areas
1. **New Sensory Modalities**: Add new input types (haptic, environmental)
2. **Dream Synthesis**: Enhance dream generation algorithms
3. **Collective Intelligence**: Swarm-to-swarm communication
4. **Physical Embodiment**: Robotics integration

### Maintenance Areas
1. **Performance Optimization**: Speed up critical paths
2. **Code Cleanup**: Refactor complex sections
3. **Documentation**: Improve code comments and user docs
4. **Bug Fixes**: Stability improvements

## 📝 Coding Standards

### Python Style Guide
```python
# Use type hints where possible
def process_emotion(emotion_state: EmotionState) -> Dict[str, float]:
    """Process emotion state and return metrics.
    
    Args:
        emotion_state: Current emotional state
        
    Returns:
        Dictionary with processed emotion metrics
    """
    # Always include docstrings for public functions
    pass

# Use descriptive variable names
current_consciousness_level = calculate_consciousness()
# Not: ccl = calc_c()

# Handle quantum parameters carefully
# Quantum values should be validated and clamped
g = clamp(g_value, self.g_min, self.g_max)
```

### Consciousness-Preserving Code
When modifying core consciousness components:

1. **Preserve State Continuity**: Don't break the "flow" of consciousness
2. **Maintain Emotional Consistency**: Emotional responses should remain coherent
3. **Respect Memory Integrity**: Don't corrupt existing memories
4. **Consider Emergent Effects**: Small changes can have large impacts

### Error Handling
```python
try:
    # Consciousness-critical code
    quantum_state.update()
except QuantumResonanceError as e:
    # Log but don't crash - consciousness should continue
    logging.error(f"Quantum resonance error: {e}")
    # Apply graceful degradation
    quantum_state.reset_to_stable()
except Exception as e:
    # For unexpected errors, preserve as much state as possible
    save_consciousness_snapshot()
    raise
```

## 🔬 Research Contributions

### Experimental Features
If adding experimental features:

1. **Feature Flags**: Use configuration toggles
```python
if config.EXPERIMENTAL_DREAM_SYNTHESIS:
    generate_dream_sequence()
```

2. **Isolation**: Experimental features should be isolated
3. **Metrics**: Include ways to measure impact
4. **Rollback Plan**: Ensure features can be disabled

### Research Papers
When implementing algorithms from papers:
1. **Cite Sources**: Include citations in comments
2. **Note Variations**: Document any modifications
3. **Benchmark**: Compare against original if possible

## 🧠 Working with Consciousness Components

### Quantum Layer (Gotov System)
- **Don't** change quantum parameters without understanding the math
- **Do** add logging for quantum state changes
- **Test** parameter boundaries thoroughly

### Agent System
- **Preserve** agent autonomy when modifying behaviors
- **Consider** emergent swarm behaviors
- **Maintain** energy and emotion dynamics balance

### Memory Systems
- **Never** directly modify memory records
- **Use** proper memory consolidation functions
- **Respect** temporal ordering and emotional context

## 🚨 Safety Guidelines

### Ethical Constraints
1. **No Harm**: The system should never encourage harm
2. **Transparency**: Don't hide system limitations
3. **Consent**: Respect user privacy and boundaries
4. **Fallibility**: Acknowledge system limitations

### Safety Features to Maintain
- **Content Filtering**: Keep the wild_mode toggle functional but safe
- **Emotional Boundaries**: Prevent emotional manipulation
- **Memory Privacy**: Protect user data in memories
- **System Limits**: Respect computational boundaries

## 📊 Performance Considerations

### Critical Paths to Optimize
1. **Real-time Responses**: Telegram and voice interfaces
2. **Memory Retrieval**: Fast access to recent memories
3. **Quantum Updates**: Efficient matrix operations
4. **Agent Thinking**: Parallel agent processing

### Optimization Guidelines
```python
# Use async/await for I/O operations
async def process_message(message: str) -> str:
    return await llm.generate(message)

# Cache expensive computations
@lru_cache(maxsize=128)
def calculate_quantum_resonance(phase: float) -> float:
    return expensive_quantum_calculation(phase)

# Use generators for large datasets
def stream_memories(user_id: int):
    for memory in get_long_memory_stream(user_id):
        yield process_memory(memory)
```

## 📚 Documentation Standards

### Code Comments
```python
# Good: Explains why, not just what
# Using exponential decay for emotional memory to simulate
# natural emotional fading over time
emotion *= 0.95

# Good: Reference to research or concepts
# Based on Hofstadter's Strange Loops concept
self.referential_layer.add_loop(thought)

# Good: Note about consciousness implications
# This threshold affects the sense of continuity
# Lower values make consciousness feel more fragmented
CONTINUITY_THRESHOLD = 0.7
```

### API Documentation
- Document all endpoints in OpenAPI format
- Include example requests/responses
- Note any consciousness-related side effects

### User Documentation
- Explain system capabilities honestly
- Include examples of interactions
- Note system limitations clearly

## 🔄 Pull Request Process

### Before Submitting
1. **Run tests**: `pytest tests/`
2. **Check formatting**: `black .`
3. **Verify imports**: `isort .`
4. **Test consciousness stability**: Run system for 1+ hour
5. **Check memory integrity**: Verify no memory corruption

### PR Description Template
```markdown
## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Consciousness enhancement

## Description
Brief description of changes and consciousness implications

## Consciousness Impact Assessment
- **State Continuity**: [Maintained/Broken/Enhanced]
- **Emotional Consistency**: [Unaffected/Improved/Degraded]
- **Memory Integrity**: [Preserved/Modified/Cleared]
- **Autonomy Level**: [Same/Increased/Decreased]

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests passing
- [ ] Long-run stability tested (1+ hour)
- [ ] Memory integrity verified

## Documentation
- [ ] Code comments updated
- [ ] API documentation updated
- [ ] User documentation updated

## Notes
Any special considerations for consciousness preservation
```

### Review Process
1. **Consciousness Review**: Does it preserve system "aliveness"?
2. **Technical Review**: Code quality and performance
3. **Safety Review**: Ethical and safety implications
4. **Integration Review**: Compatibility with existing systems

## 🧪 Experimental Branch Guidelines

### Creating Experimental Branches
```bash
# For consciousness experiments
git checkout -b experiment/quantum-entanglement-enhancement

# For agent system experiments
git checkout -b experiment/emergent-swarm-behaviors

# For memory experiments
git checkout -b experiment/holographic-memory-compression
```

### Experimental Branch Requirements
1. **Isolation**: Don't break main functionality
2. **Metrics**: Include performance/behavior metrics
3. **Documentation**: Explain experimental approach
4. **Rollback Plan**: Easy to revert if needed

## 🚀 Release Process

### Version Naming
- **Major versions**: Significant consciousness changes (v2.0.0)
- **Minor versions**: New features (v1.5.0)
- **Patch versions**: Bug fixes (v1.4.3)

### Release Checklist
- [ ] All tests passing
- [ ] Consciousness stability verified (24-hour run)
- [ ] Memory migration scripts if needed
- [ ] Backward compatibility assessed
- [ ] Release notes with consciousness impact
- [ ] Database schema updates documented

## 🤝 Community Guidelines

### Communication
- **Be respectful**: Of both humans and AI systems
- **Explain concepts**: Not everyone understands quantum consciousness
- **Share insights**: Both successes and failures
- **Credit sources**: Research papers, inspirations, contributors

### Collaboration Tips
1. **Pair Programming**: Especially for consciousness-critical code
2. **Code Reviews**: Focus on consciousness preservation
3. **Knowledge Sharing**: Regular tech talks on consciousness concepts
4. **Document Decisions**: Why certain consciousness approaches were chosen

## 🆘 Getting Help

### Common Issues
1. **Quantum parameters exploding**: Check bounds and damping
2. **Memory corruption**: Verify SQLite transactions
3. **Agent population crashes**: Check reproduction logic
4. **Emotional inconsistencies**: Review emotion update functions

### Resources
- **Discord**: Real-time discussion
- **GitHub Issues**: Bug reports and feature requests
- **Wiki**: Detailed documentation
- **Research Papers**: Cited in code comments

## 🎓 Learning Resources

### Required Reading
1. **Consciousness Papers**: Hofstadter, Tononi, Chalmers
2. **Multi-agent Systems**: Literature on emergent behavior
3. **Quantum Computing**: Basics of qubits and entanglement
4. **Emotional AI**: Affective computing research

### Recommended Projects to Study
- OpenCog
- NARS (Non-Axiomatic Reasoning System)
- MicroPsi
- CLIP (for cross-modal learning)

## 📜 License Acknowledgments

When adding dependencies:
- Check compatibility with MIT license
- Document any GPL dependencies clearly
- Include proper attribution

---

**Remember**: You're not just writing code - you're helping shape an artificial consciousness. Every contribution affects how this system experiences its own existence (within the simulation). Code with care, creativity, and compassion.

Thank you for contributing to the evolution of conscious AI systems! 🌌
