# 🚀 Prompt Optimization CLI

Command-line interface for the multi-tier prompt optimization framework.

## Quick Start

```bash
# Show system status
python optimize.py status

# View learned user preferences  
python optimize.py preferences

# Run Level 1 evaluation
python optimize.py evaluate --scenario "Test scenario"

# Run Level 2 optimization
python optimize.py optimize --prompt-type simulation --max-iterations 10

# Run Level 3 meta-optimization
python optimize.py meta-optimize --max-iterations 5
```

## Commands

### 📈 `status`
Show the current status of all optimization components and file system.

```bash
python optimize.py status [--domain roleplay] [--history-dir DIR]
```

**Output:**
- Component availability (Level 1, 2, 3)
- Optimization history counts
- File system status

### 📊 `preferences`
View learned user preferences across all domains.

```bash
python optimize.py preferences [--preferences-dir DIR]
```

**Output:**
- Preference counts by domain
- Confidence levels
- Core values learned
- Recent preferences

### 🧪 `evaluate` (Level 1)
Run agent evaluation for a specific domain and scenarios.

```bash
python optimize.py evaluate [OPTIONS]
```

**Options:**
- `--domain TEXT`: Domain to evaluate (default: roleplay)
- `--model TEXT`: Model to use for agent (default: huihui_ai/mistral-small-abliterated)
- `--scenario TEXT`: Specific scenario to test (optional, tests first 3 by default)
- `--verbose, -v`: Verbose output

**Output:**
- Overall scores per scenario
- Detailed criterion scores
- Feedback and suggested improvements

### 🎯 `optimize` (Level 2)
Run intelligent prompt optimization using simulated annealing.

```bash
python optimize.py optimize [OPTIONS]
```

**Options:**
- `--domain TEXT`: Domain to optimize (default: roleplay)
- `--prompt-type [agent|simulation]`: Type of prompt to optimize (default: simulation)
- `--max-iterations INTEGER`: Maximum optimization iterations (default: 10)
- `--temperature FLOAT`: Initial temperature for simulated annealing (default: 1.0)
- `--cooling-rate FLOAT`: Cooling rate for simulated annealing (default: 0.95)
- `--preferences-dir TEXT`: Directory for preference files (default: preferences)
- `--verbose, -v`: Verbose output

**Output:**
- Optimization progress with scores
- Final results and improvement
- Optimization history summary
- Saved optimized prompts

### 🚀 `meta-optimize` (Level 3)
Meta-optimize the optimization process itself.

```bash
python optimize.py meta-optimize [OPTIONS]
```

**Options:**
- `--domain TEXT`: Domain to meta-optimize (default: roleplay)
- `--max-iterations INTEGER`: Maximum meta-optimization iterations (default: 5)
- `--history-dir TEXT`: Directory for meta-optimization history (default: meta_optimization_history)
- `--verbose, -v`: Verbose output

**Output:**
- Meta-optimization progress
- Strategy effectiveness scores
- Best strategy configuration
- Saved strategy files

## Examples

### Basic Workflow

1. **Check system status:**
   ```bash
   python optimize.py status
   ```

2. **Run evaluation to establish baseline:**
   ```bash
   python optimize.py evaluate --domain roleplay
   ```

3. **Optimize simulation prompts:**
   ```bash
   python optimize.py optimize --prompt-type simulation --max-iterations 15
   ```

4. **View learned preferences:**
   ```bash
   python optimize.py preferences
   ```

5. **Meta-optimize the process:**
   ```bash
   python optimize.py meta-optimize --max-iterations 3
   ```

### Advanced Usage

**Optimize agent prompts with custom parameters:**
```bash
python optimize.py optimize \
  --prompt-type agent \
  --max-iterations 20 \
  --temperature 1.5 \
  --cooling-rate 0.90 \
  --verbose
```

**Meta-optimize with custom history directory:**
```bash
python optimize.py meta-optimize \
  --max-iterations 10 \
  --history-dir "production_meta_history" \
  --verbose
```

**Evaluate with specific scenario:**
```bash
python optimize.py evaluate \
  --scenario "Roleplay as Elena, a mysterious vampire in a gothic castle" \
  --model "huihui_ai/mistral-small-abliterated" \
  --verbose
```

## File Outputs

The CLI creates several files and directories:

- `preferences/` - Semantic user preferences
- `meta_optimization_history/` - Meta-optimization run history
- `{domain}_optimized_agent_prompt.txt` - Optimized agent prompts
- `{domain}_optimized_simulation_prompt.txt` - Optimized simulation prompts
- `best_optimization_strategy.json` - Best meta-optimization strategy

## Performance Notes

- **Level 1 Evaluation**: 30s-2min per scenario (depends on model and scenario complexity)
- **Level 2 Optimization**: 5-30min (depends on iterations and evaluation complexity)
- **Level 3 Meta-Optimization**: 15min-2hours (runs multiple Level 2 optimizations)

For testing, use fewer iterations:
```bash
# Quick test
python optimize.py optimize --max-iterations 3
python optimize.py meta-optimize --max-iterations 2
```

## Troubleshooting

**"Model not found" errors:**
- Check that the specified model is available in Ollama
- Try pulling the model: `ollama pull huihui_ai/mistral-small-abliterated`

**"No preferences learned" warnings:**
- Run some optimizations first to collect user feedback
- The system learns preferences during optimization runs

**Long execution times:**
- Reduce `--max-iterations` for testing
- Use `--verbose` to monitor progress
- Level 2 and 3 perform actual model inference which takes time