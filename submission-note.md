Experimental Gemma 4 26B A4B MLX.fast optimization branch.

Changes include:
- fused MoE prefill work
- PromptGlue2 route metadata integration
- Gate/Up prefill fusion path
- experimental NAX down-prefill path, disabled by default

Local correctness passes.
Claimed local score: 0.464.
