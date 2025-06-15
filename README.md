# vLLM vs SGLang Performance Benchmark

A simple benchmarking tool to compare inference performance between vLLM and SGLang on the same hardware, model, and prompts. Intended for use on RunPod, but should work anywhere. The notebook will download all dependencies and install both packages, and run a selected prompt against both engines. vLLM and SGLang excel at different things and have different use cases, and both packages might perform differently with the same case across different GPU specs -- so this will tell you which is faster for both your use case and your environment using that real-world hardware.

## 🚀 Quick Start

Simply download the notebook and all .py files, throw it into a Pod, run the installer cell, and then modify your prompt and run that cell.

```python
exec(open('hello_world_script.py').read())
change_prompt("Write a story about a robot learning to paint")
run_comparison()
```

## Understanding the Results
The script provides detailed metrics for each engine:

```
vLLM: 17.06s, ~1024 tokens, 60.0 tok/s
SGLang: 15.49s, ~817 tokens, 52.7 tok/s
🥇 vLLM was 1.1x faster in tokens/second
```

-   **Time**: Total generation time in seconds
-   **Tokens**: Number of tokens generated
-   **tok/s**: Tokens per second (higher is better)
-   **Winner**: Determined by tokens/second, not raw time
