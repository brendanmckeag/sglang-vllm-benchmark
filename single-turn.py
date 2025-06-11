"""
LLM Speed Benchmark Script - vLLM vs sglang
Block 1: Prompt Configuration for Performance Testing
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BenchmarkPrompt:
    """Configuration for a speed benchmark prompt."""
    name: str
    prompt: str
    expected_tokens: int  # Approximate expected output length for consistent testing
    description: Optional[str] = None


class BenchmarkConfig:
    """Container for benchmark prompts focused on performance testing."""
    
    def __init__(self):
        self.prompts: List[BenchmarkPrompt] = []
    
    def add_prompt(self, 
                   name: str, 
                   prompt: str, 
                   expected_tokens: int,
                   description: Optional[str] = None):
        """Add a benchmark prompt to the configuration."""
        benchmark_prompt = BenchmarkPrompt(
            name=name,
            prompt=prompt,
            expected_tokens=expected_tokens,
            description=description
        )
        self.prompts.append(benchmark_prompt)
        return benchmark_prompt
    
    def get_prompt_by_name(self, name: str) -> Optional[BenchmarkPrompt]:
        """Retrieve a prompt by its name."""
        for prompt in self.prompts:
            if prompt.name == name:
                return prompt
        return None
    
    def list_prompts(self) -> List[str]:
        """Get a list of all prompt names."""
        return [prompt.name for prompt in self.prompts]


def create_benchmark_prompts() -> BenchmarkConfig:
    """Create prompts specifically designed for speed benchmarking."""
    config = BenchmarkConfig()
    
    # Short response benchmark (~50 tokens)
    config.add_prompt(
        name="short_response",
        prompt="Explain what machine learning is in 2-3 sentences.",
        expected_tokens=50,
        description="Short response benchmark (~50 tokens)"
    )
    
    # Medium response benchmark (~200 tokens)
    config.add_prompt(
        name="medium_response",
        prompt="Write a detailed explanation of how neural networks work, including key concepts like layers, weights, and backpropagation.",
        expected_tokens=200,
        description="Medium response benchmark (~200 tokens)"
    )
    
    # Long response benchmark (~500 tokens)
    config.add_prompt(
        name="long_response",
        prompt="Provide a comprehensive overview of the history of artificial intelligence, covering major milestones from the 1950s to present day, including key researchers, breakthroughs, and applications.",
        expected_tokens=500,
        description="Long response benchmark (~500 tokens)"
    )
    
    # Very long response benchmark (~1000 tokens)
    config.add_prompt(
        name="very_long_response",
        prompt="Write a detailed technical analysis of transformer architecture in deep learning, explaining attention mechanisms, multi-head attention, positional encoding, layer normalization, and how these components work together. Include examples of popular transformer models and their applications.",
        expected_tokens=1000,
        description="Very long response benchmark (~1000 tokens)"
    )
    
    return config


if __name__ == "__main__":
    # Demonstrate the benchmark configuration
    config = create_benchmark_prompts()
    
    print("Speed Benchmark Prompts:")
    print("=" * 50)
    
    for prompt in config.prompts:
        print(f"Name: {prompt.name}")
        print(f"Expected tokens: {prompt.expected_tokens}")
        print(f"Description: {prompt.description}")
        print(f"Prompt: {prompt.prompt}")
        print("-" * 50)
