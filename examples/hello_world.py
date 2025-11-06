"""Hello World example for TNFR.

This is the absolute simplest TNFR example - perfect for beginners!

Run this file to see TNFR in action in just 3 lines of code.
"""

from tnfr.sdk import TNFRNetwork

def hello_world():
    """The simplest possible TNFR example."""
    print("="*70)
    print(" "*20 + "Hello, TNFR! 🎵")
    print("="*70)
    print()
    print("Creating your first TNFR network in 3 lines of code...")
    print()
    
    # Line 1: Create a network
    print("1. Creating network...")
    network = TNFRNetwork("hello_world")
    
    # Line 2: Add nodes and connect them
    print("2. Adding 10 nodes and connecting them...")
    network.add_nodes(10).connect_nodes(0.3, "random")
    
    # Line 3: Apply operators and measure
    print("3. Activating network and measuring coherence...")
    results = network.apply_sequence("basic_activation", repeat=3).measure()
    
    print()
    print("="*70)
    print(" "*20 + "Results")
    print("="*70)
    print()
    print(results.summary())
    print()
    print("="*70)
    print()
    print("🎉 Congratulations! You just ran your first TNFR simulation!")
    print()
    print("What just happened?")
    print("  • Created 10 resonating nodes (like musical notes)")
    print("  • Connected them randomly (30% probability)")
    print("  • Applied structural operators 3 times")
    print("  • Measured network coherence C(t)")
    print()
    print("Next steps:")
    print("  • Try: from tnfr.tutorials import hello_tnfr; hello_tnfr()")
    print("  • Read: docs/source/getting-started/QUICKSTART_NEW.md")
    print("  • Explore: examples/sdk_example.py")
    print()
    print("="*70)
    
    return results


if __name__ == "__main__":
    hello_world()
