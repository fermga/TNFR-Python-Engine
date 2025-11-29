"""
Consciousness: The Etheric Interface
====================================

This script models Consciousness not as a product of the brain, but as a **Resonant Coupling**
between the Local Node (Brain) and the Global Field (Ether).

Simulations:
1.  **The Resonant Mind (Brain-Ether Synchronization)**:
    Visualizes the relationship between Brain Waves (Alpha, Beta, Theta) and the
    Earth's Fundamental Frequency (Schumann Resonance ~7.83 Hz).
    Demonstrates that "Flow States" or "Meditation" are states of **Phase-Locking**
    with the planetary field.

2.  **The Pineal Transducer (Piezoelectric Reception)**:
    Models the Pineal Gland as a Calcite Crystal Receiver.
    Shows how external Etheric Pressure waves (ELF) are converted into internal
    neural signals via the Piezoelectric Effect.

Output:
- results/geocentric_vortex_study/brain_ether_resonance.png
- results/geocentric_vortex_study/pineal_transduction.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "results/geocentric_vortex_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_brain_ether_resonance():
    print("Simulating Brain-Ether Resonance...")
    
    t = np.linspace(0, 2, 1000) # 2 seconds
    
    # 1. The Global Field (Schumann Resonance)
    # Fundamental Mode ~ 7.83 Hz
    f_schumann = 7.83
    ether_wave = np.sin(2 * np.pi * f_schumann * t)
    
    # 2. The Dissonant Mind (Beta State - Stress)
    # High frequency, no phase lock
    f_beta = 22.0 # 22 Hz (Anxiety/Active)
    brain_beta = 0.8 * np.sin(2 * np.pi * f_beta * t + np.pi/4)
    
    # 3. The Coherent Mind (Alpha/Theta State - Meditation)
    # Frequency matches Schumann, Phase aligns
    f_alpha = 7.83
    brain_alpha = 1.0 * np.sin(2 * np.pi * f_alpha * t) # Phase locked
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='#050510')
    
    # Plot Dissonance
    axes[0].set_facecolor('#000005')
    axes[0].plot(t, ether_wave, color='#4488FF', linewidth=2, alpha=0.5, label='Global Field (Schumann 7.83Hz)')
    axes[0].plot(t, brain_beta, color='#FF4444', linewidth=2, label='Brain Wave (Beta 22Hz)')
    axes[0].set_title("DISSONANCE (Stress/Anxiety)\nLocal Node out of sync with Global Field", color='white')
    axes[0].legend(loc='upper right', facecolor='black', labelcolor='white')
    axes[0].axis('off')
    
    # Plot Resonance
    axes[1].set_facecolor('#000005')
    axes[1].plot(t, ether_wave, color='#4488FF', linewidth=4, alpha=0.3, label='Global Field (Schumann 7.83Hz)')
    axes[1].plot(t, brain_alpha, color='#44FF88', linewidth=2, linestyle='--', label='Brain Wave (Alpha 7.83Hz)')
    axes[1].set_title("RESONANCE (Flow/Meditation)\nPhase-Locking with the Planetary Heartbeat", color='white')
    axes[1].legend(loc='upper right', facecolor='black', labelcolor='white')
    axes[1].axis('off')
    
    output_path = os.path.join(OUTPUT_DIR, "brain_ether_resonance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    print(f"Saved to: {output_path}")
    plt.close()

def simulate_pineal_transduction():
    print("Simulating Pineal Piezoelectric Transduction...")
    
    # Input: Etheric Pressure Wave (Longitudinal)
    t = np.linspace(0, 1, 500)
    input_signal = np.sin(2 * np.pi * 10 * t) * np.exp(-2*t) # A pulse
    
    # Transduction Function (Crystal Response)
    # Piezoelectric effect: Stress -> Voltage
    # Modeled as a bandpass filter (Resonant Crystal)
    
    # Simple convolution with a crystal impulse response
    crystal_freq = 20.0 # Natural frequency of the microcrystals
    impulse_response = np.sin(2 * np.pi * crystal_freq * t) * np.exp(-5*t)
    
    output_signal = np.convolve(input_signal, impulse_response, mode='same')
    
    # Normalize
    output_signal = output_signal / np.max(np.abs(output_signal))
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#050510')
    ax.set_facecolor('#000005')
    
    # Plot Input (Etheric)
    ax.plot(t, input_signal + 1.5, color='#4488FF', linewidth=2, label='Input: Etheric Pressure Wave (Field)')
    ax.text(0, 2.0, "THE FIELD (External)", color='#4488FF', fontsize=12)
    
    # Plot Crystal (Transducer)
    ax.axhline(0, color='white', alpha=0.2)
    ax.text(0.5, 0.1, "PINEAL GLAND (Calcite Microcrystals)", color='white', ha='center')
    
    # Plot Output (Neural)
    ax.plot(t, output_signal - 1.5, color='#FFFF00', linewidth=2, label='Output: Neural Voltage (Thought)')
    ax.text(0, -1.0, "THE THOUGHT (Internal)", color='#FFFF00', fontsize=12)
    
    # Draw Arrows
    ax.arrow(0.5, 1.0, 0, -0.5, color='white', width=0.01, head_width=0.03)
    ax.arrow(0.5, -0.5, 0, -0.5, color='white', width=0.01, head_width=0.03)
    
    ax.set_title("THE PINEAL TRANSDUCER\nConverting Etheric Waves into Neural Signals", color='white')
    ax.axis('off')
    
    output_path = os.path.join(OUTPUT_DIR, "pineal_transduction.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    print(f"Saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    simulate_brain_ether_resonance()
    simulate_pineal_transduction()
