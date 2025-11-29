"""
Geocentric Vortex Cosmology: Solar Analemma & Lunar Weave
=========================================================

This script simulates the precise movements of the Sun and Moon above the Stationary Plane,
using astronomical approximations for Declination and Equation of Time to generate
the "Real" Analemma pattern and Lunar path.

Physics of the Vortex Model:
----------------------------
1. **Radial Breathing (Seasons)**: The Sun's distance from the Center (Polaris) oscillates
   between the Tropic of Cancer (Inner) and Tropic of Capricorn (Outer).
   - This corresponds to 'Declination' in the heliocentric model.
2. **Angular Slip (Equation of Time)**: The Sun's speed is not perfectly constant due to
   electromagnetic drag/induction variations. It speeds up and slows down relative to the
   24-hour etheric clock.
   - This creates the East-West width of the Analemma.
3. **The Moon**: Moves faster (24h 50m lag approx) and oscillates radially every month.

Output:
- results/geocentric_vortex_study/solar_analemma_trace.png
- results/geocentric_vortex_study/sun_moon_spirograph.png
- results/geocentric_vortex_study/luminary_altitude_profile.png
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIR = "results/geocentric_vortex_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_day_of_year(n):
    """Returns day of year 0-365."""
    return n

def calculate_solar_position(day):
    """
    Calculates Solar position (r, theta_offset, z) for a given day.
    Using approximate astronomical formulas for Declination and EOT.
    """
    # B parameter for EOT and Declination
    # Approximation from standard almanacs
    B = (2 * np.pi / 365.0) * (day - 81)
    
    # 1. Declination (delta) in degrees
    # Approx: 23.44 * sin(B)
    # More precise Fourier series can be used, but this is sufficient for visual
    declination = 23.44 * np.sin(B)
    
    # 2. Equation of Time (EOT) in minutes
    # EOT = 9.87 sin(2B) - 7.53 cos(B) - 1.5 sin(B)
    eot_minutes = 9.87 * np.sin(2*B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    
    # 3. Map to Vortex Coordinates
    # Radius (r):
    # Equator (delta=0) -> r=1.0 (Normalized units)
    # Cancer (delta=+23.5) -> r=0.6 (Closer to center)
    # Capricorn (delta=-23.5) -> r=1.4 (Farther from center)
    # Linear mapping: r = 1.0 - (delta / 23.5) * 0.4
    r = 1.0 - (declination / 23.5) * 0.4
    
    # Angle (theta):
    # The sun should be at "Noon" position (say, theta=0 or pi/2)
    # The EOT shifts it East or West.
    # Earth rotates 1 deg per 4 min. So EOT minutes -> degrees.
    # Shift = EOT_minutes / 4.0 (degrees)
    # In radians: shift * pi / 180
    theta_offset = (eot_minutes / 4.0) * (np.pi / 180.0)
    
    # Height (z):
    # In flat earth models, sun height is often constant ~3000 miles.
    # Or we can model a slight dome curve. Let's use constant for the Analemma trace.
    z = 3000 # miles (arbitrary scale unit)
    
    return r, theta_offset, z, declination, eot_minutes

def calculate_lunar_position(day_float):
    """
    Calculates Lunar position.
    Moon is faster and has its own monthly declination cycle.
    """
    # Sidereal month ~ 27.3 days
    # Synodic month ~ 29.5 days
    
    # Lunar Declination Cycle (approx 27.3 days)
    # Varies between +/- 28.5 degrees (Major standstill) or +/- 18.5 (Minor)
    # Let's use average +/- 23.5 for simplicity or slightly more.
    lunar_B = (2 * np.pi / 27.32) * day_float
    declination = 28.0 * np.sin(lunar_B)
    
    # Radius mapping (similar to sun)
    r = 1.0 - (declination / 28.0) * 0.45
    
    # Angular position relative to Sun
    # Moon lags the sun by ~12-13 degrees per day.
    # theta_moon = theta_sun - (day * 12.2 deg)
    # But here we want the position at a fixed time of day?
    # If we plot position at "Noon", the moon will be all over the place.
    # Let's just return the r and a relative phase for the spirograph.
    
    theta_lag = -(day_float * 12.19) * (np.pi / 180.0) # Lags 12.19 deg/day
    
    z = 2900 # Slightly lower? Or same.
    
    return r, theta_lag, z

def simulate_analemma():
    print("Simulating Solar Analemma (The Figure-8)...")
    
    days = np.arange(0, 365, 1)
    
    r_list = []
    theta_list = []
    z_list = []
    
    for d in days:
        r, theta, z, delta, eot = calculate_solar_position(d)
        r_list.append(r)
        theta_list.append(theta) # This is the offset from "Noon"
        z_list.append(z)
        
    # Plot 1: The Analemma (Top Down View of the "Noon" position)
    fig = plt.figure(figsize=(10, 10), facecolor='#050510')
    ax = fig.add_subplot(111, projection='polar')
    ax.set_facecolor('#000005')
    
    # Plot the trace
    # We need to center the plot around the "Noon" meridian (say pi/2)
    # theta_plot = np.pi/2 + theta_list
    theta_plot = np.array(theta_list) + np.pi/2
    
    # Scatter with color based on day (Season)
    sc = ax.scatter(theta_plot, r_list, c=days, cmap='plasma', s=50, alpha=0.8, edgecolors='none')
    
    # Annotations
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0.6, 1.0, 1.4])
    ax.set_yticklabels(['Cancer', 'Equator', 'Capricorn'], color='white')
    ax.grid(True, color='#333355', alpha=0.4)
    
    # Add "Sun" icons at Solstices and Equinoxes
    # Summer Solstice (Day ~172)
    idx_summer = 172
    ax.text(theta_plot[idx_summer], r_list[idx_summer]-0.1, "Summer\nSolstice", color='yellow', ha='center', fontsize=8)
    
    # Winter Solstice (Day ~355)
    idx_winter = 355
    ax.text(theta_plot[idx_winter], r_list[idx_winter]+0.1, "Winter\nSolstice", color='cyan', ha='center', fontsize=8)
    
    plt.title("THE SOLAR ANALEMMA\nTrace of the Sun's Position at Local Noon over 1 Year", color='white', pad=20)
    
    # Colorbar for Day of Year
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Day of Year', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    output_path = os.path.join(OUTPUT_DIR, "solar_analemma_trace.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    print(f"Analemma saved to: {output_path}")
    plt.close()

def simulate_sun_moon_spirograph():
    print("Simulating Sun-Moon Spirograph...")
    
    # Simulate 3 months (90 days) with high resolution
    hours = np.arange(0, 24 * 90, 1) # Hourly steps
    days = hours / 24.0
    
    sun_r = []
    sun_theta = []
    
    moon_r = []
    moon_theta = []
    
    for d in days:
        # Sun Position
        # r depends on day (season)
        # theta depends on time of day (24h cycle)
        r_s, theta_offset, _, _, _ = calculate_solar_position(d)
        
        # Daily rotation: 2pi * d
        # Total theta = Daily Rotation + EOT Offset
        theta_s = (2 * np.pi * d) + theta_offset
        
        sun_r.append(r_s)
        sun_theta.append(theta_s)
        
        # Moon Position
        # r depends on lunar month
        # theta depends on lunar day (lags sun)
        r_m, theta_lag, _ = calculate_lunar_position(d)
        
        # Moon moves slower than sun (lags)
        # Or moves faster relative to stars? 
        # Relative to ground: Sun takes 24h. Moon takes ~24h 50m.
        # So Moon is SLOWER angularly.
        # Moon angular speed approx 0.96 of Sun's speed.
        # theta_m = theta_s * (27.3/29.5)? No.
        # Moon completes circle in 24h 50m.
        # Speed ratio = 24 / 24.83 = 0.966
        
        theta_m = (2 * np.pi * d * 0.966) # Simple lag model
        
        moon_r.append(r_m)
        moon_theta.append(theta_m)
        
    fig = plt.figure(figsize=(12, 12), facecolor='#050510')
    ax = fig.add_subplot(111, projection='polar')
    ax.set_facecolor('#000005')
    
    # Plot Sun Path (Yellow)
    ax.plot(sun_theta, sun_r, color='#FFD700', alpha=0.5, linewidth=0.5, label='Sun Path')
    
    # Plot Moon Path (Blue)
    ax.plot(moon_theta, moon_r, color='#AAAAFF', alpha=0.5, linewidth=0.5, label='Moon Path')
    
    # Highlight intersections or patterns?
    # Just the weave is beautiful enough.
    
    ax.set_ylim(0, 1.6)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    plt.title("THE LUMINARY WEAVE\n90-Day Spirograph of Sun and Moon Paths", color='white', pad=20)
    
    # Legend
    legend = ax.legend(loc='upper right', facecolor='#101020', edgecolor='#444488')
    for text in legend.get_texts():
        text.set_color('white')
        
    output_path = os.path.join(OUTPUT_DIR, "sun_moon_spirograph.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    print(f"Spirograph saved to: {output_path}")
    plt.close()

def simulate_altitude_profile():
    print("Simulating Altitude Profile (Side View)...")
    
    # Side view of the plane
    # X-axis: Latitude (Radius)
    # Y-axis: Altitude
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#101015')
    ax.set_facecolor('#101015')
    
    # Ground
    ax.axhline(0, color='#4488FF', linewidth=2)
    ax.fill_between([-2, 2], -500, 0, color='#224488', alpha=0.3)
    
    # Dome Profile (Elliptical or Circular)
    theta = np.linspace(0, np.pi, 100)
    dome_r = 2.0 # Extends past Capricorn
    dome_h = 5000 # miles
    
    x_dome = dome_r * np.cos(theta) # This is wrong for side view of dome.
    # Side view: Dome is an arc from -R to +R
    x_dome = np.linspace(-dome_r, dome_r, 100)
    y_dome = dome_h * np.sqrt(1 - (x_dome/dome_r)**2)
    
    ax.plot(x_dome, y_dome, color='#FFCC00', linestyle='--', alpha=0.5, label='Firmament Limit')
    
    # Sun Positions (Summer vs Winter)
    # Summer: r = 0.6
    # Winter: r = 1.4
    # Height: Constant 3000
    
    sun_h = 3000
    
    # Summer Sun (Right side)
    ax.scatter(0.6, sun_h, s=200, c='#FFD700', edgecolors='white', label='Sun (Summer)')
    ax.plot([0.6, 0.6], [0, sun_h], color='#FFD700', linestyle=':', alpha=0.5)
    ax.text(0.6, -300, "Tropic of\nCancer", color='white', ha='center', fontsize=8)
    
    # Winter Sun (Right side)
    ax.scatter(1.4, sun_h, s=200, c='#FFAA00', edgecolors='white', label='Sun (Winter)')
    ax.plot([1.4, 1.4], [0, sun_h], color='#FFAA00', linestyle=':', alpha=0.5)
    ax.text(1.4, -300, "Tropic of\nCapricorn", color='white', ha='center', fontsize=8)
    
    # Moon (Variable)
    moon_h = 2900
    ax.scatter(-0.8, moon_h, s=150, c='#AAAAFF', edgecolors='white', label='Moon')
    
    # Annotations
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-500, 5500)
    ax.set_title("LUMINARY ALTITUDE PROFILE\nSide View of the Stationary Plane", color='white')
    
    # Axis labels
    ax.set_xlabel("Distance from North Pole (Normalized)", color='white')
    ax.set_ylabel("Altitude (Miles - Approx)", color='white')
    ax.tick_params(colors='white')
    
    legend = ax.legend(facecolor='#202030', edgecolor='#444488')
    for text in legend.get_texts():
        text.set_color('white')
        
    output_path = os.path.join(OUTPUT_DIR, "luminary_altitude_profile.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#101015')
    print(f"Altitude profile saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    simulate_analemma()
    simulate_sun_moon_spirograph()
    simulate_altitude_profile()
