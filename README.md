# beamformRecorder

## Overview

A collection of Pythonic implementation of beamforming algorithm scripts, including LCMV (Frost) and MVDR beamforming scripts and interactive script to track the source, as well as create null beam to reject unwanted azimuth range.

The scripts were created for the miniDSP UMA-16 microphone array, but should be easily adapted for other microphones that exposes multiple input channels.

## How It Works
- Captures all 16 channels from your UMA 16 microphone array
- Applies time delays to each channel based on the target steering direction
- Sums all delayed channels to create a directional beamformed output
- Saves the result as mono (1-channel) audio

## Array Configuration
- **Array Type**: 4×4 square grid
- **Microphone Spacing**: 0.02 meters (2 cm) by default
- **Total Microphones**: 16 channels

# Determining Azimuth and Elevation Angles for UMA-16 Array

## Understanding the Angles

### Azimuth (Horizontal Direction)
The compass direction in the horizontal plane:
- **0°** = Front of the array (direction MIC8, MIC7, MIC10, MIC9 face)
- **90°** = Right side (direction towards MIC13, MIC15, MIC11, MIC9)
- **-90° (or 270°)** = Left side (direction towards MIC6, MIC4, MIC8, MIC2)
- **180°** = Back (opposite of front)

### Elevation (Vertical Angle)
The angle above or below the horizontal plane:
- **0°** = Sound at same height as array (horizontal)
- **90°** = Sound directly above the array
- **-90°** = Sound directly below the array
- **45°** = Sound at 45° angle upward from horizontal

## Methods to Determine Angles

### 1. **Visual Estimation (Simplest)**
Standing at the array looking towards the sound source:
- **Azimuth**: Imagine a compass on the array. What direction is the source?
- **Elevation**: What angle above/below horizontal? (0°=level, 45°=halfway up, 90°=straight up)

### 2. **Using Your Hand**
- Point your hand toward the sound source
- Your arm points to the sound direction
- **Azimuth**: Based on compass direction relative to array
- **Elevation**: Estimated from the angle of your arm

### 3. **Reference Points on Array (Most Accurate)**
Looking from above your array (top-down view):

```
         Front (0°)
              ↑
       MIC8  MIC7  MIC10  MIC9
       
Left ← MIC6  MIC5  MIC12  MIC11 → Right
(-90°) MIC4  MIC3  MIC14  MIC13  (90°)
       
       MIC2  MIC1  MIC16  MIC15
              ↓
         Back (180°)
```

- Identify which microphones face the sound source
- Determine the azimuth from that direction

## Physical Array Setup

Based on your UMA-16 diagram:
- **Array dimensions**: 132mm × 132mm
- **Microphone spacing**: 42mm between adjacent mics
- **Facing direction**: Front edge = MIC8, MIC7, MIC10, MIC9
- **Center**: Approximately MIC5/MIC12 area

## Common Scenarios with Your UMA-16

| Scenario | Azimuth | Elevation | Usage |
|----------|---------|-----------|-------|
| Speaker directly in front | 0° | 0° | Default presentation |
| Speaker to the right | 45° | 0° | Person on your right |
| Speaker to the left | -45° | 0° | Person on your left |
| Speaker above (like ceiling) | 0° | 90° | Overhead sound source |
| Speaker above-right (45°) | 45° | 45° | Elevated source at angle |
| Speaker in front but ground level | 0° | -20° | Low-mounted microphone |
| Omnidirectional (reject noise) | — | — | Set multiple angles and average |

## Tips for Best Results

1. **Measure if critical**: Use a protractor or phone app to measure angles if high precision is needed
2. **Account for array orientation**: Mark which edge is "front" on your physical array
3. **Test incrementally**: Start with 0°/0° and adjust by 15-30° intervals
4. **Listen to the recording**: Audio quality will improve as you get closer to the actual source direction
5. **Document your setup**: Once you find good angles, note them for future recordings

## Example: Recording a Speaker

If you want to record someone speaking in front of you:
1. Position your UMA-16 array facing toward them
2. If they move side-to-side, adjust azimuth (±30-45°)
3. If they move up/down vertically, adjust elevation (±20-30°)
4. Listen to the recorded audio to fine-tune

Once you identify the best angles for your typical recording setup, you can use them consistently!

## Beamforming algorithms

Based on current real-time tests in this repo:

1. **Frost** (best overall)
	- Most stable in real-time
	- Fewest audible gaps/cuts
	- Recommended default method

# python bf_Frost_adaptive_3d.py --output recorder_output\records\audio_frost_adaptive_3d.wav --track-duration 5.0 --gain 42 --mu 0.05  

2. **MVDR**
	- Good quality after optimizations
	- Occasional brief overrun spikes during tracking updates

# python bf_MVDR_adaptive_3d.py --output recorder_output\records\audio_mvdr_opt.wav --track-duration 10.0 --gain 42 --reg 0.05 --diag-load 0.15 2>&1 | Tee-Object -FilePath debug_mvdr_opt.log


3. **LCMV**
	- Most compute-heavy
	- Most prone to overrun/clicks during beam update transitions, can cause audio to suddenely 'speed up' as buffer cannot chase

# python bf_LCMV_adaptive_3d.py --output recorder_output\records\audio_lcmv_fast.wav --track-duration 10.0 --gain 42 --reg 0.05 --diag-load 0.15 2>&1 | Tee-Object -FilePath debug_lcmv_fast.log

## Output
- **With Beamforming**: Mono audio file (1 channel) at 16 kHz
- Files saved to: `recorder_output/records/audio.wav`

## Performance Notes
- Real-time beamforming with minimal latency
- Delay-and-sum is computationally efficient and suitable for real-time use
- Good for far-field speech recognition and spatial filtering of noise

## Advanced Customization
To modify the microphone array geometry (spacing, layout), edit `audio/beamformer.py`:
- `mic_spacing` parameter: Distance between adjacent microphones in meters

## Troubleshooting
- If audio is silent: Check that microphones are properly detected (watch debug output)
- If audio quality is poor: Try different steering angles that match your sound source location
- For louder output: Adjust the beamforming gain in `beamformer.py` (multiply by a factor > 1)
