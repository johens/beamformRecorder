# recorder

Proof of concept for capturing video and audio.

## Record video and audio
```
./record.sh
ctrl+c
```

## Merge video and audio
```
cd recorder_output/records/
./merge.sh
```

## Beamforming recap (current)

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

