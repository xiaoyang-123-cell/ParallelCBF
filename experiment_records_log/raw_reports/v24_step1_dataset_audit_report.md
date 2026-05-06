# V24 Step 1.0 Dataset Audit

Status: `PASS`
TensorBoard log dir: `logs/v24_step1_dataset_audit`
JSON: `data/v24_step1_dataset_audit.json`

```json
{
  "status": "PASS",
  "episodes": 31415,
  "timesteps": 4109264,
  "zero_action_fraction": 0.0,
  "zero_action_threshold": 0.0001,
  "zero_action_max_fraction": 0.4,
  "shape_error_count": 0,
  "nan_episode_count": 0,
  "bptt_sample_requested": 100,
  "bptt_sample_checked": 100,
  "bptt_max_abs_diff": 8.940696716308594e-08,
  "bptt_max_abs_tolerance": 1e-05,
  "attempt_distribution": {
    "open": 0.18,
    "single_static": 0.36,
    "multi_obstacle": 0.26,
    "dynamic_obstacle": 0.2
  },
  "attempt_target_mix": {
    "open": 0.18,
    "single_static": 0.36,
    "multi_obstacle": 0.26,
    "dynamic_obstacle": 0.2
  },
  "attempt_mix_max_abs_error": 0.0,
  "attempt_mix_tolerance": 0.025,
  "per_shard_attempt_mix_tolerance": 0.02,
  "per_shard_attempt_audit": [],
  "accepted_distribution": {
    "open": 0.2855642209135763,
    "single_static": 0.4214547190832405,
    "multi_obstacle": 0.18449785134489893,
    "dynamic_obstacle": 0.10848320865828426
  },
  "accepted_target_mix": {
    "open": 0.29654036243822074,
    "single_static": 0.42998352553542013,
    "multi_obstacle": 0.171334431630972,
    "dynamic_obstacle": 0.10214168039538715
  },
  "accepted_mix_max_abs_error": 0.013163419713926927,
  "accepted_mix_tolerance": 0.08,
  "prediction_match": {
    "zero_action_fraction_le_40pct": true,
    "bptt_hidden_continuity": true,
    "attempt_mix_matches": true,
    "accepted_mix_matches": true,
    "shape_and_nan_clean": true
  }
}
```
