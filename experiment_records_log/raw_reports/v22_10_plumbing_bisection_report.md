# V22.10 Plumbing Bisection Report

```json
{
  "status": "FAIL",
  "step1_action_pipeline": [
    {
      "name": "Step 1A direct +X acceleration",
      "passed": true,
      "metrics": {
        "dx_mean": 0.41056200861930847,
        "dx_min": 0.4105262756347656,
        "dx_max": 0.4105854034423828,
        "expected_ideal_dx": 0.5
      },
      "notes": []
    },
    {
      "name": "Step 1B direct +Z acceleration",
      "passed": true,
      "metrics": {
        "dz_mean": 0.4923388957977295,
        "z_final_mean": 1.4923388957977295,
        "dz_min": 0.4923388957977295,
        "dz_max": 0.4923388957977295,
        "expected_ideal_dz": 0.5
      },
      "notes": []
    }
  ],
  "step2_teacher_logic": {
    "name": "Step 2 V22.10 zero-residual teacher",
    "passed": false,
    "metrics": {
      "teacher_success_rate": 0.0,
      "final_goal_dist_mean": 1.2412073612213135,
      "min_goal_dist_mean": 0.3001721203327179,
      "final_x_mean": -0.04667032137513161,
      "final_z_mean": 0.946009635925293,
      "first_u_nom_x_mean": 0.8999999761581421,
      "first_u_safe_x_mean": 0.8999990820884705,
      "first_qp_delta_norm_mean": 8.940696716308594e-07,
      "hard_cap_binding_rate_final": 0.0,
      "terminated_rate_final": 0.0,
      "truncated_rate_final": 0.0
    },
    "notes": [
      "Zero-residual V22.10 teacher failed the clear-path goal-reaching gate."
    ]
  },
  "step3_demo_data": {
    "path": "data/v22_10_teacher_demos.pt",
    "exists": true,
    "num_samples": 70000,
    "obs_shape": [
      70000,
      35
    ],
    "action_shape": [
      70000,
      3
    ],
    "metadata": {
      "clear_teacher_success_rate": 0.0,
      "obstacle_residual_action_norm_mean": 1.4048929691314698,
      "clear_fraction": 0.7142857142857143
    },
    "goal_reached_rate_from_obs": 0.014628571458160877,
    "goal_dist_mean": 0.4286962151527405,
    "goal_dist_min": 0.010421899147331715,
    "goal_dist_p05": 0.32524412870407104,
    "goal_dist_p50": 0.433474063873291,
    "goal_dist_p95": 0.5079686641693115,
    "clear_count": 50176,
    "obstacle_count": 19824,
    "clear_goal_reached_rate_from_obs": 0.020408162847161293,
    "obstacle_goal_reached_rate_from_obs": 0.0,
    "action_norm_mean": 0.3997410535812378,
    "clear_action_norm_mean": 0.0,
    "obstacle_action_norm_mean": 1.4115148782730103
  },
  "failure_point": "Step 2 failure: V22.10 zero-residual teacher fails in clear path"
}
```

## Verdict

Step 2 failure: V22.10 zero-residual teacher fails in clear path
