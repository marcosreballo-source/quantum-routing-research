#!/usr/bin/env python
"""Test all combinations of baseline and critic types in job naming."""
import sys
sys.path.insert(0, '.')

from launch_sagemaker import _extract_specs_from_config, _build_job_name

# Test configurations
test_cases = [
    # (name, config_dict, expected_job_name_pattern)
    
    # === baseline: null (auto-resolve) ===
    ("PPO with null baseline (auto→critic, default critic type)", {
        "training": {"method": "ppo", "baseline": None},
        "quantum": {"enabled": False, "critic": {"enabled": False, "type": "shared_conv1d"}},
    }, "bqc-qgat-ppo-critic-sc-C"),  # auto→critic, default sc
    
    ("Rollout with null baseline (auto→rollout)", {
        "training": {"method": "rollout", "baseline": None},
        "quantum": {"enabled": False, "critic": {"enabled": False, "type": "shared_conv1d"}},
    }, "bqc-qgat-rollout-rollout-C"),
    
    # === baseline: "critic" with different critic types ===
    ("baseline='critic' with shared_conv1d (sc)", {
        "training": {"method": "ppo", "baseline": "critic"},
        "quantum": {"enabled": False, "critic": {"enabled": True, "type": "shared_conv1d"}},
    }, "bqc-qgat-ppo-critic-sc-C"),
    
    ("baseline='critic' with shared_fc (sf)", {
        "training": {"method": "ppo", "baseline": "critic"},
        "quantum": {"enabled": False, "critic": {"enabled": True, "type": "shared_fc"}},
    }, "bqc-qgat-ppo-critic-sf-C"),
    
    ("baseline='critic' with own_encoder_conv1d (oec)", {
        "training": {"method": "ppo", "baseline": "critic"},
        "quantum": {"enabled": False, "critic": {"enabled": True, "type": "own_encoder_conv1d"}},
    }, "bqc-qgat-ppo-critic-oec-C"),
    
    ("baseline='critic' with own_encoder_mlp (oem)", {
        "training": {"method": "ppo", "baseline": "critic"},
        "quantum": {"enabled": False, "critic": {"enabled": True, "type": "own_encoder_mlp"}},
    }, "bqc-qgat-ppo-critic-oem-C"),
    
    # === baseline: "rollout" ===
    ("baseline='rollout'", {
        "training": {"method": "ppo", "baseline": "rollout"},
        "quantum": {"enabled": False, "critic": {"enabled": False}},
    }, "bqc-qgat-ppo-rollout-C"),
    
    # === baseline: "exponential" ===
    ("baseline='exponential'", {
        "training": {"method": "ppo", "baseline": "exponential"},
        "quantum": {"enabled": False, "critic": {"enabled": False}},
    }, "bqc-qgat-ppo-exponential-C"),
    
    # === baseline: "none" ===
    ("baseline='none'", {
        "training": {"method": "ppo", "baseline": "none"},
        "quantum": {"enabled": False, "critic": {"enabled": False}},
    }, "bqc-qgat-ppo-none-C"),
    
    # === quantum enabled cases ===
    ("baseline='critic' + quantum with oec", {
        "training": {"method": "ppo", "baseline": "critic"},
        "quantum": {
            "enabled": True, 
            "topology": "ring",
            "rotation": "RYRZ",
            "critic": {"enabled": True, "type": "own_encoder_conv1d"}
        },
    }, "bqc-qgat-ppo-critic-oec-Qring-RYRZ"),
    
    ("baseline=null (ppo) + quantum with default critic", {
        "training": {"method": "ppo", "baseline": None},
        "quantum": {
            "enabled": True, 
            "topology": "brickwall",
            "rotation": "RXRYRZ",
            "critic": {"enabled": False, "type": "shared_conv1d"}
        },
    }, "bqc-qgat-ppo-critic-sc-Qbw-RXRYRZ"),  # auto→critic, default sc
    
    ("baseline='rollout' + quantum", {
        "training": {"method": "ppo", "baseline": "rollout"},
        "quantum": {
            "enabled": True, 
            "topology": "chain",
            "rotation": "RY",
            "critic": {"enabled": False}
        },
    }, "bqc-qgat-ppo-rollout-Qchn-RY"),
]

print("=" * 100)
print("TESTING ALL BASELINE & CRITIC TYPE COMBINATIONS")
print("=" * 100)

passed = 0
failed = 0
failures = []

for test_name, cfg, expected in test_cases:
    specs = _extract_specs_from_config(cfg)
    job_name = _build_job_name(cfg, 'bqc', 'qgat')
    
    status = "✅ PASS" if job_name == expected else "❌ FAIL"
    if job_name == expected:
        passed += 1
    else:
        failed += 1
        failures.append((test_name, expected, job_name))
    
    print(f"\n{status} | {test_name}")
    print(f"    Expected: {expected}")
    print(f"    Got:      {job_name}")
    print(f"    Specs:    baseline={specs['baseline']}, critic={specs['critic_abbrev']}, q={specs['q_enabled']}")

print("\n" + "=" * 100)
print(f"SUMMARY: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print("=" * 100)

if failed > 0:
    print("\nFAILED TESTS:")
    for test_name, expected, got in failures:
        print(f"  - {test_name}")
        print(f"    Expected: {expected}")
        print(f"    Got:      {got}")
    sys.exit(1)
else:
    print("\n🎉 ALL TESTS PASSED!")
    sys.exit(0)
