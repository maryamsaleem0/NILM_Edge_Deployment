# Week 2 data note: UK-DALE windows for NILMFormer (Person 1, AI)

**Status:** pipeline built and run on the real data; 3 appliances x train/valid/test saved to Drive (`FYP_NILM/processed/w128/`). Files: `data_pipeline.py`, `nilm_config.json`, `timestamp_ref.py`.

## 1. Data
- UK-DALE **2017 release, plain-text** (`ukdale.zip`, 3,585,155,959 bytes, `unzip -t`: no errors) from the UKERC EDC at CEDA (catalogue ID `7d78f943-f9fe-413b-af52-1816f9d968b0`). Only the aggregate (`channel_1`) and the kettle, fridge and microwave channels were extracted.
- Appliances (plan update 23 Nov 2026): **kettle, fridge, microwave**. One model and one data folder per appliance.
- Aggregate = apparent power (EcoManager whole-house transmitter); kettle, fridge, microwave = active power (plug-in transmitters). The aggregate can therefore exceed the sum of the appliances by the reactive part. For houses 1, 2 and 5 the metadata flags the 6 s mains meter as "disabled"; we still use `channel_1.dat`, as the official builder does.

## 2. Settings (all in `nilm_config.json`)
| Setting | Value |
|---|---|
| Window length | **128** samples (a setting; `--window 64` or `96` also works) |
| Sampling | 1 min (builder: mean to 10 s, forward-fill up to 60 s, values < 5 W set to 0, clip at 6000 W, then mean to 1 min) |
| Windows | non-overlapping, any window with a missing value dropped |
| Scaling | power / **6000** (fixed; the repo default divides by the data maximum), status stays 0/1 |
| Train houses | 1, 3, 4, 5 (official). Test house: **2**. Seed 0 |
| Validation | random 20% of the training windows (what the official script does) |
| Saved per split | `*_data.npy` (N,2,2,128) float32 [aggregate/appliance x power/status], `*_start.npy` (Unix s, UTC), `*_house.npy` |

## 3. Window counts (official run)
| Appliance | Train | Valid | Test (house 2) | Active test windows |
|---|---|---|---|---|
| kettle | 15,421 | 3,856 | 1,669 | 571 (34.2%) |
| fridge | 16,608 | 4,152 | 1,344 | 1,344 (100%) |
| microwave | 16,697 | 4,175 | 1,344 | 339 (25.2%) |
| microwave, no house 5 (`microwave_no5`) | 15,517 | 3,880 | 1,344 | 339 (25.2%) |

Train+valid windows per house (kettle / fridge / microwave): house 1: 16,367 / 17,153 / 17,265; house 3: 380 / 381 / 381; house 4: 1,751 / 1,751 / 1,751; house 5: 779 / 1,475 / 1,475. House 1 supplies about 85% of all training windows.

## 4. Differences from the plan (decisions for the instructor are marked **[decide]**)
1. **Appliances:** kettle, fridge, microwave instead of kettle, fridge, dishwasher.
2. **Gap filling:** 60 s (builder, at 10 s resolution), plan said 3 min. Values below 5 W set to 0 (not in the plan). On/off status columns are also saved (not in the plan, not used by the loss).
3. **Zero-target houses [decide]:** the official split trains on houses 1, 3, 4, 5 for every appliance. Where a house has no exact channel name in `labels.dat`, the builder fills zeros. Kettle: house 4 (`kettle_radio` is not used), 9% of train+valid windows. Fridge: houses 3, 4, 5 (house 5's `fridge_freezer` is not matched), 17%. Microwave: houses 3, 4 (house 4's meter is a combined washing-machine/microwave/breadmaker meter), 10%. Option: `--train_mode with_appliance` excludes them.
4. **Validation split [decide]:** the official code takes a random 20% of all training windows (not per house, not by time), so neighbouring windows can fall in train and valid. The plan said "20% of house 1". Alternative: `--valid_mode time` (last 20% of each house).
5. **Scaling and leakage:** the official example script fits its scaler on all houses, including the test house 2, with max-scaling. The fixed 6000 divisor avoids this.
6. **Times are UTC** (no time zone, no daylight saving).

## 5. Findings to keep in mind
- **Fridge:** the only real fridge training house (house 1) is a *fridge-freezer* unit (metadata type "fridge freezer", up to about 880 W); the test house 2 has a plain fridge (max about 320 W). Fridge steps in the aggregate (about 55-60 W) are smaller than the fridge trace (about 90-100 W), and the fridge reads about 10 W when off. Expect the fridge model to be the weakest.
- **Microwave, house 5 [decide]:** its channel reads a constant **50 W** (median and 90th percentile both 50.0 W; 89.7% of samples between 5 and 100 W), while the test house is about 0 W. House 5 has only 52 active windows against 4,226 from house 1. Option B: exclude house 5 (`--train_houses 1,3,4 --tag _no5`), built and saved. Baseline for the paper comparison stays option A (official).
- **Alignment:** plots of active test windows show on/off events lining up with the aggregate. A lag scan (-5..+5 min) over the whole test set gives best lag **0 min** for kettle, fridge and microwave (the fridge number is weak, because large appliances dominate the aggregate's changes; the plots show the fridge edges coincide). One microwave test window shows a 185 W blip with no response in the aggregate (likely plug glitch).

## 6. Timestamp inputs (for Person 2, verified against the repo's `NILMDataset`, max difference 0)
Model input = 9 x 128: channel 0 = aggregate / 6000; channels 1-8 for sample i use `t_i = window_start + 60*i` seconds (UTC): sin/cos of 2pi*minute/60, 2pi*hour/24, 2pi*dayofweek/7 (Monday = 0, no -1), 2pi*month/12 (month 1-12, no -1), in the order minute, hour, dow, month. Reference code and a C lookup-table generator: `timestamp_ref.py` (`--emit_c`). Test vectors: `processed/w128/<appliance>/testvec_*.npy` (5 windows each: input 9x128, scaled target, start time). Open for the weekly meeting: send the window start time (one number per window; the board computes the 8 channels with a few lines of C, using the lookup tables from `--emit_c`) or send the precomputed 8x128 channels (about 4 KB of float32 per window).

## 7. How to reproduce (Colab, from `/content/NILMFormer`, files copied from `FYP_NILM/scripts/`)
```
uv run python data_pipeline.py --check_only
uv run --with matplotlib python data_pipeline.py --preview
uv run --with matplotlib python data_pipeline.py --appliances microwave --train_houses 1,3,4 --tag _no5 --preview
```
Each appliance takes about a minute. Outputs go to `FYP_NILM/processed/w128/<appliance>/` (settings.json inside records every choice).
