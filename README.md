# Askins-UoA-Research
Repository created for Master Thesis Reseach by Trevor Askins, UoA class of 2026.

Thesis ext is included in the Repo Admin folder

General instruction on how to clone and use this repository are below. 

Explanation of what each python script does is included inside the script as a comment at the top of the script.

Explanations of each block of code to help future users update/adapt the code to their use is included inside the code itself.

Good luck

# Abstract:
Efficient irrigation management requires balancing adequate root-zone water availability with the prevention of excessive deep percolation, nutrient loss, and unnecessary water consumption. Currently, site specific practices rely on interpolation of previous scientific studies and lessons learned. Manually creating site-specific models to guide sensor network design would be prohibitively time consuming due to the sheer number of potential sensor combinations. This creates a gap between existing crop data and site-specific applications. This study evaluates cost-aware decision tree regression as a methodology for optimizing soil sensor placement in irrigated agricultural systems. HYDRUS-1D simulations were used to generate synthetic datasets representing pressure head, volumetric water content, and water flux for 45 representative agricultural soils cultivated with corn and beans under flood irrigation. Variable root depths throughout the growing season were incorporated to produce representative training data for machine-learning models. Decision trees were trained to predict deep percolation, average root-zone water content, and a combined monitoring objective. A modified cost-aware decision tree algorithm was then used to balance predictive accuracy against sensor installation, equipment, and measurement costs. Results indicate that moderate cost weighting frequently improved prediction accuracy while substantially reducing the number and overall cost of required sensors, suggesting that cost weighting acts as an effective regularization strategy. The optimized monitoring networks reduced deployment costs while maintaining high predictive performance, with root-zone water-content predictions achieving errors below those commonly associated with high-quality field measurements. Sensor selection varied with crop type and monitoring objective, indicating that no universally optimal sensor layout exists across agricultural settings. The proposed methodology provides a practical framework for designing cost-effective irrigation monitoring networks and demonstrates the potential of integrating numerical vadose zone modeling with machine-learning techniques to support precision irrigation.

# To Use This Repository/Reproduce this work:
# 1. Clone both repos

This project depends on a custom fork of scikit-learn with a cost-aware
decision tree splitter — it must be cloned and built separately.

Execute:
git clone https://github.com/TKAskins/Askins-UoA_Research.git

Execute:
git clone https://github.com/TKAskins/scikit-learn_dt-opt_askins.git

# 2. Create the conda environment
The environment spec lives inside the sklearn fork.

Execute:
cd scikit-learn_dt-opt_askins

Excecute:
conda env create -f scikit_dev_env.yml

Execute:
conda activate scikit_dev

# 3. Build the custom scikit-learn fork into that environment
Still inside scikit-learn_dt-opt_askins/:

Execute:
pip install --editable . --no-build-isolation


If this fails, check scikit-learn's own "building from source" docs for
your OS (a C/C++ compiler is required) — the exact flags can shift between
sklearn versions.

Verify it worked before going further:

Execute:
python -c "import sklearn; print(sklearn.__file__)"


This must print a path inside scikit-learn_dt-opt_askins/, not a
site-packages copy. If it doesn't, the plain pip/conda scikit-learn is
shadowing the fork and every cost-weighted tree below will fail with
TypeError: unexpected keyword argument 'initial_cost'.

# 4. Generate Hydrus Training Data
The HYDRUS simulation outputs this project trains on are **already included**
in this repo:

- `Hydrus/python_hydrus_code/results/Flood/daily/*.mat` — 1,530 raw per-trial
  files (945 corn + 585 beans), one per soil-texture × root-depth combination
- `Hydrus/python_hydrus_code/Flood_corn_days.csv` and `Flood_beans_days.csv` —
  the aggregated files the DT training pipeline actually reads

**You do not need to redo this step to train the decision trees or reproduce
the results/figures from the initial research** — skip straight to "Train the decision trees."
Only do this if you want to change the simulation setup itself (different
soil textures, root depths, irrigation timing, etc.).

If you do need to regenerate it:

1. **Install HYDRUS-1D** — this is separate, third-party software (from
   PC-Progress), not included in this repo. It ships `H1D_CALC.exe`, the
   actual solver `HYDRUSmain_basic.py` calls via subprocess. **Windows only.**

2. Open `Hydrus/python_hydrus_code/HYDRUSmain_basic.py` and update:
   - `hydrus_exe_dir` (line ~31) — point this at wherever you installed
     HYDRUS-1D (the folder containing `H1D_CALC.exe`).
   - `PlantType` (line ~39) — `'corn'` or `'beans'`. Manual toggle, not a
     CLI flag — you'll run this script once per crop.
     - Adjust all variables associated with your crops and irrigation simulation. This is where the vast majority of editing will occur, primarily focused around .IN files that define experiment runs

3. Run it:
   ```bash
   conda run -n scikit_dev python HYDRUSmain_basic.py
This runs 45 soil textures × 21 root depths (corn) = 945, or
45 × 13 (beans) = 585, individual HYDRUS-1D simulations in a nested
loop. This is slow — expect a long run (hours, depending on your
machine) since each combination is a separate finite-difference solve.
Repeat with the other PlantType for the second crop.

Aggregate the raw per-trial .mat outputs into the CSV the DT pipeline reads:
Open process_hydrus_results.py and set plant_type (near the bottom, inside main()) to 'corn' or 'beans' — again, run once per crop.
Run it:

conda run -n scikit_dev python process_hydrus_results.py
This writes Flood_<plant_type>_days.csv into Hydrus/python_hydrus_code/, overwriting the version already there.

# 5. Train the decision trees
From Askins-UoA_Research/DT/soilexture_dt_askins/:

Execute:
conda run -n scikit_dev python dt_cost_by_level_main.py


Open the file first and set PLANT_TYPE = 'corn' or PLANT_TYPE = 'beans' near the top (line ~26) — it's a manual toggle, not a command-line flag. Run it once per crop (edit, save, rerun) to get both.
This reads the pre-computed HYDRUS output CSVs already checked into Hydrus/python_hydrus_code/ (no need to re-run any HYDRUS simulations) and writes all trained trees/train-test splits to results/Flood/... (gitignored — this step is what regenerates it locally).

# 6. Generate the report tables (CSV)
Each of these is standalone; run from DT/soilexture_dt_askins/:

Execute:
conda run -n scikit_dev python compare_trees.py
(edit PLANT_TYPE near the top first, run once per crop)

Execute:
conda run -n scikit_dev python generate_train_tables.py
(both crops in one run)

Execute:
conda run -n scikit_dev python save_sensor_efficiency_csv.py

Execute:
conda run -n scikit_dev python extract_sensor_time_freq.py

Execute:
conda run -n scikit_dev python extract_sensor_time_freq_beans.py


Output lands in report_tables/.

A few scripts (extract_sensor_freq.py, extract_sensor_efficiency.py,
extract_sensor_variance.py, cost_efficiency.py) print their results as
JSON/text to the console rather than saving a file — read their docstrings
at the top of each file before running.

# 7. Generate the PNG figures
dt_cost_by_level_plot.py is config-driven, not a single script you just
run:

Open the file and set _PLANT_TYPE, _OBJECTIVE, and _COST_THRESHOLD near the top to whichever tree variant you want plotted.
Call the plotting function you want (e.g. plot_imp_cost_panel(), pred_plot_2panel(), regdt_scatter_plots_v3()) — either interactively or by adding a call at the bottom of the file.
Output lands in figs/<objective_tag>/.

# 8. About report_figures/*.html
The interactive chart files in report_figures/ (cost efficiency, sensor
selection frequency, train-vs-test R², etc.) are hand-built visualizations,
not auto-generated by a script. Their data was manually transcribed from the
console output of the scripts in steps 5–6. If you retrain with different
data, these charts won't update automatically — the numbers embedded in each
file's <script> block need to be recomputed and pasted in by hand.


