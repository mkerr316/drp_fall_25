# A Multiscale Topological Framework for Landscape Characterization and Process Prediction

* **Author:** Michael Kerr
* **Date:** September 27, 2025
* **Timeline:** 15 Weeks (Fall 2025)

### Introduction & Objective

Significant methodological challenges limit the application of foundational geomorphometric techniques. Standard metrics, such as slope and curvature, are highly sensitive to the chosen analytical scale. In contrast, the "implementation crisis" surrounding curvature—where inconsistent sign conventions across major GIS platforms undermine reproducibility—presents a critical barrier to robust science. This project addresses these limitations directly by introducing a novel framework that moves beyond measuring local form to quantifying intrinsic, multiscale shape. The core innovation is the creation of Topological Land-Surface Parameters (TLSPs), a new suite of data layers derived from a sliding-window Topological Data Analysis (TDA) pipeline. A key intellectual contribution of this work will be the development of an interpretive workflow that translates abstract topological signatures into a physically meaningful geomorphic vocabulary, bridging the gap between quantitative shape analysis and process-based understanding.

This project is strategically situated at the intersection of geography, mathematics, and computer science, directly addressing the National Science Foundation's interest in fostering interdisciplinary research in Topological Data Analysis and Artificial Intelligence. By developing novel topological features for predictive environmental modeling, this work not only advances geomorphic theory but also aligns with national research priorities in trustworthy and interpretable AI.

### Core Hypotheses

To provide a rigorous statistical structure, this project will first establish that TLSPs are effective descriptors of landscape structure, then demonstrate their utility in a predictive context.

* **$H_1$ (Landscape Characterization Hypothesis):**
    * **$H_{A1}$:** Geomorphic provinces exhibit statistically distinct multiscale topological signatures, as measured by formal statistical tests including MANOVA and non-parametric permutation tests on persistence landscapes.
    * **$H_{01}$:** There is no statistically significant difference in the multiscale topological signatures between geomorphic provinces.

* **$H_2$ (Process Prediction Hypothesis):**
    * **$H_{A2}$:** Multiscale topological signatures are significant predictors of process-based erosion, providing a statistically significant improvement in explanatory power (e.g., higher adjusted $R^2$) over a baseline model, as evaluated through k-fold cross-validation.
    * **$H_{02}$:** The inclusion of multiscale topological signatures does not significantly improve the predictive performance of a model for process-based erosion compared to a baseline model.

### Methods and Implementation

#### Computational Framework

We will execute all analyses within a Docker container to guarantee full reproducibility and implement a parallel processing architecture designed to analyze larger-than-memory DEMs. This project uses the Dynamic Resource Allocation Framework, which follows a three-step process at runtime:
1. System Interrogation: Programmatically query system resources using `psutil`.
2. Worker Configuration: Configure a Dask `LocalCluster`, allocating a safe fraction of available memory per worker.
3. Dynamic Chunk Calculation: Translate the per-worker memory limit into an optimal data chunk size, explicitly aligning the chunk dimensions with the internal block size of the source raster (e.g., a Cloud-Optimized GeoTIFF) to maximize storage-aware I/O efficiency.

To ensure computational feasibility, we will leverage GPU acceleration via `CuPy` and `GUDHI`. The sliding-window implementation will incorporate overlap or "halo" regions to prevent edge artifacts. The Dask Dashboard will monitor performance.

#### TDA Implementation and Justification

The TDA workflow will be implemented in Python using the `scikit-tda` ecosystem.

* Filtration Method: We will use sublevel and superlevel set filtrations directly on DEM elevation values. This approach is not only more computationally efficient for gridded data but also more geomorphically interpretable than the more general Vietoris-Rips filtration, which is based on pairwise distances, as it naturally captures the emergence and merging of topographic features, such as peaks and pits.

* Homology Computation: We will compute persistent homology for dimensions 0 ($H_0$) and 1 ($H_1$) on both the raw and inverted DEMs.

* Feature Vectorization & Spatialization: To create spatially explicit TLSP rasters, we will apply the TDA workflow within a moving window across the DEM. For each window, a persistence diagram will be generated and vectorized into both Persistence Landscapes and Persistence Images to allow for a formal comparison of their effectiveness. The central pixel of the window will contain a summary statistic of each vector representation.

#### Sampling Design

A stratified, doubly balanced sampling design will ensure a statistically powerful and efficient test of the hypotheses. To ensure the sample is representative across multiple process scales, strata will be defined by cross-classifying a broad-scale variable (equal-area elevation quantiles) with a fine-scale textural variable (quantiles of wavelet-based roughness). We will use the Local Cube method (via `BalancedSampling`) because it minimizes the variance of the Horvitz-Thompson estimator by simultaneously achieving spatial and strong covariate balance.

#### Target Process Metric Justification

For the predictive track ($H_2$), we will model erosion potential using the Water Erosion Prediction Project (WEPP) model. This choice represents a commitment to the highest standard of scientific rigor. By directly linking our novel topological parameters to the outputs of a physically based, state-of-the-art erosion model, we move beyond empirical correlations and test for a more fundamental connection between multiscale landscape shape and the physics of erosional processes.

### 15-Week Project Plan

#### Section I: WEPP Parameterization & Foundational Design (Weeks 1-5)

* **Week 1: Scoping, Data Acquisition & Toolchain Finalization**
    * [x] Finalize study area polygons and hold-out validation area.
    * [x] Finalize Toolchain: Document libraries for TDA, Geomorphometry, Sampling, and Parallelism.
    * [ ] Acquire Raw Data: Download all necessary data for WEPP modeling (DEMs, climate, soils).
    * [ ] Define Covariate Layers: Finalize a comprehensive, three-tiered suite of predictor covariates:
        * Tier 1 (Foundational): DEM, slope, multiscale local relief, and multiscale wavelet-based roughness.
        * Tier 2 (Process-Proxy): TWI and SPI.
        * Tier 3 (Exploratory): Geomorphons, TLSPs, and scalar summaries (e.g., persistence entropy).
* **Weeks 2-3: WEPP Climate & Soils Parameterization**
    * [ ] Develop and execute scripts to process raw climate and soil data into WEPP-required `.cli` and `.sol` files.
* **Week 4: WEPP Model Execution & Target Raster Generation**
    * [ ] Run the WEPP model to generate the primary target variable: a raster of average annual sediment yield.
* **Week 5: Pilot Study, Power Analysis & Final Sampling**
    * [ ] Conduct a pilot analysis on the WEPP output and key covariates to estimate variance and inform a formal power analysis.
    * [ ] Design and execute the Final Sample Draw using the multiscale stratification scheme and the Local Cube method.

#### Section II: TDA Pipeline Development & Feature Engineering (Weeks 6-10)

* **Week 6: Covariate Generation & Environment Finalization**
    * [ ] Perform hydrological conditioning on all DEMs.
    * [ ] Apply targeted smoothing to the conditioned DEM before calculating sensitive indices like TWI to ensure results reflect landform-scale processes rather than micro-topographic noise.
    * [ ] Generate all Tier 1 and Tier 2 covariate layers at the empirically-derived characteristic scales.
    * [ ] Finalize Docker container with dynamic Dask and GPU-enabled parallel processing setup.
* **Weeks 7-8: Core Pipeline Validation & TLSP Generation**
    * [ ] Develop and validate the core sliding-window TDA function on synthetic landscapes.
    * [ ] Benchmark serial vs. Dask-CPU vs. Dask-GPU implementations on a pilot subset to quantify performance gains.
    * [ ] Wrap the validated TDA function using the optimal parallel backend, incorporating halo regions.
    * [ ] Generate the final suite of TLSP raster layers using both Persistence Landscape and Persistence Image vectorizations.
* **Week 9: Data Extraction & Final Dataset Assembly**
    * [ ] Extract values from all generated covariate rasters at the final sample locations.
    * [ ] Assemble and clean the master dataset for modeling.
* **Week 10: Exploratory Data Analysis**
    * [ ] Conduct a thorough exploratory analysis of the final dataset, examining distributions, correlations, and interactions.

#### Section III: Analysis, Interpretation & Synthesis (Weeks 11-13)

* **Week 11: Track A - Landscape Characterization & Classification ($H_1$)**
    * [ ] Apply PCA/UMAP for dimensionality reduction.
    * [ ] Formally test $H_1$ using MANOVA and permutation tests.
    * [ ] Train and validate a Random Forest classifier.
    * [ ] As an exploratory step, compute $H_2$ homology on a representative data subset to assess the potential of volumetric features (voids) for landscape characterization.
* **Week 12: Track B - Environmental Process Prediction ($H_2$)**
    * [ ] Train and validate parallel Gradient Boosting models using k-fold cross-validation to predict WEPP-derived sediment yield. Formally compare the performance of models using Persistence Landscape features vs. Persistence Image features to identify the most effective representation for this task.
    * [ ] Physical Interpretation: Use SHAP to identify the most predictive features from the best-performing model.
* **Week 13: Advanced Interpretation & Synthesis**
    * [ ] External Validation: Apply the final, optimized model to the independent, held-out validation dataset.
    * [ ] Topological Interpretation: Use the Mapper algorithm as the primary method for generating a "topological skeleton" of the high-dimensional feature space. This graph will visualize and interpret the relationships between feature importance (as determined by SHAP) and landscape structure.
    * [ ] Geomorphic Vocabulary Mapping: Conduct a targeted analysis to create an explicit interpretive mapping between topological features and physical landforms by quantitatively correlating the most persistent features with delineated geomorphic objects (e.g., ridge networks, enclosed basins).
    * [ ] Synthesize & Visualize: Create plots, maps, and figures that weave all results into a coherent geomorphic narrative.

#### Section IV: Documentation & Final Deliverables (Weeks 14-15)

* **Week 14: Final Report & Presentation**
    * [ ] Write final report, framing the work as a direct link between a new generation of geomorphometrics and process-based physical models.
    * [ ] Prepare a presentation for both technical and general academic audiences.
* **Week 15: Code Publication & Project Archiving**
    * [ ] Finalize GitHub repository with all commented scripts, Dockerfile, and a replication notebook.
    * [ ] Project Debrief & Future Work: Formally document a plan for publishing the results and a direct extension of the framework to predict other process metrics, such as habitat suitability or landslide susceptibility.