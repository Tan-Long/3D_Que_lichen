# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end pipeline for processing 3D tree bark scans (Stray Scanner) to detect and quantify lichen (địa y) on 92 cinnamon trees (*Cinnamomum cassia*, quế). Input: iPhone LiDAR scan data. Output: colored meshes and lichen area estimates classified by type (white/green) and slope side (upslope/downslope), reported in cm² and %.

---

## The analysis pipeline begins with field data collection using a LiDAR-equipped mobile device, capturing synchronized RGB video, depth maps, camera odometry, and IMU data around the tree. To reconstruct the trunk geometry, the system applies spatial filtering to isolate the target tree from background noise, followed by TSDF volumetric fusion to generate an initial 3D mesh.

This base model is then optimized: high-resolution RGB textures are projected onto the vertices, non-trunk artifacts are trimmed, and the surface is refined using volume-preserving Taubin smoothing. Finally, the mesh is capped at both ends and accurately aligned to global gravity using the IMU data.

For lichen detection, the algorithm evaluates local contrast and color saturation in the HSV space directly on the raw point cloud. It employs adaptive thresholds based on natural bark brightness to accurately isolate lichen colonies, effectively filtering out green vegetation. These classification labels are then mapped onto the finalized mesh. By calculating the physical area of each mesh triangle and determining the trunk's primary axis via Principal Component Analysis (PCA), the system quantifies the total coverage. To classify the spatial orientation, the global gravity vector is projected onto a plane perpendicular to the trunk axis to establish the "upslope" direction. By evaluating each mesh face's centroid against this directional vector, the system precisely categorizes and reports the lichen distribution across the upslope and downslope sides of the tree.

---

### Web Application (`3d_scan/app/`)

FastAPI + vanilla JS frontend. Key modules:

- `app/main.py` — entry point
- `app/config.py` — paths and settings; override `SCAN_DATA_DIR` env var to point to raw data; `TEAM_MEMBERS` env var (comma-separated) for identity
- `app/services/pipeline.py` — runs pipeline scripts as subprocesses using `venv311` Python (not `.venv`); parses progress from `(N/total)` stdout pattern
- `app/services/job_manager.py` — async job queue
- `app/services/database.py` — SQLite via aiosqlite
- `app/routers/` — upload, import, jobs, refine, samples, viewer endpoints
- `app/static/` — HTML/JS frontend (no framework)

**Note:** The web app's `pipeline.py` calls `venv311/Scripts/python.exe` — it does NOT use the main `.venv`.

---

## Notes

- Comments and variable names are primarily in Vietnamese; code structure follows standard Python conventions.
- Scripts support `--all`, `--skip-existing`, and `--scan NAME` flags consistently.
- CloudCompare (`tools/CloudCompare_v2.13.2_setup_x64.exe`) is optional for manual mesh inspection only.
- `3d_scan/` has its own `.git` repo tracked separately from the main project directory.

Method 1 

The analysis pipeline begins with field data collection using a LiDAR-equipped mobile device (iPhone 12+ Pro/Pro Max). The integration of consumer-grade LiDAR in these devices has been validated as a reliable and cost-effective tool for capturing high-resolution 3D point clouds in forest inventories (Gollob et al., 2021) and broader geoscientific applications (Luetzenburg et al., 2021). Synchronized Red, Green, Blue (RGB) video, depth maps, camera odometry, and Inertial Measurement Unit (IMU) data were captured using the Stray Scanner application (Stray Robots, 2021), a tool which has been successfully utilized for extracting high-fidelity iPhone LiDAR data in recent spatial studies (Tondo et al., 2023). To reconstruct the trunk geometry, the system applies spatial filtering to isolate the target tree from background noise, followed by Truncated Signed Distance Function (TSDF) volumetric fusion (Curless & Levoy, 1996) to generate an initial 3D mesh.

This base model is then optimized: high-resolution RGB textures are projected onto the vertices, non-trunk artifacts are trimmed, and the surface is refined using volume-preserving Taubin smoothing (Taubin, 1995). Finally, the mesh is capped at both ends using the gravity direction estimated from IMU accelerometer data.

For lichen detection, the algorithm evaluates local contrast and color saturation in the Hue, Saturation, Value (HSV) space directly on the raw point cloud, a robust approach for color-based segmentation under varying outdoor lighting (Cheng et al., 2001). It employs adaptive thresholds based on natural bark brightness to accurately isolate lichen colonies, effectively filtering out green vegetation. These classification labels are then mapped onto the finalized mesh. By calculating the physical area of each mesh triangle and determining the trunk's primary axis via Principal Component Analysis (PCA) (Pfeifer et al., 2004), the system quantifies the total coverage. To classify the spatial orientation, the empirical global gravity vector (hardcoded to the world Y-axis) is projected onto a plane perpendicular to the trunk axis to establish the "upslope" direction. By evaluating each mesh face's centroid against this directional vector, the system precisely categorizes and reports the lichen distribution across the upslope and downslope sides of the tree.

References

Cheng, H. D., Jiang, X. H., Sun, Y., & Wang, J. (2001). Color image segmentation: advances and prospects. Pattern Recognition, 34(12), 2259-2281.

Curless, B., & Levoy, M. (1996). A volumetric method for building complex models from range images. In Proceedings of the 23rd annual conference on Computer graphics and interactive techniques (pp. 303-312).

Gollob, C., Ritter, T., Kraßnitzer, R., Tockner, A., & Nothdurft, A. (2021). Measurement of forest inventory parameters with Apple iPad Pro and integrated LiDAR technology. Remote Sensing, 13(16), 3129.

Luetzenburg, G., Kroon, A., & Bjørk, A. A. (2021). Evaluation of the Apple iPhone 12 Pro LiDAR for an application in geosciences. Scientific Reports, 11(1), 22221.

Pfeifer, N., Gorte, B., & Winterhalder, D. (2004). Automatic reconstruction of single trees from terrestrial laser scanner data. In Proceedings of the ISPRS Congress Istanbul (Vol. XXXV, pp. 114-119).

Stray Robots. (2021). Stray Scanner (Version 1.0) [Mobile application software]. Retrieved from [https://github.com/strayrobots/scanner](https://github.com/strayrobots/scanner)

Taubin, G. (1995). A signal processing approach to fair surface design. In Proceedings of the 22nd annual conference on Computer graphics and interactive techniques (pp. 351-358).

Tondo, G. R., Riley, C., & Morgenthal, G. (2023). Characterization of the iPhone LiDAR-Based Sensing System for Vibration Measurement and Modal Analysis. Sensors, 23(18), 7832.

Method 2

The analysis pipeline begins with in situ data acquisition using a LiDAR-enabled mobile device, specifically an iPhone 12 Pro, operated via the Stray Scanner application. This setup enables the synchronized capture of Red – Green – Blue (RGB) imagery, depth maps, camera odometry, and Inertial Measurement Unit (IMU) measurements, facilitating comprehensive spatial sensing around the target tree. Such multimodal acquisition is consistent with established visual–inertial reconstruction frameworks (Leutenegger et al., 2015; Qin et al., 2018). To reconstruct trunk geometry, the collected data undergo spatial filtering to isolate the target structure from background noise, followed by volumetric integration using the Truncated Signed Distance Function (TSDF) representation. This approach enables robust surface reconstruction from range data and produces a consistent 3D mesh (Curless & Levoy, 1996; Newcombe et al., 2011). The resulting base mesh is subsequently refined through several post-processing steps. High-resolution RGB textures are projected onto the mesh vertices to enhance visual fidelity, following established large-scale texturing techniques (Waechter et al., 2014).

Non-trunk artifacts are removed through geometric filtering, and the surface is smoothed using volume-preserving Taubin filtering, which effectively reduces noise while preserving geometric features (Taubin, 1995). The mesh is then capped at both ends using the gravity direction estimated from IMU accelerometer data, ensuring physically consistent alignment (Madgwick, 2010).

For lichen detection, the method operates directly on the raw point cloud by analyzing local contrast and color saturation in the Hue–Saturation–Value (HSV) color space, a widely adopted representation for color-based segmentation (Smith, 1978). Adaptive thresholding based on bark brightness is applied to isolate lichen colonies while minimizing confusion with surrounding vegetation. The resulting classification labels are then mapped onto the reconstructed mesh. Quantification of lichen coverage is performed by computing the surface area of each triangular mesh element and aggregating contributions from classified regions. The principal axis of the trunk is estimated using Principal Component Analysis (PCA), a standard technique for geometric orientation analysis (Jolliffe, 2002). To determine spatial orientation, the global gravity vector is projected onto a plane orthogonal to the trunk axis to define the upslope direction. By evaluating each mesh face centroid relative to this directional vector, the system systematically categorizes and quantifies lichen distribution across upslope and downslope regions of the tree.

Curless, B., & Levoy, M. (1996). A volumetric method for building complex models from range images. Proceedings of the 23rd Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH), 303–312. [https://doi.org/10.1145/237170.237269](https://doi.org/10.1145/237170.237269)

Jolliffe, I. T. (2002). Principal component analysis (2nd ed.). Springer. [https://doi.org/10.1007/b98835](https://doi.org/10.1007/b98835)

Leutenegger, S., Lynen, S., Bosse, M., Siegwart, R., & Furgale, P. (2015). Keyframe-based visual–inertial odometry using nonlinear optimization. The International Journal of Robotics Research, 34(3), 314–334.

Madgwick, S. O. H. (2010). An efficient orientation filter for inertial and inertial/magnetic sensor arrays. University of Bristol.

Newcombe, R. A., Izadi, S., Hilliges, O., Molyneaux, D., Kim, D., Davison, A. J., Kohli, P., Shotton, J., Hodges, S., & Fitzgibbon, A. (2011). KinectFusion: Real-time dense surface mapping and tracking. Proceedings of the IEEE International Symposium on Mixed and Augmented Reality, 127–136. [https://doi.org/10.1109/ISMAR.2011.6092378](https://doi.org/10.1109/ISMAR.2011.6092378)

Qin, T., Li, P., & Shen, S. (2018). VINS-Mono: A robust and versatile monocular visual–inertial state estimator. IEEE Transactions on Robotics, 34(4), 1004–1020. [https://doi.org/10.1109/TRO.2018.2853729](https://doi.org/10.1109/TRO.2018.2853729)

Smith, A. R. (1978). Color gamut transform pairs. Proceedings of the 5th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH), 12–19. [https://doi.org/10.1145/800248.807361](https://doi.org/10.1145/800248.807361)

Taubin, G. (1995). A signal processing approach to fair surface design. Proceedings of the 22nd Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH), 351–358. [https://doi.org/10.1145/218380.218473](https://doi.org/10.1145/218380.218473)

Waechter, M., Moehrle, N., & Goesele, M. (2014). Let there be color! Large-scale texturing of 3D reconstructions. European Conference on Computer Vision (ECCV), 836–850. [https://doi.org/10.1007/978-3-319-10602-1_54](https://doi.org/10.1007/978-3-319-10602-1_54)