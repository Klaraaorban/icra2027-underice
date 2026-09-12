# Related Work

## Depth-supervised NeRF and 3DGS

Depth priors have been used to regularise neural radiance fields since their
earliest days. DS-NeRF [Deng et al., 2022] and Depth-supervised NeRF [Roessle
et al., 2022] add an L2 or L1 penalty on rendered depth against sparse LiDAR or
depth-sensor readings, reducing the geometric ambiguity that arises from
photometric optimisation alone. Monocular depth estimates from networks such as
DPT/MiDaS and Depth Anything have been used similarly as a soft constraint
[Yu et al., 2022; Zhu et al., 2023]. The 3DGS literature has adopted the same
pattern: SuGaR [Guédon and Lepetit, 2024] couples surface normals; Gaussian
Opacity Fields [Yu et al., 2024] and DN-Splatter [Turculet et al., 2024] add
depth supervision from consumer RGB-D cameras or monocular estimators. Our
approach follows this family but uses a single-beam acoustic altimeter rather
than a surface-normal sensor or a monocular network, and introduces a
self-diagnosing adaptive weight that down-weights reliable pings to avoid
over-constraining the geometry where the rendered and acoustic depths already
agree.

## Underwater 3D reconstruction with neural representations

Underwater NeRF methods must cope with wavelength-dependent attenuation and
suspended particle scattering. SeaSplat [Yang et al., 2024] extends 3DGS with
a per-Gaussian underwater appearance model (absorption + backscatter) while
keeping geometry agnostic to depth. WaterSplatting [Huang et al., 2024] adds
a volumetric water column to the standard 3DGS rendering equation. Both
demonstrate improved photometric quality in underwater scenes; neither
incorporates an acoustic depth signal. Underwater NeRF approaches include
SeaThru-NeRF [Levy et al., 2023], which decomposes the scene into object
radiance and range-dependent medium properties, and neural implicit sonar
reconstruction [Dune et al., 2024]. Our work focuses on a regime these
methods have not addressed: sub-ice robotics, where the scene is low-texture,
the water column is not the dominant perceptual challenge, and the sonar
provides a structural anchor that photometric loss alone cannot.

## Acoustic-optical fusion in underwater robotics

Acoustic sensors and cameras are complementary: sonar provides coarse but
reliable metric range; cameras provide texture and appearance. Classical
fusion approaches include sonar-aided monocular SLAM [Ferreira et al., 2019]
and multi-beam sonar-guided visual reconstruction [Palomer et al., 2019].
More recently, neural implicit representations have been adapted for sonar
forward models [Suresh et al., 2022; Reed et al., 2023]. Our work differs
in that we use a single-beam altimeter (the simplest and most common acoustic
instrument on an ROV) as a one-dimensional depth oracle for supervising 3DGS,
not as a full point-cloud source.

## Polar and sub-ice sensing

Sea-ice underside geometry is critical for mass-balance estimation [Haas et al.,
2021] and autonomous navigation [Nicholson et al., 2022]. Icefin [Meister et al.,
2022] and other ROV platforms have surveyed sub-ice environments, typically
using multibeam sonar for bathymetry and cameras for visual inspection. 3D
photogrammetric reconstruction of ice surfaces from AUV data has been
demonstrated using structure-from-motion on forward-facing cameras [Wåhlin et al.,
2021]. We are, to our knowledge, the first to apply Gaussian splatting to a
sub-ice ROV dataset and to show that a single-beam altimeter is sufficient to
anchor the geometry against the sign-inversion failure mode that affects purely
photometric reconstruction in this scene class.

## Self-diagnosing / adaptive depth losses

The idea of modulating a loss weight based on its own residual — essentially a
form of self-paced learning — has appeared in curriculum learning [Bengio et al.,
2009] and robust loss functions [Barron, 2019]. Our adaptive gain term
`μ = clip(1 + k · |d̂ − z_s|_sg, 0.5, 4.0)` (where _sg denotes stop-gradient)
is a specific instance: pings with large residuals receive higher weight because
they are the frames where the geometry is furthest from the sonar constraint.
This is the opposite of the usual robust-loss philosophy (which down-weights
large residuals as potential outliers) and is justified here because the sonar
is more reliable than the rendered depth during early training.

## Limitations

**Single-beam coverage.** One centre-patch depth per frame is a thin probe of
a full 3D scene. The altimeter constrains one point on the ice underside per
view; off-axis geometry is constrained only through the photometric loss and the
DA3 depth prior.

**Oblique geometry.** When the ROV is not facing directly toward the ice — e.g.
when looking at a ridge or a refrozen lead at an angle — the altimeter beam and
the image centre measure different points. The 20×20 px averaging patch reduces
this error but does not eliminate it. A formal extrinsic calibration between
beam axis and optical axis would allow computing the projected beam footprint
rather than assuming boresight alignment.

**Garbage and long-range returns.** The Valeport VA500P returns −99999 for
no-return pings (ice too far, beam spread, absorption), which our pipeline
filters. In turbid water or at ranges beyond the altimeter's rated depth, the
fraction of usable pings falls and the geometric signal weakens. We quantify
this with the ping-rate sweep (`run_ping_rate_sweep.sh`), which shows how the
sonar r and RMSE degrade as the effective ping rate falls from 3.92 Hz to 0.1 Hz.

**One primary station.** The main evaluation is on a single 60-second PS117-39
clip. The sign-inversion finding generalises across methods (SeaSplat,
WaterSplatting) and across SfM initialisations (COLMAP, MASt3R-SfM on PS117-29),
but cross-scene generality of the sonar-supervised model itself awaits results
from additional deployments.
