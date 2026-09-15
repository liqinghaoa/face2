# SO-R1-A2-D0 Formal Full Dataset Protocol

## Objective and evidence boundary
This frozen protocol plans 13,500 synthetic M/H-sensitive latent maps and 67,500 paired linear-sRGB acquisitions. M/H maps are synthetic supervision targets, not true human melanin or hemoglobin concentrations. Camera identity denotes public spectral-response profiles, and OOD is held-out camera/light identity rather than real-device adaptation or extreme visual-domain shift. No geometry, normals, or directional relighting are included; S is low-frequency spatial shading and P is a specular-like nuisance.

## Frozen acquisition domain
The training-seen cameras are Canon 5DMarkII, Nikon D80, Olympus E-PL2, and Canon 300D. Held-out cameras are Canon 1DMarkIII and Nikon D5100. Seen lights are D65, A, and FL2; FL11 is held out. The 24 allowlisted camera-light pairs comprise 12 ID, 6 camera-OOD, 4 light-OOD, and 2 joint-OOD pairs. Nokia N900, Pentax Q, and SONY NEX-5N are excluded because of high-M fluorescent-light camera-to-linear-sRGB clipping artifacts.

## Latents, masks, and paired roles
The split sizes are Train 10,000, Validation 1,000, ID Test 1,000, Camera-OOD 500, Light-OOD 500, and Joint-OOD 500. Every latent has exactly five acquisitions: A0 reference (c0/l0/appearance0), A1 camera-only (c1/l0/appearance0), A2 light-only (c0/l1/appearance0), A3 appearance-only (c0/l0/appearance1), and A4 joint (c1/l1/appearance2). M/H/mask are invariant within a latent. Full/Mild/Strong masks are 40/40/20 percent, binary, stored once per latent, and generated with the C2 deterministic smooth-mask sampler.

## Samplers, QC, and seeds
M/H use the validated `_field` contract: independent stratified bases in [0,1], low-frequency variation, local Gaussian variation, clipping to [0,1], and nonconstant finite maps. All component seeds use BLAKE2b-64 over protocol/version/root seed/split/index/component/attempt; Python hash, clock, PID, and unordered enumeration are forbidden. Appearance0 is shared by A0/A1/A2; A3 and A4 use independent appearances. Exposure is frozen to [-0.50,+0.31] EV: R1 found +0.36 EV safe on the four causal scenes and applies a 0.05 EV margin. QC permits deterministic S/P/exposure-only retry, never M/H/mask/camera/light retry, up to 128 attempts.

## Storage, manifests, resources, and G1 gate
Each latent will be a compressed NPZ with RGB [5,3,256,256] float16, one float16 M, one float16 H, one uint8 mask, and canonical metadata. Required float16 RGB bounds are max error <=0.00025 and p99 <=0.00023. Content hashes are SHA-256 over canonical decompressed C-order content and metadata. C2-derived estimates are 28.84 GiB uncompressed, 14.10 GiB compressed, 12.8 h point / 16.0 h conservative single-thread runtime, and 0.68 GiB peak RSS. G1 must achieve 13,500/67,500/54,000 counts, 24/24 coverage, zero leakage/QC violations, 128-latent deterministic replay, loader validation, and unchanged protected assets.

## Scope and next step
No formal arrays, RGB, baseline, or proposed training have been started by D0. If all G1 gates pass, later training can consume the same manifest-defined split; D0 itself only authorizes G1. We evaluate acquisition robustness over four training-seen and two held-out representative camera spectral-response identities, together with three seen and one held-out illuminant SPD.
