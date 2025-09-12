# %% [markdown]
# ## Get histogram of the morphed PILR values
# fig, axs = plt.subplots(
#     1, len(STRUCTURE_IDS), figsize=(len(STRUCTURE_IDS) * 2, 4), dpi=300
# )
# for ct, (structure_id, morphed_pilr) in enumerate(morphed_pilrs.items()):
#     for dim in range(2):
#         ax = axs[ct] if len(morphed_pilrs) > 1 else axs
#         ax.hist(
#             morphed_pilr.flatten(),
#             bins=100,
#             alpha=0.5,
#             label=STRUCTURE_IDS[structure_id],
#         )
#         ax.set_title(STRUCTURE_IDS[structure_id])
# plt.tight_layout()
# plt.show()
# # %% [markdown]
# # ## Count pixels above threshold
# fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
# threshold_pct = np.arange(0, 101)
# for ct, (structure_id, morphed_pilr) in enumerate(morphed_pilrs.items()):
#     above_threshold_fraction = np.zeros_like(threshold_pct, dtype=float)
#     for pct_ix, pct in tqdm(enumerate(threshold_pct)):
#         threshold = np.percentile(morphed_pilr, pct)
#         above_threshold = morphed_pilr > threshold
#         above_threshold_fraction[pct_ix] = np.sum(above_threshold) / np.prod(
#             above_threshold.shape
#         )
#     ax.plot(threshold_pct, above_threshold_fraction, label=STRUCTURE_IDS[structure_id])
# ax.set_xlabel("Percentile threshold")
# ax.set_ylabel("Fraction of pixels above threshold")
# ax.axvline(87, ls="--", color="black", label="87th percentile")
# ax.legend()
# plt.tight_layout()
# plt.show()


# %% [markdown]
# ## Calculate overlap fraction for different thresholds
# pct_values = np.arange(0, 101)
# fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
# overlap_fractions = np.zeros((len(morphed_pilrs), len(morphed_pilrs), len(pct_values)))
# for ct1, (structure_id1, morphed_pilr1) in enumerate(morphed_pilrs.items()):
#     for ct2, (structure_id2, morphed_pilr2) in enumerate(morphed_pilrs.items()):
#         if ct1 >= ct2:
#             continue
#         for pct_ix, pct in tqdm(enumerate(pct_values)):
#             threshold1 = np.percentile(morphed_pilr1, pct)
#             threshold2 = np.percentile(morphed_pilr2, pct)
#             above_threshold1 = morphed_pilr1 > threshold1
#             above_threshold2 = morphed_pilr2 > threshold2
#             overlap_fractions[ct1, ct2, pct_ix] = np.sum(
#                 np.logical_and(above_threshold1, above_threshold2)
#             ) / np.sum(np.logical_or(above_threshold1, above_threshold2))
#         ax.plot(
#             pct_values,
#             overlap_fractions[ct1, ct2],
#             label=f"{structure_id1} vs {structure_id2}",
#         )
# ax.set_xlabel("Percentile threshold")
# ax.set_ylabel("Overlap fraction")
# ax.legend()
# plt.tight_layout()
# plt.show()
