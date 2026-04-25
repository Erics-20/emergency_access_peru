# Analytical Methodology — District Healthcare Access Index

Pipeline module: `src/metrics.py`  |  Entry point: `run_metrics.py`  
Figures: `output/figures/`  |  Tables: `output/tables/`

---

## Overview

This document answers the four analytical questions using a composite district-level access index.  
The index has **three components** (facility availability, emergency activity, spatial access) and two **specifications** (baseline, alternative) to test sensitivity to the access definition.

| File | Description |
|---|---|
| `district_scores_baseline.csv` | All component and composite scores — baseline specification (1,762 districts) |
| `district_scores_alternative.csv` | All component and composite scores — alternative specification (1,762 districts) |
| `specification_comparison.csv` | Side-by-side comparison with tier labels, rank shifts, and tier-change flags |

---

## Question 1 — Which districts appear to have lower or higher availability of health facilities and emergency care activity?

### How it is measured

**Component 1 — Facility Availability (`fac_score`)**  
Indicator: active-facility density = `n_active / area_km²`  

- *Why density, not raw count*: districts vary enormously in area (0.8 to 45,896 km²). A raw facility count would mechanically favour large territories. Normalising by area makes urban micro-districts and sprawling Amazonian ones comparable.  
- *Why active facilities only*: the IPRESS registry includes facilities with status INOPERATIVO, CIERRE TEMPORAL, or RESTRICCIÓN DE SERVICIOS. These sites do not provide care; including them would overstate coverage.  
- *Score*: percentile rank of density across all 1,762 districts (0–100; higher = denser coverage).

**Component 2 — Emergency Activity (`activity_score`)**  
Indicator: total outpatient consultations at emergency-level facilities (II-1, II-2, II-E, III-1, III-2, III-E) in 2025, per district.

- *Why consultation volume*: facility counts measure supply. Activity volume — drawn from MINSA's HIS system — measures realised demand, capturing whether emergency-grade infrastructure is actually being used. A district may have a hospital on paper that generates zero recorded visits; consultation volume detects this.  
- *Why not per-capita*: reliable population denominators at the district level are unavailable for 2025. Percentile ranking makes the volume measure implicitly comparable without requiring a denominator.  
- *Zeros*: 1,062 districts recorded zero emergency-level consultations in HIS (they have only primary-care facilities, or their facilities reported nothing). These are not penalised beyond receiving a low percentile rank; they are not dropped.  
- *Score*: percentile rank (0–100).

**Key findings (Q1)**

| Metric | Value |
|---|---|
| Districts with ≥ 1 emergency-level facility | 195 of 1,762 (11.1%) |
| Districts with zero emergency consultations in HIS | ~1,062 (60.3%) |
| Median active-facility density | 0.014 facilities/km² |
| Facility density range | 0 – 89.5 facilities/km² |

Figures: `map_fac_score.png`, `map_activity_score.png`

---

## Question 2 — Which districts have populated centres with weaker spatial access to emergency-related health services?

### Spatial access logic (baseline)

**Component 3 — Spatial Access (`access_score`)**  
Indicator: % of CCPP (populated place) points within the district that fall within **30 km** of the nearest **emergency-level facility** (II-1 through III-E).

**Why CCPP as the demand unit**  
Peru's 136,587 populated places in the IGN/INEI registry are the smallest geographic unit for which we have complete spatial coverage. Each point represents a settlement where people live. Using CCPP avoids the modifiable area unit problem (MAUP) of assigning everyone to a district centroid.

**Why a threshold share, not a mean distance**  
A district mean can be dominated by a few extremely remote hamlets or by one very close hospital. The threshold share answers a more policy-relevant question: *what fraction of a district's settlements has a realistic travel distance to emergency care?* A share below 50 % signals that the majority of populated centres are beyond the access standard.

**Why 30 km**  
30 km is a commonly used operational threshold in Peruvian health-access studies and aligns with the 60-minute driving time standard under typical Andean road conditions. It is not claimed to be exact; it is a baseline assumption that the alternative specification relaxes.

**Pre-computation note**  
Distances are Euclidean (straight-line) computed in EPSG:32718 (UTM Zone 18S, metres). This understates true travel time on mountain roads but is consistent across districts and is the standard in the absence of road-network data.

**Key findings (Q2)**

| Metric | Value |
|---|---|
| Districts where ≥ 50% CCPP are within 30 km | 966 of 1,762 (54.8%) |
| Districts where 0% CCPP are within 30 km | 180 of 1,762 (10.2%) |
| Median % within 30 km | 70.8% |
| Districts with worst access (< 10% CCPP within 30 km) | 220 of 1,762 (12.5%) |

Figure: `map_access_score_baseline.png`

---

## Question 3 — Which districts appear most underserved and which appear best served?

### Composite index construction

$$\text{score} = \frac{1}{3}\, \text{fac\_score} + \frac{1}{3}\, \text{activity\_score} + \frac{1}{3}\, \text{access\_score}$$

Each component is a percentile rank (0–100) before weighting, so the three are in the same unit regardless of their original scale.  
Higher score = better served; lower = more underserved.

### Service tier classification (quartile-based)

| Tier | Score range | Districts |
|---|---|---|
| 4 — Best served | ≥ 75th percentile | 441 |
| 3 — Moderately served | 50th–75th | 440 |
| 2 — Weakly served | 25th–50th | 440 |
| 1 — Underserved | < 25th percentile | 441 |

Equal-count tiers were chosen (rather than equal-interval) because the composite score distribution is approximately uniform by construction (it is a mean of three percentile ranks). Quartile cuts therefore closely correspond to natural groupings.

### Evidence behind the classification

A district scores low (Tier 1) when it has:
- sparse active-facility coverage (low fac_score) **and/or**
- few or no emergency-level consultations recorded in HIS (low activity_score) **and/or**
- most of its populated places > 30 km from the nearest emergency facility (low access_score)

The component heatmap (`heatmap_underserved_components.png`) shows that underserved districts tend to have uniformly low scores across all three dimensions — it is not the case that they have good activity but poor facility coverage, for instance. This suggests a systemic deficit rather than a partial-coverage pattern.

**Top findings (Q3)**

- Underserved districts concentrate in the **jungle** regions (Loreto, Ucayali, Madre de Dios) and high-altitude **sierra** (Puno, Apurímac, Huancavelica), consistent with known geographic access barriers.
- Best-served districts cluster around **Lima, Arequipa, Trujillo, and Chiclayo** and other coastal metro areas.
- Only **195 districts** (11.1%) have any emergency-level facility. The remaining 89% rely entirely on the 7,941 geolocated facilities for geographic access — most of which are primary-care posts (I-1 to I-4).

Figures: `map_baseline_tiers.png`, `bar_top_bottom_districts.png`, `heatmap_underserved_components.png`

---

## Question 4 — How much do results change if the access definition changes?

### Two specifications

| Parameter | Baseline | Alternative |
|---|---|---|
| Component weights | ⅓ / ⅓ / ⅓ | 0.25 / 0.25 / 0.50 |
| Distance threshold | 30 km | 15 km |
| Facility target | Emergency-level only (II-1→III-E) | Any registered facility |
| Rationale shift | Balanced, emergency-focus | Access-dominant; tests any-care proximity |

The alternative doubles the weight on the spatial access component (50%) to reflect the view that geographic proximity to *some* care — even primary — is the most binding constraint for isolated communities. It also uses a stricter 15-km threshold but against a lower-bar target (any facility), capturing whether communities can reach *something* close.

### Results of the comparison

| Metric | Value |
|---|---|
| Spearman ρ (rank correlation) | 0.857 |
| Districts that changed service tier | **628 / 1,762 (35.6%)** |
| Districts that rose ≥ 2 tiers | small subset (mainly coastal and mid-valley) |
| Districts that fell ≥ 2 tiers | mainly peripheral sierra/jungle |

**Why the correlation is high (0.857) yet 35.6% change tier**  
Spearman ρ measures global rank agreement, which is strong — the two indices broadly agree on the ordering. But the quartile tier cut-offs are sensitive to the distribution of scores near the boundaries. Districts close to a percentile threshold in the baseline can shift to the adjacent tier under the alternative even if their rank moves modestly.

**What drives the changes**  
The alternative access component (15 km to any facility) has a much tighter distribution than the baseline (30 km to emergency). The median % within threshold jumps from 70.8% (baseline) to **100%** (alternative), meaning the access component becomes a **weak discriminator** in the alternative: almost all districts score near 100 on access, so the composite is driven more by facility density and emergency activity.

This flips the hierarchy for districts that have dense primary care but poor emergency coverage (e.g., small coastal districts with many clinics but no hospital). Those districts rise in the alternative because their primary-care access is rewarded. Districts with large areas and dispersed settlements that do not have dense primary care (some Andean and jungle districts) fall, because even under the lax 15 km threshold they cannot cover all their CCPP places.

**Interpretation for policy**  
The choice of access definition matters substantially. If the policy question is *"can communities access any health service?"*, the alternative paints a more optimistic picture and shifts attention to districts with physical isolation from even primary care. If the question is *"can communities reach emergency-grade care in time?"*, the baseline is more conservative and appropriate — it identifies a larger and more geographically concentrated set of underserved districts.

Figures: `scatter_specification_comparison.png`, `bar_tier_transitions.png`, `map_baseline_tiers.png`, `map_alternative_tiers.png`

---

## Summary of analytical decisions

| Decision | Choice | Justification |
|---|---|---|
| Facility component normalisation | Density (per km²) | Controls for district area heterogeneity |
| Activity indicator | Consultation volume at II+III facilities | Demand-side signal that filters out inactive facilities |
| Access unit | CCPP populated places | Sub-district resolution; avoids MAUP of centroid-only analysis |
| Access aggregation | % within threshold (not mean distance) | Threshold share is policy-interpretable; robust to outlier villages |
| Baseline threshold | 30 km to emergency facility | Operationally meaningful for emergency transport in Peru |
| Alternative threshold | 15 km to any facility | Tests sensitivity to both service standard and distance |
| Component scaling | Percentile rank before weighting | Makes components unit-free and comparably scaled |
| Classification | Quartile tiers | Equal-count groups; interpretable for geographic reporting |
| Distance metric | Euclidean in EPSG:32718 | Consistent; road-network data unavailable |
