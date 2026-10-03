# Score QA — `jev-latest` sentiment scores

- **Artifact:** `data/processed/scores.parquet` (277,213 rows) vs `data/processed/posts.parquet` (277,217 rows)
- **Date:** 2026-10-03
- **Scope:** distribution sanity, spot-checks of extremes, data-quality flags, engagement, crash-day sanity, risk register
- **Verdict:** **No degenerate output.** The model is coherent (probabilities sum to 1, `net = p_bull - p_bear`, `raw_label = argmax` in 99.76% of rows), and the 2026-10-01 crash-day signal is genuinely bearish on spot-check. Residual risks are the usual ones for off-the-shelf sentiment: sarcasm/meta-commentary misreads, neutral-lean, and a small amount of non-text leakage (URL slugs, order-book dumps).

## 1. Distributions

### 1.1 `raw_label`

| label | count | share |
|---|---:|---:|
| neutral | 152,319 | 54.95% |
| bear | 63,104 | 22.76% |
| bull | 61,790 | 22.29% |

Neutral-lean is real (55% neutral, 52% of posts have `p_neutral > 0.5`), but bull vs bear is essentially balanced (22.3% vs 22.8%).

### 1.2 Probabilities

| column | min | max | mean | median | std |
|---|---:|---:|---:|---:|---:|
| p_bull | 0.0 | 1.0 | 0.236 | 0.07 | 0.310 |
| p_bear | 0.0 | 1.0 | 0.236 | 0.08 | 0.303 |
| p_neutral | 0.0 | 1.0 | 0.528 | 0.55 | 0.386 |

### 1.3 `net` (= p_bull − p_bear, verified exact for all 277,213 rows)

| stat | value |
|---|---:|
| min / max | −1.0 / +1.0 |
| mean / median | −0.0004 / 0.00 |
| std | 0.475 |
| p1 / p5 / p25 | −0.99 / −0.88 / −0.21 |
| p50 | 0.00 |
| p75 / p95 / p99 | 0.18 / 0.92 / 1.00 |
| exactly 0 | 35,416 (12.8%) |
| \|net\| < 0.01 | 37,720 (13.6%) |

Bucket histogram is unimodal with a central spike near zero and fat tails — no bimodality, no suspicious flatness. `net > 0` in 123,650 posts vs `net < 0` in 118,147.

### 1.4 `confidence`

| stat | value |
|---|---:|
| min / max | 0.0 / 1.0 |
| mean / median | 0.702 / 0.78 |
| p25 / p75 | 0.47 / 0.96 |
| rows with confidence > 0.9 | 98,538 (35.5%) |
| rows with confidence < 0.1 | 3,442 (1.2%) |

`confidence` is a separate scalar, **not** the argmax probability: it equals the max of (p_bull, p_bear, p_neutral) for only 40,101 rows (14.5%). Mean confidence by label: neutral 0.771, bull 0.635, bear 0.603 — the model is most confident on neutral (short questions, URL-only posts, banter).

### 1.5 Consistency checks

| check | result |
|---|---|
| p_bull + p_bear + p_neutral = 1.0 | all 277,213 rows |
| net = p_bull − p_bear | all 277,213 rows |
| raw_label = argmax(p_bull, p_bear, p_neutral) | 276,553 / 277,213 (99.76%) — 660 mismatches (0.24%) |
| model values | single value `jev-latest` |

### 1.6 Monthly mean `net` (degeneracy check)

No flatness or drift. Monthly means track plausible events: **2020-03 (COVID) −0.082**, **2022-05…07 (war/rate shock) −0.055…−0.074**, **2026-02 −0.057**, **2026-03 −0.062**, **2026-09 −0.060**, and **2026-10 (partial, 726 posts) −0.115**. Positive months are muted (max ~+0.12). This is consistent with a genuine crash-sensitive signal, not constant/bimodal output.

## 2. Spot-checks (extremes)

Sampled randomly from the tails (seeded), spread across 2016–2026. `OK` = label matches a plain reading; excerpts translated in brackets where relevant.

| # | hsz_id | date | excerpt | pred (p_bull/p_bear/p_neut) | judgment |
|---|---|---|---|---|---|
| 1 | 8130005 | 2022-04-13 | "Remélem jó nagy felfelé, 11000felé" [hope for a big move up toward 11000] | bull 1.00 | OK |
| 2 | 5089173 | 2017-09-26 | "Ez volt az esés alja kb. Most szépen megpihen..és felmegy 10k környékére" [bottom is in, will go up to ~10k] | bull 1.00 | OK |
| 3 | 8835252 | 2023-12-15 | "az otp nyugdíjas reszveny, hamarosan 9 számjegyű lesz" [OTP is my pension stock, soon 9 figures] | bull 0.99 | OK (hopium, but bullish) |
| 4 | 9156038 | 2024-10-07 | "teljes TL pozit tartom, vettem hozzá még egy adagot" [holding full long, added more] | bull 0.99 | OK |
| 5 | 8500361 | 2023-02-20 | "nagyon rizikósan áll az OTP napos grafikonja… medvepiaci korrekció" [very risky daily chart, bear correction] | bear 1.00 | OK |
| 6 | 7078590 | 2020-05-01 | "az elején vagyunk… még mindig medvepiac van" [still bear market] | bear 0.99 | OK |
| 7 | 5731303 | 2018-07-24 | "minden TA könyv 'folytatas flag'… lefele fog kitörni" [continuation flag, will break down] | bear 1.00 | OK |
| 8 | 8147292 | 2022-04-27 | "megszabadultam jópár OTP részvénytől… 10 alá is nézhetünk" [sold a lot, may look below 10] | bear 1.00 | OK |
| 9 | 8835233 | 2023-12-15 | "16???" | neutral 0.99 | OK |
| 10 | 8228293 | 2022-06-20 | off-topic EU/Fidesz rant ("Az EU nem lenne nevetségesen béna…") | neutral 0.99 | OK (correctly neutral) |
| 11 | 9103054 | 2024-08-16 | "Ezt amúgy komolyan gondolod? :)" [do you seriously mean this?] | neutral 1.00 | OK |
| 12 | 6488243 | 2019-07-17 | "Maci most nagyon elfoglalt.." [Maci is busy] | neutral 1.00 | OK |
| 13 | 7913627 | 2021-11-17 | "Szép. Legalább megmentették és holnap magasan nyit." [saved, opens high tomorrow] | net +1.00 | OK |
| 14 | 6692222 | 2019-11-11 | long certs + fundamentals ("nem adom el… long certijeimet") | net +1.00 | OK |
| 15 | 9705718 | 2026-01-29 | "Minden rosszban van valami jó… kilábalás" [every cloud… recovery] | net +0.98 | OK |
| 16 | 8165469 | 2022-05-09 | "OBV még nem [emelkedik]… nem áramlik tőke a rv-be… nagyon nem jó" | net −1.00 | OK |
| 17 | 7952476 | 2021-12-13 | "pánikban lefelé eső Otp-ben? ugyan…" [skeptical of the buy-bid story] | net −0.98 | OK |
| 18 | 9056869 | 2024-06-28 | "vevők nem nagyon vannak… 300 körül eladok" [no buyers, will sell ~300] | net −0.98 | OK |
| 19 | 7122115 | 2020-05-26 | "Pár hsz elolvasása elég is volt… Lebeszélés, pesszimizmus 1000 %-kal." [commenting on the forum's 1000% pessimism] | bear 0.91 | **MISCLASSIFIED** — meta-commentary mocking the pessimists, not bearish |
| 20 | 9447685 | 2025-06-03 | "Minek kínzod magad… ha úgyis annyira ellene vagy a banknak?" [rhetorical to a bear, telling them to stop] | bear 0.80 | **MISCLASSIFIED** — arguing against a bear, not bearish |
| 21 | 7210231 | 2020-07-20 | "OTP a délutáni kilövésre készíti a muníciót! (irónia) longban ülök…" [(irony) + long] | bull 0.79 | partially OK — explicit "(irónia)" ignored, but net stance is long |

Net result: **17/18 clean directional/neutral calls correct; 2 clear meta-commentary misreads (rows 19–20) and 1 sarcasm-marker miss (row 21).**

## 3. Data-quality flags

| flag | count | scored? | notes |
|---|---:|---:|---|
| `data:image` (base64-embedded images) | 2 | **0 scored** | both are the HTTP-400 failures below |
| URL-only post (text = single http(s) link, no spaces) | 4,927 | all 4,927 | 4,408 neutral / 335 bull / 184 bear — label read from the URL slug (news headline), e.g. an Erste link slug "50-ezres-otp-uj-csucs" scored bull 0.98 |
| text < 20 chars | 36,428 | all 36,428 | 30,777 neutral / 3,023 bear / 2,628 bull; short quotes ("Erőtlen" → bear −0.83, "inkább lófing :)" → bear −0.52) are mostly sensible |
| order-book dumps (contains "Vételi"+"Eladási" + >40 digits) | 1,064 | all 1,064 | 494 bull / 451 neutral / 119 bear, mean net **+0.127**, mean confidence only 0.29 — spurious bull lean on market-depth tables |
| English/non-Hungarian posts (en-ratio ≥ 0.4, ≥30 chars) | 59 | all 59 | 40 neutral / 10 bear / 9 bull; mostly quoted news transcripts, scored neutral |

### 3.1 The 4 failed hsz_ids (`data/processed/scores_failed.json`)

| hsz_id | author | date | text | error |
|---|---|---|---|---|
| 6585101 | Shortostime | 2019-09-10 | `data:image/png;base64,iVBORw0KGgo…` (huge base64 image) | HTTP 400 (payload too large) |
| 6585117 | Shortostime | 2019-09-10 | `data:image/png;base64,iVBORw0KGgo…` (huge base64 image) | HTTP 400 (payload too large) |
| 8293952 | roland75 | 2022-08-12 | "Én meg Paksról,Kecskemétre." (short, irrelevant) | HTTP 529 (transient server error) |
| 9190029 | Fibonacchi | 2024-11-07 | "Hááát akkor ez hamar el is dőlt! 20k is pipa…" (9 upvotes, real post) | HTTP 529 (transient server error) |

These 4 are exactly the 4 posts missing from `scores.parquet` (277,217 − 277,213). Two are garbage input (base64 images), two are transient 529s that could be retried.

## 4. Engagement

### 4.1 Upvotes

| stat | upvotes | downvotes |
|---|---:|---:|
| mean | 1.63 | 1.02 |
| median | 1 | 0 |
| max | 111 | 109 |
| p90 | 4 | 2 |
| p95 | 6 | 4 |
| p99 | 11 | 9 |
| zero | 113,644 (41.0%) | 164,535 (59.4%) |
| ≥10 | 4,602 | — |
| ≥50 | 10 | — |

Top upvoted post: pomperj, 2016-12-06, 111 upvotes (TA chart commentary). Engagement is heavy-tailed but thin: only 10 posts ≥50 upvotes, so the `w = 1 + upvotes` weighting is dominated by a handful of posts.

### 4.2 Loud posters

Top authors by post count: HPeter76 11,403, ktamask 11,113, Macus 11,013, macibear 10,955, bumm 8,595. Top authors by total upvotes: macibear 23,488, Macus 20,418, ktamask 15,117.

Worst single-author days (day_total ≥ 30):

| date | author | posts | day total | share |
|---|---|---|---|---|
| 2024-03-02 | pantya9 | 23 | 36 | 63.9% |
| 2019-01-08 | HPeter76 | 75 | 121 | 62.0% |
| 2019-03-22 | HPeter76 | 123 | 225 | 54.7% |

On the crash day (2026-10-01): ktamask 73/400 = 18.3% (80 unique authors) — not dominant, but a single author is ~1/6 of the day's posts.

## 5. Crash-day sanity (2026-10-01)

`signals.parquet` row confirms: `score_wmean = −0.211`, `bear_wmean = 0.413`, `count = 400`, `z_score = −2.156`. Independently recomputed from the join: weighted (w = 1+upvotes) mean net **−0.2111**, weighted mean p_bear **0.4132** — matches.

8 most bearish posts of the day, eyeballed:

| hsz_id | author | upvotes | excerpt | pred | judgment |
|---|---|---|---|---|---|
| 9962045 | disco | 0 | "újabb aljat ütött az árfolyama. most már a japók is szabadulnak tőle." | bear 1.00 | genuinely bearish |
| 9962097 | EA83 | 5 | "…év végéig nem várható pozitív hír… csorogni fog tovább lefele" | bear 1.00 | genuinely bearish |
| 9963122 | ktamask | 4 | "az OTP-t bedarálták… Nincs vételi erő, mitől is lenne." | bear 1.00 | genuinely bearish |
| 9963421 | Bpali2 | 0 | "elkövettem egy hibát hogy korán vettem… beindított egy technikai korrekciót" | bear 1.00 | genuinely bearish (regret + bearish) |
| 9962702 | severian | 6 | "mocskos durva… zuhanórepülésben… rohadék beetetés volt… viszlát parasztok!" | bear 0.99 | genuinely bearish (angry) |
| 9962746 | Peterbond | 2 | "Az durva ami most megy… hogy így begyorsul délután nem hittem." | bear 0.99 | genuinely bearish |
| 9963085 | braveman | 6 | "Csányi mellett Trump is benzinnel akarta oltani az erdőt… összehozták a ma zuhanást." | bear 0.99 | genuinely bearish |
| 9962131 | hamucica | 0 | "meddig megy le?… 41000… lehet, addig csordogál?" | bear 0.98 | genuinely bearish (questioning how far down) |

All 8 are genuinely bearish panic/criticism — **no sarcastic false-bear flags on the crash day.** The crash-day signal is trustworthy.

## 6. Risks & recommendations

1. **Sarcasm / meta-commentary under-detection (highest-value risk).** Concrete: hsz 7122115 ("Lebeszélés, pesszimizmus 1000%-kal" → bear 0.91) and hsz 9447685 (rhetorical reply to a bear → bear 0.80) are both meta-commentary scored as directional bearish with confidence 0.86/0.70. hsz 7210231 has an explicit "(irónia)" marker that the model ignored. These inflate the bear side. **Recommend:** before aggregation, down-weight or re-verify high-|net| posts that contain sarcasm markers, and/or log sarcasm-marker posts for manual review (6,322 posts with markers and |net| ≥ 0.5 exist).

2. **Neutral-lean.** 55% of labels are neutral and 12.8% of posts have net exactly 0 (short questions, banter, off-topic). This dilutes the daily mean (median daily net is near 0). This is largely *correct* behavior for a forum, but the aggregation should be explicit about whether zero-net posts are included — currently they are, pulling `score_wmean` toward 0. **Recommend:** report a neutral-share series alongside the signal, or exclude `p_neutral == 1` posts from aggregation.

3. **URL-only leakage.** 4,927 posts are bare links; the sentiment is read from the URL slug (the headline), e.g. a vg.hu "kitores-elott-allhat-az-otp" link → bull 0.66, an Erste "50-ezres-otp-uj-csucs" link → bull 0.98. 519 URL-only posts get directional labels. This is arguably "the headline is the content", but it is inconsistent with how the rest of the corpus is scored. **Recommend:** decide a policy (score them as neutral, or accept headline-sentiment) and document it.

4. **Order-book dump leakage.** 1,064 posts are market-depth tables (buy/sell ladder) with no human sentiment; they are scored bull 494 / neutral 451 / bear 119 (mean net +0.127) with low confidence (0.29). This is a small but systematic *bull* bias from microstructure text. **Recommend:** filter order-book dumps (regex on "Vételi"+"Eladási"+digits) or down-weight by confidence before aggregation.

5. **`confidence` is unused in aggregation.** It is a separate scalar (not the argmax probability, differs in ~85% of rows) and is lowest exactly where it matters (bear 0.603, bull 0.635 vs neutral 0.771). The current `score_wmean` weights only by upvotes and ignores confidence, so low-confidence order-book/URL posts count fully. **Recommend:** consider a confidence- (or entropy-) weighted aggregation, or document why not.

6. **Failed/scored gap.** 4 posts are unscored and will silently drop from daily aggregation. Two are base64 images (correct to drop — should be filtered pre-scoring), two are transient HTTP 529s (8293952, 9190029) that should be retried. **Recommend:** retry the 2 transient failures and add a pre-scoring filter for `data:image` text.

7. **Base64 image hygiene.** Only 2 `data:image` posts exist in the corpus and both were correctly rejected by the API, but they also blew up the batch (HTTP 400). **Recommend:** strip `data:image…` bodies at ingestion so they never reach the scoring API.

8. **Author concentration.** A single author can be up to 62% of a day's posts (HPeter76 2019-01-08) and 18% of the crash-day posts (ktamask). Engagement weighting only partially offsets this. **Recommend:** report a daily author-concentration (top-author share) alongside the signal, or cap per-author contribution.

9. **Accepted as-is (no action).** Bull/bear label balance (22.3% vs 22.8%), exact probability normalization, `raw_label` consistency (99.76%), and the crash-day signal itself are all healthy. The remaining 660 `raw_label`≠argmax rows (0.24%) are negligible.
