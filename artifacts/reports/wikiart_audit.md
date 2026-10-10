# Dataset audit: wikiart

```json
{
 "dataset": "wikiart",
 "raw": {
  "n_images": 81444,
  "missing": {
   "style": 0,
   "artist": 0,
   "genre": 16452
  },
  "n_styles": 27,
  "n_artists": 1119,
  "n_genres": 10,
  "images_per_style": {
   "min": 98.0,
   "median": 1615.0,
   "mean": 3016.44,
   "max": 13060.0
  },
  "style_imbalance": {
   "n_classes": 27,
   "max_min_ratio": 133.27,
   "gini": 0.528,
   "normalized_entropy": 0.855
  },
  "artists_per_style": {
   "min": 5.0,
   "median": 72.0,
   "mean": 78.07,
   "max": 237.0
  },
  "styles_per_artist": {
   "min": 1.0,
   "median": 1.0,
   "mean": 1.88,
   "max": 11.0
  },
  "images_per_artist": {
   "min": 1.0,
   "median": 30.0,
   "mean": 72.78,
   "max": 1889.0
  },
  "share_artists_single_style": 0.538,
  "images_per_genre": {
   "min": 1902.0,
   "median": 4785.0,
   "mean": 6499.2,
   "max": 14112.0
  },
  "genres_per_style": {
   "min": 2.0,
   "median": 9.0,
   "mean": 8.07,
   "max": 10.0
  },
  "genre_imbalance": {
   "n_classes": 10,
   "max_min_ratio": 7.42,
   "gini": 0.367,
   "normalized_entropy": 0.905
  },
  "n_images_all_three_labels": 64992
 },
 "cleaned": {
  "n_images": 78278,
  "missing": {
   "style": 0,
   "artist": 0,
   "genre": 14810
  },
  "n_styles": 27,
  "n_artists": 1119,
  "n_genres": 10,
  "images_per_style": {
   "min": 85.0,
   "median": 1486.0,
   "mean": 2899.19,
   "max": 12783.0
  },
  "style_imbalance": {
   "n_classes": 27,
   "max_min_ratio": 150.39,
   "gini": 0.531,
   "normalized_entropy": 0.853
  },
  "artists_per_style": {
   "min": 5.0,
   "median": 65.0,
   "mean": 75.74,
   "max": 235.0
  },
  "styles_per_artist": {
   "min": 1.0,
   "median": 1.0,
   "mean": 1.83,
   "max": 11.0
  },
  "images_per_artist": {
   "min": 1.0,
   "median": 28.0,
   "mean": 69.95,
   "max": 1882.0
  },
  "share_artists_single_style": 0.548,
  "images_per_genre": {
   "min": 1855.0,
   "median": 4642.0,
   "mean": 6346.8,
   "max": 13747.0
  },
  "genres_per_style": {
   "min": 1.0,
   "median": 9.0,
   "mean": 8.0,
   "max": 10.0
  },
  "genre_imbalance": {
   "n_classes": 10,
   "max_min_ratio": 7.41,
   "gini": 0.367,
   "normalized_entropy": 0.904
  },
  "n_images_all_three_labels": 63468
 },
 "cleaning": {
  "n_raw": 81444,
  "n_unreadable": 0,
  "n_after_clean": 78278,
  "n_exact_duplicate_images": 1455,
  "n_near_duplicate_pairs": 1754,
  "n_near_pairs_identical_phash": 1422,
  "n_dup_groups_multi": 1731,
  "max_dup_group_size": 5,
  "near_duplicate_max_hamming": 4,
  "version": "9ace752be421",
  "reason_counts": {
   "near_duplicate_group_conflicting_style": 1373,
   "exact_duplicate_group_conflicting_style": 1348,
   "near_duplicate_of": 273,
   "near_duplicate_group_conflicting_artist": 110,
   "exact_duplicate_of": 42,
   "exact_duplicate_group_conflicting_artist": 20,
   "near_duplicate_group_conflicting_genre": 16,
   "exact_duplicate_group_conflicting_genre": 1
  }
 }
}
```

## Per style (cleaned)

| style                      |   n_images |   n_artists |   n_missing_artist |   n_genres |   n_missing_genre |   top_genre_share |
|:---------------------------|-----------:|------------:|-------------------:|-----------:|------------------:|------------------:|
| impressionism              |      12783 |         227 |                  0 |          9 |              1336 |             0.379 |
| realism                    |      10488 |         176 |                  0 |          9 |              1272 |             0.297 |
| romanticism                |       6859 |         104 |                  0 |          9 |              1639 |             0.3   |
| post impressionism         |       6256 |         181 |                  0 |         10 |               799 |             0.297 |
| expressionism              |       6247 |         235 |                  0 |         10 |              1349 |             0.34  |
| symbolism                  |       4149 |          84 |                  0 |         10 |              1225 |             0.368 |
| art nouveau modern         |       4148 |          78 |                  0 |          9 |              1601 |             0.251 |
| baroque                    |       4148 |          71 |                  0 |         10 |               816 |             0.307 |
| abstract expressionism     |       2563 |         140 |                  0 |         10 |               282 |             0.962 |
| northern renaissance       |       2534 |          26 |                  0 |          9 |               381 |             0.594 |
| naive art primitivism      |       2294 |          65 |                  0 |         10 |               566 |             0.316 |
| rococo                     |       2054 |          30 |                  0 |          9 |               402 |             0.601 |
| cubism                     |       1994 |         134 |                  0 |         10 |               464 |             0.235 |
| color field painting       |       1486 |          97 |                  0 |          2 |                54 |             0.998 |
| pop art                    |       1445 |          76 |                  0 |         10 |               903 |             0.417 |
| early renaissance          |       1384 |          24 |                  0 |          7 |                77 |             0.836 |
| high renaissance           |       1325 |          18 |                  0 |          8 |               222 |             0.613 |
| mannerism late renaissance |       1275 |          19 |                  0 |          8 |               264 |             0.622 |
| minimalism                 |       1251 |         109 |                  0 |          2 |               420 |             0.996 |
| ukiyo e                    |       1159 |          15 |                  0 |          9 |               376 |             0.352 |
| fauvism                    |        743 |          65 |                  0 |          9 |               191 |             0.33  |
| pointillism                |        488 |          31 |                  0 |          8 |                44 |             0.43  |
| contemporary realism       |        481 |          15 |                  0 |          8 |                93 |             0.446 |
| new realism                |        313 |           6 |                  0 |          7 |                33 |             0.357 |
| synthetic cubism           |        216 |           7 |                  0 |          7 |                 1 |             0.847 |
| analytical cubism          |        110 |           5 |                  0 |          6 |                 0 |             0.455 |
| action painting            |         85 |           7 |                  0 |          1 |                 0 |             1     |

## Style x genre (cleaned)

```
genre                       <missing>  abstract painting  cityscape  genre painting  illustration  landscape  nude painting  portrait  religious painting  sketch and study  still life
style                                                                                                                                                                                  
abstract expressionism            282               2195          3               7            16         12             13        20                   4                 7           4
action painting                     0                 85          0               0             0          0              0         0                   0                 0           0
analytical cubism                   0                  0          3              31             0          5              5        16                   0                 0          50
art nouveau modern               1601                  0        194             591           640        264             68       521                 104               154          11
baroque                           816                  1         53             764             6        177             34       956                1024               196         121
color field painting               54               1429          0               0             3          0              0         0                   0                 0           0
contemporary realism               93                  1          8              53             0        173             54        71                   0                 1          27
cubism                            464                128        120             247             5        170            107       293                  19                81         360
early renaissance                  77                  0          1              12            12          3              0       101                1092                86           0
expressionism                    1349                 60        325            1068           166        568            505      1664                  68               250         224
fauvism                           191                  1         46              45             1        182             48        98                   0                 9         122
high renaissance                  222                  0          1              17             0          4             20       201                 676               180           4
impressionism                    1336                  0       1529            2653             1       4338            356      1880                  18               241         431
mannerism late renaissance        264                  0         10              10             0          3              7       303                 629                47           2
minimalism                        420                828          0               0             0          0              0         0                   0                 3           0
naive art primitivism             566                  1        122             546           131        155             37       226                 309               131          70
new realism                        33                  0         57             100             1         43             23        47                   0                 9           0
northern renaissance              381                  0         18             112            12         47             13       468                1278               202           3
pointillism                        44                  1         65              97             0        191             17        60                   0                 7           6
pop art                           903                 40         71              59             4         39             25       226                   5                 4          69
post impressionism                799                  4        880             855             1       1622            218       838                  13               496         530
realism                          1272                  0        409            1552            76       2734            141      2620                 141              1196         347
rococo                            402                  0         76             287            11         62              4       993                 107                74          38
romanticism                      1639                  0        356             852           578       1108             65      1568                 418               245          30
symbolism                        1225                  1        148             377           204       1075             63       287                 463               271          35
synthetic cubism                    1                  2          1              10             0          4              0        14                   0                 2         182
ukiyo e                           376                  0         11             274            16        138             32       276                   1                21          14
```
