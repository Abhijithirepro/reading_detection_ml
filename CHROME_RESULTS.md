# Chrome (browser) — per-video results

Scored entirely in the browser: `frontend/` — MediaPipe JS FaceLandmarker + ONNX Runtime Web,
no server, nothing uploaded.  
Model: 1-D CNN (`models/model.json`, `kind: cnn`), 6,181 parameters, threshold **0.9590**.  
Each row is the **first 180 seconds** of the recording.

> **These are shipped-model scores on its own training data, so they are optimistic.**
> The honest generalisation figures are the cross-validated ones in `models/metrics.json`
> (recall 0.56, precision 0.94), where each video is scored by a fold that never saw it.
> Do not quote the accuracy below as production accuracy.

## Summary

| | result |
|---|---|
| Cheating correctly flagged | **15/21** (71.4%) |
| Not-cheating correctly cleared | **75/78** (96.2%) |
| Overall accuracy | **90/99** (90.9%) |
| Missed cheaters | 6 |
| False accusations | 3 |
| Precision | 0.83 |
| Recall | 0.71 |
| Not scoreable | 1 (damaged file, see Notes) |

## Cheating videos (21)

Source: `cheating_folder/`. Ground truth for every row is **cheating**.

| # | video | Chrome score | verdict | correct |
|---|---|---|---|---|
| 1 | `combined_ir8256777_Gaurav_Kumar.mp4` | 0.9999 | CHEATING | yes |
| 2 | `combined_ir8277721_Sakshi_Mishra.webm` | 0.9993 | CHEATING | yes |
| 3 | `combined_ir8156365_Anirudh_N_R.webm` | 0.9989 | CHEATING | yes |
| 4 | `combined_ir8257151_Shyam_Sundar_Vallapatla.webm` | 0.9987 | CHEATING | yes |
| 5 | `combined_ir8277339_Dhiraj_Sanjay_Kharche.mp4` | 0.9978 | CHEATING | yes |
| 6 | `combined_ir8299831_K_Chandana.mp4` | 0.9978 | CHEATING | yes |
| 7 | `combined_ir8232529_Sakamuri_Pavithra.mp4` | 0.9975 | CHEATING | yes |
| 8 | `combined_ir8241161_Kalaiselvan_Karthikeyan.mp4` | 0.9937 | CHEATING | yes |
| 9 | `combined_ir8244089_Marthala_Ashok_Reddy.mp4` | 0.9900 | CHEATING | yes |
| 10 | `combined_ir8251421_Chippgiri_Reddy.mp4` | 0.9879 | CHEATING | yes |
| 11 | `combined_ir8275457_Abhilasha_M_E.mp4` | 0.9860 | CHEATING | yes |
| 12 | `combined_ir8273455_Bhaskar_R.webm` | 0.9845 | CHEATING | yes |
| 13 | `combined_ir8248195_Shweta_Chandrakant_Gambhire.webm` | 0.9669 | CHEATING | yes |
| 14 | `combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1).mp4` | 0.9651 | CHEATING | yes |
| 15 | `combined_ir7233051_Ishika_Vijaywargiya.webm` | 0.9611 | CHEATING | yes |
| 16 | `combined_ir7236381_Sampath_Pujari.webm` | 0.9324 | clean | **NO** |
| 17 | `combined_ir8241299_Kalluri_Praneeth (2).webm` | 0.9309 | clean | **NO** |
| 18 | `combined_ir8308053_Muthakani_Venkateswarlu.webm` | 0.9238 | clean | **NO** |
| 19 | `combined_ir7611187_Ram_Shekhawat.webm` | 0.9052 | clean | **NO** |
| 20 | `combined_ir8289961_Thrinath_Reddy_Dadam.webm` | 0.8340 | clean | **NO** |
| 21 | `combined_ir8307595_Jagruti_Patil.mp4` | 0.5424 | clean | **NO** |

**Missed (6):** `7236381_Sampath_Pujari` (0.9324), `8241299_Kalluri_Praneeth (2)` (0.9309), `8308053_Muthakani_Venkateswarlu` (0.9238), `7611187_Ram_Shekhawat` (0.9052), `8289961_Thrinath_Reddy_Dadam` (0.8340), `8307595_Jagruti_Patil` (0.5424)

## Non-cheating videos (79)

Source: `new_not_cheating/` and `not_cheating_folder/`. Ground truth for every row is **clean**.

| # | video | folder | Chrome score | verdict | correct |
|---|---|---|---|---|---|
| 1 | `combined_ir8298679_Nichitha_Merugu.webm` | new_not_cheating | 0.9858 | CHEATING | **NO** |
| 2 | `combined_ir8324779_Mayur_Vijay_Bodle.webm` | new_not_cheating | 0.9810 | CHEATING | **NO** |
| 3 | `combined_ir8294719_Ganesh_Guttula.mp4` | not_cheating_folder | 0.9784 | CHEATING | **NO** |
| 4 | `combined_ir8250885_Shruthi_D_A.mp4` | not_cheating_folder | 0.9252 | clean | yes |
| 5 | `combined_ir8302911_Avinash_Kumar.webm` | new_not_cheating | 0.8938 | clean | yes |
| 6 | `combined_ir8306667_Shashank_Verma.webm` | new_not_cheating | 0.8891 | clean | yes |
| 7 | `combined_ir8295905_Nitish_Kaushik.webm` | new_not_cheating | 0.8185 | clean | yes |
| 8 | `combined_ir8325779_Sheeba_M.webm` | new_not_cheating | 0.8106 | clean | yes |
| 9 | `combined_ir8294759_Bhuvnesvar_Vasistha.mp4` | not_cheating_folder | 0.8069 | clean | yes |
| 10 | `combined_ir8306701_Divyanshu_Dhusia.webm` | new_not_cheating | 0.7925 | clean | yes |
| 11 | `combined_ir8279075_Mohanambal_Nanjappan.mp4` | not_cheating_folder | 0.7687 | clean | yes |
| 12 | `combined_ir8307103_Ranganathan_K.webm` | new_not_cheating | 0.7628 | clean | yes |
| 13 | `combined_ir8311131_Dheerajsing_Darbarsing_Girase.webm` | new_not_cheating | 0.7585 | clean | yes |
| 14 | `combined_ir8270165_Ajay_Kumar_Mahato.mp4` | not_cheating_folder | 0.7536 | clean | yes |
| 15 | `combined_ir8268331_Arnab_Kumar_Kumar_Das.webm` | new_not_cheating | 0.7223 | clean | yes |
| 16 | `combined_ir8287587_Vishakha_Sumant_Deshpande.mp4` | not_cheating_folder | 0.7043 | clean | yes |
| 17 | `combined_ir8279051_Kethireddy_Nirmala_Devi.mp4` | not_cheating_folder | 0.6786 | clean | yes |
| 18 | `combined_ir8237215_Palle_Sindhuja.mp4` | not_cheating_folder | 0.6751 | clean | yes |
| 19 | `combined_ir8299029_Kattamuri_Sai_Dora.webm` | new_not_cheating | 0.6750 | clean | yes |
| 20 | `combined_ir8314429_Abhishek_Chavan.webm` | new_not_cheating | 0.6377 | clean | yes |
| 21 | `combined_ir8286421_Gogineni_Priya.mp4` | not_cheating_folder | 0.6358 | clean | yes |
| 22 | `combined_ir8307855_Sai_Kumar_Ganta.webm` | new_not_cheating | 0.6356 | clean | yes |
| 23 | `combined_ir8287867_Komanduri_Suhas.mp4` | not_cheating_folder | 0.6262 | clean | yes |
| 24 | `combined_ir8314881_Pushwanth_Reddy_Munagala.webm` | new_not_cheating | 0.6141 | clean | yes |
| 25 | `combined_ir8285271_Pemmadi_Adinarayana.mp4` | not_cheating_folder | 0.6093 | clean | yes |
| 26 | `combined_ir8298919_Salman_Mohammed_Salar.webm` | new_not_cheating | 0.6079 | clean | yes |
| 27 | `combined_ir8307835_Aditya_Hemendra_Upadhyay.webm` | new_not_cheating | 0.5581 | clean | yes |
| 28 | `combined_ir8289417_Debashish_Deka.webm` | new_not_cheating | 0.5498 | clean | yes |
| 29 | `combined_ir8302027_Shipra_Ganti.webm` | new_not_cheating | 0.5488 | clean | yes |
| 30 | `combined_ir8299817_Padmanabhuni_Jagadeshwari.webm` | new_not_cheating | 0.5405 | clean | yes |
| 31 | `combined_ir8303145_Anilesh_K.webm` | new_not_cheating | 0.5143 | clean | yes |
| 32 | `combined_ir8286765_Aravind_Yesugari.webm` | not_cheating_folder | 0.4910 | clean | yes |
| 33 | `combined_ir8305671_Bhagyesh_Patil.webm` | new_not_cheating | 0.4764 | clean | yes |
| 34 | `combined_ir8295465_Anasuya_Chakraborty.webm` | new_not_cheating | 0.4705 | clean | yes |
| 35 | `combined_ir8317951_Tandra_Bhowmick.webm` | new_not_cheating | 0.4370 | clean | yes |
| 36 | `combined_ir8312461_Pathan_Arshiya_Anjum.webm` | new_not_cheating | 0.4350 | clean | yes |
| 37 | `combined_ir8293625_Babu_Singh.webm` | new_not_cheating | 0.4177 | clean | yes |
| 38 | `combined_ir8287281_Arun_Yadav.webm` | new_not_cheating | 0.4099 | clean | yes |
| 39 | `combined_ir8279129_Rakshitha_Keerthi_B.mp4` | not_cheating_folder | 0.3788 | clean | yes |
| 40 | `combined_ir8280295_Ramashetty_N_Biradar.webm` | not_cheating_folder | 0.3781 | clean | yes |
| 41 | `combined_ir8311015_Mayur_Tambralli.webm` | new_not_cheating | 0.3711 | clean | yes |
| 42 | `combined_ir8323113_Akash_Chakradhar_Yewale.webm` | new_not_cheating | 0.3692 | clean | yes |
| 43 | `combined_ir8268871_Renuka_Joshi.webm` | new_not_cheating | 0.3677 | clean | yes |
| 44 | `combined_ir8301567_Ananthan_Ms.webm` | new_not_cheating | 0.3508 | clean | yes |
| 45 | `combined_ir8244067_Eswara_Moorthi_M.webm` | new_not_cheating | 0.3377 | clean | yes |
| 46 | `combined_ir8251579_Aishwarya_Sanjay_Mokase.webm` | new_not_cheating | 0.3141 | clean | yes |
| 47 | `combined_ir8276317_Girish_Subhash_Patil.mp4` | not_cheating_folder | 0.3127 | clean | yes |
| 48 | `combined_ir8311233_Aadil_Ameen_Noormohamed.webm` | new_not_cheating | 0.2982 | clean | yes |
| 49 | `combined_ir8306733_Akash_Deep.webm` | new_not_cheating | 0.2901 | clean | yes |
| 50 | `combined_ir8269593_Kamatam_Venkata_Rohan.mp4` | not_cheating_folder | 0.2724 | clean | yes |
| 51 | `combined_ir8305727_Priyanka_Shah.webm` | new_not_cheating | 0.2719 | clean | yes |
| 52 | `combined_ir8319815_Chandra_Prakash_Bendil.webm` | new_not_cheating | 0.2619 | clean | yes |
| 53 | `combined_ir8308231_Komal_Shashikant_Mule.webm` | new_not_cheating | 0.2541 | clean | yes |
| 54 | `combined_ir8305837_Devalla_Sai_Kiran.webm` | new_not_cheating | 0.2413 | clean | yes |
| 55 | `combined_ir8252691_Panthangi_Priyanka.mp4` | not_cheating_folder | 0.2271 | clean | yes |
| 56 | `combined_ir8288469_Pankaj_Bhandari.mp4` | not_cheating_folder | 0.2260 | clean | yes |
| 57 | `combined_ir8316515_Nigar_Fathima.webm` | new_not_cheating | 0.2250 | clean | yes |
| 58 | `combined_ir8299153_Koribilli_Parameswara_Rao.webm` | new_not_cheating | 0.2249 | clean | yes |
| 59 | `combined_ir8286813_Kanimozhi_Manoharan.webm` | not_cheating_folder | 0.2151 | clean | yes |
| 60 | `combined_ir8288683_Shubham_Akarte.mp4` | not_cheating_folder | 0.2099 | clean | yes |
| 61 | `combined_ir8301419_Pallavi_Ravindranath_Mayekar.webm` | new_not_cheating | 0.1848 | clean | yes |
| 62 | `combined_ir8283753_Sakshi_Suwas_Dudhe.webm` | not_cheating_folder | 0.1679 | clean | yes |
| 63 | `combined_ir8280585_Sahil_Malik.mp4` | not_cheating_folder | 0.1272 | clean | yes |
| 64 | `combined_ir8310591_Megha_Mangal.webm` | new_not_cheating | 0.1086 | clean | yes |
| 65 | `combined_ir8305029_Patibandla_Deviprasad.webm` | new_not_cheating | 0.0968 | clean | yes |
| 66 | `combined_ir8294907_Ramarjun_Ganguly.webm` | new_not_cheating | 0.0955 | clean | yes |
| 67 | `combined_ir8298825_Navaneeth_Remesh.webm` | new_not_cheating | 0.0950 | clean | yes |
| 68 | `combined_ir8294849_Tammi_Goutham.webm` | new_not_cheating | 0.0927 | clean | yes |
| 69 | `combined_ir8266715_Akhila_Reddy_Karnati.mp4` | not_cheating_folder | 0.0919 | clean | yes |
| 70 | `combined_ir8320509_Sk_Gulap.webm` | new_not_cheating | 0.0849 | clean | yes |
| 71 | `combined_ir8252235_Prasanth_Suresh.mp4` | not_cheating_folder | 0.0828 | clean | yes |
| 72 | `combined_ir8288293_Naveen_Vigrahalla.mp4` | not_cheating_folder | 0.0710 | clean | yes |
| 73 | `combined_ir8298153_Praveen_Babu_Kanipakam.webm` | new_not_cheating | 0.0480 | clean | yes |
| 74 | `combined_ir8269065_Rajesh_Chandrasekar.mp4` | not_cheating_folder | 0.0348 | clean | yes |
| 75 | `combined_ir8320039_Sreejith_V_R.webm` | new_not_cheating | 0.0246 | clean | yes |
| 76 | `combined_ir8289939_Dhanshree_Ghanshyam_Mul.mp4` | not_cheating_folder | 0.0190 | clean | yes |
| 77 | `combined_ir8306881_Kiran_Kishor_Vyawahare.webm` | new_not_cheating | 0.0065 | clean | yes |
| 78 | `combined_ir8306335_Ranjith_Murugan.webm` | new_not_cheating | 0.0016 | clean | yes |
| 79 | `combined_ir8286773_R_Chandra_Babu.mp4` | not_cheating_folder | n/a | — | not scored |

**Falsely accused (3):** `8298679_Nichitha_Merugu` (0.9858), `8324779_Mayur_Vijay_Bodle` (0.9810), `8294719_Ganesh_Guttula` (0.9784)

## Notes

- **`combined_ir8286773_R_Chandra_Babu.mp4` cannot be scored in a browser.** The file is damaged — ffmpeg reports `Invalid NAL unit size` and decodes 2750 of a claimed 39909 frames before the stream breaks. Chrome refuses to play it and raises an error rather than returning a score from partial data.
- **The false positives outscore the weakest true positives.** `Mayur_Vijay_Bodle` (0.9810) and `Nichitha_Merugu` (0.9858) score above the genuine cheaters `Shweta` (0.9669), `Sai_Koti` (0.9651) and `Ishika` (0.9611). No threshold separates them, so this is a ranking limit of the model rather than a calibration problem — raising the threshold to clear the false alarms would also drop three real detections.
- **Two misses are near the line, three are not.** `Kalluri_Praneeth` (0.9309) and `Ram_Shekhawat` (0.9052) sit just under the threshold. `Thrinath_Reddy_Dadam` (0.8340) and especially `Jagruti_Patil` (0.5424) are not near-misses — the model does not see reading behaviour in them at all, and no threshold change recovers them.
- **The threshold is tuned for precision.** At 0.9590 the system is strong at clearing innocent candidates and weaker at catching cheaters, which is the intended trade for an integrity product — but it means roughly one in four cheating interviews is not flagged.
