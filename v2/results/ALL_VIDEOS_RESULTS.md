# All-video results — v2 (15 fps) vs the previous v2 (25 fps) and v1

Every on-disk recording (21 in `cheating_folder/`, 28 in `not_cheating_folder/`, 51 in `new_not_cheating/`), each scored by a model that **never trained on it** (out-of-fold cross-validation).

- **v2 verdict** (`v2-15fps-xgb-selftrain-hardneg-SHIPPED`): CHEATING when the average reading probability while the candidate speaks is ≥ 0.166. *Reading time* = total length of detected reading segments.
- **v2 @ 25 fps** (`v2-xgb-selftrain-hardneg-SHIPPED`, archived in `archive/25fps/`): same rule, threshold 0.205.
- **v1 verdict**: CHEATING when the top-k window mean is ≥ 0.959 (v1's own out-of-fold scores).

| | v2 (15 fps) | v2 (25 fps, previous) | v1 |
|---|---|---|---|
| Cheating caught | 10 / 21 | 9 / 21 | 10 / 21 |
| Clean cleared | 78 / 79 | 78 / 79 | 78 / 79 |
| Overall | 88 / 100 | 87 / 100 | 88 / 100 |

**Verdicts that changed with the switch to 15 fps:** ir8251421_Chippgiri_Reddy (cheating: clean → CHEATING, now right)

Near the v2 threshold (within 5 %, treat as borderline): ir8277721_Sakshi_Mishra (cheating, 0.166)

**Processing:** 100 videos, 97 sampled at 15 fps; 3 at another rate — recordings slower than 15 fps keep every frame (ir8273455_Bhaskar_R 11.1 fps, ir8325779_Sheeba_M 14.3 fps, ir8280295_Ramashetty_N_Biradar 14.0 fps). Face found in 99.8% of frames (median; lowest 82.7%).


## Cheating videos (21)

| # | video | folder | v2 score | v2 verdict | correct | reading time | segments | 25 fps score | 25 fps verdict | v1 score | v1 verdict | v1 correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ir8232529_Sakamuri_Pavithra | cheating_folder | 0.704 | CHEATING | ✓ | 5:56 | 12 | 0.739 | CHEATING | 0.998 | CHEATING | ✓ |
| 2 | ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) *(reviewed)* | cheating_folder | 0.589 | CHEATING | ✓ | 13:17 | 35 | 0.658 | CHEATING | 0.990 | CHEATING | ✓ |
| 3 | ir8156365_Anirudh_N_R | cheating_folder | 0.510 | CHEATING | ✓ | 12:51 | 24 | 0.556 | CHEATING | 0.994 | CHEATING | ✓ |
| 4 | ir8241161_Kalaiselvan_Karthikeyan | cheating_folder | 0.449 | CHEATING | ✓ | 7:19 | 18 | 0.420 | CHEATING | 0.982 | CHEATING | ✓ |
| 5 | ir8299831_K_Chandana | cheating_folder | 0.347 | CHEATING | ✓ | 3:51 | 17 | 0.359 | CHEATING | 0.999 | CHEATING | ✓ |
| 6 | ir8308053_Muthakani_Venkateswarlu *(reviewed)* | cheating_folder | 0.319 | CHEATING | ✓ | 10:06 | 28 | 0.242 | CHEATING | 0.680 | clean | ✗ |
| 7 | ir8244089_Marthala_Ashok_Reddy *(reviewed)* | cheating_folder | 0.304 | CHEATING | ✓ | 5:42 | 15 | 0.376 | CHEATING | 0.996 | CHEATING | ✓ |
| 8 | ir8275457_Abhilasha_M_E | cheating_folder | 0.234 | CHEATING | ✓ | 4:35 | 20 | 0.205 | CHEATING | 0.959 | CHEATING | ✓ |
| 9 | ir8251421_Chippgiri_Reddy | cheating_folder | 0.182 | CHEATING | ✓ | 3:08 | 16 | 0.107 | clean | 0.861 | clean | ✗ |
| 10 | ir8277721_Sakshi_Mishra | cheating_folder | 0.166 | CHEATING | ✓ | 6:57 | 26 | 0.211 | CHEATING | 0.992 | CHEATING | ✓ |
| 11 | ir8277339_Dhiraj_Sanjay_Kharche | cheating_folder | 0.100 | clean | ✗ | 2:05 | 10 | 0.119 | clean | 0.910 | clean | ✗ |
| 12 | ir8273455_Bhaskar_R | cheating_folder | 0.078 | clean | ✗ | 1:39 | 5 | 0.082 | clean | 0.982 | CHEATING | ✓ |
| 13 | ir8248195_Shweta_Chandrakant_Gambhire | cheating_folder | 0.062 | clean | ✗ | 1:28 | 7 | 0.105 | clean | 0.135 | clean | ✗ |
| 14 | ir8256777_Gaurav_Kumar | cheating_folder | 0.046 | clean | ✗ | 0:19 | 3 | 0.047 | clean | 0.912 | clean | ✗ |
| 15 | ir8241299_Kalluri_Praneeth (2) | cheating_folder | 0.043 | clean | ✗ | 0:21 | 3 | 0.045 | clean | 0.562 | clean | ✗ |
| 16 | ir8257151_Shyam_Sundar_Vallapatla | cheating_folder | 0.036 | clean | ✗ | 0:35 | 4 | 0.033 | clean | 0.981 | CHEATING | ✓ |
| 17 | ir7236381_Sampath_Pujari | cheating_folder | 0.015 | clean | ✗ | 0:12 | 2 | 0.006 | clean | 0.887 | clean | ✗ |
| 18 | ir7233051_Ishika_Vijaywargiya | cheating_folder | 0.014 | clean | ✗ | 0:16 | 1 | 0.020 | clean | 0.661 | clean | ✗ |
| 19 | ir8289961_Thrinath_Reddy_Dadam | cheating_folder | 0.003 | clean | ✗ | 0:00 | 0 | 0.011 | clean | 0.696 | clean | ✗ |
| 20 | ir8307595_Jagruti_Patil | cheating_folder | 0.003 | clean | ✗ | 0:00 | 0 | 0.004 | clean | 0.014 | clean | ✗ |
| 21 | ir7611187_Ram_Shekhawat | cheating_folder | 0.000 | clean | ✗ | 0:00 | 0 | 0.000 | clean | 0.772 | clean | ✗ |

## Clean videos (79)

| # | video | folder | v2 score | v2 verdict | correct | reading time | segments | 25 fps score | 25 fps verdict | v1 score | v1 verdict | v1 correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ir8287867_Komanduri_Suhas | not_cheating_folder | 0.343 | CHEATING | ✗ | 7:30 | 16 | 0.259 | CHEATING | 0.619 | clean | ✓ |
| 2 | ir8312461_Pathan_Arshiya_Anjum | new_not_cheating | 0.126 | clean | ✓ | 4:22 | 16 | 0.065 | clean | 0.927 | clean | ✓ |
| 3 | ir8298919_Salman_Mohammed_Salar | new_not_cheating | 0.102 | clean | ✓ | 5:29 | 24 | 0.115 | clean | 0.711 | clean | ✓ |
| 4 | ir8307103_Ranganathan_K | new_not_cheating | 0.089 | clean | ✓ | 2:57 | 17 | 0.139 | clean | 0.639 | clean | ✓ |
| 5 | ir8302911_Avinash_Kumar | new_not_cheating | 0.083 | clean | ✓ | 2:21 | 5 | 0.022 | clean | 0.674 | clean | ✓ |
| 6 | ir8298825_Navaneeth_Remesh | new_not_cheating | 0.065 | clean | ✓ | 0:28 | 5 | 0.117 | clean | 0.348 | clean | ✓ |
| 7 | ir8288469_Pankaj_Bhandari | not_cheating_folder | 0.063 | clean | ✓ | 1:07 | 5 | 0.060 | clean | 0.256 | clean | ✓ |
| 8 | ir8320039_Sreejith_V_R | new_not_cheating | 0.057 | clean | ✓ | 1:13 | 10 | 0.020 | clean | 0.445 | clean | ✓ |
| 9 | ir8305671_Bhagyesh_Patil | new_not_cheating | 0.056 | clean | ✓ | 1:03 | 6 | 0.077 | clean | 0.572 | clean | ✓ |
| 10 | ir8295905_Nitish_Kaushik | new_not_cheating | 0.054 | clean | ✓ | 2:18 | 7 | 0.038 | clean | 0.689 | clean | ✓ |
| 11 | ir8294759_Bhuvnesvar_Vasistha | not_cheating_folder | 0.044 | clean | ✓ | 1:03 | 5 | 0.032 | clean | 0.844 | clean | ✓ |
| 12 | ir8298153_Praveen_Babu_Kanipakam | new_not_cheating | 0.043 | clean | ✓ | 2:08 | 10 | 0.026 | clean | 0.486 | clean | ✓ |
| 13 | ir8285271_Pemmadi_Adinarayana | not_cheating_folder | 0.042 | clean | ✓ | 1:38 | 10 | 0.039 | clean | 0.650 | clean | ✓ |
| 14 | ir8294849_Tammi_Goutham | new_not_cheating | 0.042 | clean | ✓ | 0:58 | 7 | 0.068 | clean | 0.571 | clean | ✓ |
| 15 | ir8288683_Shubham_Akarte | not_cheating_folder | 0.040 | clean | ✓ | 1:01 | 7 | 0.040 | clean | 0.181 | clean | ✓ |
| 16 | ir8294719_Ganesh_Guttula | not_cheating_folder | 0.035 | clean | ✓ | 0:40 | 5 | 0.052 | clean | 0.958 | clean | ✓ |
| 17 | ir8311131_Dheerajsing_Darbarsing_Girase | new_not_cheating | 0.035 | clean | ✓ | 0:37 | 6 | 0.081 | clean | 0.520 | clean | ✓ |
| 18 | ir8324779_Mayur_Vijay_Bodle | new_not_cheating | 0.034 | clean | ✓ | 0:00 | 0 | 0.005 | clean | 0.980 | CHEATING | ✗ |
| 19 | ir8307855_Sai_Kumar_Ganta | new_not_cheating | 0.032 | clean | ✓ | 0:26 | 4 | 0.011 | clean | 0.152 | clean | ✓ |
| 20 | ir8283753_Sakshi_Suwas_Dudhe | not_cheating_folder | 0.030 | clean | ✓ | 0:42 | 4 | 0.069 | clean | 0.548 | clean | ✓ |
| 21 | ir8299029_Kattamuri_Sai_Dora | new_not_cheating | 0.026 | clean | ✓ | 0:28 | 4 | 0.011 | clean | 0.304 | clean | ✓ |
| 22 | ir8279129_Rakshitha_Keerthi_B | not_cheating_folder | 0.024 | clean | ✓ | 0:09 | 2 | 0.038 | clean | 0.307 | clean | ✓ |
| 23 | ir8303145_Anilesh_K | new_not_cheating | 0.022 | clean | ✓ | 0:56 | 7 | 0.031 | clean | 0.765 | clean | ✓ |
| 24 | ir8307835_Aditya_Hemendra_Upadhyay | new_not_cheating | 0.022 | clean | ✓ | 0:15 | 3 | 0.003 | clean | 0.718 | clean | ✓ |
| 25 | ir8299817_Padmanabhuni_Jagadeshwari | new_not_cheating | 0.020 | clean | ✓ | 1:17 | 10 | 0.022 | clean | 0.422 | clean | ✓ |
| 26 | ir8286765_Aravind_Yesugari | not_cheating_folder | 0.019 | clean | ✓ | 0:09 | 2 | 0.011 | clean | 0.586 | clean | ✓ |
| 27 | ir8314429_Abhishek_Chavan | new_not_cheating | 0.018 | clean | ✓ | 0:10 | 2 | 0.004 | clean | 0.415 | clean | ✓ |
| 28 | ir8317951_Tandra_Bhowmick | new_not_cheating | 0.015 | clean | ✓ | 0:14 | 2 | 0.010 | clean | 0.763 | clean | ✓ |
| 29 | ir8250885_Shruthi_D_A | not_cheating_folder | 0.014 | clean | ✓ | 0:09 | 1 | 0.011 | clean | 0.339 | clean | ✓ |
| 30 | ir8269593_Kamatam_Venkata_Rohan | not_cheating_folder | 0.014 | clean | ✓ | 0:24 | 3 | 0.010 | clean | 0.099 | clean | ✓ |
| 31 | ir8293625_Babu_Singh | new_not_cheating | 0.014 | clean | ✓ | 0:00 | 0 | 0.021 | clean | 0.651 | clean | ✓ |
| 32 | ir8279075_Mohanambal_Nanjappan | not_cheating_folder | 0.012 | clean | ✓ | 0:05 | 1 | 0.007 | clean | 0.901 | clean | ✓ |
| 33 | ir8289939_Dhanshree_Ghanshyam_Mul | not_cheating_folder | 0.012 | clean | ✓ | 0:05 | 1 | 0.007 | clean | 0.027 | clean | ✓ |
| 34 | ir8305837_Devalla_Sai_Kiran | new_not_cheating | 0.012 | clean | ✓ | 0:00 | 0 | 0.002 | clean | 0.108 | clean | ✓ |
| 35 | ir8325779_Sheeba_M | new_not_cheating | 0.012 | clean | ✓ | 0:05 | 1 | 0.016 | clean | 0.693 | clean | ✓ |
| 36 | ir8237215_Palle_Sindhuja | not_cheating_folder | 0.009 | clean | ✓ | 0:00 | 0 | 0.004 | clean | 0.124 | clean | ✓ |
| 37 | ir8268331_Arnab_Kumar_Kumar_Das | new_not_cheating | 0.009 | clean | ✓ | 0:20 | 2 | 0.002 | clean | 0.645 | clean | ✓ |
| 38 | ir8298679_Nichitha_Merugu | new_not_cheating | 0.009 | clean | ✓ | 0:13 | 2 | 0.012 | clean | 0.747 | clean | ✓ |
| 39 | ir8323113_Akash_Chakradhar_Yewale | new_not_cheating | 0.008 | clean | ✓ | 0:04 | 1 | 0.011 | clean | 0.730 | clean | ✓ |
| 40 | ir8306701_Divyanshu_Dhusia | new_not_cheating | 0.007 | clean | ✓ | 0:00 | 0 | 0.007 | clean | 0.635 | clean | ✓ |
| 41 | ir8252235_Prasanth_Suresh | not_cheating_folder | 0.006 | clean | ✓ | 0:04 | 1 | 0.012 | clean | 0.034 | clean | ✓ |
| 42 | ir8270165_Ajay_Kumar_Mahato | not_cheating_folder | 0.006 | clean | ✓ | 0:00 | 0 | 0.007 | clean | 0.465 | clean | ✓ |
| 43 | ir8306335_Ranjith_Murugan | new_not_cheating | 0.006 | clean | ✓ | 0:00 | 0 | 0.013 | clean | 0.006 | clean | ✓ |
| 44 | ir8306881_Kiran_Kishor_Vyawahare | new_not_cheating | 0.006 | clean | ✓ | 0:00 | 0 | 0.001 | clean | 0.608 | clean | ✓ |
| 45 | ir8252691_Panthangi_Priyanka | not_cheating_folder | 0.005 | clean | ✓ | 0:00 | 0 | 0.006 | clean | 0.307 | clean | ✓ |
| 46 | ir8269065_Rajesh_Chandrasekar | not_cheating_folder | 0.005 | clean | ✓ | 0:00 | 0 | 0.004 | clean | 0.035 | clean | ✓ |
| 47 | ir8286773_R_Chandra_Babu | not_cheating_folder | 0.005 | clean | ✓ | 0:00 | 0 | 0.007 | clean | 0.500 | clean | ✓ |
| 48 | ir8244067_Eswara_Moorthi_M | new_not_cheating | 0.004 | clean | ✓ | 0:00 | 0 | 0.002 | clean | 0.654 | clean | ✓ |
| 49 | ir8294907_Ramarjun_Ganguly | new_not_cheating | 0.004 | clean | ✓ | 0:00 | 0 | 0.014 | clean | 0.053 | clean | ✓ |
| 50 | ir8305727_Priyanka_Shah | new_not_cheating | 0.004 | clean | ✓ | 0:00 | 0 | 0.003 | clean | 0.618 | clean | ✓ |
| 51 | ir8287587_Vishakha_Sumant_Deshpande | not_cheating_folder | 0.003 | clean | ✓ | 0:00 | 0 | 0.002 | clean | 0.497 | clean | ✓ |
| 52 | ir8302027_Shipra_Ganti | new_not_cheating | 0.003 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.847 | clean | ✓ |
| 53 | ir8279051_Kethireddy_Nirmala_Devi | not_cheating_folder | 0.002 | clean | ✓ | 0:00 | 0 | 0.004 | clean | 0.130 | clean | ✓ |
| 54 | ir8286421_Gogineni_Priya | not_cheating_folder | 0.002 | clean | ✓ | 0:05 | 1 | 0.001 | clean | 0.576 | clean | ✓ |
| 55 | ir8268871_Renuka_Joshi | new_not_cheating | 0.002 | clean | ✓ | 0:00 | 0 | 0.014 | clean | 0.899 | clean | ✓ |
| 56 | ir8289417_Debashish_Deka | new_not_cheating | 0.002 | clean | ✓ | 0:00 | 0 | 0.003 | clean | 0.519 | clean | ✓ |
| 57 | ir8305029_Patibandla_Deviprasad | new_not_cheating | 0.002 | clean | ✓ | 0:00 | 0 | 0.001 | clean | 0.264 | clean | ✓ |
| 58 | ir8314881_Pushwanth_Reddy_Munagala | new_not_cheating | 0.002 | clean | ✓ | 0:00 | 0 | 0.007 | clean | 0.946 | clean | ✓ |
| 59 | ir8266715_Akhila_Reddy_Karnati | not_cheating_folder | 0.001 | clean | ✓ | 0:00 | 0 | 0.002 | clean | 0.145 | clean | ✓ |
| 60 | ir8276317_Girish_Subhash_Patil | not_cheating_folder | 0.001 | clean | ✓ | 0:00 | 0 | 0.001 | clean | 0.202 | clean | ✓ |
| 61 | ir8280295_Ramashetty_N_Biradar | not_cheating_folder | 0.001 | clean | ✓ | 0:00 | 0 | 0.001 | clean | 0.229 | clean | ✓ |
| 62 | ir8288293_Naveen_Vigrahalla | not_cheating_folder | 0.001 | clean | ✓ | 0:00 | 0 | 0.001 | clean | 0.010 | clean | ✓ |
| 63 | ir8251579_Aishwarya_Sanjay_Mokase | new_not_cheating | 0.001 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.274 | clean | ✓ |
| 64 | ir8299153_Koribilli_Parameswara_Rao | new_not_cheating | 0.001 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.124 | clean | ✓ |
| 65 | ir8301567_Ananthan_Ms | new_not_cheating | 0.001 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.151 | clean | ✓ |
| 66 | ir8306733_Akash_Deep | new_not_cheating | 0.001 | clean | ✓ | 0:00 | 0 | 0.001 | clean | 0.147 | clean | ✓ |
| 67 | ir8310591_Megha_Mangal | new_not_cheating | 0.001 | clean | ✓ | 0:00 | 0 | 0.002 | clean | 0.153 | clean | ✓ |
| 68 | ir8311233_Aadil_Ameen_Noormohamed | new_not_cheating | 0.001 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.430 | clean | ✓ |
| 69 | ir8320509_Sk_Gulap | new_not_cheating | 0.001 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.185 | clean | ✓ |
| 70 | ir8280585_Sahil_Malik | not_cheating_folder | 0.000 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.953 | clean | ✓ |
| 71 | ir8286813_Kanimozhi_Manoharan | not_cheating_folder | 0.000 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.333 | clean | ✓ |
| 72 | ir8287281_Arun_Yadav | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.430 | clean | ✓ |
| 73 | ir8295465_Anasuya_Chakraborty | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.681 | clean | ✓ |
| 74 | ir8301419_Pallavi_Ravindranath_Mayekar | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.047 | clean | ✓ |
| 75 | ir8306667_Shashank_Verma | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.484 | clean | ✓ |
| 76 | ir8308231_Komal_Shashikant_Mule | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.172 | clean | ✓ |
| 77 | ir8311015_Mayur_Tambralli | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.112 | clean | ✓ |
| 78 | ir8316515_Nigar_Fathima | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.114 | clean | ✓ |
| 79 | ir8319815_Chandra_Prakash_Bendil | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.000 | clean | 0.523 | clean | ✓ |

_Generated by `src/all_videos_results.py` from `results/oof_video_scores__v2-15fps-xgb-selftrain-hardneg-SHIPPED.csv`; CSV copy: `results/ALL_VIDEOS_RESULTS.csv`._
