# All-video results — v2 vs v1

Every on-disk recording (21 in `cheating_folder/`, 28 in `not_cheating_folder/`, 51 in
`new_not_cheating/`), each scored by a model that **never trained on it** (out-of-fold
cross-validation). v1 columns are v1's own out-of-fold scores (`models/cv_video_scores.csv`).

- **v2 verdict**: CHEATING when the average reading probability while the candidate speaks
  is ≥ 0.205. *Reading time* = total length of detected reading segments.
- **v1 verdict**: CHEATING when the top-k window mean is ≥ 0.959.

| | v2 | v1 |
|---|---|---|
| Cheating caught | 9 / 21 | 9 / 21 |
| Clean cleared | 78 / 79 | 78 / 79 |
| Overall | 87 / 100 | 87 / 100 |

Abhilasha M E sits exactly on the v2 threshold (borderline).

## Cheating videos (21)

| # | video | folder | v2 score | v2 verdict | correct | reading time | segments | v1 score | v1 verdict | v1 correct |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ir8232529_Sakamuri_Pavithra | cheating_folder | 0.739 | CHEATING | ✓ | 6:26 | 10 | 0.998 | CHEATING | ✓ |
| 2 | ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) *(reviewed)* | cheating_folder | 0.658 | CHEATING | ✓ | 13:41 | 29 | 0.990 | CHEATING | ✓ |
| 3 | ir8156365_Anirudh_N_R | cheating_folder | 0.556 | CHEATING | ✓ | 13:02 | 25 | 0.994 | CHEATING | ✓ |
| 4 | ir8241161_Kalaiselvan_Karthikeyan | cheating_folder | 0.420 | CHEATING | ✓ | 7:16 | 20 | 0.982 | CHEATING | ✓ |
| 5 | ir8244089_Marthala_Ashok_Reddy *(reviewed)* | cheating_folder | 0.376 | CHEATING | ✓ | 6:22 | 14 | 0.996 | CHEATING | ✓ |
| 6 | ir8299831_K_Chandana | cheating_folder | 0.359 | CHEATING | ✓ | 3:46 | 15 | 0.999 | CHEATING | ✓ |
| 7 | ir8308053_Muthakani_Venkateswarlu *(reviewed)* | cheating_folder | 0.242 | CHEATING | ✓ | 7:37 | 24 | 0.680 | clean | ✗ |
| 8 | ir8277721_Sakshi_Mishra | cheating_folder | 0.211 | CHEATING | ✓ | 8:34 | 36 | 0.992 | CHEATING | ✓ |
| 9 | ir8275457_Abhilasha_M_E | cheating_folder | 0.205 | CHEATING | ✓ | 4:11 | 17 | 0.959 | clean | ✗ |
| 10 | ir8277339_Dhiraj_Sanjay_Kharche | cheating_folder | 0.119 | clean | ✗ | 2:40 | 11 | 0.910 | clean | ✗ |
| 11 | ir8251421_Chippgiri_Reddy | cheating_folder | 0.107 | clean | ✗ | 2:03 | 12 | 0.861 | clean | ✗ |
| 12 | ir8248195_Shweta_Chandrakant_Gambhire | cheating_folder | 0.105 | clean | ✗ | 3:25 | 14 | 0.135 | clean | ✗ |
| 13 | ir8273455_Bhaskar_R | cheating_folder | 0.082 | clean | ✗ | 1:53 | 5 | 0.982 | CHEATING | ✓ |
| 14 | ir8256777_Gaurav_Kumar | cheating_folder | 0.047 | clean | ✗ | 0:21 | 2 | 0.912 | clean | ✗ |
| 15 | ir8241299_Kalluri_Praneeth (2) | cheating_folder | 0.045 | clean | ✗ | 0:26 | 3 | 0.562 | clean | ✗ |
| 16 | ir8257151_Shyam_Sundar_Vallapatla | cheating_folder | 0.033 | clean | ✗ | 0:32 | 2 | 0.981 | CHEATING | ✓ |
| 17 | ir7233051_Ishika_Vijaywargiya | cheating_folder | 0.020 | clean | ✗ | 0:19 | 1 | 0.661 | clean | ✗ |
| 18 | ir8289961_Thrinath_Reddy_Dadam | cheating_folder | 0.011 | clean | ✗ | 0:00 | 0 | 0.696 | clean | ✗ |
| 19 | ir7236381_Sampath_Pujari | cheating_folder | 0.006 | clean | ✗ | 0:06 | 1 | 0.887 | clean | ✗ |
| 20 | ir8307595_Jagruti_Patil | cheating_folder | 0.004 | clean | ✗ | 0:00 | 0 | 0.014 | clean | ✗ |
| 21 | ir7611187_Ram_Shekhawat | cheating_folder | 0.000 | clean | ✗ | 0:00 | 0 | 0.772 | clean | ✗ |

## Clean videos (79)

| # | video | folder | v2 score | v2 verdict | correct | reading time | segments | v1 score | v1 verdict | v1 correct |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ir8287867_Komanduri_Suhas | not_cheating_folder | 0.259 | CHEATING | ✗ | 5:37 | 14 | 0.619 | clean | ✓ |
| 2 | ir8307103_Ranganathan_K | new_not_cheating | 0.139 | clean | ✓ | 4:54 | 23 | 0.639 | clean | ✓ |
| 3 | ir8298825_Navaneeth_Remesh | new_not_cheating | 0.117 | clean | ✓ | 2:12 | 10 | 0.348 | clean | ✓ |
| 4 | ir8298919_Salman_Mohammed_Salar | new_not_cheating | 0.115 | clean | ✓ | 7:13 | 27 | 0.711 | clean | ✓ |
| 5 | ir8311131_Dheerajsing_Darbarsing_Girase | new_not_cheating | 0.081 | clean | ✓ | 2:52 | 14 | 0.520 | clean | ✓ |
| 6 | ir8305671_Bhagyesh_Patil | new_not_cheating | 0.077 | clean | ✓ | 1:53 | 9 | 0.572 | clean | ✓ |
| 7 | ir8283753_Sakshi_Suwas_Dudhe | not_cheating_folder | 0.069 | clean | ✓ | 1:50 | 10 | 0.548 | clean | ✓ |
| 8 | ir8294849_Tammi_Goutham | new_not_cheating | 0.068 | clean | ✓ | 2:23 | 13 | 0.571 | clean | ✓ |
| 9 | ir8312461_Pathan_Arshiya_Anjum | new_not_cheating | 0.065 | clean | ✓ | 2:12 | 12 | 0.927 | clean | ✓ |
| 10 | ir8288469_Pankaj_Bhandari | not_cheating_folder | 0.060 | clean | ✓ | 0:51 | 4 | 0.256 | clean | ✓ |
| 11 | ir8294719_Ganesh_Guttula | not_cheating_folder | 0.052 | clean | ✓ | 1:32 | 9 | 0.958 | clean | ✓ |
| 12 | ir8288683_Shubham_Akarte | not_cheating_folder | 0.040 | clean | ✓ | 1:00 | 6 | 0.181 | clean | ✓ |
| 13 | ir8285271_Pemmadi_Adinarayana | not_cheating_folder | 0.039 | clean | ✓ | 1:51 | 11 | 0.650 | clean | ✓ |
| 14 | ir8279129_Rakshitha_Keerthi_B | not_cheating_folder | 0.038 | clean | ✓ | 0:57 | 4 | 0.307 | clean | ✓ |
| 15 | ir8295905_Nitish_Kaushik | new_not_cheating | 0.038 | clean | ✓ | 1:43 | 8 | 0.689 | clean | ✓ |
| 16 | ir8294759_Bhuvnesvar_Vasistha | not_cheating_folder | 0.032 | clean | ✓ | 1:07 | 6 | 0.844 | clean | ✓ |
| 17 | ir8303145_Anilesh_K | new_not_cheating | 0.031 | clean | ✓ | 1:49 | 13 | 0.765 | clean | ✓ |
| 18 | ir8298153_Praveen_Babu_Kanipakam | new_not_cheating | 0.026 | clean | ✓ | 1:13 | 4 | 0.486 | clean | ✓ |
| 19 | ir8299817_Padmanabhuni_Jagadeshwari | new_not_cheating | 0.022 | clean | ✓ | 1:11 | 8 | 0.422 | clean | ✓ |
| 20 | ir8302911_Avinash_Kumar | new_not_cheating | 0.022 | clean | ✓ | 0:15 | 3 | 0.674 | clean | ✓ |
| 21 | ir8293625_Babu_Singh | new_not_cheating | 0.021 | clean | ✓ | 0:07 | 1 | 0.651 | clean | ✓ |
| 22 | ir8320039_Sreejith_V_R | new_not_cheating | 0.020 | clean | ✓ | 0:05 | 1 | 0.445 | clean | ✓ |
| 23 | ir8325779_Sheeba_M | new_not_cheating | 0.016 | clean | ✓ | 0:07 | 1 | 0.693 | clean | ✓ |
| 24 | ir8268871_Renuka_Joshi | new_not_cheating | 0.014 | clean | ✓ | 0:08 | 2 | 0.899 | clean | ✓ |
| 25 | ir8294907_Ramarjun_Ganguly | new_not_cheating | 0.014 | clean | ✓ | 0:00 | 0 | 0.053 | clean | ✓ |
| 26 | ir8306335_Ranjith_Murugan | new_not_cheating | 0.013 | clean | ✓ | 0:00 | 0 | 0.006 | clean | ✓ |
| 27 | ir8252235_Prasanth_Suresh | not_cheating_folder | 0.012 | clean | ✓ | 0:13 | 2 | 0.034 | clean | ✓ |
| 28 | ir8298679_Nichitha_Merugu | new_not_cheating | 0.012 | clean | ✓ | 0:15 | 2 | 0.747 | clean | ✓ |
| 29 | ir8250885_Shruthi_D_A | not_cheating_folder | 0.011 | clean | ✓ | 0:06 | 1 | 0.339 | clean | ✓ |
| 30 | ir8286765_Aravind_Yesugari | not_cheating_folder | 0.011 | clean | ✓ | 0:00 | 0 | 0.586 | clean | ✓ |
| 31 | ir8299029_Kattamuri_Sai_Dora | new_not_cheating | 0.011 | clean | ✓ | 0:16 | 3 | 0.304 | clean | ✓ |
| 32 | ir8307855_Sai_Kumar_Ganta | new_not_cheating | 0.011 | clean | ✓ | 0:04 | 1 | 0.152 | clean | ✓ |
| 33 | ir8323113_Akash_Chakradhar_Yewale | new_not_cheating | 0.011 | clean | ✓ | 0:00 | 0 | 0.730 | clean | ✓ |
| 34 | ir8269593_Kamatam_Venkata_Rohan | not_cheating_folder | 0.010 | clean | ✓ | 0:08 | 1 | 0.099 | clean | ✓ |
| 35 | ir8317951_Tandra_Bhowmick | new_not_cheating | 0.010 | clean | ✓ | 0:16 | 2 | 0.763 | clean | ✓ |
| 36 | ir8270165_Ajay_Kumar_Mahato | not_cheating_folder | 0.007 | clean | ✓ | 0:00 | 0 | 0.465 | clean | ✓ |
| 37 | ir8279075_Mohanambal_Nanjappan | not_cheating_folder | 0.007 | clean | ✓ | 0:05 | 1 | 0.901 | clean | ✓ |
| 38 | ir8286773_R_Chandra_Babu | not_cheating_folder | 0.007 | clean | ✓ | 0:00 | 0 | 0.500 | clean | ✓ |
| 39 | ir8289939_Dhanshree_Ghanshyam_Mul | not_cheating_folder | 0.007 | clean | ✓ | 0:04 | 1 | 0.027 | clean | ✓ |
| 40 | ir8306701_Divyanshu_Dhusia | new_not_cheating | 0.007 | clean | ✓ | 0:00 | 0 | 0.635 | clean | ✓ |
| 41 | ir8314881_Pushwanth_Reddy_Munagala | new_not_cheating | 0.007 | clean | ✓ | 0:00 | 0 | 0.946 | clean | ✓ |
| 42 | ir8252691_Panthangi_Priyanka | not_cheating_folder | 0.006 | clean | ✓ | 0:00 | 0 | 0.307 | clean | ✓ |
| 43 | ir8324779_Mayur_Vijay_Bodle | new_not_cheating | 0.005 | clean | ✓ | 0:00 | 0 | 0.980 | CHEATING | ✗ |
| 44 | ir8237215_Palle_Sindhuja | not_cheating_folder | 0.004 | clean | ✓ | 0:00 | 0 | 0.124 | clean | ✓ |
| 45 | ir8269065_Rajesh_Chandrasekar | not_cheating_folder | 0.004 | clean | ✓ | 0:00 | 0 | 0.035 | clean | ✓ |
| 46 | ir8279051_Kethireddy_Nirmala_Devi | not_cheating_folder | 0.004 | clean | ✓ | 0:00 | 0 | 0.130 | clean | ✓ |
| 47 | ir8314429_Abhishek_Chavan | new_not_cheating | 0.004 | clean | ✓ | 0:00 | 0 | 0.415 | clean | ✓ |
| 48 | ir8289417_Debashish_Deka | new_not_cheating | 0.003 | clean | ✓ | 0:00 | 0 | 0.519 | clean | ✓ |
| 49 | ir8305727_Priyanka_Shah | new_not_cheating | 0.003 | clean | ✓ | 0:00 | 0 | 0.618 | clean | ✓ |
| 50 | ir8307835_Aditya_Hemendra_Upadhyay | new_not_cheating | 0.003 | clean | ✓ | 0:00 | 0 | 0.718 | clean | ✓ |
| 51 | ir8266715_Akhila_Reddy_Karnati | not_cheating_folder | 0.002 | clean | ✓ | 0:00 | 0 | 0.145 | clean | ✓ |
| 52 | ir8287587_Vishakha_Sumant_Deshpande | not_cheating_folder | 0.002 | clean | ✓ | 0:00 | 0 | 0.497 | clean | ✓ |
| 53 | ir8244067_Eswara_Moorthi_M | new_not_cheating | 0.002 | clean | ✓ | 0:00 | 0 | 0.654 | clean | ✓ |
| 54 | ir8268331_Arnab_Kumar_Kumar_Das | new_not_cheating | 0.002 | clean | ✓ | 0:00 | 0 | 0.645 | clean | ✓ |
| 55 | ir8305837_Devalla_Sai_Kiran | new_not_cheating | 0.002 | clean | ✓ | 0:00 | 0 | 0.108 | clean | ✓ |
| 56 | ir8310591_Megha_Mangal | new_not_cheating | 0.002 | clean | ✓ | 0:00 | 0 | 0.153 | clean | ✓ |
| 57 | ir8276317_Girish_Subhash_Patil | not_cheating_folder | 0.001 | clean | ✓ | 0:00 | 0 | 0.202 | clean | ✓ |
| 58 | ir8280295_Ramashetty_N_Biradar | not_cheating_folder | 0.001 | clean | ✓ | 0:00 | 0 | 0.229 | clean | ✓ |
| 59 | ir8286421_Gogineni_Priya | not_cheating_folder | 0.001 | clean | ✓ | 0:00 | 0 | 0.576 | clean | ✓ |
| 60 | ir8288293_Naveen_Vigrahalla | not_cheating_folder | 0.001 | clean | ✓ | 0:00 | 0 | 0.010 | clean | ✓ |
| 61 | ir8305029_Patibandla_Deviprasad | new_not_cheating | 0.001 | clean | ✓ | 0:00 | 0 | 0.264 | clean | ✓ |
| 62 | ir8306733_Akash_Deep | new_not_cheating | 0.001 | clean | ✓ | 0:00 | 0 | 0.147 | clean | ✓ |
| 63 | ir8306881_Kiran_Kishor_Vyawahare | new_not_cheating | 0.001 | clean | ✓ | 0:00 | 0 | 0.608 | clean | ✓ |
| 64 | ir8280585_Sahil_Malik | not_cheating_folder | 0.000 | clean | ✓ | 0:00 | 0 | 0.953 | clean | ✓ |
| 65 | ir8286813_Kanimozhi_Manoharan | not_cheating_folder | 0.000 | clean | ✓ | 0:00 | 0 | 0.333 | clean | ✓ |
| 66 | ir8251579_Aishwarya_Sanjay_Mokase | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.274 | clean | ✓ |
| 67 | ir8287281_Arun_Yadav | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.430 | clean | ✓ |
| 68 | ir8295465_Anasuya_Chakraborty | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.681 | clean | ✓ |
| 69 | ir8299153_Koribilli_Parameswara_Rao | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.124 | clean | ✓ |
| 70 | ir8301419_Pallavi_Ravindranath_Mayekar | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.047 | clean | ✓ |
| 71 | ir8301567_Ananthan_Ms | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.151 | clean | ✓ |
| 72 | ir8302027_Shipra_Ganti | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.847 | clean | ✓ |
| 73 | ir8306667_Shashank_Verma | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.484 | clean | ✓ |
| 74 | ir8308231_Komal_Shashikant_Mule | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.172 | clean | ✓ |
| 75 | ir8311015_Mayur_Tambralli | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.112 | clean | ✓ |
| 76 | ir8311233_Aadil_Ameen_Noormohamed | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.430 | clean | ✓ |
| 77 | ir8316515_Nigar_Fathima | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.114 | clean | ✓ |
| 78 | ir8319815_Chandra_Prakash_Bendil | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.523 | clean | ✓ |
| 79 | ir8320509_Sk_Gulap | new_not_cheating | 0.000 | clean | ✓ | 0:00 | 0 | 0.185 | clean | ✓ |

_Generated from `results/oof_video_scores__v2-xgb-selftrain-hardneg-SHIPPED.csv`; CSV copy: `results/ALL_VIDEOS_RESULTS.csv`._
