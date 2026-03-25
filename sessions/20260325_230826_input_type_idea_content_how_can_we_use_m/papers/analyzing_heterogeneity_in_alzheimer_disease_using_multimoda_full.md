# Analyzing heterogeneity in Alzheimer Disease using multimodal normative modeling on imaging-based ATN biomarkers

## Metadata

- **Authors:** Sayantan Kumar, Tom Earnest, Braden Yang, Deydeep Kothapalli, Andrew J. Aschenbrenner
- **Year:** 2024
- **Arxiv Id:** 2404.05748v2
- **Url:** http://arxiv.org/abs/2404.05748v2

## Abstract

INTRODUCTION: Previous studies have applied normative modeling on a single neuroimaging modality to investigate Alzheimer Disease (AD) heterogeneity. We employed a deep learning-based multimodal normative framework to analyze individual-level variation across ATN (amyloid-tau-neurodegeneration) imaging biomarkers.
  METHODS: We selected cross-sectional discovery (n = 665) and replication cohorts (n = 430) with available T1-weighted MRI, amyloid and tau PET. Normative modeling estimated individual-level abnormal deviations in amyloid-positive individuals compared to amyloid-negative controls. Regional abnormality patterns were mapped at different clinical group levels to assess intra-group heterogeneity. An individual-level disease severity index (DSI) was calculated using both the spatial extent and magnitude of abnormal deviations across ATN.
  RESULTS: Greater intra-group heterogeneity in ATN abnormality patterns was observed in more severe clinical stages of AD. Higher DSI was associated with worse cognitive function and increased risk of disease progression.
  DISCUSSION: Subject-specific abnormality maps across ATN reveal the heterogeneous impact of AD on the brain.

## Introduction

neuroimaging modality to investigate Alzheimer Disease (AD) heterogeneity. We employed a 
deep learning-based multimodal normative framework to analyze individual-level variation 
across ATN (amyloid-tau-neurodegeneration) imaging biomarkers. 
METHODS: We selected cross-sectional discovery (n = 665) and replication cohorts (n = 430) 
with available T1-weighted MRI, amyloid and tau PET. Normative modeling estimated 
individual-level abnormal deviations in amyloid-positive individuals compared to amyloid-
negative controls. Regional abnormality patterns were mapped at different clinical group levels 
to assess intra-group heterogeneity. An individual-level disease severity index (DSI) was 
calculated using both the spatial extent and magnitude of abnormal deviations across ATN. 
RESULTS: Greater intra-group heterogeneity in ATN abnormality patterns was observed in 
more severe clinical stages of AD. Higher DSI was associated with worse cognitive function 
and increased risk of disease progression. 
DISCUSSION: Subject-specific abnormality maps across ATN reveal the heterogeneous 
impact of AD on the brain. 

1.Background 

Alzheimer Disease (AD) is the leading cause of dementia, characterized by cognitive and 
functional impairments that disrupt daily activities.[1,2] AD is highly heterogeneous, 
exhibiting considerable variability in clinical manifestations, cognitive decline, disease 
progression, and neuropathological changes, even within specific diagnostic categories.[3] 
However, traditional statistical approaches in AD research often overlook this heterogeneity, 
relying on case-control designs and group averages, effectively treating AD patients as a 
homogenous group. To progress toward precision medicine in AD, it is essential to move 
beyond the “average AD patient” approach and the assumption that AD affects all patients in 
the same way, and characterize disease abnormalities at the individual-level.[4]

## Methodology

[58] Doering S, McCullough AA, Gordon BA, Chen CD, McKay NS, Hobbs DA, et al. 
Evaluating Regional Importance for Tau Spatial Spread in Predicting Cognitive 
Impairment with Machine Learning. Alzheimers Dement 2023;19:e082553. 
[59] Cummings J, Apostolova L, Rabinovici GD, Atri A, Aisen P, Greenberg S, et al. 
Lecanemab: appropriate use recommendations. J Prev Alzheimers Dis 2023;10:362–77. 
[60] Tanzi RE. FDA approval of aduhelm paves a new path for Alzheimer’s disease. vol. 12. 
ACS Publications; 2021. 
[61] Orlhac F, Eertink JJ, Cottereau A-S, Zijlstra JM, Thieblemont C, Meignan M, et al. A 
guide to ComBat harmonization of imaging biomarkers in multicenter studies. J Nucl 
Med 2022;63:172–9. 

Acknowledgements 

We would like to thank the staff for the Washington University Center for High Performance 
Computing who helped enable this work. 
Competing Interests 

Author AS has received personal compensation for serving as a grant reviewer for BrightFocus 
Foundation. The remaining authors have no conflicting interests to report. 
Funding Sources 

The preparation of this report was supported by the Centene Corporation contract (P19-00559) 
for the Washington University-Centene ARCH Personalized Medicine Initiative and the 
National Institutes of Health (NIH) (R01-AG067103). Computations were performed using the 
facilities of the Washington University Research Computing and Informatics Facility, which 
were partially funded by NIH grants S10OD025200, 1S10RR022984-01A1 and 

1S10OD018091-01. Additional support is provided by the McDonnell Center for Systems 
Neuroscience. 
Data collection and sharing for this project was funded by the Alzheimer's Disease 
Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and 
DOD ADNI (Department of Defense award number W81XWH-12-2-0012). ADNI is funded 
by the National Institute on Aging, the National Institute of Biomedical Imaging and 
Bioengineering, and through generous contributions from the following: AbbVie, Alzheimer’s 
Association; Alzheimer’s Drug Discovery Foundation; Araclon Biotech; BioClinica, Inc.; 
Biogen; Bristol-Myers Squibb Company; CereSpir, Inc.; Cogstate; Eisai Inc.; Elan 
Pharmaceuticals, Inc.; Eli Lilly and Company; EuroImmun; F. Hoffmann-La Roche Ltd and 
its affiliated company Genentech, Inc.; Fujirebio; GE Healthcare; IXICO Ltd.; Janssen 
Alzheimer Immunotherapy Research & Development, LLC.; Johnson & Johnson 
Pharmaceutical Research & Development LLC.; Lumosity; Lundbeck; Merck & Co., Inc.; 
Meso Scale Diagnostics, LLC.; NeuroRx Research; Neurotrack Technologies; Novartis 
Pharmaceuticals Corporation; Pfizer Inc.; Piramal Imaging; Servier; Takeda Pharmaceutical 
Company; and Transition Therapeutics. The Canadian Institutes of Health Research is 
providing funds to support ADNI clinical sites in Canada. Private sector contributions are 
facilitated by the Foundation for the National Institutes of Health (www.fnih.org). The grantee 
organization is the Northern California Institute for Research and Education, and the study is 
coordinated by the Alzheimer’s Therapeutic Research Institute at the University of Southern 
California. ADNI data are disseminated by the Laboratory for Neuro Imaging at the University 
of Southern California. 
Data were also provided (in part) by Knight Alzheimer Disease Research Center 
(ADRC), supported by the Alzheimer’s Disease Research Center grant [P50-AG05681], 
Healthy Aging and Senile Dementia [P01 AG03991], and Adult Children Study [P01 

AG026276] and P30 NS048056 awarded to Dr Morris. AV-45 doses were provided by Avid 
Radiopharmaceuticals, a wholly owned subsidiary of Eli Lilly. 
Author contributions 
All authors contributed to the conceptualization and design of the study. SK implemented all 
data analyses and experiments and wrote the first draft of the manuscript. AS contributed to 
the interpretation of data. TE, BG and DK provided technical support. All authors were 
involved with manuscript revision, and all approved of the final draft. 
Data availability and consent statement 

All ADNI participants provided written informed consent, and study protocols were approved 
by each local site’s institutional review board. ADNI data used in this study are publicly 
available and can be requested following ADNI Data Sharing and Publications Committee 
guidelines: https://adni.loni.usc.edu/data-samples/access-data/. All protocols for Knight 
ADRC were approved by the Institutional Review Board at Washington University in St. 
Louis, and all participants provided informed consent before all procedures. Knight ADRC 
data can be obtained by submitting a data request through https://knightadrc.wustl.edu/data-
request-form/. 
Keywords 

Alzheimer’s disease, heterogeneity, multimodal, normative modeling, ATN biomarkers, MRI, 
AV45 amyloid PET, AV1451 tau PET, abnormal deviations, Disease Severity Index (DSI) 
Figures 

Figure 1: Flow chart of ADNI (1A) and Knight ADRC (1B) study participants. 

Figure 2: Brain atlas maps (Desikan-Killiany atlas for 66 cortical regions and Aseg atlas for 
24 subcortical regions) showing the pairwise group differences in magnitude of deviations at 
each region between the amyloid negative CU group and each of the CDR groups in ADNI 
(2A) and Knight ADRC (2B). The figures from left to right indicate the brain maps 
corresponding to MRI, amyloid and tau, respectively. The color bar represents the effect size 
(Cohen’s d statistic). Effect sizes of d = 0.2, d = 0.5, and d = 0.8 are typically categorized as 
small, medium, and large, respectively. Gray regions represent the regions with no statistically 
significant deviations after FDR correction. 

Figure 3: Brain atlas maps (Desikan-Killiany atlas for 66 cortical regions and Aseg atlas for 
24 subcortical regions) showing the proportion of abnormal deviations for each region in ADNI 
(3A) and Knight ADRC (3B). The figures from left to right indicate the brain maps 
corresponding to MRI, amyloid and tau respectively. The color bar represents the proportion 
of abnormal deviations of each region from 0 to 100%. Gray represents that no participants 
have abnormal deviations for that region. 

Figure 4: Hamming distance density (KDE plot) which illustrates the spread of dissimilarity 
in abnormality patterns (calculated by the Hamming distance for all modalities or 
hamming_all; see Section 2.5.4) within each CDR group for ADNI (4A) and Knight ADRC 
(4B). Higher hamming distance values indicated intra-group more heterogeneity in 
abnormality patterns. 

Figure 5: Box plot showing DSI_all (DSI across all modalities; see Section 2.5.5) for both 
ADNI (5A) and Knight ADRC (5B). The x-axis shows the different CDR groups in the ADS 
and CU-test (Section 2.2.1 and 2.3.1). FDR-corrected post hoc Tukey comparisons used to 
assess pairwise group differences. Abbreviations: DSI: Disease Severity Index, CDR = Clinical 
Dementia Rating. Statistical annotations: ns: not significant 0.05 < p <= 1, * 0.01 < p <= 0.05, 
** 0.001 < p < 0.01, *** p < 0.001. 

Figure 6: Kaplan-Meier plot of conversion from CDR < 1 to CDR >=1 for ADNI-ADS (6A) 
and ADRC-ADS (6B) participants. The x-axis and the y-axis represent the follow-up period 
(in months) and the probability of progressing from CDR <1 to CDR >= 1 respectively. The 
four lines represent the four quantiles of DSI_all (DSI across all modalities), shown by blue, 
red, green and orange respectively. The filled color span represents the 95% confidence 
intervals.

## Results

[31] Landau SM, Breault C, Joshi AD, Pontecorvo M, Mathis CA, Jagust WJ, et al. 
Amyloid-β imaging with Pittsburgh compound B and florbetapir: comparing 
radiotracers and quantification methods. J Nucl Med 2013;54:70–7. 
[32] Landau SM, Thomas BA, Thurfjell L, Schmidt M, Margolin R, Mintun M, et al. 
Amyloid PET imaging in Alzheimer’s disease: a comparison of three radiotracers. Eur J 
Nucl Med Mol Imaging 2014;41:1398–407. 
[33] Clark CM, Schneider JA, Bedell BJ, Beach TG, Bilker WB, Mintun MA, et al. Use of 
florbetapir-PET for imaging β-amyloid pathology. Jama 2011;305:275–83. 
[34] Joshi AD, Pontecorvo MJ, Clark CM, Carpenter AP, Jennings DL, Sadowsky CH, et al. 
Performance characteristics of amyloid PET with florbetapir F 18 in patients with 
Alzheimer’s disease and cognitively normal subjects. J Nucl Med 2012;53:378–84. 
[35] Su Y, Flores S, Hornbeck RC, Speidel B, Vlassenko AG, Gordon BA, et al. Utilizing 
the Centiloid scale in cross-sectional and longitudinal PiB PET studies. NeuroImage 
Clin 2018;19:406–16. 
[36] Morris JC. The Clinical Dementia Rating (CDR): current version and scoring rules. 
Neurology 1993. 
[37] Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful 
approach to multiple testing. J R Stat Soc Ser B Methodol 1995;57:289–300. 
[38] Mowinckel AM, Vidal-Piñeiro D. Visualization of brain statistics with R packages 
ggseg and ggseg3d. Adv Methods Pract Psychol Sci 2020;3:466–83. 
[39] Davatzikos C, Xu F, An Y, Fan Y, Resnick SM. Longitudinal progression of 
Alzheimer’s-like patterns of atrophy in normal older adults: the SPARE-AD index. 
Brain 2009;132:2026–35. https://doi.org/10.1093/brain/awp091. 

[40] Davatzikos C, Fan Y, Wu X, Shen D, Resnick SM. Detection of prodromal Alzheimer’s 
disease via pattern classification of magnetic resonance imaging. Neurobiol Aging 
2008;29:514–23. https://doi.org/10.1016/j.neurobiolaging.2006.11.010. 
[41] Wong DF, Rosenberg PB, Zhou Y, Kumar A, Raymont V, Ravert HT, et al. In vivo 
imaging of amyloid deposition in Alzheimer disease using the radioligand 18F-AV-45 
(flobetapir F 18). J Nucl Med 2010;51:913–20. 
[42] Levitis E, Vogel JW, Funck T, Hachinski V, Gauthier S, Vöglein J, et al. Differentiating 
amyloid beta spread in autosomal dominant and sporadic Alzheimer’s disease. Brain 
Commun 2022;4:fcac085. 
[43] Palmqvist S, Schöll M, Strandberg O, Mattsson N, Stomrud E, Zetterberg H, et al. 
Earliest accumulation of β-amyloid occurs within the default-mode network and 
concurrently affects brain connectivity. Nat Commun 2017;8:1214. 
[44] Mishra S, Gordon BA, Su Y, Christensen J, Friedrichsen K, Jackson K, et al. AV-1451 
PET imaging of tau pathology in preclinical Alzheimer disease: defining a summary 
measure. Neuroimage 2017;161:171–8. 
[45] Vogel JW, Young AL, Oxtoby NP, Smith R, Ossenkoppele R, Strandberg OT, et al. 
Four distinct trajectories of tau deposition identified in Alzheimer’s disease. Nat Med 
2021;27:871–81. 
[46] Aksman LM, Oxtoby NP, Scelsi MA, Wijeratne PA, Young AL, Alves IL, et al. A data-
driven study of Alzheimer’s disease related amyloid and tau pathology progression. 
Brain 2023;146:4935–48. 
[47] Dong A, Toledo JB, Honnorat N, Doshi J, Varol E, Sotiras A, et al. Heterogeneity of 
neuroanatomical patterns in prodromal Alzheimer’s disease: links to cognition, 
progression and biomarkers. Brain 2017;140:735–47. 
[48] Yang Z, Wen J, Davatzikos C. Smile-GANs: Semi-supervised clustering via GANs for 
dissecting brain disease heterogeneity from medical images. ArXiv Prepr 
ArXiv200615255 2020. 
[49] Dong A, Honnorat N, Gaonkar B, Davatzikos C. CHIMERA: Clustering of 
heterogeneous disease effects via distribution matching of imaging patterns. IEEE Trans 
Med Imaging 2015;35:612–21. 
[50] Poulakis K, Ferreira D, Pereira JB, Smedby Ö, Vemuri P, Westman E. Fully Bayesian 
longitudinal unsupervised learning for the assessment and visualization of AD 
heterogeneity and progression. Aging 2020;12:12622. 
[51] Young AL, Marinescu RV, Oxtoby NP, Bocchetta M, Yong K, Firth NC, et al. 
Uncovering the heterogeneity and temporal complexity of neurodegenerative diseases 
with Subtype and Stage Inference. Nat Commun 2018;9:4273. 
[52] Varol E, Sotiras A, Davatzikos C, Initiative ADN. HYDRA: Revealing heterogeneity of 
imaging and genetic patterns through a multiple max-margin discriminative analysis 
framework. Neuroimage 2017;145:346–64. 
[53] Zhang X, Mormino EC, Sun N, Sperling RA, Sabuncu MR, Yeo BT, et al. Bayesian 
model reveals latent atrophy factors with dissociable cognitive trajectories in 
Alzheimer’s disease. Proc Natl Acad Sci 2016;113:E6535–44. 
[54] Lee HJ, Lee E-C, Seo S, Ko K-P, Kang JM, Kim W-R, et al. Identification of 
heterogeneous subtypes of mild cognitive impairment using cluster analyses based on 
PET imaging of tau and astrogliosis. Front Aging Neurosci 2021;12:615467. 
[55] Toledo JB, Liu H, Grothe MJ, Rashid T, Launer L, Shaw LM, et al. Disentangling tau 
and brain atrophy cluster heterogeneity across the Alzheimer’s disease continuum. 
Alzheimers Dement Transl Res Clin Interv 2022;8:e12305. 

[56] Sun Y, Zhao Y, Hu K, Wang M, Liu Y, Liu B, et al. Distinct spatiotemporal subtypes of 
amyloid deposition are associated with diverging disease profiles in cognitively normal 
and mild cognitive impairment individuals. Transl Psychiatry 2023;13:35. 
[57] McCullough AA, Gordon BA, Christensen J, Dincer A, Keefe S, Flores S, et al. P3‐401: 
EXAMINING THE ABILITY OF A TAU SPATIAL SPREAD METRIC TO 
INDICATE DISEASE PROGRESSION COMPARED TO AN INTENSITY‐BASED

## Discussion

In this study, we applied a deep learning based normative modeling framework across multiple 
neuroimaging modalities to assess heterogeneity in neuroanatomical and neuropathological 
changes in the brain of individuals with AD. Results showed evidence of (i) heterogeneous 
patterns of abnormal deviations in regional volumetric measurements as well as amyloid and 
tau deposition between patients with AD; (ii) increased dissimilarity in spatial patterns of 
abnormal deviations for AD patients at more severe dementia stages; (iii) associations of DSI, 
which distils spatial patterns of abnormal deviations across multiple modalities in a single 
index for each subject, with cognitive performance, as well as (iv) associations of DSI with 
increased risk of disease progression. Our observations were reproducible in both the discovery 
and replication datasets, which demonstrated the generalizability of our scientific findings.

## Conclusion

In this paper, we assessed the heterogeneity in AD through the lens of multiple neuroimaging 
modalities by estimating 
regional statistically significant 
neurodegenerative 
and 
neuropathological deviations at the individual level. We studied these subject-specific maps of 
regional abnormal deviations across gray matter volume, amyloid burden and tau deposition 
and observed higher variability in the spatial patterns of MRI atrophy compared to amyloid 
and tau burden. Additionally, we showed higher within-group heterogeneity for ADS patients 
at increased dementia stages. Lastly, we developed an individualized metric of brain health that 
summarizes the extent and severity of neurodegeneration and neuropathology. Together the 
individualized disease severity index and the subject-specific maps of abnormal deviations 
have the potential to assist in clinical decision making and monitor patient response to anti-
amyloid treatments. Our results were reproducible in both the discovery and replication 
datasets, demonstrating the generalizability of our findings.

## References

[1] Kumar S, Oh I, Schindler S, Lai AM, Payne PR, Gupta A. Machine learning for 
modeling the progression of Alzheimer disease dementia using clinical data: a 
systematic literature review. JAMIA Open 2021;4:ooab052. 
[2] Richards M, Brayne C. What do we mean by Alzheimer’s disease? BMJ 2010;341. 
[3] Jack CR, Knopman DS, Jagust WJ, Shaw LM, Aisen PS, Weiner MW, et al. Hypothetical 
model of dynamic biomarkers of the Alzheimer’s pathological cascade. Lancet Neurol 
2010;9:119–28. 
[4] Verdi S, Marquand AF, Schott JM, Cole JH. Beyond the average patient: how 
neuroimaging models can address heterogeneity in dementia. Brain 2021;144:2946–53. 

[5] Habes M, Grothe MJ, Tunc B, McMillan C, Wolk DA, Davatzikos C. Disentangling 
heterogeneity in Alzheimer’s disease and related dementias using data-driven methods. 
Biol Psychiatry 2020;88:70–82. 
[6] Kia SM, Marquand AF. Neural processes mixed-effect models for deep normative 
modeling of clinical neuroimaging data. Int. Conf. Med. Imaging Deep Learn., PMLR; 
2019, p. 297–314. 
[7] Marquand AF, Rezek I, Buitelaar J, Beckmann CF. Understanding heterogeneity in 
clinical cohorts using normative models: beyond case-control studies. Biol Psychiatry 
2016;80:552–61. 
[8] Verdi S, Kia SM, Yong KX, Tosun D, Schott JM, Marquand AF, et al. Revealing 
individual neuroanatomical heterogeneity in Alzheimer disease using neuroanatomical 
normative modeling. Neurology 2023;100:e2442–53. 
[9] Loreto F, Verdi S, Kia SM, Duvnjak A, Hakeem H, Fitzgerald A, et al. Examining real-
world Alzheimer’s disease heterogeneity using neuroanatomical normative modelling. 
medRxiv 2022:2022.11. 02.22281597. 
[10] Earnest T, Bani A, Ha SM, Hobbs DA, Kothapalli D, Yang B, et al. Data‐driven 
decomposition and staging of flortaucipir uptake in Alzheimer’s disease. Alzheimers 
Dement 2024. 
[11] Lee WJ, Brown JA, Kim HR, La Joie R, Cho H, Lyoo CH, et al. Regional Aβ-tau 
interactions promote onset and acceleration of Alzheimer’s disease tau spreading. 
Neuron 2022;110:1932-1943. e5. 
[12] Pinaya WH, Scarpazza C, Garcia-Dias R, Vieira S, Baecker L, F da Costa P, et al. Using 
normative modelling to detect disease progression in mild cognitive impairment and 
Alzheimer’s disease in a cross-sectional multi-cohort study. Sci Rep 2021;11:1–13. 
[13] Pinaya WH, Mechelli A, Sato JR. Using deep autoencoders to identify abnormal brain 
structural patterns in neuropsychiatric disorders: A large‐scale multi‐sample study. Hum 
Brain Mapp 2019;40:944–54. 
[14] Lawry Aguila A, Chapman J, Janahi M, Altmann A. Conditional vaes for confound 
removal and normative modelling of neurodegenerative diseases. Int. Conf. Med. Image 
Comput. Comput.-Assist. Interv., Springer; 2022, p. 430–40. 
[15] Fraza CJ, Dinga R, Beckmann CF, Marquand AF. Warped Bayesian linear regression 
for normative modelling of big data. NeuroImage 2021;245:118715. 
[16] Loreto F, Verdi S, Kia SM, Duvnjak A, Hakeem H, Fitzgerald A, et al. Alzheimer’s 
disease heterogeneity revealed by neuroanatomical normative modeling. Alzheimers 
Dement Diagn Assess Dis Monit 2024;16:e12559. 
[17] Jack CR Jr, Wiste HJ, Therneau TM, Weigand SD, Knopman DS, Mielke MM, et al. 
Associations of Amyloid, Tau, and Neurodegeneration Biomarker Profiles With Rates 
of Memory Decline Among Individuals Without Dementia. JAMA 2019;321:2316–25. 
https://doi.org/10.1001/jama.2019.7437. 
[18] Aschenbrenner AJ, Gordon BA, Benzinger TL, Morris JC, Hassenstab JJ. Influence of 
tau PET, amyloid PET, and hippocampal volume on cognition in Alzheimer disease. 
Neurology 2018;91:e859–66. 
[19] Ebenau JL, Timmers T, Wesselman LMP, Verberk IMW, Verfaillie SCJ, Slot RER, et 
al. ATN classification and clinical progression in subjective cognitive decline. 
Neurology 2020;95:e46–58. https://doi.org/10.1212/WNL.0000000000009724. 
[20] Ezzati A, Abdulkadir A, Jack Jr. CR, Thompson PM, Harvey DJ, Truelove-Hill M, et al. 
Predictive value of ATN biomarker profiles in estimating disease progression in 
Alzheimer’s disease dementia. Alzheimers Dement 2021;17:1855–67. 
https://doi.org/10.1002/alz.12491. 

[21] Peretti DE, Ribaldi F, Scheffler M, Chicherio C, Frisoni GB, Garibotto V. Prognostic 
value of imaging-based ATN profiles in a memory clinic cohort. Eur J Nucl Med Mol 
Imaging 2023;50:3313–23. https://doi.org/10.1007/s00259-023-06311-3. 
[22] Kumar S, Payne P, Sotiras A. Improving Normative Modeling for Multi-modal 
Neuroimaging Data using mixture-of-product-of-experts variational autoencoders. 
ArXiv Prepr ArXiv231200992 2023. 
[23] Lawry Aguila A, Chapman J, Altmann A. Multi-modal Variational Autoencoders for 
normative modelling across multiple imaging modalities. Int. Conf. Med. Image 
Comput. Comput.-Assist. Interv., Springer; 2023, p. 425–34. 
[24] Kumar S, Payne PR, Sotiras A. Normative modeling using multimodal variational 
autoencoders to identify abnormal brain volume deviations in Alzheimer’s disease. 
Med. Imaging 2023 Comput.-Aided Diagn., vol. 12465, SPIE; 2023, p. 1246503. 
[25] Revised Criteria for Diagnosis and Staging of Alzheimer’s | AAIC. Revis Criteria Diagn 
Staging Alzheimers AAIC n.d. https://aaic.alz.org/diagnostic-criteria.asp (accessed June 
18, 2024). 
[26] Van Der Flier WM, Scheltens P. The ATN framework—moving preclinical Alzheimer 
disease to clinical relevance. JAMA Neurol 2022;79:968–70. 
[27] Desikan RS, Ségonne F, Fischl B, Quinn BT, Dickerson BC, Blacker D, et al. An 
automated labeling system for subdividing the human cerebral cortex on MRI scans into 
gyral based regions of interest. Neuroimage 2006;31:968–80. 
[28] Fischl B, Salat DH, Busa E, Albert M, Dieterich M, Haselgrove C, et al. Whole brain 
segmentation: automated labeling of neuroanatomical structures in the human brain. 
Neuron 2002;33:341–55. 
[29] Su Y, Blazey TM, Snyder AZ, Raichle ME, Marcus DS, Ances BM, et al. Partial 
volume correction in quantitative amyloid imaging. Neuroimage 2015;107:55–64. 
[30] Su Y, D’Angelo GM, Vlassenko AG, Zhou G, Snyder AZ, Marcus DS, et al.

## Preamble

Analyzing heterogeneity in Alzheimer Disease using multimodal 
normative modeling on imaging-based ATN biomarkers 
Sayantan Kumar a,b,* , Tom Earnest c, Braden Yang c, Deydeep Kothapalli c, Andrew J. 
Aschenbrenner d, Jason Hassenstab d, Chengie Xiong b, Beau Ances d, John Morris d, 
Tammie L. S. Benzinger c, Brian A. Gordon c, Philip Payne a,b, Aristeidis Sotiras b,c, for the 
Alzheimer’s Disease Neuroimaging Initiative† 
a Department of Computer Science and Engineering, Washington University in St Louis; 1 
Brookings Drive, Saint Louis, MO 63130 
b Institute for Informatics, Data Science & Biostatistics, Washington University School of 
Medicine in St Louis; 660 S. Euclid Ave, Campus Box 8132, Saint Louis, MO 63110 
c Mallinckrodt Institute of Radiology, Washington University School of Medicine in St 
Louis; 4525 Scott Ave, Saint Louis, MO 63110 
d Department of Neurology, Washington University School of Medicine, 660 S Euclid Ave, 
Campus Box 8111, St louis, MO 63110 
*Corresponding author: 
§ sayantan.kumar@wustl.edu, 660 S. Euclid Ave, Campus Box 8132, Saint Louis, MO 63110 

† Data used in preparation of this article were obtained from the Alzheimer’s Disease 
Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators 
within the ADNI contributed to the design and implementation of ADNI and/or provided data 
but did not participate in analysis or writing of this report. A complete listing of ADNI 
investigators can be found at: 
http://adni.loni.usc.edu/wpcontent/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf
