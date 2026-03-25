# DeepAD: A Robust Deep Learning Model of Alzheimer's Disease Progression for Real-World Clinical Applications

## Metadata

- **Authors:** Somaye Hashemifar, Claudia Iriondo, Evan Casey, Mohsen Hejrati, for Alzheimer's Disease Neuroimaging Initiative
- **Year:** 2022
- **Arxiv Id:** 2203.09096v5
- **Url:** http://arxiv.org/abs/2203.09096v5

## Abstract

The ability to predict the future trajectory of a patient is a key step toward the development of therapeutics for complex diseases such as Alzheimer's disease (AD). However, most machine learning approaches developed for prediction of disease progression are either single-task or single-modality models, which can not be directly adopted to our setting involving multi-task learning with high dimensional images. Moreover, most of those approaches are trained on a single dataset (i.e. cohort), which can not be generalized to other cohorts. We propose a novel multimodal multi-task deep learning model to predict AD progression by analyzing longitudinal clinical and neuroimaging data from multiple cohorts. Our proposed model integrates high dimensional MRI features from a 3D convolutional neural network with other data modalities, including clinical and demographic information, to predict the future trajectory of patients. Our model employs an adversarial loss to alleviate the study-specific imaging bias, in particular the inter-study domain shifts. In addition, a Sharpness-Aware Minimization (SAM) optimization technique is applied to further improve model generalization. The proposed model is trained and tested on various datasets in order to evaluate and validate the results. Our results showed that 1) our model yields significant improvement over the baseline models, and 2) models using extracted neuroimaging features from 3D convolutional neural network outperform the same models when applied to MRI-derived volumetric features.

## Introduction

Alzheimer’s disease (AD) is the most common cause
of dementia in people over 65, with 26.6 million people
suffering worldwide [3]. AD is a slowly progressing dis-
ease caused by the degeneration of brain cells, with patients
showing clinical symptoms years after the onset of the dis-
ease. Therefore, accurate diagnosis and treatment of AD
in its early stage, i.e., mild cognitive impairment (MCI), is
critical to prevent non-reversible and fatal brain damage.

## Related Work

Over the past decade, machine learning and deep learn-
ing based approaches, including the support vector ma-
chine, random forest, recurrent neural network (RNN),
and convolutional neural networks (CNN) have been pro-
posed for prognosis, predicting disease progression, mon-
itoring treatment effects and for stratifying AD patients
[16,17,24,25]. A multi-modal GRU-based RNN was used
to integrate longitudinal clinical information and cross-
sectional tabular imaging features for classifying the MCI
patients into converter to AD or not-converter to AD [15].
MinimalRNN [4] employed similar features for regressing
a couple of endpoints and for stratifying patients to cogni-
tively normal (CN), MCI, and AD [18]. An ensemble model
based on stacked CNN and a bidirectional long short-term
memory (BiLSTM) was utilized to jointly predict multiple
endpoints on the fusion of time series clinical features and
Freesurfer derived imaging features [6]. Recently, several
methods have started to employ 3D CNN based models to
extract features from MRIs. A 3D convolutional autoen-
coder (CAE) with transfer learning is employed on MRIs
to stratify the patients into progressor and non-progressor
groups [19]. A stacked denoising auto-encoder approach
was used to extract features from clinical and genetic data,
and a 3D CNN for MRIs to categorize patients into different
stages of the disease [23].
Previous research has focused on developing single-task
and/or single modality models such as predicting MCI to
AD conversion [15], classification into AD stages [21,23] ,
or predicting one or few cognitive endpoints [6,18], which
is not applicable to personalized medicine for AD. Besides,
single-task and single-modalities models exploit neither the
complementary information among modalities nor the cor-
relation between tasks.

## Methodology

CDRSB
MMSE
ADAS-COG12
SMST
DeepAD-MRI - rand init
0.0345 (0.01960)
0.0395 (0.02410)
0.0452 (0.00852)
DeepAD-MRI - pretrained
0.1550 (0.00301)
0.1660 (0.00879)
0.1300 (0.00206)
SMMT
DeepAD-MRI - rand init
0.1310 (0.00876)
0.1090 (0.02620)
0.1080 (0.02900)
DeepAD-MRI - pretrained
0.1610 (0.00315)
0.1190 (0.01030)
0.1550 (0.00844)
MMST
DeepAD-MRI+Clin
0.2600 (0.00629)
0.2280 (0.00522)
0.1940 (0.01320)
MMMT
DeepAD-MRI+Clin
0.2470 (0.00756)
0.2010 (0.01040)
0.2170 (0.00740)
10

## Experiments

A.1. Stratification of predicted cognitive trajecto-
ries
The patients in the out-study test set were divided into
tertiles based on predicted interpolated CDRSB. Tertile 3
(greatest predicted interpolated CDRSB) was associated
with faster disease progression compared to other tertiles.
Figures 5 and 6 respectively present the plots for the out-
study test when DeepAD(Clin+MRI) and DeepAD(MRI)
models are used for inference. Stratified analysis demon-
strates DeepAD predictions for interpolated CDRSB at 12
months (time=4) separates patients with different rates of
progression up to 24 months (time=8).

## Results

puts and tasks for 5 random seeds, reported as mean (stan-
dard deviation) weighted R2 for interpolated CDRSB, in-
terpolated MMSE, and interpolated ADAS-COG12.

## Conclusion

We propose DeepAD, a multimodal multi-task deep
learning approach to predict the progression of Alzheimer’s
Disease in terms of different endpoints by using longitudi-
nal clinical features and raw neuroimaging data. We ap-
ply a 3D convolutional neural network to extract the spa-
tiotemporal features of MR images and then integrate those
features with other information sources. In order to alle-
viate inter-study domain shift and improve generalization,
DeepAD utilizes an adversarial loss and sharpness-aware-
minimization. Our result show an improvement in predic-
tion accuracy and more robust prediction performance for
patients in early stages of Alzheimer’s disease. The pro-
posed multimodal multi-task deep learning approach has
7

potential to identify patients at higher risk of progressing
to AD and help develop better therapies at lower cost to so-
ciety.
7. Acknowledgements
We would like to thank all of the study participants and
their families, and all of the site investigators, study coor-
dinators, and staff. Assistance in preparing this article for
publication was provided by Genentech, Inc.
Part of data collection and sharing for this project was
funded by the ADNI (National Institutes of Health Grant
U01 AG024904). ADNI is funded by the National Insti-
tute on Aging (NIA), the National Institute of Biomedical
Imaging and Bioengineering (NIBIB), and through gener-
ous contributions from the following: Alzheimer’s Associa-
tion; Alzheimer’s Drug Discovery Foundation; BioClinica;
Biogen Idec; Bristol-Myers Squibb Company; Eisai; Elan
Pharmaceuticals; Eli Lilly and Company; F. Hoffmann-
La Roche and its affiliated company Genentech, Inc.; GE
Healthcare; Innogenetics NV; IXICO; Janssen Alzheimer
Immunotherapy Research & Development, LLC.; John-
son & Johnson Pharmaceutical Research & Development
LLC.; Medpace; Merck & Co.; Meso Scale Diagnostics,
LLC.; NeuroRx Research; Novartis Pharmaceuticals Cor-
poration; Pfizer; Piramal Imaging; Servier; Synarc; and
Takeda Pharmaceutical Company. The Canadian Institutes
of Health Research is providing funds to support ADNI
clinical sites in Canada.
Private sector contributions are
facilitated by the Foundation for the National Institutes of
Health (www.fnih.org).
The grantee organization is the
Northern California Institute for Research and Education,
and the study is coordinated by the Alzheimer’s Disease Co-
operative Study at the University of California, San Diego.
ADNI data are disseminated by the Laboratory for Neu-
roImaging at the University of California, Los Angeles
8

## References

[1] Ane Alberdi, Asier Aztiria, and Adrian Basarab. On the early
diagnosis of alzheimer’s disease from multimodal signals: A
survey. Artificial intelligence in medicine, 71:1–29, 2016. 1
[2] Benjamin Billot, Douglas N Greve, Oula Puonti, Axel
Thielscher, Koen Van Leemput, Bruce Fischl, Adrian V
Dalca, and Juan Eugenio Iglesias. Synthseg: Domain ran-
domisation for segmentation of brain mri scans of any con-
trast and resolution. arXiv preprint arXiv:2107.09559, 2021.
6
[3] Ron Brookmeyer, Elizabeth Johnson, Kathryn Ziegler-
Graham, and H Michael Arrighi.
Forecasting the global
burden of alzheimer’s disease.
Alzheimer’s & dementia,
3(3):186–191, 2007. 1
[4] Minmin Chen.
Minimalrnn: Toward more interpretable
and trainable recurrent neural networks.
arXiv preprint
arXiv:1711.06788, 2017. 3
[5] Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya
Sutskever, and Pieter Abbeel. Infogan: Interpretable repre-
sentation learning by information maximizing generative ad-
versarial nets. In Proceedings of the 30th International Con-
ference on Neural Information Processing Systems, pages
2180–2188, 2016. 4
[6] Shaker El-Sappagh, Tamer Abuhmed, SM Riazul Islam, and
Kyung Sup Kwak.
Multimodal multitask deep learning
model for alzheimer’s disease progression detection based
on time series data. Neurocomputing, 412:197–215, 2020. 3
[7] Bruce Fischl. Freesurfer. Neuroimage, 62(2):774–781, 2012.
6
[8] Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam
Neyshabur.
Sharpness-aware minimization for efficiently
improving generalization. arXiv preprint arXiv:2010.01412,
2020. 2, 3
[9] Yaroslav Ganin and Victor Lempitsky. Unsupervised domain
adaptation by backpropagation. In International conference
on machine learning, pages 1180–1189. PMLR, 2015. 2, 3
[10] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pas-
cal Germain, Hugo Larochelle, Franc¸ois Laviolette, Mario
Marchand, and Victor Lempitsky. Domain-adversarial train-
ing of neural networks.
The journal of machine learning
research, 17(1):2096–2030, 2016. 3, 4
[11] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing
Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and
Yoshua Bengio. Generative adversarial nets. Advances in
neural information processing systems, 27, 2014. 4
[12] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kil-
ian Q Weinberger.
Densely connected convolutional net-
works. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pages 4700–4708, 2017. 3
[13] Mohammad Khanahmadi, Dariush D Farhud, and Maryam
Malmir. Genetic of alzheimer’s disease: A narrative review
article. Iranian journal of public health, 44(7):892, 2015. 2
[14] Byungju Kim, Hyunwoo Kim, Kyungsu Kim, Sungjin Kim,
and Junmo Kim. Learning not to learn: Training deep neural
networks with biased data, 2019. 2, 3, 4
[15] Garam Lee, Kwangsik Nho, Byungkon Kang, Kyung-Ah
Sohn, and Dokyoon Kim.
Predicting alzheimer’s disease
progression using multi-modal deep learning approach. Sci-
entific reports, 9(1):1–12, 2019. 3
[16] PJ Moore, TJ Lyons, John Gallacher, and Alzheimer’s Dis-
ease Neuroimaging Initiative. Random forest prediction of
alzheimer’s disease using pairwise selection from time series
data. PloS one, 14(2):e0211558, 2019. 3
[17] Elaheh Moradi, Antonietta Pepe, Christian Gaser, Heikki
Huttunen, Jussi Tohka, Alzheimer’s Disease Neuroimag-
ing Initiative, et al. Machine learning framework for early
mri-based alzheimer’s conversion prediction in mci subjects.
Neuroimage, 104:398–412, 2015. 3
[18] Minh Nguyen, Tong He, Lijun An, Daniel C Alexander,
Jiashi Feng, BT Thomas Yeo, Alzheimer’s Disease Neu-
roimaging Initiative, et al.
Predicting alzheimer’s disease
progression using deep recurrent neural networks. NeuroIm-
age, 222:117203, 2020. 3
[19] Kanghan Oh, Young-Chul Chung, Ko Woon Kim, Woo-Sung
Kim, and Il-Seok Oh.
Classification and visualization of
alzheimer’s disease using volumetric convolutional neural
network and transfer learning. Scientific Reports, 9(1):1–16,
2019. 3
[20] L Palumbo, P Bosco, ME Fantacci, E Ferrari, P Oliva, G
Spera, and A Retico.
