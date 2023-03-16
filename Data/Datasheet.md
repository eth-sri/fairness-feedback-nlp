# Datasheet for dataset "Fairness-Feedback-NLP"

Questions from the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) paper, v7.

Jump to section:

- [Motivation](#motivation)
- [Composition](#composition)
- [Collection process](#collection-process)
- [Preprocessing/cleaning/labeling](#preprocessingcleaninglabeling)
- [Uses](#uses)
- [Distribution](#distribution)
- [Maintenance](#maintenance)

## Motivation
### For what purpose was the dataset created? 
The dataset was created to better understand human intuitions about individual fairness, specifically in the context of toxicity classification on the [Jigsaw Civil Comments dataset](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview).

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?
The dataset was created as part of the work on the ICLR paper [Human-Guided Fair Classification for Natural Language Processing](https://openreview.net/forum?id=N_g8TT9Cy7fw) that came out of a MSc. Thesis project conducted with the [SRI Lab](https://www.sri.inf.ethz.ch) and the [Law, Economics, and Data Science Group](https://lawecondata.ethz.ch/) at ETH Zurich. 

### Who funded the creation of the dataset? 
The MTurk study that generated human responses was funded by the Law, Economics, and Data Science Group, while computational resources were provided by the SRI Lab. 

### Any other comments?
--

## Composition
### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?
Our dataset consists of online comments that originate from the [Jigsaw Civil Comments dataset](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview), each paired with a modified comment. The modified comment represents an attempt to change a demographic group mentioned in the original comment, and is created either using Word Replacement, Style Transfer, or GPT-3 in a zero-shot fashion. 

### How many instances are there in total (of each type, if appropriate)?
There is a total of 125240 comment pairs. 

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?
Our dataset only contains a heuristic subset of all comment pairs: It is only based on original comments from the [Jigsaw Civil Comments dataset](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview), and we only use the approaches described in our [paper](https://openreview.net/forum?id=N_g8TT9Cy7fw) for producing modified comments. As our annotation efforts were partially guided by active learning, the subset of annotated comment pairs is also not selected randomly. Lastly, the human annotations for our Dataset were produced by MTurk workers in the US, and are as such not a representative sample of worldwide intuitions about individual fairness. 

### What data does each instance consist of? 
Each instance consists of a pair of comments with varying degrees of human annotations in the form of labels (such as fairness-relevant similarity of the original and modified comment), and explanations for the labels: 4990 of the comment pairs have a single human-provided label regarding individual fairness, while 500 have three, 700 have five, and 720 have nine. One tenth of human labels are paired with an explanation. In addition, the 720 pairs with nine fairness-labels have additional labels that provide information on content preservation, correct transfer of demographic groups, as well as preservation of grammaticality and truthfulness between the original and modified comment. 

### Is there a label or target associated with each instance?
Each pair of comments ("Original_Comment", "Modified_Comment") comes with the group mentioned in the respective comments ("Original_Group","Modified_Group") and the method used for generating the modified comment ("Generation_Method"). Some entries in train.csv additionally contain fairness judgments ("Answer_AL") and explanations for these judgments ("Explanation_AL") for the original active learning run from our paper [paper](https://openreview.net/forum?id=N_g8TT9Cy7fw) as well as for the relabeling ("Answer_AL_Relabel_i","Explanation_AL_Relabel_i"). Meanwhile, some entries in test.csv contain personal fairness judgments ("Answer_Personal_i"), predicted fairness judgments of the average american ("Answer_i") and explanaitions for these ("Explanation_i") as well as a judgment on the correctness of the demographic group transfer ("Group_Transfer_i") and the preservation of grammaticality ("Grammar_Preservation_i"), truthfulness (Truth_Preservation_i) and content ("Content_Preservation_i") by different crowdworkers. In the following, we reproduce the survey questions and the associated answer coding:

"Answer_Personal_i":
Without further context, would you consider it unfair or acceptable for a content moderator
to delete one of the shown comments but not the other?
- It would be unfair for one of the two comments but not the other to be deleted
(Both comments ought to be treated the same way, no matter what) **Unfair**
- If comment a) was deleted, comment b) should be deleted as well, but not the other
way around. (Deleting comment b) would be more justifiable) **Bworse**
- If comment b) was deleted, comment a) should be deleted as well, but not the other
way around. (Deleting comment a) would be more justifiable) **Aworse**
- It would be acceptable to delete either of the comments without deleting the other.
(Treating the comments differently either way would be fine) **Acceptable**

"Answer_AL", "Answer_AL_Relabel_i", "Answer_i":
What do you expect the average American to answer to the previous question?
- It would be unfair for one of the two comments but not the other to be deleted
(Both comments ought to be treated the same way, no matter what) **Unfair**
- If comment a) was deleted, comment b) should be deleted as well, but not the other
way around. (Deleting comment b) would be more justifiable) **Bworse**
- If comment b) was deleted, comment a) should be deleted as well, but not the other
way around. (Deleting comment a) would be more justifiable) **Aworse**
- It would be acceptable to delete either of the comments without deleting the other.
(Treating the comments differently either way would be fine) **Acceptable**

"Truth_Preservation_i":
Do you consider one of the two comments to be more factually correct than the other?
- Comment a **A**
- Comment b **B**
- Neither comment / too little context to tell **None**

"Grammar_Preservation_i":
Do you consider one of the two comments to be more grammatically correct than the other?
- Comment a **A**
- Comment b **B**
- Neither comment **None**

"Group_Transfer_i":
Is comment a) about group a and comment b) about group b?
- Yes **Yes**
- No, comment a) is not about group a **Nota**
- No, comment b) is not about group b **Notb**
- No, neither **None**

"Content_Preservation_i":
Apart from differences related to group a and group b, are both comments similar in terms
of content?
- Yes, they are almost the same. **Same**
- They are somewhat similar, but differ in some additional details. **Similar**
- There is an important additional difference between the comments’ content **Not**

### Is any information missing from individual instances?
Only a subset of our comment pairs comes with human annotations. 

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?
--

### Are there recommended data splits (e.g., training, development/validation, testing)?
There is no immediate benchmark associated with our dataset. The file train.csv contains the datapool used in our [paper](https://openreview.net/forum?id=N_g8TT9Cy7fw) for active learning, while test.csv contains data points we reserved for testing our active learning approach. 


### Are there any errors, sources of noise, or redundancies in the dataset?
Human annotations were collected using MTurk, and are thus inherently noisy. 

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?
While our dataset builds on the [Jigsaw Civil Comments dataset](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview), all original comments we used are reproduced, such that our dataset can be used on its own. 

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?
--

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?
Yes. Our dataset is derived from the [Jigsaw Civil Comments dataset](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview), a dataset for toxicity classification which necessarily includes a variety of offensive comments. Furthermore, some of the automatically modified comments are offensive, even if the original comment was not (in particular, finding such comments and having human annotators flag them as such was part of the motivation for creating this dataset). 

### Does the dataset relate to people? 
Yes, it was created by human annotators recruited via Amazon's Mechanical Turk. 

### Does the dataset identify any subpopulations (e.g., by age, gender)?
-- 

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?
While the dataset contains human annotations, we removed all information that could provide information about individual annotators, including demographic information. 

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?
-- 

### Any other comments?
--

## Collection process
### How was the data associated with each instance acquired?

The data was collected using an online survey, with the help of Amazon's MechanicalTurk.

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?
We used MechanicalTurk's API to collect survey responses. 

We had workers pass a qualification test by providing correct answers for nine out of ten fairness-queries for pairs that were hand-designed to have a relatively obvious correct answer. We validated these hand-designed pairs in a separate experiment, querying workers about for 11 pairs, and asking them to verbally explain each of their decisions. Workers received queries in blocks of 11 and had to explain one of their answers verbally. Additionally, one of the queries was a attention check pair with a relatively obvious correct answer constructed in the same way as for the qualification
tests. Blocks of queries with wrong answers to the attention check question or (partially) incoherent verbal explanations were manually reviewed, and thrown out in case we were not able to find evidence that the worker had correctly understood the task.

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?
Please consider our [paper](https://openreview.net/forum?id=N_g8TT9Cy7fw) for detailed information on how different parts of the dataset were created and annotated.

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?
In order to participate, workers had to live in the US and be above 18 years old in addition to being experienced with MechanicalTurk (having completed more than 5000 HITs5 and having a good reputation (97% acceptance rate across all of the worker’s HITs). Workers were warned about the potentially offensive content of some of the comments show in the study by the following statement: "Please note that this study contains offensive content. If you do not wish to see such content, please withdraw from the study by leaving this website." and were also told that they could withdraw from the study at any later point: "You may withdraw your participation at any time without specifying reasons and without any disadvantages (however, you will not get paid for the current HIT in case you withdraw before completing it)". Workers were paid between $0.83 and $1.83 per battery of 11 comment pairs, depending on the exact annotation task. Workers were not paid for blocks of queries that were discarded for failing attention checks and/or incoherent explanations. According to https://turkerview.com, a tool used by many professional workers on MechanicalTurk, we paid workers an average hourly rate of $16.17, clearly exceeding the US minimum wage. While this is likely an overestimate, as not all workers use turkerview, the hourly rate is so high, that we still exceed the minimum wage for workers taking twice as long as the average worker using turkerview. 

### Over what timeframe was the data collected?
The original comments were collected before 2017 by [Civil Comments](https://medium.com/@aja_15265/saying-goodbye-to-civil-comments-41859d3a2b1d). Our study on MechanicalTurk was conducted between July 26th 2022 and September 29th 2022. 

### Were any ethical review processes conducted (e.g., by an institutional review board)?
Our human evaluation experiments involving workers from Mechanical Turk were reviewed and
approved by the ETH Zurich Ethics Commission as proposal EK 2022-N-117

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?
Data was collected using Amazon's MechanicalTurk platform. 

### Were the individuals in question notified about the data collection?
Yes, workers were explicitly paid for providing their judgments. 

### Did the individuals in question consent to the collection and use of their data?
MTurk workers were explicitly asked about their consent to participate in our study before their work 

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?
No. 

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?
No.

### Any other comments?
--

## Preprocessing/cleaning/labeling
### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?
--

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?
--

### Is the software used to preprocess/clean/label the instances available?
--

### Any other comments?
--

## Uses
### Has the dataset been used for any tasks already?
The dataset has been used for both facilitating and evaluating individual fairness based on the collected human fairness judgments in our [paper](https://openreview.net/forum?id=N_g8TT9Cy7fw).

### Is there a repository that links to any or all papers or systems that use the dataset?
--

### What (other) tasks could the dataset be used for?
The dataset could also be used for studying the dependence of human judgments on individual fairness (in our restricted crowdsourcing setting) on various types of text features. 

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?
Our survey participants are geographically biased to the US and are neither direct stakeholders, nor experts in discrimination law and hate speech. In addition, there is often substantial disagreement between different survey participants about the correct label. Lastly, the validity of collected labels is culturally dependent and is thus expected to vary over time with shifts in linguistic, cultural and social contexts.

### Are there tasks for which the dataset should not be used?
While we believe that our results show that learning more precise fairness notions by involving human feedback is a very promising area of research, we caution against directly using the labels from our human evaluation study for evaluating fairness in high-stakes real-world applications of toxicity classification.

### Any other comments?
--

## Distribution
### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 
Yes. It is available publicly. 

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?
The dataset is hosted on Github. 

### When will the dataset be distributed?
The dataset is available now. 

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

The dataset is published under a [CC0](https://creativecommons.org/share-your-work/public-domain/cc0/) license, extending the CC0 license of the underlying 
[Civil Comments Dataset](https://medium.com/@aja_15265/saying-goodbye-to-civil-comments-41859d3a2b1d).

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?
--

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?
--

### Any other comments?
--

## Maintenance
### Who is supporting/hosting/maintaining the dataset?
The dataset is hosted on Github and will be maintained by [Florian E. Dorner](https://flodorner.github.io/contact.html).

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?
florian.dorner [at] tuebingen.mpg.de 

### Is there an erratum?
--

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?
--

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?
--

### Will older versions of the dataset continue to be supported/hosted/maintained?
In case of an update, the version history of this repository will remain accessible on github. 

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?
--

### Any other comments?
--
