from typing import List


# Summary FewShot Template
usr_for_summary: List = [
    "###Human: Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
    "X.W., H.Y., X.L., N.G. and Y.Z. analyzed and interpreted the data."
    "B.G., H.Z. and X.W. drafted the paper. All authors critically revised"
    "the manuscript and approved the final version for submission.\n"
    "Key Points:\n"
    "• We apply both 3D protein structure and sequence information with a coevolutionary strategy.\n"
    "• We addressed the oversmoothing problem in graph representation of compounds.\n"
    "• FeatNN achieved highly enhanced affinity prediction on well-known databases compared with the state-of-the-Supplementary Figure 20\n"
    "available online at http://bib.oxfordjournals.org/)." 
    "The detailed designs of the compound extraction module, protein extraction module and affinity"
    "learning module are described in the Methods and Supplementary"
    "sections available online at http://bib.oxfordjournals.org/."
    "FeatNN outperformed the SOTA model in CPA prediction"
    "To assess the performance of FeatNN, seven kinds of models"
    "mentioned above were trained on the dataset generated from"
    "The data that support the findings of this study are included in"
    "the paper, and further data are available from the corresponding"
    "author upon reasonable request."
    "Code and data availability"
    "Source codes, original data for figures and datasets (generated from PDBbind and BindingDB databases) used to train and test"
    "the models are available at: https://github.com/StarLight1212/"
    "FeatNN. Data in this paper is available at https://drive.google.com/file/d/12Z9AwrAfYto4-2JplLBxo-KGhMQ3314a/view?usp= deviation problem (Supplement Figure 2 available online at http://bib.oxfordjournals.org/); (2) We then clustered the input"
    "compound and protein information according to a certain threshold (0.3, 0.4, 0.5 and 0.6) [9] to avoid the potential data\n"
    "leakage problem that could occur due to data similarities. In this evaluation, we assessed the similarity of the proteins using\n"
    "their multi-sequence alignment (MSA) scores and calculated the \n"
    "Question: Please summarize this paper as best as you can\n"
    "Helpful Answer:###Assistant:",

]

ai_response_for_summary: List = [
            "Basic Information:\n"
            "Title: Protein 3D structure-based neural networks highly improve the accuracy in compound-protein binding affinity prediction\n"
            "Authors: Binjie Guo, Hanyu Zheng, Huan Huang, Haohan Jiang, Xiaodan Li, Naiyu Guan, Yanming Zuo, Yicheng Zhang, Xuhua Wang, Hengfu Yang\n"
            "Affiliation: Department of Neurobiology, Zhejiang University School of Medicine, Hangzhou, Zhejiang Province, China\n"
            "Keywords: compound-protein binding affinity; protein 3D structure; neural networks; deep learning;\n"
            "URLs: https://doi.org/10.1101/2022.04.06.487274, GitHub: None\n"
            "Summarize:\n"
            "a. The research background of this paper:\n"
            "The importance of incorporating protein three-dimensional structure information into compound-protein binding affinity (CPA) prediction is illustrated.\n"
            "b. Past methods, problems and motivations:\n"
            "Traditional structure-free models suffer from low accuracy due to the lack of efficient protein feature encoding. "
            "Structure-based methods, such as docking simulations, rely heavily on high-quality 3D-structure data, which limits their application.\n"
            "c. The research method proposed in this paper:\n"
            "The author developed a new neural network structure named Fast Evolutional Attention and Thoroughgoing-graph Neural Networks (FeatNN), which combines pair interaction and protein three-dimensional structure information to solve the limitations of traditional models.\n"
            "d. The performance of the method in this paper in terms of tasks and performance:\n"
            "FeatNN significantly outperforms various state-of-the-art models in CPA prediction, provides a high-precision CPA prediction method, and facilitates high-throughput virtual screening of drug candidates.\n"            
            "Background:\n"
            "a. Themes and features:\n"
            "The subject of this paper is to use neural network model to predict the affinity of non-covalent interaction between compounds and proteins. This non-covalent interaction is of great significance for high-throughput screening of drugs.\n"
            "b. Historical development:\n"
            "In the past, structural algorithms had high requirements for high-quality protein three-dimensional structure data, and unstructured algorithms lacked effective protein feature encoding, which led to low accuracy.\n"
            "c. Past methods:\n"
            "The past methods are mainly divided into structural algorithms and unstructured algorithms. The structural algorithms include molecular docking simulation and structure-based classification methods, and the unstructured algorithms include rule-based methods and machine learning methods.\n"
            "d. Past research shortcomings:\n"
            "Structure-free models suffer from low accuracy due to the lack of efficient encoding of protein features. Structure-based methods, such as molecular docking simulations, rely heavily on high-quality 3D-structure data, which limits their applications.\n"
            "e. Problems that need to be solved at present:\n"
            "Effective methods are urgently needed to improve the accuracy and speed of prediction in order to speed up the development of new drugs and reduce the cost of trial and error.\n"
            "Method:\n"
            "a. The theoretical basis of this study:\n"
            "In this paper, a neural network model is used to combine protein three-dimensional structure information and pair interaction to improve the accuracy and speed of CPA prediction.\n"
            "b. The technical route proposed in this paper (step by step):\n"
            "This paper uses a neural network model consisting of three modules: compound extractor, protein extractor, and affinity predictor. Compound and protein features are transformed into a compatible space that includes the distances between atoms in the protein, the corresponding twist angles, the sequence of residues in the protein, and the compound features. The protein extractor module aggregates sequence and structural features of proteins to predict affinity; while the compound extractor module uses a deep graph convolution unit and a multi-head attention representation strategy to extract the primary and secondary node information of compounds. The affinity prediction module uses the neuron attention mechanism to fuse the extracted protein and compound features, and predict the affinity value through the end-to-end structure.\n"
            "c. Performance of methods and techniques:\n"
            "The neural network model (Fast Evolutional Attention and Thoroughgoing-graph Neural Networks) proposed in this paper can significantly outperform other state-of-the-art models in predicting CPA, and can be applied in high-throughput virtual screening of drug candidates. The author proves the good performance of this method in CPA prediction by evaluating eight indicators of the model, including R2 score, RMSE, MAE, Pearson and Spearman, etc.\n"
            "Conclusion:\n"
            "a. The meaning of work:\n"
            "FeatNN has good predictive performance and computational efficiency, which provides a powerful tool and technical support for large-scale molecular screening.\n"
            "b. Innovation, Performance and Effort:\n"
            "For the first time, the authors propose an efficient end-to-end neural network model that combines pairwise interaction and protein 3D structure information, achieving high-precision non-covalent interaction prediction, and outperforming previous models.\n"
            "c. Conclusions of the study (list points):\n"
            "FeatNN significantly outperforms other state-of-the-art models in predicting CPA;\n"
            "FeatNN's neuron attention mechanism helps fuse protein and compound features;\n"
            "Our method provides an effective CPA prediction model, which provides support for rapid screening of drug candidates.",
    ]