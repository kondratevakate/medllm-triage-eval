# medllm-triage-eval

Benchmarking Medical Large Language Models (MedLLMs) on multimodal personal health records to evaluate triage accuracy, hallucination rates, and interpretability.

---

## 1. Literature Review

Conducted via:  
1. Google Scholar search for “TRIAGE” & “LLMs”  
2. Deep search in ChatGPT (Scholar AI add-on) & Perplexity  

**Findings:** Perplexity produced 7 nonexistent studies; ChatGPT repeated known results.

| #  | Link                                                                                              | Dataset                             | Comment                                                                                                    |
|---:|---------------------------------------------------------------------------------------------------|-------------------------------------|------------------------------------------------------------------------------------------------------------|
| 1. | https://www.jmir.org/2024/1/e53297/                                                               | Custom 124 cases                    | QWC 0.7 for GPT-4 (Mixtral, Gemini 1.5, Llama worse)                                                      |
| 2. | https://arxiv.org/pdf/2402.13448                                                                  | MIMIC-ED-Assist                     | BioGPT: AUC=0.82 (critical outcome), XGBoost: 0.81                                                          |
| 3. | https://arxiv.org/pdf/2504.16273                                                                  | ESI Handbook, KTAS, MIMIC           | GPT-4+RAG F1=0.677; BioBERT F1=0.640 (unclear over/under triage)                                           |
| 4. | https://www.medrxiv.org/content/10.1101/2024.09.27.24314505v1.full                                 | MIMIC-IV (2k cases)                 | Claude 3.5 (RAG): 65.8% accuracy vs physician 77.0% (accuracy suboptimal for imbalanced data)              |
| 5. | https://aclanthology.org/2023.findings-emnlp.167.pdf                                              | General & arithmetic tasks          | Two-step candidate generation + verification improves accuracy by +0.08                                    |
| 6. | https://www.nature.com/articles/s41598-025-86632-5                                                | Kidney-stone ED reports (500)       | GPT-4: macro-F1=0.83; high-acuity recall=87.8%                                                             |
| 7. | https://files.osf.io/v1/resources/bv5sg_v1/providers/osfstorage/...                              | Survey                              | —                                                                                                          |
| 8. | https://www.sciencedirect.com/science/article/pii/S0735675724007071                               | 1,048 ED admissions                 | ChatGPT & Copilot outperformed nurse triage on ESI-1/2 recall                                             |
| 9. | https://www.sciencedirect.com/science/article/pii/S0735675724007150                               | 392 ambulance requests              | GPT-4 agreed 76.5% groups; unanimous panels 93.8%; majority 68.2%                                         |
| 10.| https://www.tandfonline.com/doi/full/10.1080/10903127.2024.2374400                                | 100 simulated EMT cases             | GPT-3.5 > GPT-4.0                                                                                          |
| 11.| https://www.sciencedirect.com/science/article/pii/S0950705125004782                               | MIMIC-IV Notes & MIETIC             | GPT-4o EP: F1=0.88; Qwen2.5-72B FT: F1=0.91 (ablation across 12 models)                                   |
| 12.| https://pubmed.ncbi.nlm.nih.gov/38713466/                                                         | UC ED 2012–23 (251k visits)         | GPT-4: accuracy=0.89 on pairwise triage acuity                                                             |
| 13.| https://aclanthology.org/2024.findings-emnlp.329.pdf                                              | <1k custom cases                    | TRIAGEAGENT (GPT-4): error-rate ↓ 18.4%                                                                    |
| 14.| https://openreview.net/pdf?id=hFtmNJj4do                                                          | 87 mass-casualty scenarios          | GPT-4 baseline best; explores ethical bias via jailbreak prompts                                          |

---

## 2. Problem Statement

Reproduce and extend SOTA triage findings (ESI prediction) on public open datasets. Focus on:

- **[2504.16273](https://arxiv.org/pdf/2504.16273):** GPT-4 + RAG on MIMIC (F1=0.677)  
- **[0950705125004782](https://www.sciencedirect.com/science/article/pii/S0950705125004782):** Qwen2.5-72B fine-tuned (Acc=0.91)

Main metrics:

- Multiclass macro Accuracy, F1, MSE  
- Over-triage / Under-triage rates  
- Class-wise F1 (especially ESI-2 & ESI-3)  

---

## 3. Available Triage Datasets

| Dataset                                              | Period         | Size                            | Access & Terms                                                                                                                                                                                                                   |
|------------------------------------------------------|----------------|---------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **MIMIC-IV-ED** (ED module of MIMIC-IV)              | 2011–2019      | 448,972 ED stays                | PhysioNet DUA; credentialed; non-commercial; CITI required ([link](https://physionet.org/content/mimic-iv-ed/2.0/), [license](https://physionet.org/content/mimiciv/view-license/0.4/))                                             |
| **MIETIC** (MIMIC-IV-Ext Triage Instruction Corpus)   | 2025 (v1.0)    | 9,629 records                   | PhysioNet DUA; credentialed; non-commercial ([link](https://physionet.org/content/mietic/))                                                                                                                                      |
| **eICU-CRD** (Collaborative Research Database)       | 2014–2015      | 200,859 ICU stays               | PhysioNet DUA; credentialed; non-commercial; CITI required ([Nature](https://www.nature.com/articles/s41597-022-01899-x), [license](https://physionet.org/content/eicu-crd/view-license/2.0/))                                    |
| **Yale ED Triage & Patient History**                 | 2014–2017      | 560,486 ED visits               | Kaggle CC0; public domain; commercial & non-commercial use                                                                                                                                                                        |
| **ESI v4 Handbook Cases** (73 vignettes)             | 2005           | 73                              | AHRQ public domain (US govt work); commercial & non-commercial ([PDF](https://www.sgnor.ch/fileadmin/user_upload/Dokumente/Downloads/Esi_Handbook.pdf))                                                                           |
| **KTAS Simulated Cases**                             | 2020           | 762 cases                       | CC BY-NC 4.0; non-commercial only ([PDF](https://pdfs.semanticscholar.org/...))                                                                                                                                                 |
| **HCUP NEDS** (Nationwide ED Sample)                 | 2006–2022      | ~28 M unweighted ED visits      | HCUP DUA; purchase; research & aggregate only; commercial restricted ([link](https://hcup-us.ahrq.gov/db/nation/neds/NEDS_Introduction_2020.jsp))                                                                                |
| **NEMSIS** (National EMS Info System)                | 2015–present   | ~20 M EMS activations/yr        | Public domain; open commercial & non-commercial ([link](https://nemsis.org/what-is-nemsis/))                                                                                                                                     |

---

## 4. SOTA Reproduction

| #  | Link                                                                         | Model            | Metric                 | Value  |
|---:|------------------------------------------------------------------------------|------------------|------------------------|-------:|
| 3  | https://arxiv.org/pdf/2504.16273                                             | GPT-4 + RAG      | F1                     | 0.677  |
| 5  | https://www.sciencedirect.com/science/article/pii/S0950705125004782           | Qwen2.5-72B FT   | Accuracy               | 0.910  |
| 1  | https://www.jmir.org/2024/1/e53297/                                           | GPT-4            | QWC                    | 0.700  |
| 2  | https://arxiv.org/pdf/2402.13448                                             | BioGPT           | AUC (critical outcome) | 0.820  |
| 4  | https://www.medrxiv.org/content/10.1101/2024.09.27.24314505v1                 | Claude3.5 (RAG)  | Accuracy               | 0.658  |
| 6  | https://www.nature.com/articles/s41598-025-86632-5                           | GPT-4            | Macro-F1               | 0.830  |
| 8  | https://www.sciencedirect.com/science/article/pii/S0735675724007071           | ChatGPT          | Recall (ESI1&2)        | 87.8%  |
| 10 | https://www.tandfonline.com/doi/full/10.1080/10903127.2024.2374400             | GPT-3.5          | Accuracy               | (GPT-3.5 > GPT-4.0) |
| 12 | https://pubmed.ncbi.nlm.nih.gov/38713466/                                     | GPT-4            | Accuracy               | 0.890  |
| 13 | https://aclanthology.org/2024.findings-emnlp.329.pdf                          | TRIAGEAGENT(GPT-4)| Error-rate reduction   | 18.4%  |
| 15 | https://huggingface.co/spaces/m42-health/MEDIC-Benchmark                      | O3 HealthBench   | Referral score         | 0.600  |

---

## 5. Baseline & Embedding Experiments

### 5.1 Baseline Logistic Regression (numerical features)

| Model  | P    | R    | F1   | HR   | Mod. F1 | NP   | OT   | UT   | ER   |
|:-------|:----:|:----:|:----:|:-----|:-------:|:----:|:----:|:----:|:----:|
| LogReg | 0.64 | 0.60 | 0.50 | 0.53 |  0.63   | 0.66 | 0.74 | 0.18 | 0.19 |

### 5.2 BGE-M3 + MLP (serialized cases)

| Model         | P    | R    | F1   | HR   | Mod. F1 | NP   | OT   | UT   | ER   |
|:--------------|:----:|:----:|:----:|:-----|:-------:|:----:|:----:|:----:|:----:|
| BGE-M3 + MLP  | 0.60 | 0.58 | 0.45 | 0.48 |  0.60   | 0.63 | 0.71 | 0.20 | 0.20 |

### 5.3 BioBERT + MLP (serialized cases)

| Model            | Acc  | P    | R    | F1   | HR   | Mod. F1 | NP   | OT   | UT   | ER   |
|:-----------------|:----:|:----:|:----:|:----:|:-----|:-------:|:----:|:----:|:----:|:----:|
| BioBERT + MLP    | 0.60 | 0.56 | 0.48 | 0.50 | 0.50 |  0.59   | 0.63 | 0.68 | 0.20 | 0.20 |

---

## 6. Next Steps

1. **Zero-Shot GPT-4 & Qwen Fine-Tuning** (compare native API vs in-house FT)  
2. **RAG for Context Retrieval** (LLM + retrieved ED histories)  
3. **Interpretability Analysis** (LIME/SHAP on embeddings & logits)

---

*Repository code and notebooks are available under MIT License.*  
