<div align="center"><h2>Internal Consistency and Self-Feedback in Large Language Models: A Survey</h2></div>

<p align="center">
    <!-- arxiv badges -->
    <a href="#">
        <img src="https://img.shields.io/badge/Paper-red?style=flat&logo=arxiv">
    </a>
    <!-- Github -->
    <a href="https://github.com/IAAR-Shanghai/ICSFSurvey">
        <img src="https://img.shields.io/badge/Code-black?style=flat&logo=github">
    </a>
</p>

<div align="center">
    <p>
        <a href="https://scholar.google.com/citations?user=d0E7YlcAAAAJ">Xun Liang</a><sup>1*</sup>, 
        <a href="https://ki-seki.github.io/">Shichao Song</a><sup>1*</sup>, 
        <a href="https://github.com/fan2goa1">Zifan Zheng</a><sup>2*</sup>, <br>
        <a href="https://github.com/MarrytheToilet">Hanyu Wang</a><sup>1</sup>, 
        <a href="https://github.com/Duguce">Qingchen Yu</a><sup>2</sup>, 
        <a href="https://xkli-allen.github.io/">Xunkai Li</a><sup>3</sup>, 
        <a href="https://ronghuali.github.io/index.html">Rong-Hua Li</a><sup>3</sup>, 
        <a href="https://scholar.google.com/citations?user=GOKgLdQAAAAJ">Feiyu Xiong</a><sup>2</sup>, 
        <a href="https://www.semanticscholar.org/author/Zhiyu-Li/2268429641">Zhiyu Li</a><sup>2†</sup>
    </p>
    <p>
        <sup>1</sup><a href="https://en.ruc.edu.cn/">Renmin University of China</a> <br>
        <sup>2</sup><a href="https://www.iaar.ac.cn/">Institute for Advanced Algorithms Research, Shanghai</a> <br>
        <sup>3</sup><a href="https://english.bit.edu.cn/">Beijing Institute of Technology</a>
    </p>
</div>

<div align="center"><small><sup>*</sup>Equal contribution.</small></div>
<div align="center"><small><sup>†</sup>Corresponding author: Zhiyu Li (<a href="mailto:lizy@iaar.ac.cn">lizy@iaar.ac.cn</a>).</small></div>

## News

- **[2024/07/21]** Out paper is published on the arXiv platform: #.

## Introduction

Welcome to the GitHub repository for our survey paper titled *"Internal Consistency and Self-Feedback in Large Language Models: A Survey."* This repository contains all the resources, code, and references associated with the paper. Our goal is to provide a unified perspective on the self-evaluation and self-updating mechanisms in LLMs, encapsulated within the frameworks of Internal Consistency and Self-Feedback. 

![Article Framework](figures/article_framework.jpg)

Our survey includes:

- **Theoretical Framework**: 
   - **Internal Consistency**: A framework that offers unified explanations for phenomena such as the lack of reasoning and the presence of hallucinations in LLMs. It assesses the coherence among LLMs' latent layer, decoding layer, and response layer based on sampling methodologies.
   - **Self-Feedback**: Building on Internal Consistency, this framework includes two modules, Self-Evaluation and Self-Update, to enhance the model's response or the model itself.
- **Systematic Classification**: Studies are categorized by tasks and lines of work related to Self-Feedback mechanisms.
- **Evaluation Methods and Benchmarks**: Summarizes various evaluation methods and benchmarks used in the field to assess the effectiveness of Self-Feedback.
- **Critical Viewpoints**: Explores significant questions such as "Does Self-Feedback Really Work?" and proposes hypotheses like the "Hourglass Evolution of Internal Consistency," "Consistency Is (Almost) Correctness," and "The Paradox of Latent and Explicit Reasoning."
- **Future Research Directions**: Outlines promising directions for further exploration in the realm of Internal Consistency and Self-Feedback in LLMs.

## Project Structure

- **`code/`**: Contains the experimental code used in our survey.
- **`data/`**: Includes the statistical data referenced in our survey.
- **`figures/`**: Contains the figures used in this repository.
- **`latex/`**: The LaTeX source files for our survey.
- **`papers/`**: A comprehensive list of relevant papers.
- **`README.md`**: This file, providing an overview of the repository.

## Contribution

We welcome and appreciate contributions to enhance this repository. You can

* add new papers relevant to Internal Consistency or Self-Feedback, 
* or suggest modifications to improve the survey. 

Please submit an issue or a pull request with a brief description of your contribution, and we will review it promptly. Significant contributions may be acknowledged with your name included in the survey. Thank you for your support and collaboration.

## Paper List

We provide a spreadsheet containing all the papers we reviewed: [Literature](https://www.yuque.com/zhiyu-n2wnm/ugzwgf/gmqfkfigd6xw26eg?singleDoc#pBc8). A more readable table format is working in progress.

## To-Do List

- [ ] Create the Page.
- [ ] Improve paper list.
