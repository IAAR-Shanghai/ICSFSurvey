import os

import pandas as pd


def publication_info(row):
    info = (
        f"- **{row['Title']}**  \n"
        f"  {row['Affiliation']}, {row['Publication']}, 20{row['Year']}\n"
        f"  [[Paper]({row['Paper']})]\n"
    )
    info += (
        "\n"
        if pd.isna(row["Code"]) or row["Code"] == "-"
        else f"  [[Code]({row['Code']})]\n\n"
    )
    return info


curr_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(curr_dir, "ICSFSurvey-Paper-List.xlsx")
output_markdown_path = os.path.join(curr_dir, "ICSFSurvey-Paper-List.md")

sheet_section = {
    "Survey": "Related Survey Papers\n\nThese are some of the most relevant surveys related to our paper.",
    "SignalAcquisition": "Section IV: Consistency Signal Acquisition\n\nFor various forms of expressions from an LLM, we can obtain various forms of consistency signals, which can help in better updating the expressions.",
    "ReasoningElevation": "Section V: Reasoning Elevation\n\nEnhancing reasoning ability by improving LLM performance on QA tasks through Self-Feedback strategies.",
    "HallucinationAlleviation": "Section VI: Hallucination Alleviation\n\nImproving factual accuracy in open-ended generation and reducing hallucinations through Self-Feedback strategies.",
    "OtherTasks": "Section VII: Other Tasks\n\nIn addition to tasks aimed at improving consistency (enhancing reasoning and alleviating hallucinations), there are other tasks that also utilize Self-Feedback strategies.",
    "MetaEvaluation": "Section VIII.A: Meta Evaluation\n\nSome common evaluation benchmarks.",
    "Theory": "Theoretical Perspectives\n\nSome theoretical research on Internal Consistency and Self-Feedback strategies.",
}

selected_sheets = pd.read_excel(file_path, sheet_name=list(sheet_section.keys()))

markdown = (
    "## 📚 Paper List\n\n"
    "Here we list the most important references cited in our survey, as well as the papers we consider worth noting.\n\n"
    "We also provide an [online version](https://www.yuque.com/zhiyu-n2wnm/ugzwgf/gmqfkfigd6xw26eg) and an [Excel version](./ICSFSurvey-Paper-List.xlsx) for your convenience.\n\n"
    "<details><summary>Click Me to Show Table of Contents</summary>\n\n[TOC]\n\n</details>\n\n"
)

for sheet, df in selected_sheets.items():
    markdown += f"### {sheet_section[sheet]}\n\n"

    grouped = df.groupby("Task Type").size().reset_index(name="Count")
    grouped_sorted = grouped.sort_values(
        by=["Count", "Task Type"], ascending=[False, True]
    )
    task_types = grouped_sorted["Task Type"].tolist()

    for task_type in task_types:
        markdown += f"#### {task_type}\n\n"
        df_task_type = df[df["Task Type"] == task_type]
        df_sorted = df_task_type.sort_values(
            by=["Year", "Paper"], ascending=[False, False]
        )
        for idx, row in df_sorted.iterrows():
            markdown += publication_info(row)

    if task_types == []:
        df_sorted = df.sort_values(by=["Year", "Paper"], ascending=[False, False])
        for idx, row in df_sorted.iterrows():
            markdown += publication_info(row)

with open(output_markdown_path, "w") as f:
    f.write(markdown)
    print(f"Markdown file is saved to {output_markdown_path}")
