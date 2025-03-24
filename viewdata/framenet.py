# FrameNetのデータの前処理でのデータ数の変化を確認する
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from tqdm import tqdm

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 1000)


def check_lexicographic():
    ns = {"fn": "http://framenet.icsi.berkeley.edu"}
    corpus_df = pd.DataFrame(columns=["ID", "name", "description"])
    sentence_df = pd.DataFrame(
        columns=["corpID", "annotationSet_ID", "cDate", "name", "text"]
    )

    for file_name in tqdm(glob.glob("/data/data/fndata-1.7/lu/*.xml")):
        tree = ET.parse(file_name)
        root = tree.getroot()
        corpuses = root.findall("fn:header/fn:corpus", ns)
        corpus_df = pd.concat(
            [
                corpus_df,
                pd.DataFrame(
                    [corpus.attrib for corpus in corpuses],
                    columns=["ID", "name", "description"],
                ),
            ]
        ).reset_index(drop=True)

        name = root.attrib["name"]
        sentences = root.findall("fn:subCorpus/fn:sentence", ns)
        for sentence in sentences:
            row = {}
            row["name"] = name
            row["corpID"] = (
                sentence.attrib["corpID"] if "corpID" in sentence.attrib else -1
            )
            row["text"] = sentence.find("fn:text", ns).text
            annotationSets = sentence.findall("fn:annotationSet", ns)
            for annotationSet in annotationSets:
                layers = annotationSet.findall("fn:layer", ns)
                for layer in layers:
                    if layer.attrib["name"] == "Target":
                        row["annotationSet_ID"] = annotationSet.attrib["ID"]
                        row["cDate"] = annotationSet.attrib["cDate"]
                        sentence_df = pd.concat(
                            [sentence_df, pd.DataFrame([row])]
                        ).reset_index(drop=True)
                        break

    corpus_df = corpus_df.drop_duplicates()
    sentence_df = sentence_df.drop_duplicates()

    output_dir = Path("./viewdata/lexicographic")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_df.to_csv(output_dir / "corpus.csv", index=False)
    sentence_df.to_csv(output_dir / "sentence.csv", index=False)
    # with open(output_dir / "corpus.csv", "w") as f:
    #     f.write(corpus_df.to_csv())
    # with open(output_dir / "sentence.csv", "w") as f:
    #     f.write(sentence_df.to_csv())

    sentence_df["year"] = sentence_df["cDate"].apply(lambda x: x[6:10])
    with open(output_dir / "sentence_summary.txt", "w") as f:
        year_counts = sentence_df.groupby("year").size()
        cumulative_sum = year_counts.cumsum()
        percentage = (cumulative_sum / sentence_df.shape[0] * 100).round(2)
        summary_df = pd.DataFrame(
            {
                "Count": year_counts,
                "Cumulative Sum": cumulative_sum,
                "Percentage (%)": percentage,
            }
        )
        f.write(summary_df.to_markdown() + "\n\n")
        # f.write(sentence_df.groupby("year").size().to_markdown() + "\n")
        # f.write(f"{sentence_df.shape[0]}" + "\n\n")
        f.write(sentence_df.groupby("corpID").size().to_markdown() + "\n")


def check_fulltext():
    ns = {"fn": "http://framenet.icsi.berkeley.edu"}
    corpus_df = pd.DataFrame(columns=["ID", "name", "description"])
    sentence_df = pd.DataFrame(
        columns=["corpID", "annotationSet_ID", "cDate", "name", "text"]
    )

    for file_name in tqdm(glob.glob("/data/data/fndata-1.7/fulltext/*.xml")):
        tree = ET.parse(file_name)
        root = tree.getroot()
        corpuses = root.findall("fn:header/fn:corpus", ns)
        corpus_df = pd.concat(
            [
                corpus_df,
                pd.DataFrame(
                    [corpus.attrib for corpus in corpuses],
                    columns=["ID", "name", "description"],
                ),
            ]
        ).reset_index(drop=True)

        sentences = root.findall("fn:sentence", ns)
        for sentence in sentences:
            row = {}
            row["corpID"] = (
                sentence.attrib["corpID"] if "corpID" in sentence.attrib else -1
            )
            row["text"] = sentence.find("fn:text", ns).text
            annotationSets = sentence.findall("fn:annotationSet", ns)
            for annotationSet in annotationSets:
                layers = annotationSet.findall("fn:layer", ns)
                if "luID" in annotationSet.attrib.keys():
                    row["annotationSet_ID"] = annotationSet.attrib["ID"]
                    row["cDate"] = annotationSet.attrib["cDate"]
                    row["name"] = annotationSet.attrib["luName"]
                    sentence_df = pd.concat(
                        [sentence_df, pd.DataFrame([row])]
                    ).reset_index(drop=True)
                # for layer in layers:
                #     if layer.attrib["name"] == "Target":
                #         row["annotationSet_ID"] = annotationSet.attrib["ID"]
                #         row["cDate"] = annotationSet.attrib["cDate"]
                #         row["name"] = annotationSet.attrib["luName"]
                #         sentence_df = pd.concat(
                #             [sentence_df, pd.DataFrame([row])]
                #         ).reset_index(drop=True)
                #         break

    corpus_df = corpus_df.drop_duplicates().reset_index(drop=True)
    sentence_df = sentence_df.drop_duplicates().reset_index(drop=True)

    sentence_df["year"] = sentence_df["cDate"].apply(lambda x: x[6:10])

    output_dir = Path("./viewdata/fulltext")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_df.to_csv(output_dir / "corpus.csv", index=False)
    sentence_df.to_csv(output_dir / "sentence.csv", index=False)
    # with open(output_dir / "corpus.csv", "w") as f:
    #     f.write(corpus_df.to_csv())
    # with open("./viewdata/fulltext/sentence.csv", "w") as f:
    #     f.write(sentence_df.to_csv())

    with open(output_dir / "sentence_summary.txt", "w") as f:
        year_counts = sentence_df.groupby("year").size()
        cumulative_sum = year_counts.cumsum()
        percentage = (cumulative_sum / sentence_df.shape[0] * 100).round(2)
        summary_df = pd.DataFrame(
            {
                "Count": year_counts,
                "Cumulative Sum": cumulative_sum,
                "Percentage (%)": percentage,
            }
        )
        f.write(summary_df.to_markdown() + "\n\n")
        # f.write(sentence_df.groupby("year").size().to_markdown() + "\n")
        # f.write(f"{sentence_df.shape[0]}" + "\n\n")
        f.write(sentence_df.groupby("corpID").size().to_markdown() + "\n")


def read_check_lexicographic():
    input_dir = Path("./viewdata/lexicographic")
    sentence_df = pd.read_csv(input_dir / "sentence.csv")

    with open(input_dir / "sentence_summary.txt", "w") as f:
        f.write("LU全て\n")
        year_counts = sentence_df.groupby("year").size()
        cumulative_sum = year_counts.cumsum()
        percentage = (cumulative_sum / sentence_df.shape[0] * 100).round(2)
        summary_df = pd.DataFrame(
            {
                "Count": year_counts,
                "Cumulative Sum": cumulative_sum,
                "Percentage (%)": percentage,
            }
        )
        f.write(summary_df.to_markdown() + "\n\n")
        f.write(sentence_df.groupby("corpID").size().to_markdown() + "\n\n")

        lu_counts = sentence_df.groupby("name").size().sort_values(ascending=False)
        cumulative_sum = lu_counts.cumsum()
        percentage = (cumulative_sum / sentence_df.shape[0] * 100).round(2)
        summary_df = pd.DataFrame(
            {
                "Count": lu_counts,
                "Cumulative Sum": cumulative_sum,
                "Percentage (%)": percentage,
            }
        )
        f.write(summary_df.to_markdown() + "\n\n")

        f.write("動詞LU\n")
        sentence_df = sentence_df[
            sentence_df["name"].str.contains(r"\.v")
        ]  # 動詞のLUを抽出
        sentence_df = sentence_df[
            ~sentence_df["name"].str.contains(r"[\(\)\[\]]")
        ]  # ()や[]が入っている例を除外
        year_counts = sentence_df.groupby("year").size()
        cumulative_sum = year_counts.cumsum()
        percentage = (cumulative_sum / sentence_df.shape[0] * 100).round(2)
        summary_df = pd.DataFrame(
            {
                "Count": year_counts,
                "Cumulative Sum": cumulative_sum,
                "Percentage (%)": percentage,
            }
        )
        f.write(summary_df.to_markdown() + "\n\n")
        f.write(sentence_df.groupby("corpID").size().to_markdown() + "\n\n")

        lu_counts = sentence_df.groupby("name").size().sort_values(ascending=False)
        cumulative_sum = lu_counts.cumsum()
        percentage = (cumulative_sum / sentence_df.shape[0] * 100).round(2)
        summary_df = pd.DataFrame(
            {
                "Count": lu_counts,
                "Cumulative Sum": cumulative_sum,
                "Percentage (%)": percentage,
            }
        )
        with open(input_dir / "sentence_summary.csv", "w") as f2:
            f2.write(summary_df.to_csv() + "\n\n")


def read_check_fulltext():
    input_dir = Path("./viewdata/fulltext")
    sentence_df = pd.read_csv(input_dir / "sentence.csv")

    with open(input_dir / "sentence_summary.txt", "w") as f:
        f.write("LU全て\n")
        year_counts = sentence_df.groupby("year").size()
        cumulative_sum = year_counts.cumsum()
        percentage = (cumulative_sum / sentence_df.shape[0] * 100).round(2)
        summary_df = pd.DataFrame(
            {
                "Count": year_counts,
                "Cumulative Sum": cumulative_sum,
                "Percentage (%)": percentage,
            }
        )
        f.write(summary_df.to_markdown() + "\n\n")
        f.write(sentence_df.groupby("corpID").size().to_markdown() + "\n\n")

        lu_counts = sentence_df.groupby("name").size().sort_values(ascending=False)
        cumulative_sum = lu_counts.cumsum()
        percentage = (cumulative_sum / sentence_df.shape[0] * 100).round(2)
        summary_df = pd.DataFrame(
            {
                "Count": lu_counts,
                "Cumulative Sum": cumulative_sum,
                "Percentage (%)": percentage,
            }
        )
        f.write(summary_df.to_markdown() + "\n\n")

        f.write("動詞LU\n")
        sentence_df = sentence_df[
            sentence_df["name"].str.contains(r"\.v")
        ]  # 動詞のLUを抽出
        sentence_df = sentence_df[
            ~sentence_df["name"].str.contains(r"[\(\)\[\]]")
        ]  # ()や[]が入っている例を除外
        year_counts = sentence_df.groupby("year").size()
        cumulative_sum = year_counts.cumsum()
        percentage = (cumulative_sum / sentence_df.shape[0] * 100).round(2)
        summary_df = pd.DataFrame(
            {
                "Count": year_counts,
                "Cumulative Sum": cumulative_sum,
                "Percentage (%)": percentage,
            }
        )
        f.write(summary_df.to_markdown() + "\n\n")
        f.write(sentence_df.groupby("corpID").size().to_markdown() + "\n\n")

        lu_counts = sentence_df.groupby("name").size().sort_values(ascending=False)
        cumulative_sum = lu_counts.cumsum()
        percentage = (cumulative_sum / sentence_df.shape[0] * 100).round(2)
        summary_df = pd.DataFrame(
            {
                "Count": lu_counts,
                "Cumulative Sum": cumulative_sum,
                "Percentage (%)": percentage,
            }
        )

        with open(input_dir / "sentence_summary.csv", "w") as f2:
            f2.write(summary_df.to_csv() + "\n\n")


def main():
    # check_lexicographic()
    # check_fulltext()
    read_check_lexicographic()
    read_check_fulltext()


if __name__ == "__main__":
    main()
