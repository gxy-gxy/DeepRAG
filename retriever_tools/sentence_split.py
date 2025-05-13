from llama_index.core.node_parser import SentenceSplitter
import pandas as pd

# 初始化 SentenceSplitter，设置每个块包含 100 个标记，无重叠
splitter = SentenceSplitter(chunk_size=100, chunk_overlap=0)

# 读取 TSV 文件
input_file_path = "/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/xinyanguan/baseline/SRAG/data/freshqa/output_file.tsv"
docs = pd.read_csv(input_file_path, sep="\t")

# 检查是否有 text 字段
if "text" not in docs.columns:
    raise ValueError("TSV 文件中未找到 'text' 字段，请检查文件结构。")

# 对 text 字段进行分割，并保留其他字段
split_results = []
for idx, row in docs.iterrows():
    text = row['text']
    other_data = row.drop(labels=['text']).to_dict()  # 提取除了 text 以外的其他列
    try:
        # 分割当前行的文本
        chunks = splitter.split_text(text)
        for chunk in chunks:
            # 保留其他字段，添加分割后的 chunk
            result = other_data.copy()
            result['text'] = chunk
            result['original_index'] = idx
            split_results.append(result)
    except Exception as e:
        print(text)
        print(f"Error processing row {idx}: {e}")

# 将分割结果转换为 DataFrame
# id titile text放前三列
split_df = pd.DataFrame(split_results)

# 将 id, title, text 放在前三列，确保这些字段存在
columns_order = ['id', 'title', 'text']  # 指定需要放在前面的列
remaining_columns = [col for col in split_df.columns if col not in columns_order]  # 其余列
split_df = split_df[columns_order + remaining_columns]  # 按指定顺序重排列


# 保存结果到新的 TSV 文件
output_file_path = "/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/xinyanguan/baseline/SRAG/data/freshqa/split_output_file.tsv"
split_df.to_csv(output_file_path, sep="\t", index=False)

print(f"分割完成，结果已保存到 {output_file_path}")
