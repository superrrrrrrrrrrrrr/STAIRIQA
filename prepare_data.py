import pandas as pd
import json
import os

# 1. 设置文件路径
score_csv_path = 'koniq10k_scores_and_distributions(c1-c5指打1-5分的人数）.csv'
train_json_path = 'koniq_train.json'
test_json_path = 'koniq_test.json'

# 2. 读取分数 CSV
# 注意：你的 CSV 文件名可能包含特殊字符，请确保文件名准确
if not os.path.exists(score_csv_path):
    print(f"错误：找不到分数文件 {score_csv_path}")
    exit()

df_scores = pd.read_csv(score_csv_path)
# 为了方便查找，将 CSV 中的 image_name 设为索引（去除可能的引号等干扰）
df_scores['clean_name'] = df_scores['image_name'].astype(str).str.strip()
df_lookup = df_scores.set_index('clean_name')

print(f"已加载分数库，共 {len(df_scores)} 条数据。")

def process_and_save(json_path, output_csv_name):
    if not os.path.exists(json_path):
        print(f"跳过：找不到 JSON 文件 {json_path}")
        return

    with open(json_path, 'r') as f:
        data_list = json.load(f)

    # 准备收集匹配到的数据行
    processed_rows = []
    
    # 必需列（StairIQA 代码要求）
    required_cols = ['image_name','c1','c2','c3','c4','c5','c_total','MOS','SD','MOS_zscore']

    for item in data_list:
        # item 结构: {'image': 'koniq_train/10004473376.jpg', 'score': 77.38...}
        full_path = item['image']
        score_from_json = item['score']
        
        # 从路径中提取纯文件名用于去 CSV 查表
        # 例如 "koniq_train/10004473376.jpg" -> "10004473376.jpg"
        base_name = os.path.basename(full_path)
        
        # 在 CSV 中查找元数据
        if base_name in df_lookup.index:
            row_data = df_lookup.loc[base_name].to_dict()
        else:
            # 如果 CSV 里找不到（极少见情况），我们用 0 填充元数据，但必须保留 image_name 和 score
            # print(f"警告: CSV中未找到 {base_name}，使用默认值填充")
            row_data = {col: 0 for col in required_cols if col not in ['image_name', 'MOS_zscore']}
        
        # === 关键步骤：覆盖/修正关键字段 ===
        # 1. image_name 必须用 JSON 里的带路径版本 (适配 config.yaml 的 database_dir)
        row_data['image_name'] = full_path
        
        # 2. 确保 MOS_zscore 存在 (StairIQA 训练用的就是这一列)
        # 我们可以优先用 JSON 里的 score，因为它可能对应你特定的实验设置
        row_data['MOS_zscore'] = score_from_json
        
        # 3. 确保 MOS 也有值 (防止代码某些地方用到)
        if 'MOS' not in row_data:
            row_data['MOS'] = score_from_json

        processed_rows.append(row_data)

    # 转换为 DataFrame
    df_result = pd.DataFrame(processed_rows)
    
    # 补齐可能缺失的列（如 c1-c5 如果 CSV 里没有或查找失败）
    for col in required_cols:
        if col not in df_result.columns:
            df_result[col] = 0
            
    # 按正确顺序导出
    df_result = df_result[required_cols]
    
    # 保存
    df_result.to_csv(output_csv_name, index=False)
    print(f"成功生成 {output_csv_name}: 包含 {len(df_result)} 条数据")

# 3. 执行转换
process_and_save(train_json_path, 'Koniq10k_train_0.csv')
process_and_save(test_json_path, 'Koniq10k_test_0.csv')