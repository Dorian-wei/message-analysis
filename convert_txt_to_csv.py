import re
import pandas as pd
import os

# 提取日期和时间，格式为 YYYY-MM-DD HH
def extract_datetime(text):
    match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2})', text)
    return match.group(1) if match else None

# 提取用户和消息内容
def extract_user_message(text):
    match = re.search(r'(.+?)\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\):(.+)', text)
    if match:
        return match.group(1), match.group(3)
    return None, None

# 处理聊天记录文件并转换为CSV文件
def process_chat_file(file_path, output_csv_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在。请检查路径并重试。")
        return

    # 读取聊天内容
    with open(file_path, 'r', encoding='utf-8') as file:
        chat_content = file.readlines()

    # 存储处理后的数据
    data = []

    # 逐行处理聊天内容
    for line in chat_content:
        if '(' in line and ')' in line:
            speaker, message = extract_user_message(line)
            datetime = extract_datetime(line)
            if speaker and message and datetime:
                data.append([datetime, speaker, message])

    # 检查是否有提取到的数据
    if not data:
        print("警告：未从文件中提取到有效的聊天数据。请确认文件格式是否正确。")
        return

    # 转换为DataFrame
    df = pd.DataFrame(data, columns=['DateTime', 'Speaker', 'Message'])

    # 保存为CSV文件
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"成功：聊天记录已保存为 CSV 文件：{output_csv_path}")
    except Exception as e:
        print(f"错误：无法保存文件。原因：{e}")

# 命令行交互使用
def user_interface():
    print("欢迎使用聊天记录转换工具！")
    
    # 获取用户输入的文件路径
    file_path = input("请输入聊天记录文件的路径（如 /path/to/chat.txt）：")
    
    # 获取用户输入的输出文件路径
    output_csv_path = input("请输入输出的 CSV 文件路径（如 /path/to/output.csv）：")
    
    # 处理文件并生成CSV
    process_chat_file(file_path, output_csv_path)

# 启动！
if __name__ == "__main__":
    user_interface()