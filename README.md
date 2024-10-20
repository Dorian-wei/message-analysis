# 聊天记录分析工具

本项目是一个基于 Python 的聊天记录分析工具，能够处理中文聊天记录，对聊天内容进行清洗、分词、词频统计、情感分析，并生成多种可视化图表。该工具支持对单个或多个聊天记录文件进行处理，并提供对不同聊天记录情感指数的对比分析。

## 主要功能

1. **数据清洗**：去除聊天记录中的空格、表情符号和图片标识，并将时间字段转换为标准日期格式。
2. **分词处理**：使用 `jieba` 库对聊天内容进行分词，并过滤掉常见的停用词。
3. **词频统计**：统计每个词语在聊天记录中的出现频率，并生成高频词汇的可视化图表（词云和柱状图）。
4. **情感分析**：使用 `SnowNLP` 对每条消息进行情感分析，得出每条消息的情感得分（0 到 1 之间，越接近 1 表示情感越正面）。
5. **可视化展示**：
   - 每日总消息量趋势图。
   - 每个发言者的发言频率趋势图。
   - 词云图和词频前 20 的词条柱状图。
   - 每日情感变化趋势图和情感得分分布图。
6. **多文件支持**：支持处理多个聊天记录文件，合并后进行分析。
7. **情感对比分析**：在多个聊天记录中，对比不同来源的情感均值、情感峰值及每日情感趋势，并生成图表展示。

## 项目依赖

该项目使用了以下 Python 库：

- `pandas`：用于数据处理和分析。
- `jieba`：用于中文分词。
- `re`：用于正则表达式清理文本。
- `collections.Counter`：用于词频统计。
- `snownlp`：用于中文文本的情感分析。
- `matplotlib`：用于生成图表。
- `wordcloud`：用于生成词云。

可以通过以下命令安装所需依赖：

```bash
pip install pandas jieba snownlp matplotlib wordcloud
```

## 使用说明

### 1. 前期准备

可以参考这个项目：https://github.com/BlueMatthew/WechatExporter
按照这一项目的说明，导出聊天记录为txt文档；

### 2. 处理数据

将txt文档转换为符合以下格式的 CSV 文件：

- **必需的列**：
  - `DateTime`：表示消息发送的时间。
  - `Speaker`：表示消息发送者。
  - `Message`：表示消息内容。

示例 CSV 文件格式：

| DateTime            | Speaker | Message     |
|---------------------|---------|-------------|
| 2023-01-01 08:00:00 | Alice   | 早上好！     |
| 2023-01-01 08:05:00 | Bob     | [表情]       |
| 2023-01-01 08:10:00 | Alice   | 今天的天气真好！ |
| 2023-01-01 08:15:00 | Bob     | 是啊，阳光明媚。 |

可以使用convert_txt_to_csv.py 这一文件进行转换；
```bash
python convert_txt_to_csv.py
```

### 3. 运行程序

运行主程序，通过输入 CSV 文件的路径开始分析。程序支持单个文件和多个文件的输入。

运行以下命令：

```bash
python message_ana.py
```

## 案例展示
![image](https://github.com/user-attachments/assets/77e97cf4-1ad8-485b-b0d2-40a4e658fc63)
![image](https://github.com/user-attachments/assets/f2e0fa60-3c86-42df-90fe-28e7d1a0fbdf)
![image](https://github.com/user-attachments/assets/b5c34fd0-f024-4d3b-b3df-276d1158e46d)
<img width="530" alt="image" src="https://github.com/user-attachments/assets/5f845c19-c6c8-4d16-afbf-02d7c64bb5e2">
<img width="529" alt="image" src="https://github.com/user-attachments/assets/ba68e3af-ef98-4141-b8f0-4ca130cdf043">


