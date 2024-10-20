import pandas as pd
import jieba
import re
from collections import Counter
from snownlp import SnowNLP
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 停用词列表
stopwords = {"在", "的", "是", "和", "了", "就", "都", "而", "及", "与", "或", "一个", "我们", "你", "我", "吗", "哦","吧","也","呢","要","还","上","有","啊"}

# 1. 数据清洗函数
def clean_data(df):
    """ 清洗数据：去掉空格和表情符号，处理日期 """
    df['DateTime'] = df['DateTime'].str.strip()  # 去掉时间列的空格
    df['Message'] = df['Message'].replace(r'\[表情\]', '', regex=True)  # 去掉表情符号
    df['Message'] = df['Message'].replace(r'\[图片\]', '', regex=True)  # 去掉未显示的图片
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')  # 转换为日期格式
    return df

# 2. 文本预处理函数
def preprocess_text(text):
    """ 使用正则表达式去除标点符号和特殊字符 """
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text)
        return text
    return ''

# 3. 分词与停用词过滤
def tokenize_and_filter(df):
    """ 使用jieba分词并过滤掉停用词 """
    df['Processed_Message'] = df['Message'].apply(preprocess_text)
    df['Tokenized_Message'] = df['Processed_Message'].apply(lambda x: jieba.lcut(x))
    df['Filtered_Message'] = df['Tokenized_Message'].apply(
        lambda tokens: [word for word in tokens if word not in stopwords and word.strip()]
    )
    return df

# 4. 词频统计函数
def word_frequency(df):
    """ 统计词频并返回Counter对象 """
    all_words = [word for tokens in df['Filtered_Message'] for word in tokens if word.strip() != '']
    return Counter(all_words)

# 5. 情感分析函数
def sentiment_analysis(df):
    """ 使用SnowNLP对消息进行情感分析 """
    def get_sentiment(text):
        if isinstance(text, str) and text.strip():  # 确保文本非空
            s = SnowNLP(text)
            return s.sentiments
        return None
    df['Sentiment'] = df['Message'].apply(get_sentiment)
    return df

# 6.可视化函数，单独绘制每个图表（用于单个聊天记录分析）
def plot_combined_message_frequency(df, specific_people):
    """ 绘制多个发言者的每日发言频率在同一张图上 """
    df['Date'] = df['DateTime'].dt.date
    plt.figure(figsize=(10, 6))
    
    for person in specific_people:
        specific_person_df = df[df['Speaker'] == person]
        specific_person_df['Date'] = specific_person_df['DateTime'].dt.date
        person_daily_messages = specific_person_df.groupby('Date').size()
        person_daily_messages.plot(kind='line', label=person)

    plt.title('不同发言者的每日发言频率')
    plt.xlabel('日期')
    plt.ylabel('消息数量')
    plt.legend()  # 显示图例以区分不同发言者
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_total_message_frequency(df):
    """ 绘制每日总消息量 """
    df['Date'] = df['DateTime'].dt.date
    daily_messages = df.groupby('Date').size()

    plt.figure(figsize=(10, 6))
    daily_messages.plot(kind='line', color='blue')
    plt.title('每日消息总量')
    plt.xlabel('日期')
    plt.ylabel('消息数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_wordcloud(word_freq):
    """ 生成并绘制词云图 """
    wordcloud = WordCloud(
        font_path='/Users/akirawei/Library/Fonts/SimHei.ttf',
        background_color='white',
        width=800,
        height=600
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('词云 - 高频词汇')
    plt.show()

def plot_top_words(word_freq):
    """ 绘制词频前20的词条柱状图 """
    common_words = word_freq.most_common(20)
    words, freqs = zip(*common_words)

    plt.figure(figsize=(10, 6))
    plt.bar(words, freqs, color='orange')
    plt.title('词频前20的词条')
    plt.ylabel('频率')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_sentiment_trend(df):
    """ 绘制每日情感变化折线图 """
    df['Date'] = df['DateTime'].dt.date
    daily_sentiment = df.groupby('Date')['Sentiment'].mean()

    plt.figure(figsize=(10, 6))
    daily_sentiment.plot(kind='line', color='green')
    plt.title('每日情感变化')
    plt.xlabel('日期')
    plt.ylabel('情感得分')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_sentiment_distribution(df):
    """ 绘制情感得分分布柱状图 """
    plt.figure(figsize=(10, 6))
    df['Sentiment'].hist(bins=20, color='purple')
    plt.title('情感得分分布')
    plt.xlabel('情感得分')
    plt.ylabel('频率')
    plt.tight_layout()
    plt.show()

# 7. 新增：存储情感分析结果
def store_sentiment_analysis(df):
    """存储每个聊天记录的情感分析结果"""
    sentiment_results = {}

    for source in df['ChatSource'].unique():
        source_df = df[df['ChatSource'] == source]

        # 计算情感均值
        average_sentiment = source_df['Sentiment'].mean()

        # 每日情感均值
        source_df['Date'] = source_df['DateTime'].dt.date
        daily_sentiment = source_df.groupby('Date')['Sentiment'].mean()

        # 找到每日情感最高的日期
        max_sentiment_date = daily_sentiment.idxmax()
        max_sentiment_value = daily_sentiment.max()

        # 存储结果
        sentiment_results[source] = {
            'average_sentiment': average_sentiment,
            'max_sentiment_date': max_sentiment_date,
            'max_sentiment_value': max_sentiment_value,
            'daily_sentiment': daily_sentiment
        }
    
    return sentiment_results

# 8. 新增：情感对比分析
def compare_sentiment(sentiment_results):
    """对比多个聊天记录的情感指数"""
    
    # 对比平均情感得分
    print("情感均值对比：")
    for source, result in sentiment_results.items():
        print(f"{source}: 平均情感得分 = {result['average_sentiment']:.4f}")
    
    # 找出情感均值最高的聊天记录
    highest_avg_sentiment = max(sentiment_results.items(), key=lambda x: x[1]['average_sentiment'])
    print(f"\n情感均值最高的聊天记录是 {highest_avg_sentiment[0]}，得分为 {highest_avg_sentiment[1]['average_sentiment']:.4f}")

    # 对比每日情感峰值
    print("\n每日情感峰值对比：")
    for source, result in sentiment_results.items():
        print(f"{source}: 最高情感日期 = {result['max_sentiment_date']}, 最高情感得分 = {result['max_sentiment_value']:.4f}")

    # 找出每日情感峰值最高的记录
    highest_max_sentiment = max(sentiment_results.items(), key=lambda x: x[1]['max_sentiment_value'])
    print(f"\n每日情感峰值最高的聊天记录是 {highest_max_sentiment[0]}，得分为 {highest_max_sentiment[1]['max_sentiment_value']:.4f}，日期为 {highest_max_sentiment[1]['max_sentiment_date']}")

# 9. 新增：情感趋势对比图
def plot_sentiment_trend_comparison(sentiment_results):
    """绘制多个聊天记录的情感趋势对比"""
    plt.figure(figsize=(10, 6))

    for source, result in sentiment_results.items():
        daily_sentiment = result['daily_sentiment']
        daily_sentiment.plot(kind='line', label=source)

    plt.title('不同聊天记录的每日情感趋势对比')
    plt.xlabel('日期')
    plt.ylabel('情感得分')
    plt.legend()  # 显示图例以区分不同聊天记录
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 7. 主函数：整合流程
def main():
    """主函数：读取文件并依次调用各个处理步骤"""

    # 获取用户输入的文件路径
    file_paths = input("请输入要分析的聊天记录 CSV 文件路径（可输入多个路径，用逗号分隔）：")
    file_paths = file_paths.split(',')

    # 创建一个空的 DataFrame 用于存放所有聊天记录
    all_data = pd.DataFrame()

    # 处理每个文件
    for i, file_path in enumerate(file_paths):
        try:
            df = pd.read_csv(file_path.strip())
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到，请确保文件路径正确。")
            continue

        # 打印 CSV 文件的所有列名，帮助确认数据结构
        print(f"\nCSV 文件 {file_path} 的列名如下：")
        print(df.columns)

        # 确保 'Speaker' 列名正确
        if 'Speaker' not in df.columns:
            print(f"未找到 'Speaker' 列，请确认该列名是否存在，或是否使用了不同的列名。")
            continue

        # 数据清洗
        df = clean_data(df)

        # 添加来源标识
        df['ChatSource'] = f"聊天记录_{i+1}"

        # 合并所有聊天记录
        all_data = pd.concat([all_data, df], ignore_index=True)

    # 确保数据不为空
    if all_data.empty:
        print("未加载任何有效数据，程序退出。")
        return

    # 分词与停用词过滤
    all_data = tokenize_and_filter(all_data)

    # 词频统计
    word_freq = word_frequency(all_data)

    # 情感分析
    all_data = sentiment_analysis(all_data)

    # 提取所有说话者并显示，供用户选择
    speakers = all_data['Speaker'].unique()
    print("\n聊天记录中包含以下发言者：")
    for i, speaker in enumerate(speakers, 1):
        print(f"{i}. {speaker}")

    selected_indices = input("\n请选择你想查看的发言者（输入编号，用逗号分隔，例如：1,2）：")
    try:
        selected_indices = [int(i) - 1 for i in selected_indices.split(',')]
        specific_people = [speakers[i] for i in selected_indices]
    except (ValueError, IndexError):
        print("输入有误，请输入正确的编号。")
        return

    # 进行可视化分析
    plot_combined_message_frequency(all_data, specific_people)
    plot_total_message_frequency(all_data)
    plot_wordcloud(word_freq)
    plot_top_words(word_freq)
    plot_sentiment_trend(all_data)
    plot_sentiment_distribution(all_data)

    # 存储情感分析结果
    sentiment_results = store_sentiment_analysis(all_data)

    # 情感指数对比分析
    compare_sentiment(sentiment_results)

    # 可视化情感趋势对比
    plot_sentiment_trend_comparison(sentiment_results)

# 运行主函数
if __name__ == "__main__":
    main()