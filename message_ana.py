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

# 单独绘制每个图表
# 绘制不同人物的发言频率在一张图上
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

# 7. 主函数：整合流程
def main():
    """ 主函数：读取文件并依次调用各个处理步骤 """

    # 1. 获取用户输入的文件路径
    file_path = input("请输入要分析的聊天记录 CSV 文件路径：")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("文件未找到，请确保文件路径正确。")
        return
    
    # 打印 CSV 文件的所有列名，帮助确认数据结构
    print("\nCSV 文件的列名如下：")
    print(df.columns)

    # 确保 'Speaker' 列名正确
    if 'Speaker' not in df.columns:
        print("未找到 'Speaker' 列，请确认该列名是否存在，或是否使用了不同的列名。")
        return
    
    df = clean_data(df)

    # 2. 提取所有说话者并显示，供用户选择
    speakers = df['Speaker'].unique()
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

    # 继续处理文本和分析
    df = tokenize_and_filter(df)
    word_freq = word_frequency(df)
    df = sentiment_analysis(df)

    # 调用单独的可视化函数
    plot_combined_message_frequency(df, specific_people)
    plot_total_message_frequency(df)
    plot_wordcloud(word_freq)
    plot_top_words(word_freq)
    plot_sentiment_trend(df)
    plot_sentiment_distribution(df)

# 运行主函数
if __name__ == "__main__":
    main()