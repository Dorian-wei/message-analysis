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
stopwords = {"在", "的", "是", "和", "了", "就", "都", "而", "及", "与", "或", "一个", "我们", "你", "我", "吗", "哦"}

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
        if isinstance(text, str):
            s = SnowNLP(text)
            return s.sentiments
        return None
    df['Sentiment'] = df['Message'].apply(get_sentiment)
    return df

# 6. 可视化图表生成函数
def plot_visualizations(df, word_freq, specific_people):
    """ 整合所有图表并在一张图上展示 """
    df['Date'] = df['DateTime'].dt.date

    # 创建多子图的画布
    fig, axs = plt.subplots(4, 2, figsize=(18, 24))  # 4行2列布局

    # 1. 每个人的每日发言频率折线图
    for person in specific_people:
        specific_person_df = df[df['Speaker'] == person]
        specific_person_df['Date'] = specific_person_df['DateTime'].dt.date
        person_daily_messages = specific_person_df.groupby('Date').size()
        person_daily_messages.plot(kind='line', ax=axs[0, 0], label=person)

    axs[0, 0].set_title('特定人物的每日发言频率')
    axs[0, 0].set_xlabel('日期')
    axs[0, 0].set_ylabel('消息数量')
    axs[0, 0].legend()  # 显示图例

    # 2. 总体每日消息数量折线图
    daily_messages = df.groupby('Date').size()
    daily_messages.plot(kind='line', ax=axs[0, 1], color='blue')
    axs[0, 1].set_title('每日消息总量')
    axs[0, 1].set_xlabel('日期')
    axs[0, 1].set_ylabel('消息数量')

    # 3. 生成词云图
    wordcloud = WordCloud(
        font_path='/Users/akirawei/Library/Fonts/SimHei.ttf',
        background_color='white',
        width=800,
        height=600
    ).generate_from_frequencies(word_freq)
    axs[1, 0].imshow(wordcloud, interpolation='bilinear')
    axs[1, 0].axis('off')
    axs[1, 0].set_title('词云 - 高频词汇')

    # 4. 词频前20的词条柱状图
    common_words = word_freq.most_common(20)
    words, freqs = zip(*common_words)
    axs[1, 1].bar(words, freqs, color='orange')
    axs[1, 1].set_title('词频前20的词条')
    axs[1, 1].set_ylabel('频率')
    axs[1, 1].tick_params(axis='x', rotation=45)

    # 5. 每日情感变化折线图
    daily_sentiment = df.groupby('Date')['Sentiment'].mean()
    daily_sentiment.plot(kind='line', ax=axs[2, 0], color='green')
    axs[2, 0].set_title('每日情感变化')
    axs[2, 0].set_xlabel('日期')
    axs[2, 0].set_ylabel('情感得分')

    # 6. 情感得分分布柱状图
    df['Sentiment'].hist(bins=20, ax=axs[2, 1], color='purple')
    axs[2, 1].set_title('情感得分分布')
    axs[2, 1].set_xlabel('情感得分')
    axs[2, 1].set_ylabel('频率')

    # 调整布局
    plt.tight_layout()
    plt.show()

# 7. 主函数：整合流程
def main(file_path, specific_people):
    """ 主函数：读取文件并依次调用各个处理步骤 """
    df = pd.read_csv(file_path)
    df = clean_data(df)
    df = tokenize_and_filter(df)
    word_freq = word_frequency(df)
    df = sentiment_analysis(df)

    # 调用可视化函数
    plot_visualizations(df, word_freq, specific_people)

# 调用主函数
file_path = "/Users/akirawei/Desktop/kisen.csv" #替换为你的路径
specific_people = ['浮水', '孔皓宇kisen'] #替换为你要分析的聊天者名称；
main(file_path, specific_people)
