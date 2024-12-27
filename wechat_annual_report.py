import os
import sys
import pandas as pd
import jieba
import jieba.analyse
import re
from collections import Counter
from snownlp import SnowNLP
import streamlit as st
import logging
import emoji
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("annual_report.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# 1. 加载停用词
@st.cache_data
def load_stopwords(filepath='stopwords.txt'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f if line.strip())
        logging.info(f"成功加载停用词，共 {len(stopwords)} 个停用词。")
        return stopwords
    except FileNotFoundError:
        logging.error(f"停用词文件 {filepath} 未找到，请确保文件存在。")
        st.error(f"停用词文件 {filepath} 未找到，请确保文件存在。")
        sys.exit(1)
    except Exception as e:
        logging.error(f"加载停用词文件时出错: {e}")
        st.error(f"加载停用词文件时出错: {e}")
        sys.exit(1)

# 2. 数据清洗函数
def clean_data(df):
    """ 清洗数据：转换日期时间格式，去除无效记录 """
    try:
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        missing_dates = df['DateTime'].isna().sum()
        if missing_dates > 0:
            logging.warning(f"{missing_dates} 条记录的日期格式无法解析，将被忽略。")
            st.warning(f"{missing_dates} 条记录的日期格式无法解析，将被忽略。")
        df = df.dropna(subset=['DateTime'])
        df['Message'] = df['Message'].astype(str).str.strip()
        return df
    except Exception as e:
        logging.error(f"数据清洗时出错: {e}")
        st.error(f"数据清洗时出错: {e}")
        return pd.DataFrame()

# 3. 文本预处理函数
def preprocess_text(text):
    """ 使用正则表达式去除标点符号和特殊字符，仅保留中文、英文、数字和空格 """
    if isinstance(text, str):
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        return text
    return ''

# 4. 分词与停用词过滤
def tokenize_and_filter(df, stopwords):
    """ 使用jieba分词并过滤掉停用词 """
    try:
        df['Processed_Message'] = df['Message'].apply(preprocess_text)
        df['Tokenized_Message'] = df['Processed_Message'].apply(lambda x: jieba.lcut(x))
        df['Filtered_Message'] = df['Tokenized_Message'].apply(
            lambda tokens: [word for word in tokens if word not in stopwords and word.strip()]
        )
        logging.info("分词与停用词过滤完成。")
        return df
    except Exception as e:
        logging.error(f"分词与停用词过滤时出错: {e}")
        st.error(f"分词与停用词过滤时出错: {e}")
        return df

# 5. 词频统计函数
def word_frequency(df):
    """ 统计词频并返回Counter对象 """
    try:
        all_words = [word for tokens in df['Filtered_Message'] for word in tokens if word.strip() != '']
        word_freq = Counter(all_words)
        logging.info(f"词频统计完成，共统计到 {len(word_freq)} 个不同词汇。")
        return word_freq
    except Exception as e:
        logging.error(f"词频统计时出错: {e}")
        st.error(f"词频统计时出错: {e}")
        return Counter()

# 6. 情感分析函数
def sentiment_analysis(df):
    """ 使用SnowNLP对消息进行情感分析 """
    def get_sentiment(text):
        if isinstance(text, str) and text.strip():  # 确保文本非空
            try:
                s = SnowNLP(text)
                return s.sentiments
            except Exception as e:
                logging.error(f"情感分析时出错: {e}")
                return None
        return None

    try:
        df['Sentiment'] = df['Message'].apply(get_sentiment)
        sentiment_na = df['Sentiment'].isna().sum()
        if sentiment_na > 0:
            logging.warning(f"{sentiment_na} 条记录的情感得分为 NaN。")
            st.warning(f"{sentiment_na} 条记录的情感得分为 NaN。")
        logging.info("情感分析完成。")
        return df
    except Exception as e:
        logging.error(f"情感分析时出错: {e}")
        st.error(f"情感分析时出错: {e}")
        return df

# 7. 表情统计函数
def extract_emojis(df):
    """
    提取聊天记录中的表情并统计频次
    """
    try:
        # 提取 Unicode 表情符号
        def extract_emoji(text):
            return [char for char in text if char in emoji.EMOJI_DATA]

        df['Emojis'] = df['Message'].apply(lambda x: extract_emoji(x) if isinstance(x, str) else [])
        logging.info("表情提取完成。")
        return df
    except Exception as e:
        logging.error(f"提取表情时出错: {e}")
        st.error(f"提取表情时出错: {e}")
        return df

def count_emojis(df):
    """
    统计表情使用频次并返回排名前 5 的表情
    """
    try:
        all_emojis = [emoji_char for emojis in df['Emojis'] for emoji_char in emojis]
        emoji_counter = Counter(all_emojis)
        return emoji_counter.most_common(5)  # 返回排名前 5 的表情
    except Exception as e:
        logging.error(f"统计表情频次时出错: {e}")
        st.error(f"统计表情频次时出错: {e}")
        return []

# 8. 关键时刻回顾
def key_moments(df):
    """ 识别关键时刻，如生日、庆祝、聚会等 """
    try:
        # 示例：假设有特定关键词标识关键时刻
        keywords = ['生日', '庆祝', '聚会', '活动', '合作']
        moments = df[df['Message'].str.contains('|'.join(keywords))]
        
        # 只取最有趣的前5个关键时刻
        moments = moments.head(5)
    
        if not moments.empty:
            st.write("### 🎉 **关键时刻回顾**")
            for _, row in moments.iterrows():
                st.markdown(f"- **{row['DateTime'].strftime('%Y-%m-%d')}**: {row['Speaker']} 分享了 **{row['Message']}**")
        else:
            st.write("### 🎉 **关键时刻回顾**")
            st.write("未识别到重要的关键时刻。")
    except Exception as e:
        logging.error(f"识别关键时刻时出错: {e}")
        st.error(f"识别关键时刻时出错: {e}")

# 9. 趣味统计
def fun_stats(df, word_freq):
    """ 展示趣味统计，如最常用词、最长消息、最活跃日期等 """
    try:
        # 最常用词
        most_common = word_freq.most_common(5)

        # 最长消息
        df['Message_Length'] = df['Message'].apply(len)
        if not df.empty:
            longest_message = df.loc[df['Message_Length'].idxmax()]
            # 最活跃日期
            daily_messages = df.groupby(df['DateTime'].dt.date).size()
            most_active_date = daily_messages.idxmax()
            most_active_count = daily_messages.max()

            st.write("### 🕹️ **趣味统计**")
            st.markdown(f"- **最常用词汇**: {', '.join([word for word, _ in most_common])}")
            st.markdown(f"- **最长消息**: \"{longest_message['Message']}\" 由 **{longest_message['Speaker']}** 于 {longest_message['DateTime'].strftime('%Y-%m-%d')} 发送")
            st.markdown(f"- **最活跃日期**: {most_active_date}，发送了 **{most_active_count} 条消息**")
        else:
            st.write("### 🕹️ **趣味统计**")
            st.write("没有足够的数据进行趣味统计。")
    except Exception as e:
        logging.error(f"展示趣味统计时出错: {e}")
        st.error(f"展示趣味统计时出错: {e}")

# 10. 生成PDF报告
def generate_pdf(chat_name, year, report_content):
    """ 使用reportlab生成PDF报告 """
    try:
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        margin = 50
        y_position = height - margin

        # 设置标题
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(width / 2, y_position, f"群聊【{chat_name}】{year}年度报告")
        y_position -= 40

        # 设置正文
        c.setFont("Helvetica", 12)
        for line in report_content.split('\n'):
            if y_position < margin:
                c.showPage()
                y_position = height - margin
                c.setFont("Helvetica", 12)
            c.drawString(margin, y_position, line)
            y_position -= 15

        c.save()
        buffer.seek(0)
        return buffer
    except Exception as e:
        logging.error(f"生成PDF时出错: {e}")
        st.error(f"生成PDF时出错: {e}")
        return None

# 11. 生成年度报告
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        logging.info("成功读取上传的文件。")
        return df
    except FileNotFoundError:
        st.error(f"文件 {uploaded_file.name} 未找到，请确保文件路径正确。")
        logging.error(f"文件 {uploaded_file.name} 未找到。")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error(f"文件 {uploaded_file.name} 是空的。")
        logging.error(f"文件 {uploaded_file.name} 是空的。")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"读取文件 {uploaded_file.name} 时出错: {e}")
        logging.error(f"读取文件 {uploaded_file.name} 时出错: {e}")
        return pd.DataFrame()

def generate_annual_report(uploaded_file, stopwords, chat_name, year):
    df = load_data(uploaded_file)
    if df.empty:
        st.error("上传的文件没有有效数据。")
        return

    # 检查必要的列
    required_columns = {'Speaker', 'DateTime', 'Message'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        st.error(f"CSV 文件缺少以下必要列: {', '.join(missing)}")
        logging.error(f"CSV 文件缺少以下必要列: {', '.join(missing)}")
        return

    # 数据清洗
    df = clean_data(df)
    if df.empty:
        st.error("数据清洗后没有有效数据。")
        return

    # 获取群聊创建日期（第一条消息日期）
    creation_date = df['DateTime'].min().date()
    st.write(f"**群聊创建日期**：{creation_date}")

    # 筛选报告年度的消息
    start_date = pd.to_datetime(f"{year}-01-01")
    end_date = pd.to_datetime(f"{year}-12-31")
    if creation_date > start_date.date():
        start_date = pd.to_datetime(creation_date)
    df_year = df[(df['DateTime'] >= start_date) & (df['DateTime'] <= end_date)]
    if df_year.empty:
        st.error("在指定的报告年度内没有有效的聊天记录。")
        return

    # 分词与停用词过滤
    df_year = tokenize_and_filter(df_year, stopwords)
    if 'Filtered_Message' not in df_year.columns:
        st.error("分词与停用词过滤后缺少 'Filtered_Message' 列。")
        logging.error("分词与停用词过滤后缺少 'Filtered_Message' 列。")
        return

    # 词频统计
    word_freq = word_frequency(df_year)

    # 情感分析
    df_year = sentiment_analysis(df_year)

    # 表情提取与统计
    df_year = extract_emojis(df_year)
    top_emojis = count_emojis(df_year)

    # 活跃发言者
    top_speakers = df_year['Speaker'].value_counts()

    # 生成报告内容
    report_content = ""

    report_content += f"🎉 **群聊【{chat_name}】{year}年度报告** 🎉\n\n"
    report_content += "---\n\n"
    report_content += f"## 🌟 **引言**\n\n"
    report_content += f"{year}年，对于我们亲爱的群聊【{chat_name}】来说，是充满欢笑、分享与成长的一年。在这一年中，我们共同经历了无数精彩瞬间，增进了彼此的了解，建立了深厚的友谊。现在，让我们一起回顾这一年的精彩时刻，看看我们的群聊有多么活跃和有趣吧！\n\n"
    report_content += "---\n\n"

    # 年度亮点
    total_messages = len(df_year)
    active_days = df_year['DateTime'].dt.date.nunique()
    peak_hour = df_year['DateTime'].dt.hour.value_counts().idxmax()

    report_content += f"## 📈 **年度亮点**\n\n"
    report_content += f"- **总消息量**：本年度我们共发送了 **{total_messages} 条消息**，平均每天约 **{total_messages / active_days:.0f} 条**。无论是日常闲聊还是重要讨论，大家的积极参与让群聊充满了活力。\n"
    report_content += f"- **活跃天数**：全年中，有 **{active_days} 天** 的活跃交流。这显示了我们群体的高粘性和持续互动。\n"
    report_content += f"- **高峰时段**：晚上 **{peak_hour} 点到 {peak_hour +1} 点** 是消息发送的高峰期，大家下班回家后的轻松时光成为了交流的黄金时段。\n\n"
    report_content += "---\n\n"

    # 活跃成员榜单
    report_content += f"## 🏆 **活跃成员榜单**\n\n"
    report_content += f"### **Top 5 活跃发言者**\n\n"
    for i, (speaker, count) in enumerate(top_speakers.head(5).items(), start=1):
        # 计算发言字数
        speaker_messages = df_year[df_year['Speaker'] == speaker]['Message']
        total_words = speaker_messages.apply(len).sum()
        # 假设一篇本科毕业论文为20,000字
        theses = total_words / 20000
        report_content += f"{i}. **{speaker}** - **{count} 条消息**, 发言字数：**{total_words} 字**，相当于 **{theses:.2f} 篇本科毕业论文**\n"
    report_content += f"*感谢这些小伙伴们的热情参与和不懈贡献，让我们的群聊如此精彩！*\n\n"
    report_content += "---\n\n"

    # 年度热门话题
    report_content += f"## 🧩 **年度热门话题**\n\n"
    top_topics = word_freq.most_common(10)
    for i, (topic, freq) in enumerate(top_topics, start=1):
        report_content += f"{i}. **{topic}** - **{freq} 次**\n"
    report_content += f"*这些话题不仅丰富了我们的聊天内容，也让大家在不同领域有所收获和成长。*\n\n"
    report_content += "---\n\n"

    # 表情与GIF使用统计
    report_content += f"## 😄 **表情使用统计**\n\n"
    if top_emojis:
        for i, (emoji_char, freq) in enumerate(top_emojis, start=1):
            report_content += f"{i}. **{emoji_char}** - **{freq} 次**\n"
    else:
        report_content += f"本年度未使用任何表情符号。\n"
    report_content += f"*表情的活跃使用，极大地提升了我们的互动体验，让每一条消息都充满了情感和乐趣。*\n\n"
    report_content += "---\n\n"

    # 情感分析
    average_sentiment = df_year['Sentiment'].mean()
    positive_messages = (df_year['Sentiment'] > 0.6).sum()
    negative_messages = (df_year['Sentiment'] < 0.4).sum()

    report_content += f"## 💬 **情感分析**\n\n"
    report_content += f"- **平均情感得分**：**{average_sentiment:.2f}**（1.0 为最积极）\n"
    report_content += f"- **积极消息**：**{positive_messages} 条**\n"
    report_content += f"- **消极消息**：**{negative_messages} 条**\n"
    report_content += f"*积极的情感氛围是我们群聊持续健康发展的关键，感谢大家的努力与包容！*\n\n"
    report_content += "---\n\n"

    # 关键时刻回顾
    key_moments(df_year)

    report_content += "---\n\n"

    # 趣味统计
    fun_stats(df_year, word_freq)

    report_content += "---\n\n"

    # 未来展望
    report_content += f"## 🔮 **未来展望**\n\n"
    report_content += f"- **增强互动**：通过更多有趣的活动和话题，进一步增强群聊的互动性和趣味性。\n"
    report_content += f"- **拓展合作**：继续推动群内合作项目，为社区和大家带来更多实实在在的益处。\n"
    report_content += f"- **提升体验**：优化群聊管理，确保交流环境更加友好和谐。\n"
    report_content += f"*让我们携手并进，迎接更加精彩和充实的新一年！*\n\n"
    report_content += "---\n\n"

    # 感谢词
    report_content += f"## 📢 **感谢有你**\n\n"
    report_content += f"感谢每一位群聊成员的参与和贡献，是你们让【{chat_name}】成为一个温暖、有趣且充满活力的大家庭。期待在未来的日子里，我们能一起创造更多美好的回忆！\n\n"
    report_content += "---\n\n"

    # 附录
    report_content += f"# 📚 **附录**\n\n"
    report_content += f"### **成员统计**\n"
    total_members = df_year['Speaker'].nunique()
    active_members = df_year['Speaker'].value_counts().head(5).index.tolist()
    report_content += f"- **总成员数**：**{total_members}** 人\n"
    report_content += f"- **活跃成员**：{', '.join(active_members)} 等\n"
    report_content += f"*成员结构的变化体现了群聊的活跃度和吸引力。*\n\n"
    report_content += "---\n\n"

    # 结束语
    report_content += f"# 🌈 **结束语**\n\n"
    report_content += f"{year}年已经成为历史，但我们在群聊【{chat_name}】中的故事还在继续。让我们一起期待，迎接更加美好和精彩的{year + 1}年！\n\n"

    # 显示报告
    st.markdown(report_content)

    # 生成PDF
    if st.button("导出为PDF"):
        pdf_buffer = generate_pdf(chat_name, year, report_content)
        if pdf_buffer:
            st.download_button(
                label="下载年度报告 PDF",
                data=pdf_buffer,
                file_name=f"{chat_name}_{year}_Annual_Report.pdf",
                mime='application/pdf'
            )

# 10. 生成年度报告
def generate_report(chat_name, year, df, word_freq, top_speakers, top_emojis):
    try:
        # 内容已经在generate_annual_report函数中生成
        pass  # 由于generate_annual_report已经处理报告内容和PDF生成，这里不需要额外操作
    except Exception as e:
        logging.error(f"生成年度报告时出错: {e}")
        st.error(f"生成年度报告时出错: {e}")

# 主函数
def main():
    st.title("🎉 **群聊年度报告生成器** 🎉")
    st.write("上传你的群聊CSV文件，生成一份生动有趣的年度报告。")

    st.sidebar.title("上传数据文件")
    uploaded_file = st.sidebar.file_uploader("选择一个CSV文件", type=["csv"])
    chat_name = st.sidebar.text_input("请输入群聊名称", "阿拉善群聊")
    year = st.sidebar.number_input("请输入报告年度", min_value=2000, max_value=2100, value=2023)

    if uploaded_file is not None:
        st.header(f"📝 **正在生成【{chat_name}】{year}年度报告**")
        stopwords = load_stopwords('stopwords.txt')
        generate_annual_report(uploaded_file, stopwords, chat_name, year)
    else:
        st.write("请在左侧上传一个CSV文件以生成年度报告。")

if __name__ == "__main__":
    main()
