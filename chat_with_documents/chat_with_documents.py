import streamlit as st
import os
from langchain.chat_models import ChatOpenAI  # 替换旧的OpenAI
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader  # 新增MD加载器
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# 页面标题
st.title("近代史期末复习助手（人工智能课程设计版）")
st.caption("2025级物理2班邓艾果开发")

# 1. 配置OpenAI密钥（用Streamlit Secrets更安全，部署时在Secrets里填openai_api_key）
if "openai_api_key" not in st.secrets:
    st.error("请在Streamlit部署页面的「Secrets」中添加openai_api_key！")
    st.stop()
openai_api_key = st.secrets["openai_api_key"]

# 2. 加载知识点文档（指定MD加载器，用os.path处理路径）
docs_path = os.path.join(os.path.dirname(__file__), "Documents")  # 适配部署路径
try:
    loader = DirectoryLoader(
        docs_path,
        glob="*.md",
        loader_cls=UnstructuredMarkdownLoader  # 明确指定MD文件的加载器
    )
    documents = loader.load()
    if not documents:
        st.warning("Documents文件夹中未找到.md格式的知识点文档，请确认上传！")
except Exception as e:
    st.error(f"文档加载失败：{str(e)}")
    st.stop()

# 3. 分割文档
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # 增加overlap提升上下文连贯性
texts = text_splitter.split_documents(documents)

# 4. 初始化向量库
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
db.persist()  # 持久化向量库，避免重复加载

# 5. 创建问答链（用ChatOpenAI）
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3})  # 召回3个相关片段提升准确性
)

# 6. 用户交互
user_question = st.text_input("输入知识点/真题问题（比如“洋务运动的影响”）：")
if user_question:
    with st.spinner("正在检索知识点..."):
        try:
            result = qa.run(user_question)
            st.success("查询结果：")
            st.write(result)
        except Exception as e:
            st.error(f"查询失败：{str(e)}（若提示密钥错误，请检查Streamlit Secrets配置）")
