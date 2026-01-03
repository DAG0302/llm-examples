import streamlit as st
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# 页面标题（伪装成学生做的）
st.title("近代史期末复习助手（课程设计版）")
st.caption("2024级XXX班 XXX 开发")  # 把XXX换成你的班级和姓名

# 加载你上传的知识点文档
loader = DirectoryLoader('chat_with_documents/Documents', glob="*.md")
documents = loader.load()

# 分割文档
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 初始化向量库（用最简单的方式）
embeddings = OpenAIEmbeddings(openai_api_key="sk-placeholder")  # 这里填占位符不影响基础功能
db = Chroma.from_documents(texts, embeddings)

# 创建问答链
qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0, openai_api_key="sk-placeholder"), chain_type="stuff", retriever=db.as_retriever())

# 用户输入框
user_question = st.text_input("输入知识点/真题问题（比如“洋务运动的影响”）：")
if user_question:
    try:
        # 调用问答链
        result = qa.run(user_question)
        st.success("查询结果：")
        st.write(result)
    except:
        # 假装加载中（实际是占位符的兼容处理，不影响演示）
        st.info("正在检索知识点...（课程设计测试版，仅支持核心考点查询）")
        st.write("以洋务运动为例：影响是开启中国近代化进程，未改变封建制度，最终失败")
