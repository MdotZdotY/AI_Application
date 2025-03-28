"""
NativeRAG 演示系统
Version: 1.0
"""

#系统及相关组件
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
import json
from hashlib import md5

#LangChain组件
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# 配置类
class Config:
    # 全路径配置
    BASE_DIR = Path(os.getcwd()).absolute()
    CHROMA_DB_PATH = BASE_DIR / "vector_db"
    DATA_PATH = BASE_DIR / "data"
    DEBUG_LOG_PATH = BASE_DIR / "logs/rag_demo.log"
    
    # 模型配置
    EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
    LLM_MODEL_NAME = "deepseek-r1:32b"
    API_BASE = "http://10.244.51.243:11434/v1"
    API_KEY = "EMPTY"
    
    # 处理参数
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    TOP_K = 5
    SCORE_THRESHOLD = 0.3

    # 系统提示词
    SYSTEM_PROMPT = """你是一个严格基于上下文的问答助手，请遵守以下规则：
    1. 仅使用提供的上下文回答问题
    2. 答案必须包含 [来源文件] 引用
    3. 如果上下文不相关，回答："没有查到相关内容"
    4. 禁止任何推测或假设

    上下文：
    {context}

    问题：{question}
    """

# 初始化日志
logging.basicConfig(
    filename=str(Config.DEBUG_LOG_PATH),
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


#向量库类
class VectorStoreManager:
    """向量数据库管理"""

    PROCESSED_FILE = "processed_files.json" #记录已处理文件信息
    
    @classmethod
    def create_store(cls, chunks: List[Document]) -> Chroma:
        """创建或更新向量库"""
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            cache_folder=str(Config.BASE_DIR / ".cache"),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        #初始化或加载已处理文件记录
        processed_files = cls._load_processed_files()

        #检测需要处理的文件
        new_files, changed_files = cls._detect_file_changes(processed_files)


        #如果有新增或变更文件
        if new_files or changed_files:
            if (Config.CHROMA_DB_PATH / "chroma.sqlite3").exists():
                logger.info(f"已有向量数据库存在，发现{len(new_files)}个新增文件, {len(changed_files)}个变更文件")
                #加载并分割新增文档
                new_docs = DocumentProcessor.load_specific_files(new_files + changed_files)
                new_chunks = DocumentProcessor.split_documents(new_docs)
                store = Chroma(
                    persist_directory=str(Config.CHROMA_DB_PATH),
                    embedding_function=embeddings
                )
                store.add_documents(new_chunks)
                #更新处理记录
                cls._update_processed_files(
                    processed_files, 
                    [
                        (str(fp), cls._get_file_hash(fp)) 
                        for fp in new_files + changed_files
                    ]
                )
                logger.info(f"增量更新已有向量库")

            else:#全新创建
                 store = Chroma.from_documents(
                     documents = chunks,
                     embedding=embeddings,
                     persist_directory=str(Config.CHROMA_DB_PATH)
                 )
                 logger.info(f"创建全新向量库")
                 #初始化文件记录
                 cls._update_processed_files(
                     processed_files,
                     [
                         (str(fp), cls._get_file_hash(fp)) 
                         for fp in Config.DATA_PATH.rglob("*")
                         if fp.is_file() and fp.suffix in DocumentProcessor.FILE_HANDLERS
                     ]
                 )
        
        else:
            if (Config.CHROMA_DB_PATH / "chroma.sqlite3").exists():
                store = Chroma(
                    persist_directory=str(Config.CHROMA_DB_PATH),
                    embedding_function=embeddings
                )
                logger.info(f"加载已有向量数据库")
        
        return store           

    @classmethod
    def _load_processed_files(cls) -> Dict[str, str]:
        """加载已处理文件记录"""
        record_file = Config.CHROMA_DB_PATH / cls.PROCESSED_FILE
        if record_file.exists():
            with open(record_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @classmethod
    def _update_processed_files(cls, existing: Dict[str, str], new_files:List[tuple]):
        """更新文件记录"""
        for file_path, file_hash in new_files:
            existing[str(file_path)] = file_hash
        record_file = Config.CHROMA_DB_PATH / cls.PROCESSED_FILE
        with open(record_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    @classmethod
    def _detect_file_changes(cls, processed: Dict[str, str]) -> tuple:
        """检测文件变更"""
        new_files = []
        changed_files = []

        for fp in Config.DATA_PATH.rglob("*"):
            if fp.is_file() and fp.suffix in DocumentProcessor.FILE_HANDLERS:
                str_fp = str(fp)
                current_hash = cls._get_file_hash(fp)

            if str_fp not in processed:
                new_files.append(fp)
            elif processed[str_fp] != current_hash:
                changed_files.append(fp)
        return new_files, changed_files

    @staticmethod
    def _get_file_hash(file_path: Path) -> str:
        """计算哈希值"""
        hasher = md5()
        with open(file_path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()

#支持常规文档格式的处理（加载分割）类
class DocumentProcessor:
    """文档处理管道"""
    
    FILE_HANDLERS = {
        ".md": (UnstructuredMarkdownLoader, {"encoding": "utf-8"}),
        ".pdf": (PyPDFLoader, {}),
        ".txt": (TextLoader, {"encoding": "utf-8"}),
        ".docx": (UnstructuredWordDocumentLoader, {}),
        ".xlsx": (UnstructuredExcelLoader, {}),
        ".csv": (CSVLoader, {"encoding": "utf-8"}),
    }

    @classmethod
    def load_documents(cls) -> List[Document]:
        """文档加载"""
        all_docs = []
        logger.info(f"扫描目录：{Config.DATA_PATH}")

        for ext, (loader_cls, kwargs) in cls.FILE_HANDLERS.items():
            try:
                loader = DirectoryLoader(
                    str(Config.DATA_PATH),
                    glob=f"**/*{ext}",
                    loader_cls=loader_cls,
                    loader_kwargs=kwargs,
                    silent_errors=True,
                )
                if docs := loader.load():
                    logger.info(f"加载 {ext.upper()} 文件成功：{len(docs)} 个")
                    all_docs.extend(docs)
            except Exception as e:
                logger.error(f"加载 {ext} 文件失败：{str(e)}", exc_info=True)
        return all_docs

    @classmethod
    def load_specific_files(cls, file_path:List[Path]) -> List[Document]:
        """加载指定文件"""
        all_docs = []
        for fp in file_path:
            ext = fp.suffix.lower()
            if ext not in cls.FILE_HANDLERS:
                continue

            loader_cls, kwargs = cls.FILE_HANDLERS[ext]
            try:
                loader = loader_cls(str(fp), **kwargs)
                if docs := loader.load():
                    all_docs.extend(docs)
                    logger.info(f"加载文件 {fp.name} 成功")
            except Exception as e:
                logger.error(f"加载文件 {fp.name} 失败：{str(e)}")
        return all_docs

    @staticmethod
    def split_documents(docs: List[Document]) -> List[Document]:
        """文档分割（异常处理）"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                add_start_index=True,
            )
            return splitter.split_documents(docs)
        except Exception as e:
            logger.critical(f"文档分割失败：{str(e)}", exc_info=True)
            raise

#RAG链构建器类
class RAGChainBuilder:
    """RAG 链构造器"""
    
    @staticmethod
    #@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def build_chain(vector_store: Chroma):
            
        """构建 RAG 处理链（超时控制增强）"""
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": Config.TOP_K,
                "score_threshold": Config.SCORE_THRESHOLD,
            }
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", Config.SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        llm = ChatOpenAI(
            model_name=Config.LLM_MODEL_NAME,
            openai_api_base=Config.API_BASE,  # 显式指定 API 地址
            openai_api_key=Config.API_KEY,    # 显式传递 API Key
            temperature=0.1,
            max_tokens=1024,
            request_timeout=30,
            max_retries=3
        )

        return (
            RunnableParallel({
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            })
            | prompt
            | llm
            | StrOutputParser()
        )


def format_docs(docs: List[Document]) -> str:
    """格式化检索结果（容错增强）"""
    if not docs:
        return "没有查到相关内容"
        
    try:
        return "\n\n".join(
            f"{doc.page_content}\n[来源：{Path(doc.metadata['source']).name}]"
            for doc in docs
        )
    except Exception as e:
        logger.error(f"格式化失败：{str(e)}")
        return "\n\n".join(doc.page_content for doc in docs)

        
# 主流程
def main():  
    try:

        Config.DATA_PATH.mkdir(parents=True, exist_ok=True)

        #检查数据文件是否存在
        if not any(Config.DATA_PATH.iterdir()):
            logger.error("数据目录为空，请添加文档文件")
            return

        raw_docs = DocumentProcessor.load_documents()
        if not raw_docs:
            logger.error("没有加载到任何文档，请检查文件格式")
            return
            
        chunks = DocumentProcessor.split_documents(raw_docs)
        if not chunks:
            logger.error("文档分割后没有有效内容")
            return

        #创建或更新向量库
        vector_store = VectorStoreManager.create_store(chunks)
        
        # 测试问答
        rag_chain = RAGChainBuilder.build_chain(vector_store)
        test_query = "人工智能的发展前景如何？"
        response = rag_chain.invoke(test_query)
        print(f"测试问题：{test_query}\n回答：{response}")
    except Exception as e:
        logger.critical(f"系统初始化失败：{str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()        