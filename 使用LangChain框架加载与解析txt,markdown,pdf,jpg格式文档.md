## 1. LangChain 中的文档加载

大型语言模型（LLMs）存在数据实时性的问题。即使是像 GPT-4 这样强大的模型也对最近的事件一无所知。

在 LLMs 看来，世界是静止的。它们只知道通过它们的训练数据所呈现的世界。因此我们可以将最新的一些知识文档添加到LLMs中，来补充LLMs模型的知识。

在Langchain 中的通过提示文档加载类（document_loaders）来实现文档的加载，本文将详细介绍如何通过document_loaders实现txt、markdown、pdf、jpg格式文档的加载。

## 2. 加载文档
langchain提供了很多文档加载的类，以便进行不同的文件加载，这些类都通过 langchain.document_loaders 引入。

例如：UnstructuredFileLoader（txt文件读取）、UnstructuredFileLoader（word文件读取）、MarkdownTextSplitter（markdown文件读取）、UnstructuredPDFLoader（PDF文件读取）

本文准备了四种格式的文件进行加载测试，文件默认放在docs目录下，大家也可以直接打开查看。

### 2.1 导入对应的langchain库

```python
from langchain.text_splitter import CharacterTextSplitter,MarkdownTextSplitter
from langchain.document_loaders import UnstructuredFileLoader,UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import UnstructuredImageLoader
from rapidocr_onnxruntime import RapidOCR
```
这里的RapidOCR是专门针对图像格式文档加载处理的，要使用下列命令下载对应的依赖：

```python
%pip install rapidocr_onnxruntime pdf2image pdfminer.six -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### 2.2 加载文档
#### 加载txt文档

```python
#加载txt文件
def load_txt_file(txt_file):    
    loader = UnstructuredFileLoader(os.path.join(work_dir, txt_file))
    docs = loader.load()
    print(docs[0].page_content[:100])
    return docs
```
#### 加载md文档

```python
#加载md文件
def load_md_file(md_file):    
    loader = UnstructuredMarkdownLoader(os.path.join(work_dir, md_file))
    docs = loader.load()
    print(docs[0].page_content[:100])
    return docs
```
#### 加载pdf文档

```python
#加载pdf文件
def load_pdf_file(pdf_file):    
    loader = UnstructuredPDFLoader(os.path.join(work_dir, pdf_file))
    docs = loader.load()
    print('pdf:\n',docs[0].page_content[:100])
    return docs
```
#### 加载jpg文档

```python
#加载jpg文件
def load_jpg_file(jpg_file):
    ocr = RapidOCR()
    result,_ = ocr(os.path.join(work_dir,jpg_file))
    docs = ""
    if result:
        ocr_result = [line[1] for line in result]
        docs += "\n".join(ocr_result)
        print('jpg:\n',docs[:100])
    return docs
```
#### 从docs_path路径加载文件

```python
#从docs_path路径加载文件
for doc in os.listdir(docs_path):
    doc_path = f'{docs_path}/{doc}'
    if doc_path.endswith('.txt'):
        load_txt_file(doc_path)
    elif doc_path.endswith('.md'):
        load_md_file(doc_path)
    elif doc_path.endswith('.pdf'):
        load_pdf_file(doc_path)
    elif doc_path.endswith('.jpg'):
        load_jpg_file(doc_path)
```

#### 结果展示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/2b379d69efba408fbd4103d1410c8299.png)
### 2.3 文档分割
对于大语言模型，往往单次传入的token长度是有限的。因此在加载完成后，还需要对文件进行分割，这样才能更准确的被模型所理解。

分割默认有两个关键参数：chunk_size：每个分割段的最大长度；chunk_overlap：相邻两个分割段之间的重叠token数量。这两个参数可以根据实际需要来配置。
#### 分割txt文件

```python
#分割txt文件
def load_txt_splitter(txt_file, chunk_size=200, chunk_overlap=20):
    docs = load_txt_file(txt_file)
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    #默认展示分割后第一段内容
    print('split_docs[0]: ', split_docs[0])
    return split_docs
```
#### 分割md文件

```python
#分割md文件
def load_md_splitter(md_file, chunk_size=200, chunk_overlap=20):
    docs = load_md_file(md_file)
    text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    #默认展示分割后第一段内容
    print('split_docs[0]: ', split_docs[0])
    return split_docs
```
#### 分割pdf文件

```python
#分割pdf文件
def load_pdf_splitter(pdf_file, chunk_size=200, chunk_overlap=20):
    docs = load_pdf_file(pdf_file)
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    #默认展示分割后第一段内容
    print('split_docs[0]: ', split_docs[0])
    return split_docs
```
#### 分割jpg文件

```python
#分割jpg文件
def load_jpg_splitter(jpg_file, chunk_size=200, chunk_overlap=20):
    docs = load_jpg_file(jpg_file)
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.create_documents([docs])
    #默认展示分割后第一段内容
    print('split_docs[0]: ', split_docs[0])
    return split_docs
```
#### 从docs_path目录读取文件并进行分割

```python
for doc in os.listdir(docs_path):
    doc_path = f'{docs_path}/{doc}'
    if doc_path.endswith('.txt'):
        load_txt_splitter(doc_path)
    elif doc_path.endswith('.md'):
        load_md_splitter(doc_path)
    elif doc_path.endswith('.pdf'):
        load_pdf_splitter(doc_path)
    elif doc_path.endswith('.jpg'):
        load_jpg_splitter(doc_path)
```
#### 效果展示
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/297d8a51dd6c4b4a8730d86e4b2f68ce.png)

### 总结
使用LangChain库进行文档加载，对于txt,md,pdf格式的文档，都可以用LangChain类加载，UnstructuredFileLoader（txt文件读取）、UnstructuredFileLoader（word文件读取）、MarkdownTextSplitter（markdown文件读取）、UnstructuredPDFLoader（PDF文件读取），对于jpg格式的文档，我这里提供了一种思路。

源代码文件地址：[document_loader.ipynb](https://github.com/STRUGGLE1999/LangChain-ChatGLM3/blob/main/document_loader.ipynb)

