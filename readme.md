## RAG Based Document Question Answering Application

### Description:
DocQnA is a robust document question answering (QA) application built using retrieval augmented generation (RAG) technology. This application enables users to ask queries related to their documents, which can be in various formats such as txt, docx, and pdf, without requiring any chat history awareness. Powered by Microsoft's 4-bit quantized Phi-3 model from Hugging Face and the Langchain framework, DocQnA efficiently processes user queries and retrieves accurate answers from the provided documents. With its user-friendly interface and powerful functionality, DocQnA simplifies the process of extracting information from textual documents, enhancing productivity and accessibility.

## Setup Instructions

### 1. Virtual Environment Creation:
1. Create a virtual environment using Conda with Python version 3.10 or higher.
2. Navigate to the directory where you want to create the virtual environment and open the command line.
3. Execute the following command to create the virtual environment:
    ```
    conda create -p myvenv python=3.10 -y
    ```
4. Activate your virtual environment using the command:
    ```
    conda activate path\to\myvenv
    ```

### 2. Local PyTorch GPU Drivers Installation:
1. Check the CUDA version of your hardware by running the command:
    ```
    nvcc --version
    ```
2. Install PyTorch drivers based on your CUDA version. Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) page, select your hardware configurations, and generate the installation command.
3. For example, if your GPU CUDA hardware version is 12.0, use the generated installation command like:
    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    ```
4. Run 'cuda_test.py' to verify the availability of CUDA:
    ```
    python path\to\cuda_test.py
    ```
5. Upon successful installation of local PyTorch drivers, you'll receive the message 'CUDA is available.' Otherwise, you can run the application on your local CPU.

### 3. Install Dependencies:
1. Install all necessary libraries by executing the command:
    ```
    pip install langchain langchain-core langchain-community sentence-transformers llama-cpp-python pymupdf streamlit
    ```
2. Install the FAISS library according to the GPU or CPU version:
    - GPU version:
    ```
    conda install -c conda-forge faiss-gpu -y
    ```
    - CPU version:
    ```
    pip install faiss-cpu
    ```
    (In this example, faiss-gpu was used.)
3. Download the model from the provided link and place it in the 'models' folder in your working directory. Model download link: [Model Download Link](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/blob/main/Phi-3-mini-4k-instruct-q4.gguf)
4. Place your text data in the 'data' folder. (Example: text data of docx, pdf, and txt formats)
5. Run your AI application using the following command:
    ```
    streamlit run app_ui.py
    ```
    or you can test the results of your AI application in the command line prompt by running:
    ```
    python app_cmd.py
    ```
