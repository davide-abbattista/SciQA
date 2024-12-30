# SciQA - a Scientific Question Answering System with Citations
The Scientific Question Answering (SciQA) System is an end-to-end solution designed to provide accurate, contextually relevant, and citation-supported answers to user queries. The system leverages state-of-the-art NLP techniques for document retrieval, answer generation, and consistency verification. It is built to process a [dataset](https://huggingface.co/datasets/loukritia/science-journal-for-kids-data) (from Hugging Face) of scientific abstracts and provide users with reliable answers while ensuring transparency through citations and warnings when inconsistencies are detected.

**Overview of Query and Document Processing**

When a user submits a query, the system follows a structured process to retrieve, process, and generate a response. Abstracts from the dataset are first preprocessed and embedded into a vector space for efficient similarity matching. The system then retrieves the most relevant documents, and uses these documents as context for generating a coherent answer. The entire workflow ensures that the generated answer is accurate, consistent, and traceable to its source documents.

**Dataset Preparation and Preprocessing**

The system starts by loading and inspecting the dataset using the Dataset Handler:
-	Data Loading: The dataset, provided in CSV format, is loaded into a structured DataFrame. Key columns include the original abstracts from academic papers and their simplified “Kids Abstracts.”
-	Inspection: The system checks for missing values to ensure completeness and analyzes the token lengths of abstracts to verify compatibility with the model's token limits.
-	Summary: A high-level overview of the dataset, including data types and sample entries, is displayed.

**Embedding Creation and Similarity Search**

To efficiently retrieve relevant documents, the Embedding Indexer encodes the dataset into dense vector embeddings:
-	Embedding Model: The [multi-qa-mpnet-base-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1) model is used to generate embeddings. This model is specifically designed for semantic search. It has been trained on question-answer pairs from diverse sources and ensures high-quality semantic representation of textual data.
-	Indexing: The embeddings are stored in a [FAISS](https://faiss.ai/) index, a fast and scalable library for nearest neighbor search. The system uses cosine similarity as the metric to retrieve the most relevant documents for a given query.
-	Query Matching: When a query is submitted, it is also encoded into a dense vector. This query embedding is then matched against the FAISS index to retrieve the top-k documents that are semantically closest to the query.

The use of dense embeddings and FAISS ensures scalability, making the system capable of handling large datasets efficiently.

**Query Answering and Consistency Verification**

Once relevant documents are retrieved, the Query Answering module processes the query and documents to generate a contextually appropriate answer:

1. Answer Generation:
 -	A zero-shot prompting approach is used with the [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) Seq2Seq model to generate a detailed answer. The context for this model includes the abstracts of the retrieved documents and the query itself.
 - The model excels at generating coherent, human-readable answers, leveraging its pre-trained capabilities on a variety of NLP tasks.

2. Consistency Verification:
 -	To ensure the generated answer is aligned with the source documents, the system calculates semantic similarity between the generated answer and potential answer spans extracted from the retrieved documents.
 -	The [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2) model identifies potential answer spans, and the [multi-qa-mpnet-base-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1) model computes their embeddings.
 -	The similarity between the generated answer's embedding and each span's embedding is evaluated. If at least one similarity score meets or exceeds a predefined threshold, the answer is considered consistent.
 - The retrieved documents associated to the similarity scores above the threshold are used as references in the system's answer.

3. Citations and Warnings:
 -	Citations for the documents that meet the consistency check, used in generating the answer, are appended to provide transparency and traceability.
 -	If the consistency check fails, the system flags the answer with a warning, alerting the user to potential inaccuracies.

**System Integration and Workflow**

The Scientific QA System orchestrates the entire process, combining the dataset preparation, embedding indexing, and query answering components:

1. Initialization:
 - The dataset is loaded and preprocessed.
 - 	Dense vector embeddings are created for all abstracts, and a FAISS index is built for similarity search.
 -	The query-answering model and consistency-checking mechanisms are initialized.

2.	Query Handling:
 -	The user's query is encoded into an embedding and matched against the FAISS index to retrieve the most relevant documents.
 -	The retrieved documents are used to generate an answer, which is then validated for consistency.

3. Answer Delivery:
 -	The system returns the generated answer, along with citations for the source documents that meet the consistency check.
 -	If the consistency check fails, the system includes a warning.

**Justification of Implementation Choices**
-	Dense Embeddings and FAISS: Using dense embeddings ensures robust semantic matching, while FAISS provides a scalable solution for similarity search, enabling the system to handle large datasets efficiently.
-	Zero-Shot Prompting: The [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) model is leveraged for its strong generalization capabilities, ensuring high-quality answers without task-specific fine-tuning.
-	Consistency Checks: The consistency verification mechanism enhances the reliability of the system, ensuring that the answers align with the source documents.
-	Transparency: Citations and warnings provide users with the necessary context to evaluate the trustworthiness of the generated answers.
