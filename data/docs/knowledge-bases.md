# Amazon Bedrock Knowledge Bases

A knowledge base lets a foundation model answer questions using your own documents. Documents are stored in an S3 bucket that acts as the data source. Bedrock converts each document into embeddings and writes them to a vector store.

The Retrieve API returns the passages that best match a query, together with a relevance score and the location of the source document. The RetrieveAndGenerate API retrieves passages and also produces an answer with citations.

Ingestion must finish before a knowledge base can answer questions. A knowledge base with no indexed documents returns empty results instead of an error.
