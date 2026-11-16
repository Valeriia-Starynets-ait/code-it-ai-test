# Chunking strategy

Chunking splits a document into smaller passages before embedding. Fixed size chunking cuts text into windows of a set length with some overlap between neighbours. Overlap keeps a sentence that straddles a boundary retrievable from either side.

The chunking strategy cannot be changed after the data source is created. Changing it requires creating a new data source and ingesting the documents again. Choose the chunk size and overlap before the first ingestion.
