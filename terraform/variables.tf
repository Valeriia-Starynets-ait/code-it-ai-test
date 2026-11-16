variable "name" {
  type    = string
  default = "kbrag-poc"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "account_id" {
  type        = string
  description = "AWS account id; supply at plan time, never commit it."
  default     = "000000000000"
}

variable "ticket_key" {
  type    = string
  default = "AI-1"
}

variable "embedding_model_id" {
  type        = string
  description = "Titan Text Embeddings V2; vector index dimension must match (1024 default)."
  default     = "amazon.titan-embed-text-v2:0"
}

variable "vector_collection_arn" {
  type        = string
  description = "ARN of an existing OpenSearch Serverless VECTORSEARCH collection."
  default     = "arn:aws:aoss:us-east-1:000000000000:collection/placeholder"
}

variable "vector_index_name" {
  type    = string
  default = "kbrag-index"
}
