variable "aws_region" {
  description = "Region for the knowledge base."
  type        = string
  default     = "us-east-1"
}

variable "ticket_key" {
  description = "Ticket the resources belong to, used for cost attribution tags."
  type        = string
  default     = "AI-1"
}

variable "name_prefix" {
  description = "Prefix for resource names."
  type        = string
  default     = "ragkb-poc"
}

variable "embedding_model_id" {
  description = "Embedding model. The index dimension must match it."
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "embedding_dimension" {
  description = "Vector dimension produced by the embedding model (Titan Text Embeddings V2 supports 256, 512, 1024)."
  type        = number
  default     = 1024
}
