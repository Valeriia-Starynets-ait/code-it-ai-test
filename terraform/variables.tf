variable "aws_region" {
  description = "AWS region for the Knowledge Base and its vector store."
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Prefix applied to resource names."
  type        = string
  default     = "bedrock-rag-e2e"
}

variable "ticket_key" {
  description = "Delivery ticket key, applied as the code-it:ticket cost-allocation tag."
  type        = string
  default     = "co-feat-bedrock-rag-e2e-2"
}
