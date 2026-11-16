output "knowledge_base_id" {
  description = "Set as RAG_KNOWLEDGE_BASE_ID to use the bedrock backend."
  value       = aws_bedrockagent_knowledge_base.this.id
}

output "data_source_id" {
  value = aws_bedrockagent_data_source.documents.data_source_id
}

output "documents_bucket" {
  description = "Upload the staged build/kb/data folder under the data/ prefix."
  value       = aws_s3_bucket.documents.bucket
}
