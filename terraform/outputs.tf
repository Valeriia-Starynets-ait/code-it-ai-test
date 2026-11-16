output "knowledge_base_id" {
  value = aws_bedrockagent_knowledge_base.this.id
}

output "data_source_bucket" {
  value = aws_s3_bucket.docs.bucket
}
