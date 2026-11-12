output "corpus_bucket" {
  description = "Name of the S3 bucket holding the source corpus."
  value       = aws_s3_bucket.corpus.bucket
}

output "kb_service_role_arn" {
  description = "ARN of the scoped Knowledge Base service role."
  value       = aws_iam_role.kb_service.arn
}
