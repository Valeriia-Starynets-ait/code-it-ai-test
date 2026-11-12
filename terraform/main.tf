# Terraform stub for the Bedrock Knowledge Base RAG pipeline.
#
# This is a STUB: it is not applied here and provisions nothing by itself. It
# records the intended shape of the infrastructure (S3 corpus bucket, an
# OpenSearch Serverless vector store, the KB and its scoped service role) so the
# deploy is reviewable. Per the bedrock-kb-rag skill, the only approved apply
# path on a delivery box is the `code-it-apply` wrapper, never a raw
# `terraform apply`.

terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      "code-it:ticket" = var.ticket_key
      "project"        = "bedrock-rag-e2e"
    }
  }
}

# --- Corpus bucket: one prefix per source/tenant. ---------------------------
resource "aws_s3_bucket" "corpus" {
  bucket = "${var.name_prefix}-corpus"
}

resource "aws_s3_bucket_public_access_block" "corpus" {
  bucket                  = aws_s3_bucket.corpus.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "corpus" {
  bucket = aws_s3_bucket.corpus.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

# --- KB service role: scoped trust (SourceAccount + SourceArn ArnLike). ------
data "aws_caller_identity" "current" {}

resource "aws_iam_role" "kb_service" {
  name = "${var.name_prefix}-kb-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "bedrock.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
          ArnLike = {
            "aws:SourceArn" = "arn:aws:bedrock:${var.aws_region}:${data.aws_caller_identity.current.account_id}:knowledge-base/*"
          }
        }
      }
    ]
  })
}

# NOTE: The Knowledge Base, data source and OpenSearch Serverless collection are
# intentionally left as commented placeholders. Wiring them requires the
# embedding model, chunking strategy and vector-store choice recorded in
# src/rag/config.py, and an apply through `code-it-apply`.
#
# resource "aws_bedrockagent_knowledge_base" "this" { ... }
# resource "aws_bedrockagent_data_source" "this"     { ... }
# resource "aws_opensearchserverless_collection" "vectors" { ... }
