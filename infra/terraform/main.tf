# Stub: plan only. Nothing here is applied by this repository.
# Real applies go through the approved wrapper (code-it-apply), never a raw terraform apply.

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  account_id    = data.aws_caller_identity.current.account_id
  partition     = data.aws_partition.current.partition
  embedding_arn = "arn:${local.partition}:bedrock:${var.aws_region}::foundation-model/${var.embedding_model_id}"
}

# Documents staged by `ragkb ingest` (data/ plus *.metadata.json sidecars) are uploaded here.
resource "aws_s3_bucket" "documents" {
  bucket_prefix = "${var.name_prefix}-docs-"
}

resource "aws_s3_bucket_public_access_block" "documents" {
  bucket                  = aws_s3_bucket.documents.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3vectors_vector_bucket" "vectors" {
  vector_bucket_name = "${var.name_prefix}-vectors"
}

resource "aws_s3vectors_index" "chunks" {
  vector_bucket_name = aws_s3vectors_vector_bucket.vectors.vector_bucket_name
  index_name         = "chunks"
  data_type          = "float32"
  dimension          = var.embedding_dimension
  distance_metric    = "cosine"
}

data "aws_iam_policy_document" "kb_trust" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["bedrock.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:${local.partition}:bedrock:${var.aws_region}:${local.account_id}:knowledge-base/*"]
    }
  }
}

data "aws_iam_policy_document" "kb_access" {
  statement {
    actions   = ["bedrock:InvokeModel"]
    resources = [local.embedding_arn]
  }

  statement {
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.documents.arn, "${aws_s3_bucket.documents.arn}/*"]
  }

  statement {
    actions = [
      "s3vectors:GetIndex",
      "s3vectors:QueryVectors",
      "s3vectors:PutVectors",
      "s3vectors:GetVectors",
      "s3vectors:DeleteVectors",
    ]
    resources = [aws_s3vectors_index.chunks.index_arn]
  }
}

resource "aws_iam_role" "knowledge_base" {
  name_prefix        = "${var.name_prefix}-kb-"
  assume_role_policy = data.aws_iam_policy_document.kb_trust.json
}

resource "aws_iam_role_policy" "knowledge_base" {
  role   = aws_iam_role.knowledge_base.id
  policy = data.aws_iam_policy_document.kb_access.json
}

resource "aws_bedrockagent_knowledge_base" "this" {
  name     = "${var.name_prefix}-kb"
  role_arn = aws_iam_role.knowledge_base.arn

  knowledge_base_configuration {
    type = "VECTOR"

    vector_knowledge_base_configuration {
      embedding_model_arn = local.embedding_arn
    }
  }

  storage_configuration {
    type = "S3_VECTORS"

    s3_vectors_configuration {
      index_arn = aws_s3vectors_index.chunks.index_arn
    }
  }

  depends_on = [aws_iam_role_policy.knowledge_base]
}

# Chunking is done by `ragkb ingest`, so Bedrock must not re-chunk. This cannot be changed later
# without recreating the data source.
resource "aws_bedrockagent_data_source" "documents" {
  knowledge_base_id = aws_bedrockagent_knowledge_base.this.id
  name              = "${var.name_prefix}-documents"

  data_source_configuration {
    type = "S3"

    s3_configuration {
      bucket_arn         = aws_s3_bucket.documents.arn
      inclusion_prefixes = ["data/"]
    }
  }

  vector_ingestion_configuration {
    chunking_configuration {
      chunking_strategy = "NONE"
    }
  }
}
